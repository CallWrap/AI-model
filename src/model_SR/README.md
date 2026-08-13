# CallWrap — 상담 SR유형 분류 모델 (B안)

STT + 화자분리된 상담 통화 전문을 입력받아 **SR유형 8종**을 분류한다.
상담사의 상담 후처리를 자동화하는 것이 목적이며, 분류 결과는 RabbitMQ Job의
처리 결과로 Spring 백엔드에 반환된다.

> 업무 대/중/소분류(A안)와 상담등급은 **본 저장소 범위 밖**이다. 별도 과제로 진행한다.

---

## 라벨 체계

| SR유형 | 적기처리기간 | 우선순위 |
|---|---:|:---:|
| 시스템오류 | 30일 | 1 |
| 기능개선 | 30일 | 2 |
| 자료보정 | 5일 | 3 |
| 정책 및 지침문의 | 5일 | 4 |
| 고객PC환경 | 3일 | 5 |
| 자료확인 | 2일 | 6 |
| 사용법문의 | 1일 | 7 |
| 기타 | 3일 | 8 (fallback) |

참고자료 규칙: **“SR유형 설정은 적기처리기간이 긴 항목을 우선하여 설정”**

이 한 줄이 설계를 결정한다. 한 통화에 여러 유형이 섞이는 것이 정상이므로
**멀티라벨(8-way sigmoid)로 학습하고, 추론 시 우선순위 규칙으로 축약**한다.
규칙은 코드로 100% 재현되므로 적기처리기간이 개정되어도 재학습 없이 후처리만
고치면 된다.

판정 기준·혼동쌍 규칙 전문: [`docs/label_definition.md`](docs/label_definition.md)

---

## 환경

- Windows 11 / Python 3.11.9
- GPU: RTX 4080 SUPER 16GiB (bf16 네이티브 지원)
- torch 2.6.0+cu124, transformers 4.57

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

모든 실행은 `PYTHONPATH=src` 를 붙인다.

**VRAM 실측** (`klue/roberta-large`, seq 512, bf16 AMP):

| batch | peak VRAM |
|---:|---:|
| 8 | 8.20 GiB |
| 16 | 12.05 GiB (기본값) |

---

## 파이프라인

```
D0  라벨 정의서 ─────────────┐
                            ├─> P2 파일럿 생성 ─[게이트: 정합성 95%]─> P3 본 생성
P1-a FAQ 코퍼스 수집 ────────┘                                          │
     └─> P4-a DAPT (MLM)                                               │
P1-b STT 노이즈 증강기 ───────────────────────────────────────────────┤
                                                                       ▼
                                          P3-b TF-IDF 베이스라인 [품질 진단 게이트]
                                                       │
                                                       ▼
                                          P4-b 파인튜닝 (PyTorch)
                                                       │
                                                       ▼
                                          P5 임계값 캘리브레이션 + 운영점
```

### D0 — 라벨 정의서 ✅

`docs/label_definition.md`. 8종 정의, 우선순위 tie-break 확정, 혼동쌍 5종
판정 규칙, 검수 게이트 기준.

### P1-a — 도메인 코퍼스 수집 ✅

```bash
PYTHONPATH=src ./.venv/Scripts/python.exe -m callwrap.collect --all
PYTHONPATH=src ./.venv/Scripts/python.exe -m callwrap.collect \
    --pdf "장애인 활동지원제도 100문 100답_최종.pdf"
```

산출물이 두 갈래다.

| 경로 | 형식 | 용도 |
|---|---|---|
| `data/faq/*.jsonl` | `{question, answer, category, source, url}` | P2/P3 시나리오의 **사실 근거** |
| `data/corpus/*.txt` | 한 줄 = 한 문단 | P4-a **DAPT 코퍼스** |

HTTP 캐시(`.cache/http/`)가 있어 파서를 고쳐가며 재실행해도 서버를 다시 때리지 않는다.
요청 간 1초 지연, 3회 재시도.

**수집 실적** (2026-07-28):

| 출처 | 건수 | robots.txt |
|---|---:|---|
| 사회서비스 전자바우처 FAQ | 157 | 해당 경로 허용 |
| 장애인활동지원 자주묻는 질문 | 29 | robots.txt 없음 |
| 세종시지역사회서비스지원단 FAQ | 15 | `Allow:/` |
| **합계** | **201** | 총 707문단 / 43,453자 |

> ⚠️ **이 코퍼스만으로는 DAPT에 부족하다.** 43K자는 MLM 도메인 적응에 필요한
> 양(수 MB)에 크게 못 미친다. 실제로 참고자료 소분류의 핵심 용어가 코퍼스에
> 아예 없다 — `가산단가` 0회, `과오결제` 0회, `결제단말기` 0회, `자격변경` 0회,
> `예탁금` 2회. **100문100답 PDF를 확보하기 전에는 P4-a를 건너뛰고**
> `klue/roberta-large`를 그대로 백본으로 쓰는 편이 낫다.
> 자세한 판단 근거는 아래 "남은 작업" 참조.

### P1-b — STT 노이즈 증강기 ✅

```bash
PYTHONPATH=src ./.venv/Scripts/python.exe -m callwrap.stt_noise   # 데모
```

학습은 깨끗한 생성 텍스트, 추론은 STT 출력. 이 skew를 방치하면 실서비스에서
무너진다. 문장부호 제거, filler 삽입, 자기수정 반복, 도메인 용어 오인식(사전 24종),
숫자 표기 흔들림, 조사 탈락, 띄어쓰기 붕괴, 발화 분절, **diarization 화자 뒤집힘**까지
시뮬레이션한다. 학습 시 `--noise-prob 0.5`로 매 에폭 새로 샘플링된다.

### P2 — 파일럿 생성 (게이트)

```bash
export ANTHROPIC_API_KEY=...
PYTHONPATH=src ./.venv/Scripts/python.exe -m callwrap.generate \
    --out data/pilot.jsonl --limit 200 --workers 8
```

**라벨을 먼저 정하고 대화를 생성**한다(label-first). 라벨 노이즈가 원천적으로
줄고 커버리지가 격자로 관리된다. 격자 = `SR유형(8) × 업무소분류(29) × 난이도(일반/경계/복합)`.
혼동쌍 5종은 `경계` 난이도로 별도 물량을 배정한다.

> **여기가 유일한 하드 게이트다.** 사람이 200건을 정의서 §3·§4 기준으로 재라벨링해
> 최종 유형 일치율 **95% 미만**이면 정의서/프롬프트를 고치고 재실행한다.
> 라벨이 흔들린 채 본 생성으로 넘어가면 이후 전부 재작업이다.

### P3-a — 본 생성

```bash
PYTHONPATH=src ./.venv/Scripts/python.exe -m callwrap.generate \
    --out data/synth_raw.jsonl --per-cell 3 --workers 8
```

시스템 프롬프트가 요청 간 바이트 단위로 고정이라 프롬프트 캐시가 누적된다
(실행 종료 시 히트율이 출력된다).

### P3-b — 베이스라인 (품질 진단 게이트)

```bash
PYTHONPATH=src ./.venv/Scripts/python.exe -m callwrap.baseline --data data/synth_raw.jsonl
```

**성능 하한 측정용이 아니라 데이터 품질 진단용이다.** TF-IDF char n-gram +
LogisticRegression이 합성 검증셋에서 macro-F1 **0.90을 넘으면 나쁜 신호**다.
LLM이 유형별 정형 표현을 반복 생성했다는 뜻이고, 모델은 그 키워드만 외운다.
실제 통화에서는 그런 말을 안 하므로 현장에서 무너진다.

게이트 초과 시 클래스별 최고 가중치 n-gram을 자동 출력하므로 누설 표현을 눈으로 찾을 수 있다.

### P4-a — DAPT (도메인 적응 사전학습)

```bash
PYTHONPATH=src ./.venv/Scripts/python.exe -m callwrap.dapt \
    --corpus data/corpus --out runs/dapt-roberta-large --epochs 3
```

`바우처`, `본인부담금`, `가산단가`, `행복e음`, `예탁금` 같은 도메인 어휘 표현이 잡힌다.
데이터가 적은 상황에서 효과가 가장 확실한 투자다.
**산출 체크포인트는 A안 팀과 공유할 것** — 입력 텍스트가 같으므로 백본 하나를 같이 쓰면 된다.

### P4-b — 파인튜닝

```bash
PYTHONPATH=src ./.venv/Scripts/python.exe -m callwrap.train \
    --data data/synth_raw.jsonl \
    --val-data data/real_holdout.jsonl \
    --backbone runs/dapt-roberta-large \
    --out runs/v1 --epochs 8 --noise-prob 0.5
```

- 8-way sigmoid + `BCEWithLogitsLoss(pos_weight=우선순위가중 × 클래스불균형보정)`
- bf16 autocast (Ada 네이티브 → GradScaler 불필요), TF32 matmul
- AdamW + layer-wise LR decay(0.9) + linear warmup 10%
- head+tail 절단, 화자 스페셜 토큰 + 학습형 화자 임베딩(0 초기화 = 항등 시작)
- 조기종료: 검증 macro-F1

#### 진행상황 확인

에폭이 끝날 때마다 **즉시 디스크에 기록**한다. 중단되어도 그 시점까지의 기록과
실패 사유가 남고, 학습 중에도 다른 터미널에서 확인할 수 있다.

| 파일 | 내용 |
|---|---|
| `progress.jsonl` | 에폭당 1줄. loss / macro-F1 / SLA위반율 / 클래스별 F1 / lr / 소요시간 / peak VRAM. append-only |
| `status.json` | 현재 상태 스냅샷 (`running` / `done` / `failed` / `stopped`), best, ETA |
| `train.log` | 콘솔 출력 전문 |

```bash
# 학습 중 다른 터미널에서
PYTHONPATH=src ./.venv/Scripts/python.exe scripts/watch_progress.py --run runs/v1
PYTHONPATH=src ./.venv/Scripts/python.exe scripts/watch_progress.py --run runs/v1 --follow
PYTHONPATH=src ./.venv/Scripts/python.exe scripts/watch_progress.py --all   # runs/ 전체 요약
```

```
✔ runs/v1  [done]  epoch 3/3  경과 0:06
  backbone=klue/roberta-large | device=cuda | n_train=364 | n_val=36 | batch_size=16

 epoch      loss  macro-F1     SLA위반     acc       시간    VRAM
-------------------------------------------------------------------
     1    1.5007    0.0407    0.0000   0.194     0:02     2.6  <- best
     2    0.8694    0.6343    0.0000   0.722     0:02     2.6  <- best
     3    0.4242    0.8057    0.0000   0.833     0:02     2.5  <- best
```

DAPT(`callwrap.dapt`)도 같은 형식으로 loss/perplexity를 기록한다.

### P5 — 임계값 캘리브레이션

```bash
PYTHONPATH=src ./.venv/Scripts/python.exe -m callwrap.calibrate \
    --run runs/v1 --sla-penalty 0.5 --target-accuracy 0.90
```

목적함수는 단순 macro-F1이 아니라 비대칭 오류비용을 반영한다:

```
objective = macro_F1 − sla_penalty × SLA위반율
```

우선순위 상위 클래스(시스템오류/기능개선)는 임계값이 낮게 잡히는 것이 정상이다.
놓치면 SLA를 위반하기 때문이다. 실제 클래스 분포를 알면 `--prior-file`로 중요도
가중을 준다. 임계값과 운영점은 `runs/v1/config.json`에 저장된다.

---

## 스모크 테스트 (API 키 불필요)

```bash
PYTHONPATH=src ./.venv/Scripts/python.exe scripts/make_dummy_data.py --out data/dummy.jsonl -n 400
PYTHONPATH=src ./.venv/Scripts/python.exe -m callwrap.baseline --data data/dummy.jsonl
PYTHONPATH=src ./.venv/Scripts/python.exe -m callwrap.train --data data/dummy.jsonl \
    --backbone klue/roberta-base --out runs/smoke --epochs 1
PYTHONPATH=src ./.venv/Scripts/python.exe -m callwrap.calibrate --run runs/smoke
```

배선 확인용이다. 더미는 템플릿 조합이라 모델 품질과 무관하며, 베이스라인이
품질 게이트 경고를 띄우는 것이 정상 동작이다.

---

## 평가 지표

| 지표 | 역할 |
|---|---|
| **macro-F1** | 주 지표. accuracy는 쓰지 않는다 (클래스 불균형) |
| **SLA 위반율** | 부 지표. 실제가 장기유형인데 단기유형으로 예측한 비율 = 실제 손해 |
| 8×8 혼동행렬 | 우선순위 순 정렬. 시스템오류↔고객PC환경 셀을 특히 볼 것 |
| 커버리지-정확도 곡선 | 운영점 선정 (“커버리지 70%에서 정확도 92%”) |
| Top-3 정확도 | 후처리 UI가 Top-3를 제시할 때의 체감 정확도 |

---

## 남은 작업

| 항목 | 상태 |
|---|---|
| 실데이터 테스트셋 200~300건 | **가명처리 대기 중** — 확보 즉시 `--val-data`로 교체. 합성 검증셋 지표는 낙관적이다 |
| 100문100답 PDF | **미확보**. 메일 첨부본이며 공개 URL이 없다. 현재 코퍼스에 `가산단가`·`과오결제`·`결제단말기`·`자격변경`이 0회라 DAPT 효과를 기대하기 어렵다 |
| `ANTHROPIC_API_KEY` | **미설정**. P2/P3 합성 데이터 생성이 막혀 있다 |
| P6 서빙 (FastAPI worker + RabbitMQ) | 범위 밖. `runs/*/config.json`의 임계값과 `collapse_probs()`를 그대로 쓰면 된다 |
| 능동학습 루프 | P6 이후. 상담사 확정값을 PostgreSQL에 적재 → 주기 재학습 |

---

## 구조

```
model/
├── docs/label_definition.md      # D0 — 라벨 정의서 (모든 단계의 기준)
├── scripts/
│   ├── make_dummy_data.py        # 스모크 테스트용 더미 생성기
│   └── watch_progress.py         # 학습 진행상황 조회 (--follow / --all)
└── src/callwrap/
    ├── labels.py        # 8종 라벨, 우선순위 축약, pos_weight, SLA 위반 판정
    ├── collect.py       # P1-a     FAQ 3종 + PDF 수집 (캐시/지연/재시도)
    ├── prompts.py       # 생성 격자, 시스템/유저 프롬프트
    ├── generate.py      # P2/P3-a  라벨 조건부 생성 (구조화 출력)
    ├── stt_noise.py     # P1-b     STT 노이즈 증강기
    ├── dataset.py       # 전처리, head+tail 절단, 화자 인코딩, Dataset/Collator
    ├── model.py         # SRClassifier (백본 + 화자 임베딩 + 멀티라벨 헤드)
    ├── dapt.py          # P4-a     도메인 적응 MLM
    ├── train.py         # P4-b     PyTorch 학습 루프
    ├── calibrate.py     # P5       임계값 최적화 + 운영점
    ├── evaluate.py      # 지표 (macro-F1, SLA 위반율, 혼동행렬, 커버리지 곡선)
    └── progress.py      # 에폭별 진행상황 기록 (progress.jsonl / status.json)
```
