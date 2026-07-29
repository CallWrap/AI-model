# CallWrap — AI 분석

> 상담이 끝나면, 기록은 AI가

상담 통화 전사 텍스트를 받아 후처리 폼(요약·유형·분류·등급·후속처리)을 채운다.
전체 파이프라인의 **③ AI 분석** 단계이며, 외부에 노출하는 것은 함수 하나다.

```python
from callwrap_ai import analyze

result = analyze(transcript)   # Transcript -> AnalysisResult
```

**stateless** — 텍스트를 받아 결과만 돌려주고 아무것도 저장하거나 기억하지 않는다.
저장·큐잉·재시도는 `server-fastapi` 가 맡는다.

---

## 시작하기

```bash
uv sync                  # 기본 (pydantic만)
uv sync --extra train    # 파인튜닝까지 (torch·transformers·sklearn)
```

```bash
cp .env.example .env     # 토큰은 각자 발급해 채운다
```

```bash
uv run pytest tests/ -q
```

`.env` 의 `CLAUDE_CODE_OAUTH_TOKEN` 은 `claude setup-token` 으로 발급한다. 개인 구독
한도로 동작하며 별도 API 비용이 없다. 파인튜닝은 로컬 MPS라 토큰이 필요 없다.

## 구조

```
src/callwrap_ai/      패키지 — 서비스 산출물
  schemas.py            입출력 계약 (4개 레포 공용, 변경 시 전원 리뷰)
  pipeline.py           analyze() — 모듈 조립
  llm/                  LLM 백엔드 추상화 (Anthropic ↔ Ollama)
eval/                 평가 도구 — 시나리오 생성·변환·라벨링·채점
train/                파인튜닝 — 분할·전처리·학습
docs/                 라벨링 기준·설계 결정
data/                 로컬 전용 (.gitignore). 골든셋·모델 가중치
tests/                계약 테스트
```

`data/` 는 커밋하지 않는다. **재현은 코드로 한다.**

## 기능

| ID | 기능 |
|---|---|
| F1 | 상담 요약 — 요청제목·요청내용·처리내용 |
| F2 | 상담유형(8종)·상담분류(중6·소29) 분류 |
| F3 | 상담등급 판정·악성콜 감지 |
| F4 | 후속처리 판정 |
| — | FAQ 필요 판단, 캘리브레이션, PII 재탐지 |

각 항목은 `Judgment` 구조로 **값 + confidence + 근거 문장 인덱스**를 함께 반환한다.
`evidence_indices` 는 발화의 `index` 리스트이며, 웹 UI가 이걸로 원문을 하이라이트한다.

## 재현

**F2 중분류 파인튜닝**

```bash
uv run --extra train python train/split_data.py       # 골든셋 100건 → 학습 80 / 검증 20
uv run --extra train python train/train_category.py   # 학습·채점·저장
```

**파이프라인 채점**

```bash
uv run python eval/run.py --task all
```

> ⚠️ 현재 지표(중분류 Macro-F1 0.46)는 **재현 데이터 기준 내부 값이다.
> 대외 발표·보고서 인용 금지.** 파이프라인이 동작함을 확인하는 용도이며,
> 서비스 품질의 근거로 쓰지 않는다. (명세 9.1)

설정 근거와 클래스별 지표는 [train/README.md](train/README.md) 참고.

## 데이터 전략

기관 데이터는 개인정보 특성상 반출되지 않는다. 실데이터 학습은 확장 영역으로 분리하고,
아래 구조로 간다. (2026-07-29 3자 회의 확정)

| 단계 | 데이터 | 역할 |
|---|---|---|
| 생성 | 기관 FAQ → 상담 시나리오 재구성 → TTS 합성 | 학습·개발 |
| 검증 | 공개 실음성 데이터 | 파이프라인 적합성 입증 |
| 확장 | 기관 가명처리 실데이터 (확보 시) | 동일 파이프라인으로 재학습 |

재현 데이터의 품질 자체가 성과물이다. 아나운서 톤에 머물지 않고 실제 상담의
구어체·중복·끊김을 반영해야 검증 단계가 의미를 갖는다.

## 문서

| 문서 | 내용 |
|---|---|
| [docs/label_spec.md](docs/label_spec.md) | 상담유형·상담분류 판정 기준과 경계 케이스. 라벨링·평가의 전제 |
| [docs/decisions.md](docs/decisions.md) | 설계 결정과 근거 |
| [train/README.md](train/README.md) | 파인튜닝 결과·재현 절차·설정 근거 |

## 다른 레포와의 관계

| 레포 | 관계 |
|---|---|
| `CallWrap/server-fastapi` | 이 패키지를 의존한다. Celery 태스크에서 `analyze()` 호출 |
| `CallWrap/server-spring` | 업무 서버 |
| `CallWrap/web` | 상담사 검토 UI·대시보드. `confidence` 로 3-tier 렌더링 |

`schemas.py` 는 네 레포의 단일 출처다. 필드·enum을 바꾸면 전원에게 영향이 간다.

## 규칙

- `main` 직접 push 금지 — PR + 승인 1명
- 브랜치 `feat/#{이슈번호}-{설명}`, PR 본문에 `closes #N`
- **음성 파일과 기관 상담 데이터는 절대 커밋 금지.** 가명처리본도 마찬가지다
- API 키는 코드에 쓰지 않고 환경변수로만 접근한다
- 모델 비교는 seed 3개 이상의 평균·편차로 판단한다. 단일 실행으로 결론 내지 않는다

## 환경

Python 3.11 · uv · MacBook (MPS, fp32). MPS에서 fp16은 불안정하다.
