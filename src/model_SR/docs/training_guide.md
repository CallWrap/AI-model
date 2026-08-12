# 학습 실행 가이드

**본 문서는 PowerShell 기준이다.** 모든 명령은 `c:\CallWrap\model` 에서 실행한다.

## 0. 세션 준비 (터미널을 새로 열 때마다 한 번)

```powershell
cd c:\CallWrap\model
.\.venv\Scripts\Activate.ps1        # 프롬프트에 (.venv) 가 붙는다
$env:PYTHONPATH = "src"
$env:PYTHONIOENCODING = "utf-8"
```

> ⚠️ **PowerShell에는 `PYTHONPATH=src python ...` 같은 인라인 환경변수 문법이 없다.**
> 그건 bash 전용이며, PowerShell은 `PYTHONPATH=src` 를 명령어 이름으로 해석해
> `CommandNotFoundException` 을 낸다. 위처럼 `$env:` 로 먼저 설정해야 한다.
>
> 가상환경이 활성화되어 있으면 `.\.venv\Scripts\python.exe` 대신 그냥 `python`
> 을 쓰면 된다. 아래 명령들은 모두 그 상태를 전제한다.

### 동작 확인

```powershell
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`2.6.0+cu124 True NVIDIA GeForce RTX 4080 SUPER` 가 나와야 한다.

<details>
<summary>Git Bash를 쓰는 경우</summary>

```bash
cd /c/CallWrap/model
export PYTHONPATH=src PYTHONIOENCODING=utf-8
./.venv/Scripts/python.exe -c "import torch;print(torch.cuda.is_available())"
```
</details>

---

## 1. 데이터 준비

```powershell
python scripts\build_dataset.py --out data\train.jsonl --val-out data\val.jsonl
```

확인할 것:
- **라벨 8종이 모두 존재**하는가 (0건인 라벨이 있으면 학습이 그 클래스를 못 배운다)
- `sr_final 재계산으로 교정된 건` 이 많으면 생성기가 우선순위 규칙을 잘못 적용한 것
- 소분류 커버리지가 20종 이상인가

---

## 2. 품질 게이트 (학습 전 필수)

```powershell
python -m callwrap.baseline `
    --data data\train.jsonl --val-data data\val.jsonl `
    --emit-banlist configs\banned_phrases.json
```

> PowerShell의 줄바꿈 연결 문자는 백틱(`` ` ``)이다. bash의 `\` 가 아니다.
> 헷갈리면 그냥 한 줄로 이어 쓰면 된다.

**판정 기준**

| macro-F1 | 의미 | 조치 |
|---|---|---|
| **> 0.90** | 표현 다양성 부족. 모델이 정형 문구를 외운다 | **학습 금지.** 금지어를 갱신하고 재생성 |
| 0.55 ~ 0.90 | 정상 범위 | 학습 진행 |
| < 0.55 | 라벨 노이즈 또는 데이터 부족 의심 | 라벨 정합성 점검 |

게이트 초과 시 출력되는 `클래스별 최고 가중치 n-gram` 을 보면 어떤 표현이
누설되는지 바로 보인다.

---

## 3. 학습

```powershell
python -m callwrap.train `
    --data data\train.jsonl `
    --val-data data\val.jsonl `
    --backbone klue/roberta-large `
    --out runs\v1 `
    --epochs 10 --batch-size 16 --lr 1e-5 --noise-prob 0.5 --patience 4
```

첫 실행은 백본 다운로드(약 1.3GB)로 몇 분 걸린다.

### 데이터가 적을 때 (300건 미만)

`roberta-large` 는 과적합하기 쉽다. `base` 로 먼저 감을 잡는 편이 낫다.

```powershell
python -m callwrap.train `
    --data data\train.jsonl --val-data data\val.jsonl `
    --backbone klue/roberta-base `
    --out runs\v1-base --epochs 15 --batch-size 16 --lr 3e-5 --patience 5
```

### VRAM 실측 (RTX 4080 SUPER 16GiB, seq 512, bf16)

| 백본 | batch | peak |
|---|---:|---:|
| roberta-base | 16 | 2.6 GiB |
| roberta-large | 8 | 8.2 GiB |
| roberta-large | 16 | 12.1 GiB |

OOM이 나면 `--batch-size 8 --grad-accum 2` 또는 `--grad-checkpointing`.

---

## 4. 모니터링

에폭마다 즉시 디스크에 기록되므로 **다른 터미널**에서 볼 수 있다.
새 터미널이므로 0단계(세션 준비)를 다시 해야 한다.

```powershell
cd c:\CallWrap\model
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"; $env:PYTHONIOENCODING = "utf-8"

python scripts\watch_progress.py --run runs\v1 --follow
```

```
✔ runs/v1  [running]  epoch 4/10  경과 0:12
 epoch      loss  macro-F1     SLA위반     acc       시간    VRAM
     1    1.4820    0.1204    0.0000   0.216     0:41     12.0
     2    0.9133    0.4871    0.0714   0.514     0:40     12.0  <- best
     3    0.6402    0.6238    0.0357   0.649     0:41     12.0  <- best
     4    0.4915    0.6903    0.0357   0.703     0:41     12.0  <- best
```

산출 파일:

| 파일 | 내용 |
|---|---|
| `runs/v1/progress.jsonl` | 에폭당 1줄, append-only |
| `runs/v1/status.json` | 현재 상태 · best · ETA · 실패 시 오류 |
| `runs/v1/train.log` | 콘솔 출력 전문 |
| `runs/v1/model.pt` | best 체크포인트 |

### 무엇을 볼 것인가

| 신호 | 해석 | 조치 |
|---|---|---|
| macro-F1이 계속 0.05 근처 | 모델이 한 클래스로 붕괴 | lr을 낮추거나 `--priority-span 0.5` 로 pos_weight 완화 |
| train loss는 내려가는데 macro-F1 정체 | 과적합 | `--epochs` 축소, `--dropout 0.2`, 데이터 증량 |
| **SLA 위반율이 오르는데 macro-F1도 오름** | 상위 SLA 클래스를 놓치는 중 | 주의. P5 캘리브레이션에서 `--sla-penalty` 를 높인다 |
| VRAM이 계속 증가 | 메모리 누수 의심 | 배치 크기 고정 확인 |

**macro-F1이 주 지표다. accuracy는 보지 말 것** (클래스 불균형 때문에 무의미).

---

## 5. 임계값 캘리브레이션

```powershell
python -m callwrap.calibrate --run runs\v1 --sla-penalty 0.5 --target-accuracy 0.90
```

목적함수는 `macro_F1 − sla_penalty × SLA위반율` 이다. 시스템오류·기능개선처럼
우선순위 상위 클래스의 임계값이 **낮게** 잡히는 것이 정상이다 — 놓치면 SLA를
위반하기 때문이다.

결과는 `runs/v1/config.json` 의 `thresholds` 와 `operating_point` 에 저장되며,
서빙 시 `collapse_probs(probs, thresholds)` 로 그대로 쓰면 된다.

---

## 6. 중단과 재개

`Ctrl+C` 로 중단해도 `progress.jsonl` 과 best 체크포인트는 남는다.
현재 재개(resume) 기능은 없으므로 처음부터 다시 돌려야 한다.
장시간 실행은 백그라운드로 두고 `watch_progress.py --follow` 로 보는 편이 낫다.

---

## 자주 만나는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| `'PYTHONPATH=src' 용어가 ... 인식되지 않습니다` | bash 문법을 PowerShell에서 사용 | `$env:PYTHONPATH = "src"` 를 **먼저 따로** 실행 |
| `ModuleNotFoundError: callwrap` | `PYTHONPATH` 미설정 | 위와 동일 |
| 줄 끝 `\` 에서 구문 오류 | bash 줄바꿈 문자 | PowerShell은 백틱(`` ` ``), 또는 한 줄로 |
| `exit=127` (Git Bash) | cwd가 `c:\CallWrap` | `cd /c/CallWrap/model` |
| 한글이 `????` 로 출력 | 콘솔 인코딩 | `$env:PYTHONIOENCODING = "utf-8"` |
| `Activate.ps1 ... 실행할 수 없습니다` | 실행 정책 제한 | `Set-ExecutionPolicy -Scope Process RemoteSigned` |
| `FileNotFoundError: data\train.jsonl` | 1단계 미실행 | `build_dataset.py` 먼저 |
| CUDA OOM | 배치 과다 | `--batch-size 8 --grad-accum 2` |
| DataLoader 멈춤 | Windows 멀티프로세싱 | `--num-workers 0` (기본값) |
