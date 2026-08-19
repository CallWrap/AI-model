# 오프라인(망분리) 실행 안내 — AI-model(callwrap_ai)

이 번들만으로 실행한다. **밖에서 아무것도 받지 않는다** — PyPI·GitHub·ollama.com 전부 불필요.

```
code/AI-model/            callwrap_ai 소스 (analyze 하위 모듈, eval/, config/)
wheels/                    파이썬 의존성 (base + train extra, transitive 포함)
models/checkpoints/        F2(중분류·소분류)·SR유형 checkpoint — 성진에게 수령해 배치
ollama/                    F1·F3·F4·FAQ용 온디바이스 LLM (exaone3.5:7.8b) — 있으면 자동 포함
```

## 범위 밖 (이 번들이 다루지 않는 것)

- **model_SR 파인튜닝**(`src/model_SR/`) — `klue/roberta-large` + CUDA GPU 전용. 사보원 머신이
  어떤 하드웨어인지 정해지면 그때 별도로 준비한다(HF 캐시 + CUDA 휠은 GPU 스펙에 딸린다).
- **eval/의 합성데이터 생성 스크립트**(`generate_scenarios.py` 등) — anthropic API를 직접 호출한다.
  실데이터 테스트와 무관하고, 애초에 오프라인으로 옮길 수 없다.

## 1. 설치

```bash
cd code/AI-model
python -m venv .venv && . .venv/bin/activate
pip install --no-index --find-links ../wheels -r ../wheels/requirements.pypi.txt
pip install --no-index --no-deps --find-links ../wheels callwrap-ai
```

## 2. checkpoint·ollama 배치

```bash
cp -R ../models/checkpoints/* data/models/          # F2/SR checkpoint (있으면)
ollama serve &                                       # ollama가 이미 설치돼 있다면
cp -R ../ollama/models/* ~/.ollama/models/
```

## 3. 환경변수 (필수 — wheel로 설치하면 코드 안의 상대경로 기본값이 깨진다)

`grade.py`·`summary.py`·`classify.py`는 `Path(__file__).resolve().parents[2]`로 저장소
루트를 추정한다. 이건 소스에서 바로 돌릴 때(`uv run`, editable install)만 맞고, 이 번들처럼
빌드된 wheel을 설치하면 site-packages 안을 가리켜 깨진다 — 그래서 아래를 전부 명시해야 한다.

```bash
export CALLWRAP_MODEL_DIR=$(pwd)/data/models
export CALLWRAP_GRADE_RULE=$(pwd)/config/grade_rule.json
export CALLWRAP_SENTIMENT_PROMPT=$(pwd)/eval/prompts/sentiment_v1.md
export CALLWRAP_SUMMARY_PROMPT=$(pwd)/eval/prompts/summary_v6.md
export OLLAMA_URL=http://127.0.0.1:11434               # 기본값과 같으면 생략 가능
```

## 4. 동작 확인 (실데이터 넣기 전에)

```bash
pytest -q
python - <<'PY'
from callwrap_ai import classify
print(classify.predict_category("테스트 상담 대본입니다"))  # checkpoint 있으면 (라벨, 확률), 없으면 None
PY
python - <<'PY'
from callwrap_ai.grade import judge
from callwrap_ai.schemas import Transcript
# 실제 Transcript로 교체해 확인 — ollama가 켜져 있어야 None이 아닌 값이 나온다
PY
```

## 주의

- **checkpoint가 없으면 F2는 조용히 None을 돌려준다**(부분 실패로 설계됨). 정확도를 실제로
  검증하려면 반드시 `data/models/`에 checkpoint를 배치해야 한다.
- **ollama 모델(exaone3.5:7.8b)은 수 GB다.** 이 스크립트를 ollama가 설치된 머신에서 돌려야
  자동으로 담긴다. 안 그러면 `ollama/README.txt`만 생기니 별도로 반입해야 한다.
- **Python 버전을 사보원 머신과 맞춰야 한다.** wheel은 이 스크립트를 돌린 머신의 Python
  버전으로 빌드된다 — 다르면 그 버전 Python으로 다시 받아야 한다.
- **실데이터·평가 결과는 반출 금지.** 반출은 지표(정확도 등)만.
