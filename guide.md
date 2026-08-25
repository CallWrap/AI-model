# AI-model — 사보원 오프라인 셋업 가이드

## 이 레포는 단독 실행하지 않는다

`AI-model`(패키지명 `callwrap-ai`)은 `server-fastapi`가 Python 라이브러리로 가져다 쓰는
코드다. 별도 서버·프로세스가 없다 — `server-fastapi/.venv`에 이미 설치되어 있으므로
이 폴더 자체를 실행할 일은 없다.

## 확인할 것: 체크포인트 3종이 제자리에 있는지

```
data/models/
├── category-klue-roberta-base-seed42/       중분류(6종)
├── subcategory-klue-roberta-base-seed42/    소분류(29종)
└── sr-type-klue-roberta-base-v2/            상담유형/SR유형(8종)
```

이 zip에는 이미 담겨 있다. `server-fastapi`가 `CALLWRAP_MODEL_DIR` 환경변수로 이 경로를
직접 가리키므로, **별도 복사나 다운로드가 필요 없다** (Whisper·pyannote·Ollama와 다른 점 —
그것들은 사용자 캐시 경로로 복사가 필요하다, [server-fastapi/guide.md](../server-fastapi/guide.md) 참고).

## 이 폴더의 모델만 따로 테스트하고 싶으면

`server-fastapi/.venv`를 그대로 쓴다 (이 폴더 자체엔 별도 venv를 만들 필요 없음):
```bash
cd ../server-fastapi
export CALLWRAP_MODEL_DIR="$(pwd)/../AI-model/data/models"
.venv/Scripts/python.exe -c "
from callwrap_ai import classify, sr_type, grade, summary
print('중분류/소분류 모델 로드:', classify.is_ready())
print('SR유형 모델 로드:', sr_type.is_ready())
print('등급판정(Ollama) 준비:', grade.is_ready())
"
```
`grade.is_ready()`가 `False`면 Ollama가 안 떠 있거나 `exaone3.5:7.8b`가 없는 것이다
([server-fastapi/guide.md](../server-fastapi/guide.md)의 Ollama 모델 복사 단계를 확인한다).

## 참고

- `analyze()` 하나가 요약(F1)·상담유형/중분류/소분류(F2)·등급(F3)·후속처리·FAQ필요(F4)를
  전부 채운다 — 상세 계약은 [../server-fastapi/docs/message_contract.md](../server-fastapi/docs/message_contract.md).
- 이 레포의 지표(중분류 Macro-F1 0.50, 소분류 0.31~0.39 등)는 **합성 데이터 기준 내부 지표**다.
  사보원에서 실제 통화로 다시 확인하는 게 이번 테스트의 핵심 목적이다.
