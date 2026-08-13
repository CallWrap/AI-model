# eval/datasets/ — 골든셋 (커밋됨)

명세 4.2에 따라 **작으므로 커밋한다.** 두 파일 합쳐 350KB다.
이게 있어야 다른 사람 컴퓨터에서도 학습이 재현된다 (명세 11장).

| 파일 | 건수 | 내용 |
|---|---|---|
| `golden_category.jsonl` | 100 | 대본 + **중분류 정답**. 사람 검수 완료 |
| `gold_draft.jsonl` | 100 | 같은 100건의 유형·소분류·등급·후속·FAQ **라벨 초안** |

`scenario_id` 로 서로 조인된다.

## golden_category.jsonl — 확정본

기관 FAQ 333건에서 생성한 상담 시나리오를 Claude·Gemini 독립 라벨링으로
교차검증하고, 갈린 건은 사람이 판정했다 (AI 일치 74건 + 사람 검수 26건).

| 중분류 | 건수 |
|---|---|
| 대상자관리 | 31 |
| 제공기관관리 | 17 |
| 결제및비용지급 | 16 |
| 제공인력관리 | 14 |
| 지자체사업관리 | 13 |
| 부정사용 | 9 |

## gold_draft.jsonl — 초안, 검수 전

**중분류를 뺀 나머지 라벨은 아직 사람 검수를 거치지 않았다.** `label_status` 로 구분한다
(`생성라벨` 44건 = 대본 생성 시 정한 값, `기계초안` 56건 = LLM 재라벨링 결과).
명세 9.1이 "평가셋 라벨은 생성 라벨을 그대로 쓰지 않고 사람이 다시 붙인다"고 못박은
대상이므로, **이 상태로 성능을 주장하면 안 된다.**

알려진 문제: **소분류의 계층 정합이 15건 깨져 있다.** 중분류에 속하지 않는 소분류가
붙어 있다 (예: 중분류 `대상자관리` 에 소분류 `포상금` — 포상금은 `부정사용` 소속).
검수 시 이 15건부터 본다.

등급 분포는 일반 68 / 악성콜 16 / 칭찬콜 16이다.

## 여기 두지 않는 것

- `train.jsonl` · `val.jsonl` — `split_data.py` 가 만드는 파생물. seed 고정이라 재현된다
- 기관 제공 PDF·xlsx (`data/raw/`) — 기관 자료라 커밋 금지
- FAQ 원문 (`data/processed/`) — 기관 PDF에서 추출한 분량이 섞여 있다
- 모델 가중치 (`data/models/`) — 423MB × 3

## 재현

```bash
uv run --extra train python train/split_data.py       # → data/scenarios/{train,val}.jsonl
uv run --extra train python train/train_category.py   # → data/models/...
```
