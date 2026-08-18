"""소분류 데이터셋 계층 분할 — 학습셋/검증셋으로 나눈다.

subcategory_dataset.jsonl 183건을 **소분류(29종) 비율을 유지한 채** 나눈다.
중분류 기준으로 나누는 split_data.py와 다르다. 소분류로 층화하지 않으면
29클래스 중 상당수가 검증셋에 0건 들어가 F1을 계산할 수 없다.

검증셋에는 생성 증강분(source=generated)이 섞인다. 이건 의도한 것이다 — 그래야
클래스마다 표본이 들어가 측정이 성립한다. 다만 그 점수는 "생성 데이터를 맞힌
점수"이므로, 채점할 때 source별로 갈라 두 숫자를 나란히 본다.

seed를 고정하므로 같은 명령이면 누가 돌려도 같은 분할이 나온다.

실행: uv run --extra train python train/split_subcategory.py
      uv run --extra train python train/split_subcategory.py --seed 7 --val-ratio 0.25
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from callwrap_ai.schemas import SUBCATEGORIES_BY_CATEGORY

AI = Path(__file__).resolve().parents[1]
DATASET = AI / "eval/datasets/subcategory_dataset.jsonl"  # 커밋됨 (명세 4.2)
TRAIN_OUT = AI / "data/scenarios/sub_train.jsonl"
VAL_OUT = AI / "data/scenarios/sub_val.jsonl"

BY_CAT = {c.value: {s.value for s in subs} for c, subs in SUBCATEGORIES_BY_CATEGORY.items()}
VALID_SUB = {s for subs in BY_CAT.values() for s in subs}


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"데이터셋이 없습니다: {path}\n"
                         "  uv run python eval/build_subcategory_dataset.py 를 먼저 실행합니다.")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

    bad = [r["scenario_id"] for r in rows if r["subcategory"] not in VALID_SUB]
    if bad:
        raise SystemExit(f"계약에 없는 소분류: {bad}")
    broken = [r["scenario_id"] for r in rows if r["subcategory"] not in BY_CAT[r["category"]]]
    if broken:
        raise SystemExit(f"계층 위반: {broken}")
    empty = [r["scenario_id"] for r in rows if not r["text"].strip()]
    if empty:
        raise SystemExit(f"대본 텍스트가 빈 건: {empty}")
    return rows


def dump(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def report(train: list[dict], val: list[dict]) -> None:
    tc, vc = Counter(r["subcategory"] for r in train), Counter(r["subcategory"] for r in val)

    print(f"\n{'중분류':<13}{'소분류':<18}{'학습':>5}{'검증':>5}{'합계':>5}")
    print("-" * 46)
    for cat in sorted(BY_CAT, key=lambda c: -sum(tc[s] + vc[s] for s in BY_CAT[c])):
        for sub in sorted(BY_CAT[cat], key=lambda s: -(tc[s] + vc[s])):
            print(f"{cat:<13}{sub:<18}{tc[sub]:>5}{vc[sub]:>5}{tc[sub] + vc[sub]:>5}")
    print("-" * 46)
    print(f"{'합계':<31}{len(train):>5}{len(val):>5}{len(train) + len(val):>5}")

    missing = [s for s in VALID_SUB if vc[s] == 0]
    if missing:
        print(f"\n⚠️ 검증셋에 0건인 소분류 {len(missing)}종 — 이 클래스는 F1을 낼 수 없다:")
        print("   ", ", ".join(sorted(missing)))
    else:
        print("\n검증셋에 29종 모두 1건 이상 — 전 클래스 F1 산출 가능")

    print("\n검증셋 출처:", dict(Counter(r["source"] for r in val)))
    human = sum(1 for r in val if r["source"] in ("human", "draft"))
    print(f"  사람 검수·초안 {human}건 / 생성 {len(val) - human}건")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="재현용 난수 시드")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="검증셋 비율")
    args = ap.parse_args()

    rows = load(DATASET)
    train, val = train_test_split(
        rows,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=[r["subcategory"] for r in rows],  # 소분류 기준 층화
    )

    dump(train, TRAIN_OUT)
    dump(val, VAL_OUT)
    report(train, val)
    print(f"\nseed={args.seed}")
    print(f"학습 {len(train)}건 -> {TRAIN_OUT.relative_to(AI)}")
    print(f"검증 {len(val)}건 -> {VAL_OUT.relative_to(AI)}")


if __name__ == "__main__":
    main()
