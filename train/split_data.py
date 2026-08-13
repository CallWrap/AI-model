"""중분류 골든셋 계층 분할 — 학습셋/검증셋으로 나눈다.

golden_category.jsonl 100건을 중분류(category) 비율을 유지한 채 학습 80 / 검증 20으로
나눈다. 무작위로 나누면 최소 클래스(부정사용 9건)가 검증셋에 0건 들어갈 수 있어
그 클래스의 F1을 계산할 수 없다. 계층 분할(stratify)로 양쪽 모두에 배분한다.

seed를 고정하므로 같은 명령이면 누가 돌려도 같은 분할이 나온다. 산출물(jsonl)은
커밋하지 않고 이 스크립트로 재현한다.

실행: uv run --extra train python train/split_data.py
      uv run --extra train python train/split_data.py --seed 7 --val-size 25
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from callwrap_ai.schemas import ConsultationCategory

AI = Path(__file__).resolve().parents[1]
GOLDEN = AI / "eval/datasets/golden_category.jsonl"  # 커밋됨 (명세 4.2)
TRAIN_OUT = AI / "data/scenarios/train.jsonl"
VAL_OUT = AI / "data/scenarios/val.jsonl"

VALID = {c.value for c in ConsultationCategory}


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

    bad_label = [r["scenario_id"] for r in rows if r["category"] not in VALID]
    if bad_label:
        raise SystemExit(f"계약에 없는 중분류: {bad_label}")
    empty = [r["scenario_id"] for r in rows if not r["text"].strip()]
    if empty:
        raise SystemExit(f"대본 텍스트가 빈 건: {empty}")
    return rows


def dump(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def report(train: list[dict], val: list[dict]) -> None:
    tc, vc = Counter(r["category"] for r in train), Counter(r["category"] for r in val)

    print(f"\n{'중분류':<14}{'학습':>6}{'검증':>6}{'합계':>6}")
    print("-" * 32)
    for cat in sorted(VALID, key=lambda c: -(tc[c] + vc[c])):
        print(f"{cat:<14}{tc[cat]:>6}{vc[cat]:>6}{tc[cat] + vc[cat]:>6}")
    print("-" * 32)
    print(f"{'합계':<14}{len(train):>6}{len(val):>6}{len(train) + len(val):>6}")

    missing = [c for c in VALID if vc[c] == 0]
    if missing:
        print(f"\n[경고] 검증셋에 0건인 중분류: {missing} — Macro-F1 계산이 불가하다")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42, help="분할 난수 시드 (재현용)")
    p.add_argument("--val-size", type=int, default=20, help="검증셋 건수")
    args = p.parse_args()

    rows = load(GOLDEN)
    train, val = train_test_split(
        rows,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=[r["category"] for r in rows],
    )

    dump(train, TRAIN_OUT)
    dump(val, VAL_OUT)

    report(train, val)
    print(f"\nseed={args.seed}")
    print(f"학습 {len(train)}건 -> {TRAIN_OUT.relative_to(AI)}")
    print(f"검증 {len(val)}건 -> {VAL_OUT.relative_to(AI)}")


if __name__ == "__main__":
    main()
