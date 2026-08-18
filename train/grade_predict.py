"""상담등급 판정 — 발화별 감정 점수를 2축 규칙으로 칭찬/악성/일반에 매핑한다.

    uv run --extra train python train/grade_predict.py              # 데이터셋 채점
    uv run --extra train python train/grade_predict.py --sweep      # 임계값별 성능
    uv run --extra train python train/grade_predict.py --file 통화.json

판정 규칙 (config/grade_rule.json)
    score_min  < abusive_max_score_min   → 악성콜   가장 격한 발화가 얼마나 격했나
    score_tail > praise_min_score_tail   → 칭찬콜   통화가 어떻게 끝났나
    그 외                                 → 일반

한 축으로는 안 된다. score_min만 쓰면 칭찬콜 25건을 전부 일반으로 흘리고
(칭찬콜도 시작할 때는 문제를 말하므로 최저점이 낮다), score_tail만 쓰면
악성 Recall이 0.85로 떨어진다. 각 축이 잘하는 등급을 따로 맡긴다.

**우선순위는 악성콜을 놓치지 않는 것이다** (명세 7장). 악성콜을 일반으로 표시하면
아무도 모르고 넘어가지만, 일반을 악성으로 표시하면 상담사 검토에서 걸러진다.
칭찬콜을 일반으로 흘리는 것도 업무상 손해가 작아 감수한다.

evidence_indices에는 **가장 부정적인 발화의 index**를 담는다. web/이 이걸로
원문을 하이라이트한다.
"""

import argparse
import json
from pathlib import Path

from callwrap_ai.schemas import ConsultationGrade

AI = Path(__file__).resolve().parents[1]
RULE = AI / "config/grade_rule.json"
SCORES = AI / "data/scenarios/llm_sentiment/scores_exaone3.5_7.8b.jsonl"


def load_rule(path: Path = RULE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def judge(scores: list[float], indices: list[int], rule: dict) -> dict:
    """발화별 점수 -> 등급 + 확신도 + 근거 발화 index.

    scores와 indices는 **고객 발화만** 같은 순서로 들어온다.
    """
    if not scores:
        return {"grade": None, "confidence": 0.0, "evidence_indices": [], "reason": "고객 발화 없음"}

    k = max(1, len(scores) // 3)
    s_min = min(scores)
    s_tail = sum(scores[-k:]) / k
    low, high = rule["abusive_max_score_min"], rule["praise_min_score_tail"]

    if s_min < low:
        grade, margin, span = ConsultationGrade.ABUSIVE, low - s_min, 1.0 + low
    elif s_tail > high:
        grade, margin, span = ConsultationGrade.PRAISE, s_tail - high, 1.0 - high
    else:
        grade = ConsultationGrade.NORMAL
        margin, span = min(s_min - low, high - s_tail), max(high - low, 1e-6)

    # 확신도는 임계값에서 얼마나 떨어졌는지다. 캘리브레이션된 확률이 아니므로
    # 3-tier 임계값(90/60)에 그대로 물리기 전에 보정이 필요하다 (명세 9.3, Step 4).
    conf = max(0.0, min(1.0, margin / span)) if span > 0 else 0.0

    # 근거: 가장 부정적인 발화. 칭찬콜이면 가장 긍정적인 발화를 가리킨다.
    pick = scores.index(max(scores)) if grade is ConsultationGrade.PRAISE else scores.index(s_min)
    return {"grade": grade.value, "confidence": round(conf, 4),
            "evidence_indices": [indices[pick]],
            "reason": f"score_min={s_min:.2f}, score_tail={s_tail:.2f}"}


def evaluate(rows: list[dict], rule: dict) -> None:
    from sklearn.metrics import classification_report, confusion_matrix, recall_score

    grades = [g.value for g in ConsultationGrade]
    gold = [r["grade"] for r in rows]
    pred = [judge(r["scores"], r["indices"], rule)["grade"] for r in rows]

    print(classification_report(gold, pred, labels=grades, zero_division=0))
    print("혼동행렬 (행=실제, 열=예측) ", grades)
    print(confusion_matrix(gold, pred, labels=grades))
    rec = recall_score(gold, pred, labels=[ConsultationGrade.ABUSIVE.value], average="micro", zero_division=0)
    n_ab = sum(1 for g in gold if g == ConsultationGrade.ABUSIVE.value)
    print(f"\n악성콜 Recall {rec:.4f}  ({round(rec * n_ab)}/{n_ab}건)  ← 최우선 지표")
    missed = [r["scenario_id"] for r, p in zip(rows, pred)
              if r["grade"] == ConsultationGrade.ABUSIVE.value and p != ConsultationGrade.ABUSIVE.value]
    if missed:
        print(f"놓친 악성콜: {missed}")


def sweep(rows: list[dict], rule: dict) -> None:
    """악성 임계값을 풀어가며 Recall과 오탐을 함께 본다. 값 선택 근거를 남기기 위한 것."""
    from sklearn.metrics import f1_score

    grades = [g.value for g in ConsultationGrade]
    gold = [r["grade"] for r in rows]
    n_ab = sum(1 for g in gold if g == ConsultationGrade.ABUSIVE.value)
    print(f"{'악성 임계값':>10}{'악성 Recall':>12}{'잡은 수':>8}{'오탐':>7}{'Macro-F1':>10}")
    print("-" * 48)
    for low in [-0.90, -0.85, -0.80, -0.75, -0.70, -0.65, -0.60, -0.50]:
        r2 = {**rule, "abusive_max_score_min": low}
        pred = [judge(r["scores"], r["indices"], r2)["grade"] for r in rows]
        hit = sum(1 for g, p in zip(gold, pred)
                  if g == ConsultationGrade.ABUSIVE.value and p == ConsultationGrade.ABUSIVE.value)
        fp = sum(1 for g, p in zip(gold, pred)
                 if g != ConsultationGrade.ABUSIVE.value and p == ConsultationGrade.ABUSIVE.value)
        f1 = f1_score(gold, pred, average="macro", labels=grades, zero_division=0)
        mark = "  ← 현재" if abs(low - rule["abusive_max_score_min"]) < 1e-9 else ""
        print(f"{low:>10.2f}{hit / n_ab:>12.4f}{hit:>6}/{n_ab}{fp:>7}{f1:>10.4f}{mark}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", type=Path, default=RULE)
    ap.add_argument("--sweep", action="store_true", help="임계값별 Recall·오탐 비교")
    ap.add_argument("--file", type=Path, help="점수 jsonl 1건을 판정 (형식: scores/indices)")
    args = ap.parse_args()

    rule = load_rule(args.rule)
    print(f"규칙: score_min < {rule['abusive_max_score_min']} → 악성콜 / "
          f"score_tail > {rule['praise_min_score_tail']} → 칭찬콜\n")

    if args.file:
        d = json.loads(args.file.read_text(encoding="utf-8"))
        print(json.dumps(judge(d["scores"], d.get("indices", list(range(len(d["scores"])))), rule),
                         ensure_ascii=False, indent=2))
        return

    rows = [json.loads(l) for l in SCORES.open(encoding="utf-8")]
    (sweep if args.sweep else evaluate)(rows, rule)


if __name__ == "__main__":
    main()
