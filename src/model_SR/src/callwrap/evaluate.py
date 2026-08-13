"""평가 지표.

주 지표는 **최종 SR유형에 대한 macro-F1** 이다. accuracy는 쓰지 않는다
(사용법문의가 절반이면 accuracy 0.5는 아무 의미가 없다).

부 지표로 **SLA 위반율**을 반드시 함께 본다. 실제가 장기유형(30일/5일)인데
단기유형으로 예측한 비율이며, 업무적으로 진짜 손해가 나는 지표다.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .labels import (LABELS, NUM_LABELS, PRIORITY, SLA_DAYS, collapse_probs,
                     is_sla_violation)


# ---------------------------------------------------------------------------
# 최종 라벨 기준
# ---------------------------------------------------------------------------
def final_label_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """축약된 최종 SR유형 기준 지표."""
    macro_f1 = f1_score(y_true, y_pred, labels=list(LABELS),
                        average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, labels=list(LABELS),
                        average="micro", zero_division=0)
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), zero_division=0
    )

    violations = sum(is_sla_violation(t, p_) for t, p_ in zip(y_true, y_pred))
    # 분모는 '위반이 발생할 수 있는 건'(SLA가 최단이 아닌 건)으로 잡는다.
    at_risk = sum(1 for t in y_true if SLA_DAYS[t] > min(SLA_DAYS.values()))

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "accuracy": float(np.mean([t == p_ for t, p_ in zip(y_true, y_pred)])),
        "sla_violation_rate": float(violations / at_risk) if at_risk else 0.0,
        "sla_violations": int(violations),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]),
                   "f1": float(f[i]), "support": int(sup[i])}
            for i, name in enumerate(LABELS)
        },
    }


def confusion_matrix_8x8(y_true: list[str], y_pred: list[str]) -> np.ndarray:
    """행=실제, 열=예측. 우선순위 순서로 정렬해 SLA 방향을 읽기 쉽게 한다."""
    idx = {name: i for i, name in enumerate(PRIORITY)}
    cm = np.zeros((NUM_LABELS, NUM_LABELS), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def format_confusion(cm: np.ndarray) -> str:
    """혼동행렬을 콘솔 표로. 우측 하단(=SLA 위반 영역)이 핵심."""
    w = max(len(n) for n in PRIORITY) + 1
    head = " " * w + "".join(f"{n[:6]:>7}" for n in PRIORITY)
    lines = [head, "-" * len(head)]
    for i, name in enumerate(PRIORITY):
        row = f"{name:<{w}}" + "".join(f"{cm[i, j]:>7}" for j in range(NUM_LABELS))
        lines.append(row)
    lines.append("")
    lines.append("행=실제 / 열=예측. 대각선 아래쪽 = 짧은 SLA로 예측 = 위반 방향.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 멀티라벨 기준 (헤드 자체 품질)
# ---------------------------------------------------------------------------
def multilabel_metrics(y_true: np.ndarray, probs: np.ndarray,
                       thresholds: np.ndarray) -> dict:
    y_pred = (probs >= thresholds).astype(int)
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0,
        labels=list(range(NUM_LABELS))
    )
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_class": {
            LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                        "f1": float(f[i]), "support": int(sup[i])}
            for i in range(NUM_LABELS)
        },
    }


# ---------------------------------------------------------------------------
# 커버리지-정확도 곡선
# ---------------------------------------------------------------------------
def coverage_accuracy_curve(
    y_true: list[str], probs: np.ndarray, thresholds: np.ndarray,
    steps: int = 20,
) -> list[dict]:
    """자동분류 비율(커버리지)을 낮출수록 정확도가 얼마나 오르는가.

    운영점("커버리지 70%에서 정확도 92%")을 고르기 위한 곡선.
    신뢰도 = 최댓값 확률.
    """
    preds, conf = [], []
    for row in probs:
        preds.append(collapse_probs(row, thresholds))
        conf.append(float(np.max(row)))

    order = np.argsort(-np.asarray(conf))
    n = len(order)
    curve = []
    for k in range(1, steps + 1):
        cut = max(1, int(n * k / steps))
        sel = order[:cut]
        correct = sum(preds[i] == y_true[i] for i in sel)
        viol = sum(is_sla_violation(y_true[i], preds[i]) for i in sel)
        curve.append({
            "coverage": cut / n,
            "accuracy": correct / cut,
            "sla_violation_rate": viol / cut,
            "min_confidence": float(conf[order[cut - 1]]),
        })
    return curve


def topk_accuracy(y_true: list[str], probs: np.ndarray, k: int = 3) -> float:
    """상위 k개 후보 안에 정답이 있는 비율.

    후처리 UI가 Top-3를 제시하는 설계라면 이 값이 실질적 체감 정확도다.
    """
    hits = 0
    for true, row in zip(y_true, probs):
        top = np.argsort(-row)[:k]
        if true in {LABELS[i] for i in top}:
            hits += 1
    return hits / len(y_true) if y_true else 0.0


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------
def report(y_true: list[str], y_pred: list[str],
           probs: np.ndarray | None = None,
           thresholds: np.ndarray | None = None,
           title: str = "평가 결과") -> dict:
    m = final_label_metrics(y_true, y_pred)

    print(f"\n{'=' * 68}")
    print(f"{title}   (n={len(y_true)})")
    print("=" * 68)
    print(f"macro-F1            : {m['macro_f1']:.4f}   <- 주 지표")
    print(f"micro-F1            : {m['micro_f1']:.4f}")
    print(f"accuracy            : {m['accuracy']:.4f}   (참고용)")
    print(f"SLA 위반율          : {m['sla_violation_rate']:.4f} "
          f"({m['sla_violations']}건)   <- 부 지표")

    if probs is not None:
        for k in (2, 3):
            print(f"Top-{k} accuracy      : {topk_accuracy(y_true, probs, k):.4f}")

    print(f"\n{'SR유형':<18}{'P':>8}{'R':>8}{'F1':>8}{'n':>7}  SLA")
    print("-" * 68)
    for name in PRIORITY:
        c = m["per_class"][name]
        print(f"{name:<18}{c['precision']:>8.3f}{c['recall']:>8.3f}"
              f"{c['f1']:>8.3f}{c['support']:>7}  {SLA_DAYS[name]:>2}일")

    print("\n[혼동행렬]")
    print(format_confusion(confusion_matrix_8x8(y_true, y_pred)))

    if probs is not None and thresholds is not None:
        print("\n[커버리지-정확도]")
        print(f"{'커버리지':>9}{'정확도':>9}{'SLA위반':>9}{'최소신뢰도':>11}")
        for row in coverage_accuracy_curve(y_true, probs, thresholds, steps=10):
            print(f"{row['coverage']:>9.2f}{row['accuracy']:>9.3f}"
                  f"{row['sla_violation_rate']:>9.3f}{row['min_confidence']:>11.3f}")

    return m
