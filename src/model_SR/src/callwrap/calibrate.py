"""P5: 임계값 캘리브레이션 + 운영점 선정.

학습이 끝난 뒤 클래스별 임계값을 검증셋에서 최적화한다. 목적함수는 단순
macro-F1이 아니라 정의서 §1.2의 비대칭 오류비용을 반영한다:

    objective = macro_F1 - sla_penalty * SLA위반율

시스템오류/기능개선처럼 우선순위 상위 클래스는 임계값이 낮게(민감하게)
잡히는 것이 정상이다. 놓치면 SLA를 위반하기 때문이다.

실제 클래스 분포가 학습 분포와 다르면 --prior-file 로 중요도 가중을 준다.
(합성 데이터는 균형 있게 만들지만 실제는 사용법문의/자료확인 쪽으로 쏠린다.)

사용:
    python -m callwrap.calibrate --run runs/v1
    python -m callwrap.calibrate --run runs/v1 --prior-file configs/real_prior.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import coverage_accuracy_curve, report, topk_accuracy
from .labels import (LABELS, NUM_LABELS, PRIORITY, SLA_DAYS, collapse_batch,
                     is_sla_violation)


# ---------------------------------------------------------------------------
# 목적함수
# ---------------------------------------------------------------------------
def weighted_macro_f1(y_true: list[str], y_pred: list[str],
                      weights: np.ndarray) -> float:
    """샘플 가중을 반영한 macro-F1 (클래스 평균)."""
    scores = []
    for name in LABELS:
        tp = float(sum(w for t, p, w in zip(y_true, y_pred, weights)
                       if t == name and p == name))
        fp = float(sum(w for t, p, w in zip(y_true, y_pred, weights)
                       if t != name and p == name))
        fn = float(sum(w for t, p, w in zip(y_true, y_pred, weights)
                       if t == name and p != name))
        if tp + fp + fn == 0:
            continue  # 검증셋에 해당 클래스가 없으면 평균에서 제외
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def weighted_sla_rate(y_true: list[str], y_pred: list[str],
                      weights: np.ndarray) -> float:
    shortest = min(SLA_DAYS.values())
    at_risk = float(sum(w for t, w in zip(y_true, weights)
                        if SLA_DAYS[t] > shortest))
    if at_risk == 0:
        return 0.0
    viol = float(sum(w for t, p, w in zip(y_true, y_pred, weights)
                     if is_sla_violation(t, p)))
    return viol / at_risk


def objective(y_true: list[str], probs: np.ndarray, thresholds: np.ndarray,
              weights: np.ndarray, sla_penalty: float) -> tuple[float, float, float]:
    y_pred = collapse_batch(probs, thresholds)
    f1 = weighted_macro_f1(y_true, y_pred, weights)
    sla = weighted_sla_rate(y_true, y_pred, weights)
    return f1 - sla_penalty * sla, f1, sla


# ---------------------------------------------------------------------------
# 좌표 상승 탐색
# ---------------------------------------------------------------------------
def optimize_thresholds(
    y_true: list[str], probs: np.ndarray, weights: np.ndarray,
    *, sla_penalty: float = 0.5, rounds: int = 4,
    grid: np.ndarray | None = None, verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """클래스별 임계값을 좌표 상승으로 최적화한다.

    최종 라벨이 우선순위 규칙으로 결정되므로 클래스 간 상호작용이 있다.
    따라서 클래스별 독립 최적화가 아니라 좌표 상승을 여러 라운드 돌린다.
    """
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.025), 4)

    thresholds = np.full(NUM_LABELS, 0.5)
    best_obj, best_f1, best_sla = objective(
        y_true, probs, thresholds, weights, sla_penalty
    )
    if verbose:
        print(f"초기 (all=0.5): obj={best_obj:.4f} f1={best_f1:.4f} sla={best_sla:.4f}")

    for rnd in range(1, rounds + 1):
        improved = False
        # 우선순위 상위 클래스부터 조정한다. 축약에서 먼저 이기는 쪽이므로
        # 하위 클래스 임계값보다 영향이 크다.
        for name in PRIORITY:
            ci = LABELS.index(name)
            base = thresholds[ci]
            for t in grid:
                thresholds[ci] = t
                obj, f1, sla = objective(y_true, probs, thresholds,
                                         weights, sla_penalty)
                if obj > best_obj + 1e-9:
                    best_obj, best_f1, best_sla, base = obj, f1, sla, t
                    improved = True
            thresholds[ci] = base
        if verbose:
            print(f"라운드 {rnd}: obj={best_obj:.4f} f1={best_f1:.4f} "
                  f"sla={best_sla:.4f}")
        if not improved:
            break

    return thresholds, {"objective": best_obj, "macro_f1": best_f1,
                        "sla_violation_rate": best_sla}


# ---------------------------------------------------------------------------
# 실분포 중요도 가중
# ---------------------------------------------------------------------------
def importance_weights(y_true: list[str], prior: dict[str, float] | None) -> np.ndarray:
    """검증셋 분포를 실제 운영 분포로 재가중한다.

    합성 데이터는 균형 있게 만들지만 실제 콜 분포는 사용법문의/자료확인 쪽으로
    쏠린다. 그 상태의 임계값을 그대로 쓰면 운영에서 캘리브레이션이 어긋난다.
    """
    n = len(y_true)
    if not prior:
        return np.ones(n)

    total = sum(prior.values())
    target = {k: v / total for k, v in prior.items()}

    empirical: dict[str, int] = {}
    for t in y_true:
        empirical[t] = empirical.get(t, 0) + 1

    w = np.array([
        target.get(t, 1e-6) / (empirical[t] / n) if empirical.get(t) else 0.0
        for t in y_true
    ])
    return w * n / w.sum() if w.sum() > 0 else np.ones(n)


# ---------------------------------------------------------------------------
# 운영점 선정
# ---------------------------------------------------------------------------
def pick_operating_point(curve: list[dict], target_accuracy: float = 0.90) -> dict:
    """목표 정확도를 만족하는 최대 커버리지 지점."""
    feasible = [c for c in curve if c["accuracy"] >= target_accuracy]
    if feasible:
        return max(feasible, key=lambda c: c["coverage"])
    return max(curve, key=lambda c: c["accuracy"])  # 목표 미달 시 최고 정확도


def main() -> None:
    ap = argparse.ArgumentParser(description="임계값 캘리브레이션 + 운영점 선정")
    ap.add_argument("--run", type=Path, required=True, help="train.py 산출 디렉터리")
    ap.add_argument("--sla-penalty", type=float, default=0.5,
                    help="목적함수의 SLA 위반 페널티 계수. 클수록 보수적")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--prior-file", type=Path, default=None,
                    help='실제 클래스 분포 JSON: {"사용법문의": 0.42, ...}')
    ap.add_argument("--target-accuracy", type=float, default=0.90,
                    help="운영점 선정 기준")
    args = ap.parse_args()

    probs = np.load(args.run / "val_probs.npy")
    y_true = json.loads((args.run / "val_true.json").read_text(encoding="utf-8"))
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    prior = json.loads(args.prior_file.read_text(encoding="utf-8")) \
        if args.prior_file else None

    print(f"검증셋 {len(y_true)}건 / {config['val_kind']}")
    if config["val_kind"].startswith("합성"):
        print("  ⚠ 합성 검증셋으로 캘리브레이션한 임계값은 운영에서 어긋납니다.")
        print("    실데이터 holdout이 확보되면 반드시 재캘리브레이션하십시오.")
    if prior:
        print(f"실분포 중요도 가중 적용: {args.prior_file.name}")

    weights = importance_weights(y_true, prior)

    print("\n[임계값 최적화]")
    thresholds, stats = optimize_thresholds(
        y_true, probs, weights,
        sla_penalty=args.sla_penalty, rounds=args.rounds,
    )

    print(f"\n{'SR유형':<18}{'임계값':>9}{'SLA':>7}")
    print("-" * 36)
    for name in PRIORITY:
        print(f"{name:<18}{thresholds[LABELS.index(name)]:>9.3f}"
              f"{SLA_DAYS[name]:>5}일")

    y_pred = collapse_batch(probs, thresholds)
    report(y_true, y_pred, probs, thresholds, title="캘리브레이션 후 최종 성능")

    curve = coverage_accuracy_curve(y_true, probs, thresholds, steps=20)
    op = pick_operating_point(curve, args.target_accuracy)
    print(f"\n[운영점 — 목표 정확도 {args.target_accuracy:.0%}]")
    print(f"  커버리지     {op['coverage']:.1%}  (자동분류 비율)")
    print(f"  정확도       {op['accuracy']:.3f}")
    print(f"  SLA 위반율   {op['sla_violation_rate']:.3f}")
    print(f"  신뢰도 컷    {op['min_confidence']:.3f}")
    print(f"  -> 신뢰도 {op['min_confidence']:.3f} 미만은 상담사 확인 큐로 보냅니다.")

    print(f"\n[제품 관점] Top-3 정확도 {topk_accuracy(y_true, probs, 3):.3f}")
    print("  후처리 UI가 Top-3 후보를 제시하는 설계라면 이 값이 체감 정확도입니다.")

    config["thresholds"] = thresholds.tolist()
    config["calibration"] = {
        "sla_penalty": args.sla_penalty,
        "prior_file": str(args.prior_file) if args.prior_file else None,
        **stats,
    }
    config["operating_point"] = op
    (args.run / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n임계값을 {args.run / 'config.json'} 에 저장했습니다.")


if __name__ == "__main__":
    main()
