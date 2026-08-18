"""감정 점수 → 상담등급 판정 방식 비교.

    uv run --extra train python eval/compare_sentiment.py

세 방식을 같은 데이터·같은 규칙으로 비교한다.
  전용 감정 모델   KoELECTRA 6감정 (0.11B, MIT)
  EXAONE 3.5 7.8B  온디바이스 LLM
  Gemma 3 12B      온디바이스 LLM

판정 규칙은 축 하나에 임계값 두 개다.
  점수 < low  → 악성콜        점수 > high → 칭찬콜        그 사이 → 일반

**임계값을 전체 데이터로 맞추고 그 데이터로 채점하면 점수가 부풀려진다.**
5겹 교차검증으로 4/5에서 임계값을 찾고 나머지 1/5에서 채점한다. 전체로 맞춘 점수도
함께 내서 얼마나 부풀려지는지 보인다.

명세 7장이 "악성콜 Recall 우선"으로 지정했으므로 Recall을 따로 낸다.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

AI = Path(__file__).resolve().parents[1]
SCN = AI / "data/scenarios"
GRADES = ["악성콜", "일반", "칭찬콜"]

# 방식별 (파일, 후보 축). 축은 전부 "높을수록 긍정"이 되도록 부호를 맞춘다.
SOURCES = {
    "전용 감정모델": (SCN / "sentiment_scores.jsonl",
                 {"-gap_mean": -1, "-angry_mean": -1, "-angry_tail": -1, "happy_mean": 1}),
    "EXAONE 7.8B": (SCN / "llm_sentiment/scores_exaone3.5_7.8b.jsonl",
                    {"score_tail": 1, "score_mean": 1, "score_head": 1, "score_delta": 1,
                     "score_min": 1, "-neg_ratio": -1}),
    "Gemma 12B": (SCN / "llm_sentiment/scores_gemma3_12b.jsonl",
                  {"score_tail": 1, "score_mean": 1, "score_head": 1, "score_delta": 1,
                     "score_min": 1, "-neg_ratio": -1}),
}


def load(path: Path, axes: dict[str, int]) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        agg = r.get("agg") or {}
        if not agg:
            continue
        vals = {}
        for ax, sign in axes.items():
            key = ax.lstrip("-")
            if key in agg:
                vals[ax] = sign * agg[key]
        if vals:
            out[r["scenario_id"]] = {"grade": r["grade"], **vals}
    return out


def classify(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.where(x < low, "악성콜", np.where(x > high, "칭찬콜", "일반"))


def fit_thresholds(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Macro-F1이 가장 높은 임계값 두 개를 격자 탐색으로 찾는다."""
    grid = np.quantile(x, np.linspace(0.02, 0.98, 40))
    best, best_f1 = (grid[0], grid[-1]), -1.0
    for i, low in enumerate(grid):
        for high in grid[i:]:
            f1 = f1_score(y, classify(x, low, high), average="macro", labels=GRADES, zero_division=0)
            if f1 > best_f1:
                best, best_f1 = (low, high), f1
    return best


def evaluate(x: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    """5겹 교차검증 — 임계값은 학습 폴드에서만 찾는다."""
    pred = np.empty(len(y), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        low, high = fit_thresholds(x[tr], y[tr])
        pred[te] = classify(x[te], low, high)
    low_all, high_all = fit_thresholds(x, y)
    pred_all = classify(x, low_all, high_all)
    return {
        "cv_macro_f1": f1_score(y, pred, average="macro", labels=GRADES, zero_division=0),
        "cv_acc": (pred == y).mean(),
        "cv_abusive_recall": recall_score(y, pred, labels=["악성콜"], average="micro", zero_division=0),
        "fit_macro_f1": f1_score(y, pred_all, average="macro", labels=GRADES, zero_division=0),
        "thresholds": f"{low_all:.3f} / {high_all:.3f}",
    }


def main() -> None:
    loaded = {n: load(p, ax) for n, (p, ax) in SOURCES.items()}
    avail = {n: d for n, d in loaded.items() if d}
    missing = [n for n, d in loaded.items() if not d]
    if missing:
        print(f"⚠️ 아직 없는 방식: {', '.join(missing)}\n")
    if not avail:
        raise SystemExit("비교할 결과가 없습니다.")

    common = set.intersection(*(set(d) for d in avail.values()))
    print(f"공통 통화 {len(common)}건으로 비교 (방식별 실패 건은 제외)")
    ids = sorted(common)
    y = np.array([next(iter(avail.values()))[i]["grade"] for i in ids])
    print("  등급:", {g: int((y == g).sum()) for g in GRADES})

    print(f"\n{'방식':<15}{'축':<14}{'CV MacroF1':>11}{'CV Acc':>9}{'악성 Recall':>12}{'전체적합':>10}")
    print("-" * 72)
    best_of = {}
    for name, d in avail.items():
        axes = [a for a in SOURCES[name][1] if a in next(iter(d.values()))]
        rows = []
        for ax in axes:
            x = np.array([d[i][ax] for i in ids], dtype=float)
            rows.append((ax, evaluate(x, y)))
        rows.sort(key=lambda r: -r[1]["cv_macro_f1"])
        best_of[name] = rows[0]
        for k, (ax, m) in enumerate(rows):
            tag = name if k == 0 else ""
            mark = " ←" if k == 0 else ""
            print(f"{tag:<15}{ax:<14}{m['cv_macro_f1']:>11.4f}{m['cv_acc']:>9.4f}"
                  f"{m['cv_abusive_recall']:>12.4f}{m['fit_macro_f1']:>10.4f}{mark}")
        print()

    print("=" * 72)
    print(f"{'방식':<15}{'최적 축':<14}{'CV MacroF1':>11}{'악성 Recall':>12}{'임계값':>18}")
    print("-" * 72)
    for name, (ax, m) in sorted(best_of.items(), key=lambda x: -x[1][1]["cv_macro_f1"]):
        print(f"{name:<15}{ax:<14}{m['cv_macro_f1']:>11.4f}{m['cv_abusive_recall']:>12.4f}"
              f"{m['thresholds']:>18}")
    print("\n'전체적합'은 같은 데이터로 임계값을 맞추고 채점한 값이다.")
    print("CV 점수와의 차이가 곧 부풀림의 크기다.")


if __name__ == "__main__":
    main()
