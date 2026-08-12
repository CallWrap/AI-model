"""best 체크포인트의 혼동행렬을 이미지로 그린다.

val_probs.npy 는 학습 중 best 갱신 시점에만 저장되므로, 이 스크립트가 그리는
행렬은 항상 config.json 의 best epoch 결과와 일치한다.

사용:
    python scripts/plot_confusion.py --run runs/v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from callwrap.evaluate import confusion_matrix_8x8  # noqa: E402
from callwrap.labels import PRIORITY, collapse_batch  # noqa: E402

FONT_CANDIDATES = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"]


def _set_korean_font() -> None:
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for name in FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            return
    print("경고: 한글 폰트를 찾지 못했다 — 라벨이 네모(tofu)로 나올 수 있다.")


def main() -> None:
    ap = argparse.ArgumentParser(description="best 체크포인트 혼동행렬 시각화")
    ap.add_argument("--run", type=Path, default=Path("runs/v2"))
    ap.add_argument("--out", type=Path, default=None,
                    help="기본값: <run>/confusion_matrix.png")
    args = ap.parse_args()

    run = args.run
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    thresholds = np.asarray(config["thresholds"], dtype=np.float64)
    best = config["best"]

    probs = np.load(run / "val_probs.npy")
    y_true = json.loads((run / "val_true.json").read_text(encoding="utf-8"))
    y_pred = collapse_batch(probs, thresholds)

    cm = confusion_matrix_8x8(y_true, y_pred)
    n = len(PRIORITY)

    _set_korean_font()
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(PRIORITY, rotation=45, ha="right")
    ax.set_yticklabels(PRIORITY)
    ax.set_xlabel("예측 (Predicted)")
    ax.set_ylabel("실제 (Actual)")

    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            color = "white" if v > vmax * 0.6 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, str(v), ha="center", va="center",
                     color=color, fontweight=weight, fontsize=10)

    # 대각선 아래(행 우선순위 > 열 우선순위)는 짧은 SLA로 잘못 예측한 위반 방향.
    for i in range(1, n):
        for j in range(i):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False, edgecolor="red",
                                        linewidth=1.2, alpha=0.5))

    title = (f"SR유형 혼동행렬 — best epoch {best['epoch']}  "
             f"(macro-F1 {best['macro_f1']:.3f}, "
             f"SLA위반율 {best['sla_violation_rate']:.3f})\n"
             f"행=실제 / 열=예측 / 우선순위 순 정렬 / 빨간 테두리=SLA 위반 방향")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out = args.out or (run / "confusion_matrix.png")
    fig.savefig(out, dpi=150)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
