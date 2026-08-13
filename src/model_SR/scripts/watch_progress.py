"""학습 진행상황 조회.

학습이 도는 동안 **다른 터미널에서** 실행해 상태를 확인한다.
progress.jsonl 은 에폭마다 append 되므로 학습 프로세스를 건드리지 않는다.

사용:
    python scripts/watch_progress.py --run runs/v1              # 1회 출력
    python scripts/watch_progress.py --run runs/v1 --follow     # 갱신 감시
    python scripts/watch_progress.py --all                      # runs/ 전체 요약
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from callwrap.progress import (format_progress_table, read_progress,  # noqa: E402
                               read_status)

STATE_MARK = {"running": "●", "done": "✔", "failed": "✘",
              "stopped": "■", "unknown": "?"}


def _fmt_hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def show_run(run_dir: Path, show_meta: bool = True) -> str:
    status = read_status(run_dir)
    records = read_progress(run_dir)

    if not status and not records:
        return f"{run_dir}: 기록 없음 (아직 시작 전이거나 잘못된 경로)"

    state = status.get("state", "unknown")
    mark = STATE_MARK.get(state, "?")
    done = status.get("epochs_done", len(records))
    total = status.get("total_epochs")
    total_s = f"/{total}" if total else ""

    lines = [
        "=" * 72,
        f"{mark} {run_dir}  [{state}]  epoch {done}{total_s}"
        f"  경과 {_fmt_hms(status.get('elapsed_seconds', 0))}",
        f"  갱신: {status.get('updated_at', '-')}",
    ]

    if show_meta and status.get("meta"):
        meta = status["meta"]
        keys = ("backbone", "device", "n_train", "n_val", "batch_size",
                "lr", "max_length", "noise_prob", "n_blocks")
        bits = [f"{k}={meta[k]}" for k in keys if k in meta]
        if bits:
            lines.append("  " + " | ".join(bits))
        if meta.get("val_kind"):
            lines.append(f"  검증셋: {meta['val_kind']}")

    if status.get("error"):
        lines.append(f"  오류: {status['error']}")

    lines += ["", format_progress_table(records)]

    best = status.get("best")
    if best:
        if best.get("macro_f1") is not None:
            lines.append(
                f"\n  best: epoch {best['epoch']}  macro-F1 {best['macro_f1']:.4f}"
                f"  SLA위반 {best.get('sla_violation_rate', 0):.4f}"
            )
        elif best.get("perplexity") is not None:
            lines.append(f"\n  best: epoch {best['epoch']}  "
                         f"loss {best['train_loss']:.4f}  "
                         f"ppl {best['perplexity']:.2f}")

    last = status.get("last") or {}
    if state == "running" and last.get("eta_seconds"):
        lines.append(f"  남은 예상 시간: {_fmt_hms(last['eta_seconds'])}")

    return "\n".join(lines)


def list_all(runs_root: Path) -> str:
    if not runs_root.exists():
        return f"{runs_root} 없음"
    dirs = sorted(d for d in runs_root.iterdir()
                  if d.is_dir() and (d / "status.json").exists())
    if not dirs:
        return f"{runs_root} 아래에 기록된 run이 없습니다."

    rows = [f"{'run':<26}{'상태':<10}{'epoch':>8}{'best':>12}{'경과':>10}",
            "-" * 68]
    for d in dirs:
        st = read_status(d)
        best = st.get("best") or {}
        score = best.get("macro_f1", best.get("perplexity"))
        rows.append(
            f"{d.name:<26}"
            f"{STATE_MARK.get(st.get('state', 'unknown'), '?')} "
            f"{st.get('state', '-'):<8}"
            f"{st.get('epochs_done', 0):>8}"
            f"{(f'{score:.4f}' if score is not None else '-'):>12}"
            f"{_fmt_hms(st.get('elapsed_seconds', 0)):>10}"
        )
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="학습 진행상황 조회")
    ap.add_argument("--run", type=Path, default=None, help="run 디렉터리")
    ap.add_argument("--all", action="store_true", help="runs/ 전체 요약")
    ap.add_argument("--runs-root", type=Path, default=Path("runs"))
    ap.add_argument("--follow", "-f", action="store_true", help="갱신 감시")
    ap.add_argument("--interval", type=float, default=10.0, help="감시 주기(초)")
    args = ap.parse_args()

    if args.all or args.run is None:
        print(list_all(args.runs_root))
        if not args.follow:
            return

    target = args.run
    if target is None:
        return

    if not args.follow:
        print(show_run(target))
        return

    try:
        last_seen = -1
        while True:
            status = read_status(target)
            done = status.get("epochs_done", 0)
            if done != last_seen:
                print("\033[2J\033[H", end="")  # 화면 지우기
                print(show_run(target))
                last_seen = done
            if status.get("state") in {"done", "failed", "stopped"}:
                print(f"\n[{status['state']}] 감시를 종료합니다.")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n감시 중단 (학습은 계속 진행됩니다).")


if __name__ == "__main__":
    main()
