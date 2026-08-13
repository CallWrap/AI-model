"""학습 진행상황 기록.

train.py / dapt.py 가 에폭마다 즉시 디스크에 기록한다. 중간에 죽어도 그때까지의
기록이 남고, 학습 중에도 다른 터미널에서 확인할 수 있다.

산출 파일 (run 디렉터리):
    progress.jsonl  에폭당 1줄. 한 번 쓰면 고치지 않는 append-only 기록
    status.json     현재 상태 스냅샷. 매 에폭 덮어쓴다 (외부 모니터링용)
    train.log       콘솔 출력 전문

확인:
    python scripts/watch_progress.py --run runs/v1            # 1회 출력
    python scripts/watch_progress.py --run runs/v1 --follow   # 갱신 감시
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _fmt_hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


class _Tee:
    """stdout을 콘솔과 로그 파일에 동시에 쓴다."""

    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self._stream = stream
        self._log = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class ProgressLogger:
    """에폭 단위 진행상황을 즉시 디스크에 기록한다.

    with 블록으로 쓰면 콘솔 출력이 train.log 에 함께 저장되고, 예외로 중단되어도
    status.json 에 실패 상태가 남는다.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        task: str = "train",
        total_epochs: int | None = None,
        meta: dict | None = None,
        tee_stdout: bool = True,
    ) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.total_epochs = total_epochs
        self.meta = meta or {}

        self.progress_path = self.dir / "progress.jsonl"
        self.status_path = self.dir / "status.json"
        self.log_path = self.dir / "train.log"

        self.start_time = time.time()
        self.records: list[dict] = []
        self.best: dict | None = None
        self._epoch_start = self.start_time

        self._log_fh = self.log_path.open("a", encoding="utf-8")
        self._tee = tee_stdout
        self._orig_stdout: TextIO | None = None

        self._log_fh.write(f"\n{'=' * 70}\n{_now()}  {task} 시작\n{'=' * 70}\n")
        self._log_fh.flush()
        self._write_status("running")

    # -- 컨텍스트 매니저 ---------------------------------------------------
    def __enter__(self) -> "ProgressLogger":
        if self._tee:
            self._orig_stdout = sys.stdout
            sys.stdout = _Tee(sys.stdout, self._log_fh)  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
            self._orig_stdout = None
        if exc_type is not None:
            self._write_status("failed", error=f"{exc_type.__name__}: {exc}")
            self._log_fh.write(f"{_now()}  중단: {exc_type.__name__}: {exc}\n")
        elif self.status().get("state") == "running":
            self._write_status("done")
        self._log_fh.flush()
        self._log_fh.close()
        return False  # 예외를 삼키지 않는다

    # -- 기록 --------------------------------------------------------------
    def start_epoch(self, epoch: int) -> None:
        self._epoch_start = time.time()
        total = f"/{self.total_epochs}" if self.total_epochs else ""
        print(f"\n[epoch {epoch}{total}] 시작 — {_now()}", flush=True)

    def log_epoch(
        self,
        epoch: int,
        *,
        metrics: dict[str, Any],
        is_best: bool = False,
        extra: dict | None = None,
    ) -> dict:
        """에폭 1회분을 progress.jsonl 에 append 하고 status.json 을 갱신한다."""
        elapsed = time.time() - self.start_time
        epoch_sec = time.time() - self._epoch_start

        eta = None
        if self.total_epochs and epoch < self.total_epochs:
            eta = (self.total_epochs - epoch) * epoch_sec

        record = {
            "task": self.task,
            "epoch": epoch,
            "timestamp": _now(),
            "epoch_seconds": round(epoch_sec, 1),
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta, 1) if eta is not None else None,
            "is_best": is_best,
            **metrics,
        }
        if extra:
            record.update(extra)
        record["gpu_peak_gib"] = _gpu_peak_gib()

        with self.progress_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.records.append(record)
        if is_best:
            self.best = record
        self._write_status("running")

        self._print_row(record, eta)
        return record

    def _print_row(self, rec: dict, eta: float | None) -> None:
        bits = [f"epoch {rec['epoch']}"]
        for key, label, fmt in (
            ("train_loss", "loss", "{:.4f}"),
            ("val_loss", "val_loss", "{:.4f}"),
            ("macro_f1", "macro-F1", "{:.4f}"),
            ("sla_violation_rate", "SLA위반", "{:.4f}"),
            ("perplexity", "ppl", "{:.2f}"),
        ):
            if rec.get(key) is not None:
                bits.append(f"{label} {fmt.format(rec[key])}")
        bits.append(f"{_fmt_hms(rec['epoch_seconds'])}")
        if rec.get("gpu_peak_gib"):
            bits.append(f"VRAM {rec['gpu_peak_gib']:.1f}GiB")
        if eta is not None:
            bits.append(f"ETA {_fmt_hms(eta)}")
        if rec["is_best"]:
            bits.append("<- best")
        print("  " + " | ".join(bits), flush=True)

    # -- 상태 --------------------------------------------------------------
    def _write_status(self, state: str, error: str | None = None) -> None:
        status = {
            "task": self.task,
            "state": state,                # running | done | failed | stopped
            "updated_at": _now(),
            "epochs_done": len(self.records),
            "total_epochs": self.total_epochs,
            "elapsed_seconds": round(time.time() - self.start_time, 1),
            "last": self.records[-1] if self.records else None,
            "best": self.best,
            "meta": self.meta,
        }
        if error:
            status["error"] = error
        self.status_path.write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def status(self) -> dict:
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def finish(self, state: str = "done", note: str | None = None) -> None:
        if note:
            print(note, flush=True)
        self._write_status(state)

    # -- 요약 --------------------------------------------------------------
    def summary_table(self) -> str:
        return format_progress_table(self.records)


def _gpu_peak_gib() -> float | None:
    """직전 구간 peak VRAM(GiB). 읽고 나서 카운터를 리셋한다."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    peak = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    return round(peak, 2)


# ---------------------------------------------------------------------------
# 조회 (watch_progress.py 가 사용)
# ---------------------------------------------------------------------------
def read_progress(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "progress.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 기록 도중 잘린 마지막 줄
    return records


def read_status(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "status.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def format_progress_table(records: list[dict]) -> str:
    if not records:
        return "(기록 없음)"

    is_mlm = any(r.get("perplexity") is not None for r in records)
    if is_mlm:
        cols = [("epoch", "epoch", 6, "{:d}"),
                ("train_loss", "loss", 10, "{:.4f}"),
                ("perplexity", "ppl", 10, "{:.2f}")]
    else:
        cols = [("epoch", "epoch", 6, "{:d}"),
                ("train_loss", "loss", 10, "{:.4f}"),
                ("macro_f1", "macro-F1", 10, "{:.4f}"),
                ("sla_violation_rate", "SLA위반", 10, "{:.4f}"),
                ("accuracy", "acc", 8, "{:.3f}")]
    cols += [("epoch_seconds", "시간", 9, None),
             ("gpu_peak_gib", "VRAM", 8, "{:.1f}")]

    header = "".join(f"{label:>{w}}" for _, label, w, _ in cols) + "  "
    lines = [header, "-" * (len(header) + 4)]
    for rec in records:
        row = ""
        for key, _, w, fmt in cols:
            v = rec.get(key)
            if v is None:
                cell = "-"
            elif key == "epoch_seconds":
                cell = _fmt_hms(v)
            else:
                cell = fmt.format(v)
            row += f"{cell:>{w}}"
        lines.append(row + ("  <- best" if rec.get("is_best") else ""))
    return "\n".join(lines)
