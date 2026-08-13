"""생성분을 병합·검증하고 학습용 데이터셋을 만든다.

하는 일:
  1. data/gen_*.jsonl 과 지정한 추가 파일을 모두 읽는다
  2. 스키마 검증 — 잘못된 건은 버리고 사유를 보고한다
  3. sr_final 을 우선순위 규칙으로 **재계산**한다 (생성기가 틀렸을 수 있다)
  4. 중복 제거 (발화 내용 해시)
  5. 층화 분할로 train / val 저장

사용:
    python scripts/build_dataset.py --out data/train.jsonl --val-out data/val.jsonl
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from callwrap.dataset import save_jsonl, stratified_split  # noqa: E402
from callwrap.labels import LABELS, collapse  # noqa: E402

VALID_SPEAKERS = {"고객", "상담사"}
VALID_DIFFICULTY = {"일반", "경계", "복합"}


def validate(rec: dict, idx: int, src: str) -> tuple[dict | None, str | None]:
    """한 건을 검증하고 정규화한다. (정상 레코드, 오류사유) 반환."""
    rid = rec.get("id", f"{src}#{idx}")

    labels = rec.get("sr_labels")
    if not isinstance(labels, list) or not labels:
        return None, f"{rid}: sr_labels 없음"
    unknown = [l for l in labels if l not in LABELS]
    if unknown:
        return None, f"{rid}: 알 수 없는 라벨 {unknown}"

    utts = rec.get("utterances")
    if not isinstance(utts, list) or len(utts) < 4:
        return None, f"{rid}: 발화 부족 ({len(utts) if isinstance(utts, list) else 0}턴)"
    for u in utts:
        if not isinstance(u, dict) or u.get("speaker") not in VALID_SPEAKERS:
            return None, f"{rid}: 화자 값 오류"
        if not str(u.get("text", "")).strip():
            return None, f"{rid}: 빈 발화"

    # 생성기가 축약을 틀렸을 수 있으므로 규칙으로 재계산한다.
    expected = collapse(labels)
    fixed = rec.get("sr_final") != expected

    meta = rec.get("meta") or {}
    if meta.get("difficulty") not in VALID_DIFFICULTY:
        meta["difficulty"] = "일반"

    out = {
        "id": rid,
        "sr_labels": labels,
        "sr_final": expected,
        "target_label": rec.get("target_label", labels[0]),
        "utterances": [{"speaker": u["speaker"], "text": str(u["text"]).strip()}
                       for u in utts],
        "evidence_indices": rec.get("evidence_indices", []),
        "rationale": rec.get("rationale", ""),
        "meta": {**meta, "source_file": src, "collapse_fixed": fixed},
    }
    return out, None


def content_key(rec: dict) -> str:
    body = " ".join(u["text"] for u in rec["utterances"])
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description="생성분 병합 · 검증 · 분할")
    ap.add_argument("--glob", default="data/gen_*.jsonl",
                    help="병합할 파일 패턴")
    ap.add_argument("--extra", type=Path, action="append", default=[],
                    help="추가로 포함할 파일 (반복 가능)")
    ap.add_argument("--out", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--val-out", type=Path, default=Path("data/val.jsonl"))
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = sorted(Path().glob(args.glob)) + list(args.extra)
    if not paths:
        raise SystemExit(f"입력 파일이 없습니다: {args.glob}")

    records: list[dict] = []
    errors: list[str] = []
    per_file: dict[str, int] = {}

    for p in paths:
        n_ok = 0
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{p.name}#{i}: JSON 파싱 실패 ({exc.msg})")
                continue
            rec, err = validate(raw, i, p.name)
            if err:
                errors.append(err)
            else:
                records.append(rec)
                n_ok += 1
        per_file[p.name] = n_ok

    # 중복 제거
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in records:
        k = content_key(r)
        if k in seen:
            errors.append(f"{r['id']}: 내용 중복")
            continue
        seen.add(k)
        deduped.append(r)

    print("[파일별 유효 건수]")
    for name, n in per_file.items():
        print(f"  {name:<24} {n:>4}건")
    print(f"\n총 {len(deduped)}건 (중복·오류 제외)")

    fixed = sum(1 for r in deduped if r["meta"].get("collapse_fixed"))
    if fixed:
        print(f"  ⚠ sr_final 재계산으로 교정된 건: {fixed}건")

    if errors:
        print(f"\n[제외 {len(errors)}건]")
        for e in errors[:15]:
            print("  -", e)
        if len(errors) > 15:
            print(f"  ... 외 {len(errors) - 15}건")

    print("\n[최종 라벨 분포]")
    dist = collections.Counter(r["sr_final"] for r in deduped)
    for name in LABELS:
        n = dist.get(name, 0)
        bar = "#" * int(n / max(dist.values()) * 30) if dist else ""
        print(f"  {name:<18} {n:>4}  {bar}")

    print("\n[난이도]", dict(collections.Counter(
        r["meta"].get("difficulty") for r in deduped)))
    print("[소분류]", len({r["meta"].get("sub_category") for r in deduped}), "종")
    print("[멀티라벨]", sum(1 for r in deduped if len(r["sr_labels"]) > 1), "건")

    missing = [l for l in LABELS if dist.get(l, 0) == 0]
    if missing:
        print(f"\n  ⚠ 데이터가 없는 라벨: {missing}")

    train, val = stratified_split(deduped, args.val_ratio, args.seed)
    save_jsonl(train, args.out)
    save_jsonl(val, args.val_out)
    print(f"\n학습 {len(train)}건 -> {args.out}")
    print(f"검증 {len(val)}건 -> {args.val_out}")
    print("\n다음 단계:")
    print(f"  python -m callwrap.baseline --data {args.out} "
          f"--val-data {args.val_out} --emit-banlist configs/banned_phrases.json")


if __name__ == "__main__":
    main()
