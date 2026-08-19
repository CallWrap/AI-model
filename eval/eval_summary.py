"""요약·후속처리·FAQ필요 채점 — 프롬프트 개선을 위한 실패 사례 추출까지.

    uv run --extra train python eval/eval_summary.py
    uv run --extra train python eval/eval_summary.py --show 8        # 실패 사례 더 보기
    uv run --extra train python eval/eval_summary.py --pred A.jsonl B.jsonl   # 프롬프트 비교

**ROUGE는 참고용이다** (명세 Step 2). 요약은 정답이 하나가 아니라 문자 n-gram 겹침으로는
품질을 다 못 잰다. 이 스크립트의 진짜 목적은 **어디서 왜 틀렸는지 모아 보는 것**이다.
3건 감으로 프롬프트를 고치면 그 3건에만 맞춰진다.

자동으로 잡는 실패 유형:
  화자 혼입   문의내용에 상담사 행위가, 처리내용에 고객 반응이 섞임 (명세 9.2와 같은 종류)
  내용 누락   정답 대비 재현율이 낮음 — 여러 문의 중 일부만 담은 경우
  분량 이탈   목표 길이에서 크게 벗어남
  bool 오답   needs_followup / needs_faq
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

AI = Path(__file__).resolve().parents[1]
GOLD = AI / "eval/datasets/summary_dataset.jsonl"
PRED_DIR = AI / "data/scenarios/llm_summary"

FIELDS = ("title", "inquiry", "resolution")
TARGET_LEN = {"title": 21, "inquiry": 92, "resolution": 98}

# 문의내용에 있으면 안 되는 상담사 행위 (화자 혼입)
COUNSELOR_ACT = re.compile(r"(안내하|안내드|안내함|안내했|확인해 드|설명하|설명했|접수하|드렸|알려 ?드)")
# 처리내용에 있으면 안 되는 고객 반응
CUSTOMER_REACT = re.compile(r"(불만|항의|짜증|화를 냈|언성|불평|요구하며|따졌)")


def grams(s: str, n: int) -> list[str]:
    s = "".join(s.split())
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def overlap_pr(ref: str, hyp: str, n: int) -> tuple[float, float]:
    """(정밀도, 재현율). 재현율이 낮으면 정답 내용을 빠뜨린 것이다."""
    r, h = grams(ref, n), grams(hyp, n)
    if not r or not h:
        return 0.0, 0.0
    pool, hit = list(h), 0
    for g in r:
        if g in pool:
            pool.remove(g)
            hit += 1
    return hit / len(h), hit / len(r)


def f1(ref: str, hyp: str, n: int) -> float:
    p, r = overlap_pr(ref, hyp, n)
    return 2 * p * r / (p + r) if p + r else 0.0


def diagnose(g: dict, q: dict) -> list[str]:
    """이 건에서 무엇이 잘못됐는지 이름표를 붙인다."""
    tags = []
    if COUNSELOR_ACT.search(q["inquiry"]):
        tags.append("문의에 상담사 행위")
    if CUSTOMER_REACT.search(q["resolution"]):
        tags.append("처리에 고객 반응")
    for f in ("inquiry", "resolution"):
        if overlap_pr(g["summary"][f], q[f], 2)[1] < 0.30:
            tags.append(f"{f} 누락")
        if len(q[f]) > TARGET_LEN[f] * 1.7:
            tags.append(f"{f} 과다")
        elif len(q[f]) < TARGET_LEN[f] * 0.5:
            tags.append(f"{f} 과소")
    if g["needs_followup"] != q["needs_followup"]:
        tags.append(f"후속 오답({g['needs_followup']}→{q['needs_followup']})")
    if g["needs_faq"] != q["needs_faq"]:
        tags.append(f"FAQ 오답({g['needs_faq']}→{q['needs_faq']})")
    return tags


def bool_scores(pairs: list[tuple[bool, bool]]) -> dict:
    tp = sum(1 for g, p in pairs if g and p)
    fp = sum(1 for g, p in pairs if not g and p)
    fn = sum(1 for g, p in pairs if g and not p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"acc": sum(g == p for g, p in pairs) / len(pairs),
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "prec": prec, "recall": rec}


def evaluate(gold: dict, pred_path: Path, show: int) -> None:
    rows = [json.loads(l) for l in pred_path.open(encoding="utf-8")]
    rows = [r for r in rows if r["scenario_id"] in gold]
    if not rows:
        print(f"  {pred_path.name}: 채점 대상 없음")
        return

    print(f"\n{'=' * 78}\n{pred_path.stem}  ({len(rows)}건)\n{'=' * 78}")
    print(f"{'필드':<12}{'ROUGE-1':>9}{'ROUGE-2':>9}{'재현율':>9}{'평균길이':>10}{'목표':>7}")
    for f in FIELDS:
        r1 = sum(f1(gold[r["scenario_id"]]["summary"][f], r["pred"][f], 1) for r in rows) / len(rows)
        r2 = sum(f1(gold[r["scenario_id"]]["summary"][f], r["pred"][f], 2) for r in rows) / len(rows)
        rec = sum(overlap_pr(gold[r["scenario_id"]]["summary"][f], r["pred"][f], 2)[1] for r in rows) / len(rows)
        ln = sum(len(r["pred"][f]) for r in rows) / len(rows)
        print(f"{f:<12}{r1:>9.3f}{r2:>9.3f}{rec:>9.3f}{ln:>10.1f}{TARGET_LEN[f]:>7}")

    print()
    for b in ("needs_followup", "needs_faq"):
        s = bool_scores([(gold[r["scenario_id"]][b], r["pred"][b]) for r in rows])
        print(f"{b:<16} Acc {s['acc']:.3f}  F1 {s['f1']:.3f}  "
              f"(정밀도 {s['prec']:.3f} / 재현율 {s['recall']:.3f})")

    bad_ev = [r["scenario_id"] for r in rows if not r["pred"]["evidence_indices"]]
    print(f"근거 발화 없음: {len(bad_ev)}건" + (f" {bad_ev[:5]}" if bad_ev else ""))

    diags = {r["scenario_id"]: diagnose(gold[r["scenario_id"]], r["pred"]) for r in rows}
    tally = Counter(t for ts in diags.values() for t in ts)
    clean = sum(1 for ts in diags.values() if not ts)
    print(f"\n문제 없는 건: {clean}/{len(rows)} ({clean/len(rows):.1%})")
    print("실패 유형 빈도:")
    for tag, n in tally.most_common():
        print(f"  {n:4d}  {tag}")

    worst = sorted(((len(ts), sid) for sid, ts in diags.items() if ts), reverse=True)[:show]
    if worst:
        print(f"\n문제가 많은 건 {len(worst)}개 — 프롬프트 개선 재료")
        for _, sid in worst:
            g, q = gold[sid], next(r["pred"] for r in rows if r["scenario_id"] == sid)
            print(f"\n  [{sid}] {', '.join(diags[sid])}")
            print(f"    문의 정답: {g['summary']['inquiry'][:90]}")
            print(f"    문의 예측: {q['inquiry'][:90]}")
            print(f"    처리 예측: {q['resolution'][:90]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="*", type=Path, help="비우면 llm_summary의 pred_*.jsonl 전부")
    ap.add_argument("--show", type=int, default=5, help="실패 사례 표시 개수")
    ap.add_argument("--gold", type=Path, default=GOLD, help="정답 파일 (재작성본 비교용)")
    ap.add_argument("--only", type=Path, help="이 파일에 있는 scenario_id만 채점")
    args = ap.parse_args()

    gold = {json.loads(l)["scenario_id"]: json.loads(l) for l in args.gold.open(encoding="utf-8")}
    if args.only:
        keep = {json.loads(l)["scenario_id"] for l in args.only.open(encoding="utf-8")}
        gold = {k: v for k, v in gold.items() if k in keep}
    preds = args.pred or sorted(PRED_DIR.glob("pred_*.jsonl"))
    if not preds:
        raise SystemExit(f"예측 파일이 없습니다: {PRED_DIR}")
    print(f"정답 {len(gold)}건 ({args.gold.name}) | 채점 대상 {len(preds)}개")
    for p in preds:
        evaluate(gold, p, args.show)


if __name__ == "__main__":
    main()
