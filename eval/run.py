"""평가 CLI — 골든셋으로 analyze()를 채점한다. (개발명세 Step 1 / 7장)

  uv run python eval/run.py                 # 전 과제
  uv run python eval/run.py --task f2       # 분류만
  uv run python eval/run.py --gold data/scenarios/golden_confirmed.jsonl
  uv run python eval/run.py --limit 20      # 빠른 확인

지표 (명세 7장):
  F1 요약   : ROUGE 근사(참고용). 정성 평가(LLM-as-judge)는 별도
  F2 분류   : Macro-F1, ECE          목표 Macro-F1 >= 0.85 (중분류 기준)
  F3 등급   : Macro-F1 + 악성콜 Recall(우선 지표)
  F4 후속   : Accuracy, F1

정답 파일은 검수 전이면 gold_draft.jsonl(초안), 검수 후면 확정본을 쓴다.
초안으로 낸 점수는 참고치일 뿐 성능 근거가 아니다.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from callwrap_ai import Transcript, analyze

AI_DIR = Path(__file__).resolve().parents[1]
SCN_DIR = AI_DIR / "data/scenarios"
TASKS = ("f1", "f2", "f3", "f4")

# ── 지표 ────────────────────────────────────────────


def prf(pairs: list[tuple[str, str]]) -> tuple[float, dict]:
    """(정답, 예측) 목록 → Macro-F1과 클래스별 P/R/F1·지지도."""
    labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    per = {}
    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[lab] = {"P": precision, "R": recall, "F1": f1, "n": tp + fn}
    # Macro-F1은 정답에 등장한 클래스만 대상 (예측만 한 유령 클래스 제외)
    gold_labels = {g for g, _ in pairs}
    macro = sum(per[l]["F1"] for l in gold_labels) / len(gold_labels) if gold_labels else 0.0
    return macro, per


def accuracy(pairs: list[tuple]) -> float:
    return sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0.0


def ece(items: list[tuple[bool, float]], bins: int = 10) -> float:
    """Expected Calibration Error — 신뢰도와 실제 정확도의 괴리."""
    if not items:
        return 0.0
    buckets = defaultdict(list)
    for correct, conf in items:
        idx = min(int(conf * bins), bins - 1)
        buckets[idx].append((correct, conf))
    total = len(items)
    err = 0.0
    for group in buckets.values():
        acc = sum(1 for c, _ in group if c) / len(group)
        avg_conf = sum(cf for _, cf in group) / len(group)
        err += len(group) / total * abs(acc - avg_conf)
    return err


def rouge_like(ref: str, hyp: str) -> tuple[float, float]:
    """한국어 형태소 분석기 없이 쓰는 근사치 — 문자 uni/bi-gram F1. 참고용 지표."""

    def grams(s: str, n: int) -> list[str]:
        s = "".join(s.split())
        return [s[i : i + n] for i in range(len(s) - n + 1)]

    def f1(n: int) -> float:
        r, h = grams(ref, n), grams(hyp, n)
        if not r or not h:
            return 0.0
        overlap = 0
        pool = list(h)
        for g in r:
            if g in pool:
                pool.remove(g)
                overlap += 1
        precision, recall = overlap / len(h), overlap / len(r)
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return f1(1), f1(2)


# ── 로딩 ────────────────────────────────────────────


def load(gold_path: Path, limit: int | None) -> list[tuple[dict, Transcript]]:
    golds = {}
    for line in gold_path.open(encoding="utf-8"):
        g = json.loads(line)
        golds[g["scenario_id"]] = g

    pairs = []
    for line in (SCN_DIR / "transcripts.jsonl").open(encoding="utf-8"):
        rec = json.loads(line)
        gold = golds.get(rec["scenario_id"])
        if gold is None:
            continue  # 골든셋에 없는 콜(큐레이션 탈락분)은 평가 제외
        pairs.append((gold, Transcript.model_validate(rec["transcript"])))
        if limit and len(pairs) >= limit:
            break
    return pairs


# ── 리포트 ──────────────────────────────────────────


def report_classification(name: str, pairs: list[tuple[str, str]], confs: list[tuple[bool, float]], target: float | None = None) -> None:
    macro, per = prf(pairs)
    line = f"\n[{name}] Macro-F1 {macro:.3f}   Acc {accuracy(pairs):.3f}   ECE {ece(confs):.3f}   n={len(pairs)}"
    if target:
        line += f"   (목표 {target})" + ("  ✅" if macro >= target else "  ❌")
    print(line)
    print(f"  {'클래스':<20}{'P':>7}{'R':>7}{'F1':>7}{'n':>5}")
    for lab, m in sorted(per.items(), key=lambda x: -x[1]["n"]):
        if m["n"] == 0 and m["F1"] == 0:
            continue
        print(f"  {lab:<20}{m['P']:>7.2f}{m['R']:>7.2f}{m['F1']:>7.2f}{m['n']:>5}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=(*TASKS, "all"), default="all")
    ap.add_argument("--gold", type=Path, default=AI_DIR / "eval/datasets/gold_draft.jsonl")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    data = load(args.gold, args.limit)
    if not data:
        print("평가할 데이터 없음 — to_transcript.py / build_review_sheet.py를 먼저 실행하세요.")
        return

    statuses = {g.get("label_status") for g, _ in data}
    print(f"골든셋 {len(data)}건  (정답 파일: {args.gold.name}, 라벨 상태: {', '.join(sorted(map(str, statuses)))})")
    if statuses & {"기계초안", "생성라벨"}:
        print("⚠️  사람 검수 전 초안 라벨입니다. 점수는 참고치이며 성능 근거로 쓰지 마세요. (명세 9.1)")

    results = [(gold, analyze(tr)) for gold, tr in data]
    failures = [g["scenario_id"] for g, r in results if r.failures]
    if failures:
        print(f"부분 실패: {len(failures)}건 {failures[:5]}")

    want = TASKS if args.task == "all" else (args.task,)

    if "f2" in want:
        for field, name, target in (
            ("consultation_type", "F2 상담유형", 0.85),
            ("consultation_category", "F2 중분류", 0.85),
            ("consultation_subcategory", "F2 소분류(보조)", None),
        ):
            pairs, confs = [], []
            for gold, res in results:
                pred = getattr(res, field)
                if pred is None or gold.get(field) is None:
                    continue
                gv, pv = gold[field], pred.value.value
                pairs.append((gv, pv))
                confs.append((gv == pv, pred.confidence))
            if pairs:
                report_classification(name, pairs, confs, target)

    if "f3" in want:
        pairs, confs = [], []
        for gold, res in results:
            if res.grade is None or gold.get("grade") is None:
                continue
            gv, pv = gold["grade"], res.grade.value.value
            pairs.append((gv, pv))
            confs.append((gv == pv, res.grade.confidence))
        if pairs:
            report_classification("F3 상담등급", pairs, confs)
            _, per = prf(pairs)
            if "악성콜" in per:
                print(f"  ★ 악성콜 Recall {per['악성콜']['R']:.3f} (우선 지표)")

    if "f4" in want:
        for field, name in (("needs_followup", "F4 후속처리"), ("needs_faq", "FAQ 생성 필요")):
            pairs, confs = [], []
            for gold, res in results:
                pred = getattr(res, field)
                if pred is None or gold.get(field) is None:
                    continue
                gv, pv = str(gold[field]), str(pred.value)
                pairs.append((gv, pv))
                confs.append((gv == pv, pred.confidence))
            if pairs:
                report_classification(name, pairs, confs)

    if "f1" in want:
        scores = defaultdict(list)
        for gold, res in results:
            if res.summary is None:
                continue
            gs, ps = gold.get("summary", {}), res.summary.value
            for key, hyp in (("title", ps.title), ("inquiry", ps.inquiry), ("resolution", ps.resolution)):
                ref = gs.get(key) or ""
                if ref:
                    r1, r2 = rouge_like(ref, hyp)
                    scores[key].append((r1, r2))
        if scores:
            print("\n[F1 요약] 문자 n-gram 근사 ROUGE (참고용 — 정성 평가는 LLM-as-judge로 별도)")
            print(f"  {'항목':<12}{'R-1':>8}{'R-2':>8}{'n':>5}")
            for key, vals in scores.items():
                r1 = sum(v[0] for v in vals) / len(vals)
                r2 = sum(v[1] for v in vals) / len(vals)
                print(f"  {key:<12}{r1:>8.3f}{r2:>8.3f}{len(vals):>5}")


if __name__ == "__main__":
    main()
