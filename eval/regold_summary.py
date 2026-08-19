"""요약 정답 재작성 — 대본을 읽고 후처리 폼 정답을 다시 뽑는다 (Claude Opus).

    uv run python eval/regold_summary.py --limit 3
    uv run python eval/regold_summary.py

기존 정답은 `scenario_v2.md`가 요약과 대본을 **동시에** 생성한 것이라, 대화가 계획과
다르게 흘러가면 어긋난 채 남았다. `needs_followup`은 생성 스펙에서 순번(`i % 10`)으로
기계 배정된 값이다.

대본을 읽고 다시 쓰면 구조적으로 불일치가 사라진다.

출력: eval/datasets/summary_gold_v2.jsonl (기존 정답은 그대로 둔다 — 비교해야 하므로)
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

AI = Path(__file__).resolve().parents[1]
DATASET = AI / "eval/datasets/summary_dataset.jsonl"
PROMPT = AI / "eval/prompts/gold_summary_v1.md"
CACHE_ROOT = AI / "data/scenarios/regold"
OUT_ROOT = AI / "eval/datasets"

FIELDS = ("title", "inquiry", "resolution")


def load_token() -> str:
    env = AI / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("CLAUDE_CODE_OAUTH_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


def ask(prompt: str, model: str, effort: str = "low") -> str | None:
    cmd = ["env", "-u", "ANTHROPIC_BASE_URL", "-u", "CLAUDECODE", "-u", "CLAUDE_CODE_SESSION_ID"]
    if tok := load_token():
        cmd.append(f"CLAUDE_CODE_OAUTH_TOKEN={tok}")
    try:
        r = subprocess.run(cmd + ["claude", "-p", "--model", model, "--effort", effort],
                           input=prompt, capture_output=True, text=True, timeout=300)
        return r.stdout if r.returncode == 0 else None
    except subprocess.TimeoutExpired:
        return None


def parse(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(isinstance(d.get(f), str) and d[f].strip() for f in FIELDS):
        return None
    if not all(isinstance(d.get(b), bool) for b in ("needs_followup", "needs_faq")):
        return None
    return {**{f: d[f].strip() for f in FIELDS},
            "needs_followup": d["needs_followup"], "needs_faq": d["needs_faq"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="opus")
    ap.add_argument("--effort", default="low", choices=("low", "medium", "high"),
                    help="사고 수준. 대본을 읽고 정리하는 작업이라 low로 충분하다")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    tmpl = PROMPT.read_text(encoding="utf-8").split("\n## 프롬프트\n", 1)[1].strip()
    cache = CACHE_ROOT if args.model == "opus" else CACHE_ROOT.parent / f"regold_{args.model}"
    cache.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in DATASET.open(encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"모델 {args.model} | 사고 {args.effort} | {len(rows)}건")

    out, failed, t0 = [], [], time.time()
    for i, r in enumerate(rows, 1):
        cf = cache / f"{r['scenario_id']}.json"
        got = json.loads(cf.read_text(encoding="utf-8")) if cf.exists() else None
        if got is None:
            dialog = "\n".join(f"[{u['index']}] {u['role']}: {u['text']}" for u in r["utterances"])
            raw = ask(tmpl.format(dialog=dialog), args.model, args.effort)
            got = parse(raw) if raw else None
            if got is None:
                failed.append(r["scenario_id"])
                continue
            cf.write_text(json.dumps(got, ensure_ascii=False), encoding="utf-8")
        out.append({"scenario_id": r["scenario_id"], "source": r["source"],
                    "summary": {f: got[f] for f in FIELDS},
                    "needs_followup": got["needs_followup"], "needs_faq": got["needs_faq"],
                    "text": r["text"], "utterances": r["utterances"]})
        if i % 20 == 0:
            print(f"  [{i}/{len(rows)}] 경과 {time.time()-t0:.0f}초")

    out_path = OUT_ROOT / ("summary_gold_v2.jsonl" if args.model == "opus"
                           else f"summary_gold_v2_{args.model}.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    old = {json.loads(l)["scenario_id"]: json.loads(l) for l in DATASET.open(encoding="utf-8")}
    fu = sum(1 for r in out if r["needs_followup"] != old[r["scenario_id"]]["needs_followup"])
    fq = sum(1 for r in out if r["needs_faq"] != old[r["scenario_id"]]["needs_faq"])
    print(f"\n{len(out)}건 완료 ({time.time()-t0:.0f}초). 실패 {len(failed)}건 {failed[:5]}")
    print(f"  기존 정답과 다른 needs_followup: {fu}건 / needs_faq: {fq}건")
    for f_ in FIELDS:
        lens = [len(r["summary"][f_]) for r in out]
        old_lens = [len(old[r["scenario_id"]]["summary"][f_]) for r in out]
        print(f"  {f_:<11} 새 {sum(lens)/len(lens):5.1f}자  기존 {sum(old_lens)/len(old_lens):5.1f}자")
    print(f"\n저장: {out_path.relative_to(AI)}")


if __name__ == "__main__":
    main()
