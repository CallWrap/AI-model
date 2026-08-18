"""요약·후속처리·FAQ필요 — 온디바이스 LLM 1회 호출로 함께 뽑는다 (명세 Step 2).

    uv run python train/summarize_llm.py --model exaone3.5:7.8b
    uv run python train/summarize_llm.py --model exaone3.5:7.8b --limit 3
    uv run python train/summarize_llm.py --model gemma3:12b --prompt eval/prompts/summary_v2.md

F1 요약(제목·문의내용·처리내용)·F4 후속처리·FAQ필요를 한 번의 호출로 받는다.
같은 문맥을 세 번 읽을 이유가 없다.

**파싱 실패 시 1회 재시도한다** (명세 Step 2). 그래도 실패하면 그 건을 건너뛰고
리포트에 남긴다 — 부분 실패를 허용한다.

통화별 응답을 캐시하므로 중간에 끊겨도 이어서 돌릴 수 있다.
temperature=0 + seed 고정으로 같은 입력에 같은 출력이 나온다.
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

AI = Path(__file__).resolve().parents[1]
DATASET = AI / "eval/datasets/summary_dataset.jsonl"
CACHE_ROOT = AI / "data/scenarios/llm_summary"
DEFAULT_PROMPT = AI / "eval/prompts/summary_v1.md"
API = "http://127.0.0.1:11434/api/generate"

FIELDS = ("title", "inquiry", "resolution")


def load_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if "\n## 프롬프트\n" not in text:
        raise SystemExit(f"'## 프롬프트' 절이 없습니다: {path}")
    return text.split("\n## 프롬프트\n", 1)[1].strip()


def ask(model: str, prompt: str, timeout: int = 300) -> str | None:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "seed": 42}}).encode()
    req = urllib.request.Request(API, body, {"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["response"]
    except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
        return None


def parse(raw: str, n_utt: int) -> dict | None:
    """JSON 한 덩어리를 꺼내 검증한다. 필드가 빠지거나 형이 안 맞으면 None."""
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
    ev = d.get("evidence_indices") or []
    ev = [i for i in ev if isinstance(i, int) and 0 <= i < n_utt]   # 없는 발화 번호는 버린다
    return {**{f: d[f].strip() for f in FIELDS},
            "needs_followup": d["needs_followup"], "needs_faq": d["needs_faq"],
            "evidence_indices": ev[:3]}


def build_dialog(utterances: list[dict]) -> str:
    """발화에 번호를 붙인다 — evidence_indices를 받기 위해서다."""
    return "\n".join(f"[{u['index']}] {u['role']}: {u['text']}" for u in utterances)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="exaone3.5:7.8b")
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    tmpl = load_prompt(args.prompt)
    cache = CACHE_ROOT / f"{args.model.replace(':', '_')}__{args.prompt.stem}"
    cache.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in DATASET.open(encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"모델 {args.model} | 프롬프트 {args.prompt.name} | {len(rows)}건")

    out_rows, failed, retried, t0 = [], [], 0, time.time()
    for i, r in enumerate(rows, 1):
        cf = cache / f"{r['scenario_id']}.json"
        got = json.loads(cf.read_text(encoding="utf-8")) if cf.exists() else None

        if got is None:
            prompt = tmpl.format(dialog=build_dialog(r["utterances"]))
            raw = ask(args.model, prompt)
            got = parse(raw, len(r["utterances"])) if raw else None
            if got is None:                       # 재시도 1회 (명세 Step 2)
                retried += 1
                raw = ask(args.model, prompt + "\n\n반드시 JSON 하나만 출력하세요.")
                got = parse(raw, len(r["utterances"])) if raw else None
            if got is None:
                failed.append(r["scenario_id"])
                continue
            cf.write_text(json.dumps(got, ensure_ascii=False), encoding="utf-8")

        out_rows.append({"scenario_id": r["scenario_id"], "source": r["source"], "pred": got})
        if i % 20 == 0:
            print(f"  [{i}/{len(rows)}] 경과 {time.time()-t0:.0f}초")

    out = CACHE_ROOT / f"pred_{args.model.replace(':', '_')}__{args.prompt.stem}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{len(out_rows)}건 완료 ({time.time()-t0:.0f}초). 재시도 {retried}건, 실패 {len(failed)}건 {failed[:5]}")
    if out_rows:
        for f_ in FIELDS:
            lens = [len(r["pred"][f_]) for r in out_rows]
            print(f"  {f_:<11} 평균 {sum(lens)/len(lens):5.1f}자")
        nf = sum(r["pred"]["needs_followup"] for r in out_rows)
        nq = sum(r["pred"]["needs_faq"] for r in out_rows)
        print(f"  needs_followup 참 {nf}/{len(out_rows)}   needs_faq 참 {nq}/{len(out_rows)}")
    print(f"\n저장: {out.relative_to(AI)}")


if __name__ == "__main__":
    main()
