"""발화별 감정 점수 — 온디바이스 LLM(ollama) 판정. sentiment_score.py의 비교군이다.

    uv run python train/sentiment_llm.py --model exaone3.5:7.8b
    uv run python train/sentiment_llm.py --model gemma3:12b

전용 감정 모델과 달리 학습 데이터가 필요 없고, 무감정/중립을 이해한다.
대신 점수의 근거가 "모델이 그렇게 판단했다"뿐이고 값이 특정 구간에 뭉칠 수 있다
(명세 9.3이 경고한 성질). 그래서 본 후보가 아니라 비교군으로 둔다.

통화 하나를 **한 번의 요청**으로 처리한다. 발화마다 호출하면 186건 × 7발화 = 1300회가
되지만, 통화 단위로 묶으면 186회다.

temperature=0 + seed 고정으로 재현성을 확보한다. 통화별 응답을 캐시하므로
중간에 끊겨도 이어서 돌릴 수 있다.
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

from preprocess import strip_emotion_markers

AI = Path(__file__).resolve().parents[1]
DATASET = AI / "eval/datasets/grade_dataset.jsonl"
CACHE_ROOT = AI / "data/scenarios/llm_sentiment"
API = "http://localhost:11434/api/generate"

PROMPT = """다음은 상담 통화에서 민원인(고객)이 말한 발화 목록이다.
각 발화에 담긴 감정을 -1.0(매우 부정적·화남)에서 1.0(매우 긍정적·만족) 사이 실수로 매겨라.
단순한 사실 질문이나 정보 요청은 0.0 근처로 둔다.

발화:
{lines}

JSON 배열로만 답하라. 설명 금지. 형식: [{{"n":1,"score":-0.8}}, ...]"""


def ask(model: str, prompt: str, timeout: int = 300) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "seed": 42}}).encode()
    req = urllib.request.Request(API, body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["response"]


def parse(raw: str, n: int) -> list[float] | None:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    by_n = {int(d["n"]): float(d["score"]) for d in arr if "n" in d and "score" in d}
    if len(by_n) < n:
        return None
    return [max(-1.0, min(1.0, by_n.get(i + 1, 0.0))) for i in range(n)]


def aggregate(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {}
    k = max(1, len(scores) // 3)
    head = sum(scores[:k]) / k
    tail = sum(scores[-k:]) / k
    return {
        "score_mean": sum(scores) / len(scores),
        "score_min": min(scores),
        "score_head": head,                 # 초반 1/3 — 통화 시작 시 감정
        "score_tail": tail,                 # 후반 1/3 — 통화가 어떻게 끝났나
        "score_delta": tail - head,         # 개선(+) / 악화(-)
        "neg_ratio": sum(1 for s in scores if s < -0.3) / len(scores),
        "n_utt": len(scores),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--keep-markers", action="store_true")
    ap.add_argument("--limit", type=int, help="앞에서 N건만 (시험용)")
    args = ap.parse_args()

    cache = CACHE_ROOT / args.model.replace(":", "_").replace("/", "-")
    cache.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in DATASET.open(encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    out_rows, failed, t0 = [], [], time.time()
    for i, r in enumerate(rows, 1):
        sid = r["scenario_id"]
        utts = [u for u in r["utterances"] if u["role"] == "고객"]
        texts = [u["text"] if args.keep_markers else strip_emotion_markers(u["text"]) for u in utts]

        cf = cache / f"{sid}.json"
        scores = json.loads(cf.read_text(encoding="utf-8")) if cf.exists() else None
        if scores is None or len(scores) != len(texts):
            lines = "\n".join(f"{j+1}. {t}" for j, t in enumerate(texts))
            try:
                scores = parse(ask(args.model, PROMPT.format(lines=lines)), len(texts))
            except Exception as e:
                print(f"  [{i}/{len(rows)}] {sid} 호출 실패: {e}")
                scores = None
            if scores is None:
                failed.append(sid)
                continue
            cf.write_text(json.dumps(scores), encoding="utf-8")

        out_rows.append({"scenario_id": sid, "grade": r["grade"], "source": r["source"],
                         "scores": scores, "indices": [u["index"] for u in utts],
                         "agg": {k: round(v, 4) for k, v in aggregate(scores).items()}})
        if i % 20 == 0:
            print(f"  [{i}/{len(rows)}] 경과 {time.time()-t0:.0f}초")

    out = CACHE_ROOT / f"scores_{args.model.replace(':', '_')}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{len(out_rows)}건 완료 ({time.time()-t0:.0f}초). 실패 {len(failed)}건 {failed[:5]}")
    keys = ["score_mean", "score_min", "score_tail", "neg_ratio"]
    print(f"\n등급별 집계 평균 — {args.model}")
    print(f"{'등급':<8}{'건수':>5}" + "".join(f"{k:>12}" for k in keys))
    print("-" * (13 + 12 * len(keys)))
    for g in ("악성콜", "일반", "칭찬콜"):
        sel = [r for r in out_rows if r["grade"] == g and r["agg"]]
        if sel:
            line = f"{g:<8}{len(sel):>5}"
            for k in keys:
                line += f"{sum(r['agg'][k] for r in sel)/len(sel):>12.4f}"
            print(line)
    print(f"\n저장: {out.relative_to(AI)}")


if __name__ == "__main__":
    main()
