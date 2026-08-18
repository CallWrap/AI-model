"""발화별 감정 점수 산출 — 공개 한국어 감정 모델을 그대로 추론에 쓴다.

    uv run --extra train python train/sentiment_score.py
    uv run --extra train python train/sentiment_score.py --keep-markers   # 비교용
    uv run --extra train python train/sentiment_score.py --model <다른 모델>

파인튜닝하지 않는다. 발화별 감정 정답이 없어서 학습이 불가능하기 때문이다
(우리가 가진 건 통화 단위 등급 라벨 186건뿐이다).

기본 모델은 6감정(angry/anxious/embarrassed/happy/heartache/sad)을 출력한다.
**무감정 클래스가 없어 중립 문장도 6개 중 하나로 배정된다** — 실측 결과 중립적인
문의가 angry로 쏠린다. 그래서 angry 절대값을 그대로 임계값에 쓰면 안 되고,
통화 간 상대 비교나 angry-happy 격차를 봐야 한다. 이 스크립트는 그 판단에 필요한
분포를 먼저 보여준다.

감정 표지(`(욕설)` 등)는 기본으로 제거한다. 실제 STT 전사에는 없는 표지이고,
우리 데이터에서는 악성콜 27건에만 붙어 있어 그대로 두면 모델이 그것만 본다.

출력: data/scenarios/sentiment_scores.jsonl (통화별 발화 점수 + 집계)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from callwrap_ai.preprocess import strip_emotion_markers

AI = Path(__file__).resolve().parents[1]
DATASET = AI / "eval/datasets/grade_dataset.jsonl"
OUT = AI / "data/scenarios/sentiment_scores.jsonl"
DEFAULT_MODEL = "Jinuuuu/KoELECTRA_fine_tunning_emotion"

NEGATIVE = {"angry", "anxious", "sad", "heartache", "embarrassed"}
POSITIVE = {"happy"}


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def score(texts: list[str], tok, model, device, batch: int = 32) -> list[dict[str, float]]:
    names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=256, return_tensors="pt")
        p = model(input_ids=enc["input_ids"].to(device),
                  attention_mask=enc["attention_mask"].to(device)).logits.softmax(-1).cpu()
        out += [dict(zip(names, row.tolist())) for row in p]
    return out


def aggregate(scores: list[dict[str, float]]) -> dict[str, float]:
    """통화 하나의 고객 발화 점수들 -> 집계 지표.

    평균만 보면 통화 내내 평온하다가 마지막에 폭발하는 콜을 놓친다.
    최저·비율·후반 추이를 함께 낸다.
    """
    if not scores:
        return {}
    angry = [s.get("angry", 0.0) for s in scores]
    happy = [s.get("happy", 0.0) for s in scores]
    neg = [sum(v for k, v in s.items() if k in NEGATIVE) for s in scores]
    tail = max(1, len(scores) // 3)
    return {
        "angry_mean": sum(angry) / len(angry),
        "angry_max": max(angry),
        "happy_mean": sum(happy) / len(happy),
        "happy_max": max(happy),
        "neg_mean": sum(neg) / len(neg),
        "gap_mean": sum(a - h for a, h in zip(angry, happy)) / len(angry),
        "angry_tail": sum(angry[-tail:]) / tail,   # 후반 1/3
        "n_utt": len(scores),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--keep-markers", action="store_true", help="감정 표지를 지우지 않는다 (비교용)")
    ap.add_argument("--role", default="고객", choices=("고객", "상담사", "all"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in DATASET.open(encoding="utf-8")]
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    print(f"모델 {args.model} | 디바이스 {device} | 표지 {'유지' if args.keep_markers else '제거'} | 대상 {args.role}")

    flat, owner = [], []
    for r in rows:
        for u in r["utterances"]:
            if args.role != "all" and u["role"] != args.role:
                continue
            t = u["text"] if args.keep_markers else strip_emotion_markers(u["text"])
            flat.append(t)
            owner.append((r["scenario_id"], u["index"]))
    print(f"발화 {len(flat)}개 채점 중…")
    scored = score(flat, tok, model, device)

    by_call = defaultdict(list)
    for (sid, idx), s in zip(owner, scored):
        by_call[sid].append({"index": idx, **{k: round(v, 4) for k, v in s.items()}})

    out_rows = []
    for r in rows:
        us = by_call.get(r["scenario_id"], [])
        out_rows.append({"scenario_id": r["scenario_id"], "grade": r["grade"], "source": r["source"],
                         "utterances": us, "agg": {k: round(v, 4) for k, v in aggregate(us).items()}})
    with OUT.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 등급별 분포 — 이 지표로 갈리는지가 첫 관문이다
    print(f"\n등급별 집계 평균 ({args.role} 발화 기준)")
    keys = ["angry_mean", "angry_max", "angry_tail", "happy_mean", "happy_max", "gap_mean"]
    print(f"{'등급':<8}{'건수':>5}" + "".join(f"{k:>12}" for k in keys))
    print("-" * (13 + 12 * len(keys)))
    for g in ("악성콜", "일반", "칭찬콜"):
        sel = [r for r in out_rows if r["grade"] == g and r["agg"]]
        if not sel:
            continue
        line = f"{g:<8}{len(sel):>5}"
        for k in keys:
            line += f"{sum(r['agg'][k] for r in sel) / len(sel):>12.4f}"
        print(line)

    print(f"\n저장: {OUT.relative_to(AI)}")


if __name__ == "__main__":
    main()
