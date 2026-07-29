"""학습한 중분류 모델에 대본을 넣어 분류·확신도를 출력한다.

    uv run --extra train python train/predict.py --file 대본.txt
    uv run --extra train python train/predict.py --file data/scenarios/val.jsonl
    uv run --extra train python train/predict.py --text "상담원: ...\n고객: ..."

  입력은 원본 대본 그대로 넣는다. `화자: 발화` 를 줄바꿈으로 이은 평문이면 되고,
  인사말 제거·512토큰 절단은 학습 때와 똑같이 여기서 처리한다.

  jsonl을 주면 줄마다 예측하고, `category` 필드가 있으면 채점까지 한다.
  오류 분석용이다 — 무엇을 무엇으로 착각하는지 본다.

  라벨 순서는 checkpoint의 config.json(id2label)에서 읽는다. 계약 enum 순서가
  바뀌어도 이미 학습된 모델은 자기 매핑을 그대로 쓴다.

  추론만 한다. 모델을 바꾸지 않으므로 같은 입력이면 항상 같은 출력이다.
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocess import strip_boilerplate

AI = Path(__file__).resolve().parents[1]
MODELS = AI / "data/models"
DEFAULT_MODEL = MODELS / "category-klue-roberta-base-seed42"


def pick_device() -> torch.device:
    # 학습과 동일 (명세 11장). MPS가 없으면 CPU — 추론은 CPU로도 충분히 빠르다.
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def width(s: str) -> int:
    """터미널 표시 폭. 한글·전각은 두 칸을 차지해 len()과 다르다."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, w: int) -> str:
    return s + " " * max(0, w - width(s))


def load_model(model_dir: Path):
    if not (model_dir / "config.json").exists():
        sys.exit(
            f"checkpoint가 없습니다: {model_dir}\n"
            "학습 데이터와 모델은 커밋되지 않으므로 아래로 재현합니다.\n\n"
            "  uv run --extra train python train/split_data.py\n"
            "  uv run --extra train python train/train_category.py\n"
        )

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    # config의 id2label은 키가 문자열로 저장돼 있어 번호순으로 다시 세운다.
    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    return tok, model, labels


@torch.no_grad()
def predict(texts: list[str], tok, model, device, max_length: int) -> torch.Tensor:
    """대본 목록 → 클래스별 확률. 전처리는 학습과 동일하게 여기서 건다."""
    cleaned = [strip_boilerplate(t) for t in texts]
    enc = tok(cleaned, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    logits = model(
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
    ).logits
    return logits.softmax(-1).cpu()


def read_input(args) -> tuple[list[str], list[str | None], list[str | None]]:
    """(대본, id, 정답) — id·정답은 jsonl일 때만 채워진다."""
    if args.text:
        return [args.text], [None], [None]

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"파일이 없습니다: {path}")

    if path.suffix == ".jsonl":
        rows = [json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip()]
        if not rows:
            sys.exit(f"빈 파일입니다: {path}")
        return (
            [r["text"] for r in rows],
            [r.get("scenario_id") for r in rows],
            [r.get("category") for r in rows],
        )

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        sys.exit(f"빈 파일입니다: {path}")
    return [text], [None], [None]


def show_one(probs: torch.Tensor, labels: list[str], top_k: int) -> None:
    score, idx = probs.topk(min(top_k, len(labels)))
    w = max(width(labels[i]) for i in idx.tolist() + [0]) + 4
    print(f"{pad('중분류', w)}확신도")
    for s, i in zip(score.tolist(), idx.tolist()):
        print(f"{pad(labels[i], w)}{s:.3f}")


def show_batch(probs: torch.Tensor, labels: list[str], ids, golds) -> None:
    score, idx = probs.max(-1)
    preds = [labels[i] for i in idx.tolist()]
    scored = [g for g in golds if g]

    id_w = max([width(i or "-") for i in ids] + [width("ID")]) + 3
    lab_w = max(width(p) for p in preds + ["예측"]) + 3

    header = f"{pad('ID', id_w)}{pad('예측', lab_w)}확신도"
    print(header + ("  정답" if scored else ""))
    for i, p, s, g in zip(ids, preds, score.tolist(), golds):
        line = f"{pad(i or '-', id_w)}{pad(p, lab_w)}{s:.3f}"
        if g:
            line += "  O" if p == g else f"  X → {g}"
        print(line)

    if scored:
        hit = sum(p == g for p, g in zip(preds, golds) if g)
        print(f"\n{len(scored)}건 중 {hit}건 정답 ({hit / len(scored):.3f})")


def main() -> None:
    ap = argparse.ArgumentParser(description="학습한 중분류 모델로 대본을 분류한다")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="대본을 직접 입력")
    src.add_argument("--file", help="대본 파일 (.txt 단건 / .jsonl 배치)")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL, help="checkpoint 경로")
    ap.add_argument("--top-k", type=int, default=3, help="단건일 때 보여줄 후보 수")
    ap.add_argument("--max-length", type=int, default=512, help="학습과 같은 값을 쓴다")
    args = ap.parse_args()

    texts, ids, golds = read_input(args)
    tok, model, labels = load_model(args.model_dir)
    device = pick_device()
    model.to(device)

    probs = predict(texts, tok, model, device, args.max_length)

    if len(texts) == 1 and not any(golds):
        show_one(probs[0], labels, args.top_k)
    else:
        show_batch(probs, labels, ids, golds)


if __name__ == "__main__":
    main()
