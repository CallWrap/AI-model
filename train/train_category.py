"""중분류(F2) 파인튜닝 — 학습·채점·저장 한 세트. (개발명세 Step 3)

split_data.py가 만든 train.jsonl(80건)로 klue/roberta-base를 파인튜닝하고,
val.jsonl(20건)로 Macro-F1을 잰 뒤 모델과 지표를 저장한다.

  uv run --extra train python train/train_category.py
  uv run --extra train python train/train_category.py --seed 7
  uv run --extra train python train/train_category.py --model monologg/koelectra-base-v3-discriminator

평가 방식에 관한 정직성 규칙:
  마지막 에폭 모델을 최종 결과로 삼는다. 에폭마다 재서 제일 잘 나온 값을 고르면
  검증 20건에 맞춘 점수가 되어 실력이 부풀려진다. 에폭별 점수는 참고용으로만 찍는다.

  검증 20건은 클래스당 2~6건뿐이라 Macro-F1이 크게 흔들린다. 한 번 돌린 숫자로
  모델을 비교하지 않는다. seed 여러 개로 돌려 평균과 편차를 함께 본다.
    for s in 42 7 2026; do uv run --extra train python train/train_category.py --seed $s; done

  이 점수는 재현 데이터 기준 내부 지표다. 실상담 성능을 보장하지 않는다(명세 9.1).
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from callwrap_ai.schemas import ConsultationCategory
from callwrap_ai.preprocess import strip_boilerplate

AI = Path(__file__).resolve().parents[1]
SCN = AI / "data/scenarios"
MODELS = AI / "data/models"

# 라벨 순서는 계약(schemas.py) 정의 순서로 고정한다. 정렬 방식이 바뀌면 저장된
# 모델의 출력 번호와 라벨이 어긋나므로 여기서만 정한다.
LABELS = [c.value for c in ConsultationCategory]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # MPS 시드도 함께 설정된다


def pick_device() -> torch.device:
    # MPS에서 fp16은 불안정하므로 fp32를 쓴다 (명세 11장).
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode(path: Path, tok, max_length: int) -> tuple[TensorDataset, list[str], int]:
    """jsonl → 토큰 텐서. 잘린 건수를 함께 돌려준다."""
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    texts = [strip_boilerplate(r["text"]) for r in rows]

    truncated = [
        r["scenario_id"]
        for r, t in zip(rows, texts)
        if len(tok(t, add_special_tokens=True)["input_ids"]) > max_length
    ]

    enc = tok(texts, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    labels = torch.tensor([LABEL2ID[r["category"]] for r in rows])
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], labels)
    return ds, truncated, len(rows)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    gold, pred = [], []
    for ids, mask, y in loader:
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        pred += logits.argmax(-1).cpu().tolist()
        gold += y.tolist()
    return gold, pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="klue/roberta-base", help="HuggingFace 모델 이름")
    ap.add_argument("--seed", type=int, default=42, help="재현용 난수 시드")
    ap.add_argument("--max-length", type=int, default=512, help="읽을 최대 토큰 수 (roberta 상한 512)")
    ap.add_argument("--epochs", type=int, default=10, help="학습 데이터를 몇 바퀴 돌지")
    ap.add_argument("--lr", type=float, default=2e-5, help="학습률 — 한 번에 얼마나 고칠지")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device()
    slug = args.model.replace("/", "-")
    out_dir = MODELS / f"category-{slug}-seed{args.seed}"

    print(f"모델 {args.model} | 디바이스 {device} | seed {args.seed} | max_length {args.max_length}")

    tok = AutoTokenizer.from_pretrained(args.model)
    train_ds, tr_cut, n_train = encode(SCN / "train.jsonl", tok, args.max_length)
    val_ds, va_cut, n_val = encode(SCN / "val.jsonl", tok, args.max_length)
    print(f"학습 {n_train}건 (잘림 {len(tr_cut)}건) | 검증 {n_val}건 (잘림 {len(va_cut)}건)")
    if va_cut:
        print(f"  검증 잘린 건: {va_cut}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(total_steps * 0.1), total_steps)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for ids, mask, y in train_loader:
            optim.zero_grad()
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            losses.append(out.loss.item())

        gold, pred = evaluate(model, val_loader, device)
        macro = f1_score(gold, pred, average="macro", zero_division=0)
        history.append({"epoch": epoch, "loss": sum(losses) / len(losses), "val_macro_f1": macro})
        print(f"  epoch {epoch:2d} | loss {history[-1]['loss']:.4f} | val Macro-F1 {macro:.4f}")

    # 최종 = 마지막 에폭. 에폭 고르기(early stopping)는 검증 20건에 과적합되므로 안 한다.
    gold, pred = evaluate(model, val_loader, device)
    macro = f1_score(gold, pred, average="macro", zero_division=0)
    micro = f1_score(gold, pred, average="micro", zero_division=0)

    print(f"\n=== 최종 (마지막 에폭) Macro-F1 {macro:.4f} | Accuracy {micro:.4f} ===")
    print(classification_report(gold, pred, labels=range(len(LABELS)), target_names=LABELS, zero_division=0))
    print("목표 Macro-F1 >= 0.85 (명세 7장). 100건 규모에서 미달은 예상된 결과이며 데이터 부족이 원인이다.")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "seed": args.seed,
                "max_length": args.max_length,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "n_train": n_train,
                "n_val": n_val,
                "truncated_train": tr_cut,
                "truncated_val": va_cut,
                "final_macro_f1": macro,
                "final_accuracy": micro,
                "history": history,
                "labels": LABELS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n저장: {out_dir.relative_to(AI)}  (모델 + metrics.json)")


if __name__ == "__main__":
    main()
