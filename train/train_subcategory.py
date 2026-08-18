"""소분류 29종 파인튜닝 — 학습·채점·저장 한 세트.

train_category.py(중분류 6종)와 같은 구조이며, 다른 점은 두 가지다.
  1. 라벨이 소분류 29종이다
  2. 채점을 **세 가지 조건**으로 한다

세 조건을 다 재는 이유:
  마스킹 없음        29개 중에서 고른다. 비교 기준선
  정답 중분류 마스킹   중분류가 맞다고 가정하고 그 안의 2~9개 중에서만 고른다.
                    소분류 모델 자체의 실력
  예측 중분류 마스킹   중분류도 우리 모델이 예측한 값을 쓴다. 실서비스에서 나올 성능.
                    중분류가 틀리면 소분류도 같이 틀리는 연쇄가 여기서 드러난다

검증셋에는 생성 증강분이 섞여 있으므로 source별 점수도 함께 낸다 (명세 9.1).

실행: uv run --extra train python train/train_subcategory.py
      uv run --extra train python train/train_subcategory.py --seed 7
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from callwrap_ai.schemas import SUBCATEGORIES_BY_CATEGORY, ConsultationSubcategory
from preprocess import strip_boilerplate

AI = Path(__file__).resolve().parents[1]
SCN = AI / "data/scenarios"
MODELS = AI / "data/models"
CATEGORY_CKPT = MODELS / "category-klue-roberta-base-seed42"  # 예측 중분류 마스킹용

LABELS = [s.value for s in ConsultationSubcategory]
LABEL2ID = {n: i for i, n in enumerate(LABELS)}
CATS = sorted(SUBCATEGORIES_BY_CATEGORY, key=lambda c: c.value)
CAT_NAMES = [c.value for c in CATS]

# 중분류 -> 그 중분류에 속한 소분류만 True
MASK = torch.zeros(len(CATS), len(LABELS), dtype=torch.bool)
for ci, c in enumerate(CATS):
    for s in SUBCATEGORIES_BY_CATEGORY[c]:
        MASK[ci, LABEL2ID[s.value]] = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def encode(rows: list[dict], tok, max_length: int):
    texts = [strip_boilerplate(r["text"]) for r in rows]
    cut = [r["scenario_id"] for r, t in zip(rows, texts)
           if len(tok(t, add_special_tokens=True)["input_ids"]) > max_length]
    enc = tok(texts, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    y = torch.tensor([LABEL2ID[r["subcategory"]] for r in rows])
    return TensorDataset(enc["input_ids"], enc["attention_mask"], y), cut


@torch.no_grad()
def logits_of(model, loader, device) -> torch.Tensor:
    model.eval()
    out = []
    for ids, mask, _ in loader:
        out.append(model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.cpu())
    return torch.cat(out)


@torch.no_grad()
def predict_categories(rows: list[dict], device, max_length: int) -> list[str]:
    """기존 중분류 checkpoint로 중분류를 예측한다 (예측 마스킹용)."""
    if not (CATEGORY_CKPT / "config.json").exists():
        return []
    tok = AutoTokenizer.from_pretrained(CATEGORY_CKPT)
    m = AutoModelForSequenceClassification.from_pretrained(CATEGORY_CKPT).to(device).eval()
    names = [m.config.id2label[i] for i in range(len(m.config.id2label))]
    texts = [strip_boilerplate(r["text"]) for r in rows]
    enc = tok(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    preds = []
    for i in range(0, len(texts), 8):
        lg = m(input_ids=enc["input_ids"][i:i + 8].to(device),
               attention_mask=enc["attention_mask"][i:i + 8].to(device)).logits
        preds += [names[j] for j in lg.argmax(-1).cpu().tolist()]
    del m
    return preds


def masked_pred(logits: torch.Tensor, cats: list[str]) -> list[int]:
    """중분류에 속한 소분류만 후보로 남기고 고른다. 계층 위반이 원천적으로 안 나온다."""
    out = []
    for lg, c in zip(logits, cats):
        allowed = MASK[CAT_NAMES.index(c)]
        out.append(int(lg.masked_fill(~allowed, float("-inf")).argmax()))
    return out


def score(gold: list[int], pred: list[int]) -> tuple[float, float]:
    return (f1_score(gold, pred, average="macro", zero_division=0),
            f1_score(gold, pred, average="micro", zero_division=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="klue/roberta-base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device()
    slug = args.model.replace("/", "-")
    out_dir = MODELS / f"subcategory-{slug}-seed{args.seed}"
    print(f"모델 {args.model} | 디바이스 {device} | seed {args.seed}")

    tr_rows, va_rows = load(SCN / "sub_train.jsonl"), load(SCN / "sub_val.jsonl")
    tok = AutoTokenizer.from_pretrained(args.model)
    tr_ds, tr_cut = encode(tr_rows, tok, args.max_length)
    va_ds, va_cut = encode(va_rows, tok, args.max_length)
    print(f"학습 {len(tr_rows)}건 (잘림 {len(tr_cut)}) | 검증 {len(va_rows)}건 (잘림 {len(va_cut)})")

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=LABEL2ID,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = len(tr_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(steps * 0.1), steps)

    gold = [LABEL2ID[r["subcategory"]] for r in va_rows]
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for ids, mask, y in tr_loader:
            optim.zero_grad()
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            losses.append(out.loss.item())
        lg = logits_of(model, va_loader, device)
        m_plain = score(gold, lg.argmax(-1).tolist())[0]
        history.append({"epoch": ep, "loss": sum(losses) / len(losses), "val_macro_f1_plain": m_plain})
        print(f"  epoch {ep:2d} | loss {history[-1]['loss']:.4f} | val Macro-F1(마스킹없음) {m_plain:.4f}")

    # 최종 = 마지막 에폭. 에폭 고르기는 검증 37건에 과적합되므로 안 한다.
    lg = logits_of(model, va_loader, device)
    gold_cats = [r["category"] for r in va_rows]
    pred_cats = predict_categories(va_rows, device, args.max_length)

    results = {}
    conds = [("마스킹 없음", lg.argmax(-1).tolist()),
             ("정답 중분류 마스킹", masked_pred(lg, gold_cats))]
    if pred_cats:
        conds.append(("예측 중분류 마스킹", masked_pred(lg, pred_cats)))

    print(f"\n{'=' * 62}\n조건별 최종 점수 (마지막 에폭, 검증 {len(va_rows)}건)\n{'=' * 62}")
    print(f"{'조건':<20}{'Macro-F1':>10}{'Accuracy':>10}")
    for name, pred in conds:
        ma, mi = score(gold, pred)
        results[name] = {"macro_f1": ma, "accuracy": mi}
        print(f"{name:<20}{ma:>10.4f}{mi:>10.4f}")

    if pred_cats:
        cat_acc = sum(a == b for a, b in zip(pred_cats, gold_cats)) / len(gold_cats)
        print(f"\n(참고) 중분류 예측 정확도 {cat_acc:.4f} — 이게 낮으면 예측 마스킹도 같이 떨어진다")

    # source별 — 검증셋에 생성 증강분이 섞여 있다
    best_name, best_pred = conds[1]
    print(f"\n출처별 Accuracy ({best_name} 기준)")
    for src in ("human", "draft", "generated"):
        idx = [i for i, r in enumerate(va_rows) if r["source"] == src]
        if idx:
            acc = sum(gold[i] == best_pred[i] for i in idx) / len(idx)
            print(f"  {src:<10} {len(idx):3d}건  {acc:.4f}")
    orig = [i for i, r in enumerate(va_rows) if r["source"] != "generated"]
    if orig:
        print(f"  생성 제외 {len(orig):3d}건  {sum(gold[i] == best_pred[i] for i in orig) / len(orig):.4f}")

    # 중분류별 — 소분류가 몇 종인지에 따라 난이도가 다르다
    print(f"\n중분류별 Accuracy ({best_name} 기준)")
    for c in sorted(set(gold_cats)):
        idx = [i for i, g in enumerate(gold_cats) if g == c]
        n_sub = len(SUBCATEGORIES_BY_CATEGORY[next(x for x in CATS if x.value == c)])
        acc = sum(gold[i] == best_pred[i] for i in idx) / len(idx)
        print(f"  {c:<14} 소분류 {n_sub}종  검증 {len(idx):2d}건  {acc:.4f}")

    print()
    print(classification_report(gold, best_pred, labels=range(len(LABELS)),
                                target_names=LABELS, zero_division=0))

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "metrics.json").write_text(json.dumps({
        "model": args.model, "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
        "n_train": len(tr_rows), "n_val": len(va_rows),
        "truncated_train": tr_cut, "truncated_val": va_cut,
        "val_source": dict(Counter(r["source"] for r in va_rows)),
        "results": results, "history": history, "labels": LABELS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out_dir.relative_to(AI)}")


if __name__ == "__main__":
    main()
