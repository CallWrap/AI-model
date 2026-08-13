"""P4-a: 도메인 적응 사전학습 (DAPT).

FAQ / 100문100답 / 지침 코퍼스로 MLM을 이어서 학습한다.
`바우처`, `본인부담금`, `가산단가`, `행복e음`, `예탁금` 같은 도메인 어휘의
표현이 잡힌다. 코퍼스가 수 MB 수준이라 GPU 1장에 몇 시간이면 끝나고,
데이터가 적은 상황에서 효과가 가장 확실한 투자다.

산출된 체크포인트는 A안(업무분류) 팀과 공유할 수 있다. 입력 텍스트가
동일하므로 도메인 적응된 백본 하나를 두 모델이 같이 쓰면 된다.

입력: 한 줄 = 한 문단인 .txt 파일들 (또는 {"text": ...} JSONL)
사용:
    python -m callwrap.dapt --corpus data/corpus/ --out runs/dapt-roberta-large
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling,
                          get_linear_schedule_with_warmup)

from .dataset import SPEAKER_TOKENS
from .progress import ProgressLogger


def read_corpus(path: Path, min_chars: int = 30) -> list[str]:
    """디렉터리 또는 단일 파일에서 문단 리스트를 만든다."""
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    texts: list[str] = []
    for f in files:
        if f.suffix.lower() not in {".txt", ".jsonl", ".md"} or not f.is_file():
            continue
        if f.suffix.lower() == ".jsonl":
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        texts.append(json.loads(line).get("text", ""))
                    except json.JSONDecodeError:
                        texts.append(line)
        else:
            texts.extend(f.read_text(encoding="utf-8").splitlines())
    return [t.strip() for t in texts if len(t.strip()) >= min_chars]


class MLMDataset(Dataset):
    """문단을 이어붙여 max_length 블록으로 자른다 (packing)."""

    def __init__(self, texts: list[str], tokenizer, max_length: int = 512) -> None:
        ids: list[int] = []
        for t in tqdm(texts, desc="토크나이즈", leave=False):
            ids.extend(tokenizer.encode(t, add_special_tokens=False))
            ids.append(tokenizer.sep_token_id)

        block = max_length - 2
        self.blocks = [
            [tokenizer.cls_token_id] + ids[i:i + block] + [tokenizer.sep_token_id]
            for i in range(0, max(len(ids) - block, 1), block)
        ]

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor(self.blocks[idx], dtype=torch.long)}


def main() -> None:
    ap = argparse.ArgumentParser(description="도메인 적응 사전학습 (MLM)")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/dapt"))
    ap.add_argument("--backbone", default="klue/roberta-large")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--mlm-prob", type=float, default=0.15)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    texts = read_corpus(args.corpus)
    if not texts:
        raise SystemExit(f"코퍼스가 비어 있습니다: {args.corpus}")
    print(f"문단 {len(texts):,}개, 총 {sum(len(t) for t in texts):,}자")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    # 분류 단계와 어휘를 맞춰 둔다 (화자 토큰).
    tokenizer.add_special_tokens({"additional_special_tokens": SPEAKER_TOKENS})

    model = AutoModelForMaskedLM.from_pretrained(args.backbone)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    ds = MLMDataset(texts, tokenizer, args.max_length)
    print(f"MLM 블록 {len(ds):,}개 (block={args.max_length})")

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True,
                                               mlm_probability=args.mlm_prob)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collator, num_workers=0,
                    pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6)
    total_steps = math.ceil(len(dl) / args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and args.bf16) else None

    args.out.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(
        args.out, task="dapt", total_epochs=args.epochs,
        meta={"backbone": args.backbone, "device": str(device),
              "n_paragraphs": len(texts), "n_blocks": len(ds),
              "max_length": args.max_length, "lr": args.lr,
              "mlm_prob": args.mlm_prob},
    )

    model.train()
    with logger:
        best_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            logger.start_epoch(epoch)
            running, n = 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            bar = tqdm(dl, desc=f"DAPT epoch {epoch}/{args.epochs}", leave=False)
            for step, batch in enumerate(bar, 1):
                batch = {k: v.to(device, non_blocking=True)
                         for k, v in batch.items()}
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=amp_dtype is not None):
                    loss = model(**batch).loss
                (loss / args.grad_accum).backward()

                if step % args.grad_accum == 0 or step == len(dl):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                running += loss.item()
                n += 1
                bar.set_postfix(loss=f"{running / n:.4f}",
                                ppl=f"{math.exp(min(running / n, 20)):.1f}")

            epoch_loss = running / max(n, 1)
            is_best = epoch_loss < best_loss
            if is_best:
                best_loss = epoch_loss
                # MLM은 검증셋이 없으므로 loss가 개선될 때마다 저장한다.
                model.save_pretrained(args.out)
                tokenizer.save_pretrained(args.out)

            logger.log_epoch(
                epoch,
                metrics={"train_loss": epoch_loss,
                         "perplexity": math.exp(min(epoch_loss, 20)),
                         "lr": scheduler.get_last_lr()[0]},
                is_best=is_best,
            )

    print("\n[에폭별 진행상황]")
    print(logger.summary_table())
    print(f"\nDAPT 완료 -> {args.out}  (best loss {best_loss:.4f})")
    print(f"다음 단계: python -m callwrap.train --backbone {args.out} ...")


if __name__ == "__main__":
    main()
