"""P4-b: SR유형 분류 파인튜닝 (네이티브 PyTorch 루프).

- 손실: BCEWithLogitsLoss + pos_weight (우선순위 가중 x 클래스 불균형 보정)
- AMP : bf16 autocast (RTX 4080 SUPER = Ada, bf16 네이티브 지원 -> GradScaler 불필요)
- 옵티: AdamW + layer-wise LR decay + linear warmup
- 조기종료: 검증 macro-F1 (최종 축약 라벨 기준)

사용:
    python -m callwrap.train \
        --data data/synth_stt.jsonl \
        --val-data data/real_holdout.jsonl \
        --backbone klue/roberta-large \
        --out runs/v1
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .dataset import (Collator, SRDataset, load_jsonl, register_speaker_tokens,
                      stratified_split)
from .evaluate import multilabel_metrics, report
from .labels import (LABELS, NUM_LABELS, class_balance_pos_weight, collapse_batch,
                     priority_pos_weight)
from .model import SRClassifier
from .progress import ProgressLogger
from .stt_noise import NoiseConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        # Ada 이상에서 TF32를 켜면 matmul이 눈에 띄게 빨라진다.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device,
                  amp_dtype: torch.dtype | None = torch.bfloat16) -> np.ndarray:
    """검증/추론용 확률 행렬 (N, NUM_LABELS)."""
    model.eval()
    chunks: list[np.ndarray] = []
    use_amp = amp_dtype is not None and device.type == "cuda"
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        spk = batch["speaker_ids"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            logits = model(ids, mask, spk)
        chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, NUM_LABELS))


def build_pos_weight(y_train: np.ndarray, priority_span: float,
                     balance_clip: float) -> torch.Tensor:
    """정의서 §1.2의 비대칭 오류비용 + 클래스 불균형을 함께 반영."""
    w = priority_pos_weight(span=priority_span) * \
        class_balance_pos_weight(y_train, clip=balance_clip)
    return torch.tensor(w, dtype=torch.float32)


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = pick_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 데이터 -----------------------------------------------------------
    records = load_jsonl(args.data)
    if args.val_data:
        train_recs, val_recs = records, load_jsonl(args.val_data)
        val_kind = f"별도 검증셋 ({Path(args.val_data).name})"
    else:
        train_recs, val_recs = stratified_split(records, args.val_ratio, args.seed)
        val_kind = "합성 검증셋 (층화 분할)"

    print(f"device={device}  학습 {len(train_recs)}건 / 검증 {len(val_recs)}건")
    print(f"검증셋 종류: {val_kind}")
    if not args.val_data:
        print("  ⚠ 합성 검증셋으로만 조기종료를 하면 지표가 낙관적입니다. "
              "실데이터 holdout 확보 시 --val-data 로 교체하십시오.")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    register_speaker_tokens(tokenizer)

    noise_cfg = NoiseConfig()
    ds_tr = SRDataset(train_recs, tokenizer, max_length=args.max_length,
                      head_ratio=args.head_ratio, noise_prob=args.noise_prob,
                      noise_config=noise_cfg, seed=args.seed)
    ds_va = SRDataset(val_recs, tokenizer, max_length=args.max_length,
                      head_ratio=args.head_ratio, noise_prob=0.0, seed=args.seed)

    collate = Collator(tokenizer.pad_token_id)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       collate_fn=collate, num_workers=args.num_workers,
                       pin_memory=(device.type == "cuda"), drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.eval_batch_size, shuffle=False,
                       collate_fn=collate, num_workers=args.num_workers,
                       pin_memory=(device.type == "cuda"))

    # ---- 모델 -------------------------------------------------------------
    model = SRClassifier(
        args.backbone, NUM_LABELS, dropout=args.dropout, pooling=args.pooling,
        use_speaker_embedding=not args.no_speaker_embedding,
        vocab_size=len(tokenizer),
    ).to(device)
    if args.grad_checkpointing:
        model.backbone.gradient_checkpointing_enable()

    y_train = np.stack([ds_tr[i]["labels"].numpy() for i in range(len(ds_tr))]) \
        if args.auto_pos_weight else np.ones((1, NUM_LABELS), dtype=np.float32)
    pos_weight = build_pos_weight(y_train, args.priority_span,
                                  args.balance_clip).to(device)
    print("pos_weight:", {n: round(float(v), 2)
                          for n, v in zip(LABELS, pos_weight.cpu())})

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.optimizer_groups(args.lr, args.head_lr_mult,
                               args.weight_decay, args.llrd),
        betas=(0.9, 0.98), eps=1e-6,
    )

    steps_per_epoch = math.ceil(len(dl_tr) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and args.bf16) else None
    thresholds = np.full(NUM_LABELS, 0.5)

    # ---- 학습 루프 --------------------------------------------------------
    best = {"macro_f1": -1.0, "epoch": -1}
    patience_left = args.patience
    y_true = [r["sr_final"] for r in val_recs]
    # 검증셋 noise_prob=0.0 이라 결정적이므로 루프 밖에서 한 번만 계산한다.
    y_multi = np.stack([ds_va[i]["labels"].numpy() for i in range(len(ds_va))])

    logger = ProgressLogger(
        out_dir, task="train", total_epochs=args.epochs,
        meta={"backbone": args.backbone, "device": str(device),
              "n_train": len(train_recs), "n_val": len(val_recs),
              "val_kind": val_kind, "batch_size": args.batch_size,
              "grad_accum": args.grad_accum, "lr": args.lr,
              "max_length": args.max_length, "noise_prob": args.noise_prob},
    )

    with logger:
        for epoch in range(1, args.epochs + 1):
            logger.start_epoch(epoch)
            model.train()
            running, n_batches = 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            bar = tqdm(dl_tr, desc=f"epoch {epoch}/{args.epochs}", leave=False)

            for step, batch in enumerate(bar, 1):
                ids = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                spk = batch["speaker_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=amp_dtype is not None):
                    logits = model(ids, mask, spk)
                loss = criterion(logits.float(), labels) / args.grad_accum
                loss.backward()

                if step % args.grad_accum == 0 or step == len(dl_tr):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                running += loss.item() * args.grad_accum
                n_batches += 1
                bar.set_postfix(loss=f"{running / n_batches:.4f}")

            # ---- 검증 -----------------------------------------------------
            probs = predict_probs(model, dl_va, device, amp_dtype)
            y_pred = collapse_batch(probs, thresholds)
            m = report(y_true, y_pred, probs, thresholds,
                       title=f"epoch {epoch} — {val_kind}")

            # 최종 축약 라벨(m['macro_f1'])은 우선순위 규칙을 거치므로, 여러
            # 클래스가 동시에 임계값을 넘는 시기엔 항상 최상위 클래스로
            # 뭉개져 "학습이 안 되는 것"처럼 보일 수 있다. 8개 헤드가 각자
            # 독립적으로 신호를 배우고 있는지는 이 멀티라벨 지표로 따로
            # 확인한다 — 이게 오르는데 축약 macro-F1만 정체돼 있다면 데이터나
            # 모델이 아니라 pos_weight/임계값 쪽 문제다.
            ml = multilabel_metrics(y_multi, probs, thresholds)

            is_best = m["macro_f1"] > best["macro_f1"]
            if is_best:
                best = {"macro_f1": m["macro_f1"], "epoch": epoch,
                        "sla_violation_rate": m["sla_violation_rate"]}
                patience_left = args.patience
                torch.save(model.state_dict(), out_dir / "model.pt")
                np.save(out_dir / "val_probs.npy", probs)
            else:
                patience_left -= 1

            print(f"  [멀티라벨 헤드 진단] macro-F1(축약 전) {ml['macro_f1']:.4f}  "
                  f"micro-F1 {ml['micro_f1']:.4f}  "
                  f"(축약 후 macro-F1 {m['macro_f1']:.4f}과 함께 보고 판단할 것 — "
                  f"이게 오르는데 축약 후 값만 정체돼 있으면 임계값/pos_weight 문제)")

            # 에폭 결과를 즉시 디스크에 남긴다. 중단되어도 여기까지는 보존된다.
            logger.log_epoch(
                epoch,
                metrics={
                    "train_loss": running / max(n_batches, 1),
                    "lr": scheduler.get_last_lr()[0],
                    "macro_f1": m["macro_f1"],
                    "micro_f1": m["micro_f1"],
                    "accuracy": m["accuracy"],
                    "sla_violation_rate": m["sla_violation_rate"],
                    "sla_violations": m["sla_violations"],
                    "per_class_f1": {k: round(v["f1"], 4)
                                     for k, v in m["per_class"].items()},
                    "multilabel_macro_f1": ml["macro_f1"],
                    "multilabel_micro_f1": ml["micro_f1"],
                    "patience_left": patience_left,
                },
                is_best=is_best,
            )

            if is_best:
                print(f"  -> best 갱신, 체크포인트 저장 ({out_dir / 'model.pt'})")
            else:
                print(f"  -> 개선 없음 (patience {patience_left}/{args.patience})")
                if patience_left <= 0:
                    print("조기종료.")
                    logger.finish("stopped", note="early stopping")
                    break

    history = logger.records

    # ---- 산출물 -----------------------------------------------------------
    tokenizer.save_pretrained(out_dir / "tokenizer")
    config = {
        "backbone": args.backbone,
        "max_length": args.max_length,
        "head_ratio": args.head_ratio,
        "pooling": args.pooling,
        "dropout": args.dropout,
        "use_speaker_embedding": not args.no_speaker_embedding,
        "labels": list(LABELS),
        "vocab_size": len(tokenizer),
        "thresholds": thresholds.tolist(),   # P5에서 캘리브레이션으로 교체
        "best": best,
        "history": history,
        "val_kind": val_kind,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 검증셋 정답을 함께 저장해 P5 캘리브레이션이 바로 이어지게 한다.
    (out_dir / "val_true.json").write_text(
        json.dumps([r["sr_final"] for r in val_recs], ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(out_dir / "val_y_multi.npy",
            np.stack([ds_va[i]["labels"].numpy() for i in range(len(ds_va))]))

    print("\n[에폭별 진행상황]")
    print(logger.summary_table())
    print(f"\n최고 성능: epoch {best['epoch']}, macro-F1 {best['macro_f1']:.4f}")
    print(f"산출물: {out_dir}")
    print(f"  progress.jsonl  에폭별 기록")
    print(f"  status.json     현재 상태 스냅샷")
    print(f"  train.log       콘솔 출력 전문")
    print("다음 단계: python -m callwrap.calibrate --run", out_dir)
    return best


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SR유형 분류 파인튜닝 (PyTorch)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-data", default=None,
                    help="실데이터 holdout 권장. 없으면 --data에서 층화 분할")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--backbone", default="klue/roberta-large")

    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--head-ratio", type=float, default=0.5,
                    help="절단 시 앞부분 비중 (나머지는 뒷부분)")
    ap.add_argument("--pooling", default="cls_mean", choices=["cls", "mean", "cls_mean"])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-speaker-embedding", action="store_true")

    ap.add_argument("--epochs", type=int, default=8)
    # RTX 4080 SUPER(16GiB) 실측: klue/roberta-large, seq 512, bf16
    #   batch 8  -> peak  8.20 GiB
    #   batch 16 -> peak 12.05 GiB   <- 기본값. 여유 있음
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5, help="large=1e-5, base=2e-5")
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--llrd", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=3)

    ap.add_argument("--noise-prob", type=float, default=0.5,
                    help="학습 시 STT 노이즈 적용 확률")
    ap.add_argument("--priority-span", type=float, default=0.5,
                    help="우선순위 상위 클래스 pos_weight 가중 폭. "
                         "class-balance-clip과 곱해지므로 1.0 이상으로 올리지 "
                         "말 것 — 결합 가중치가 커지면 여러 클래스가 동시에 "
                         "임계값을 넘어 우선순위 최상위 클래스로 붕괴한다.")
    ap.add_argument("--balance-clip", type=float, default=4.0,
                    help="클래스 불균형 pos_weight 상한. priority-span과 곱해지므로 "
                         "너무 크게 잡지 말 것 (기본 4.0, 이전 기본값 10.0에서 축소)")
    ap.add_argument("--auto-pos-weight", action="store_true", default=True)

    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="VRAM 부족 시. 속도는 20~30% 느려짐")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="Windows에서는 0 권장")
    ap.add_argument("--seed", type=int, default=42)
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
