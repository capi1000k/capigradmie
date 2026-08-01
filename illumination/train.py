"""Yoritilganlik (bright / dark / normal) klassifikatsiyasi uchun o'qitish skripti.

Stratified K-Fold cross-validation, transfer learning (timm/torchvision),
AMP, cosine LR scheduler va har fold uchun eng yaxshi checkpoint saqlash.

Misol:
    python illumination/train.py \
        --data-dir "/opt/goinfre/zanerhon/data_освещённость" \
        --model tf_efficientnet_b0 --img-size 224 --epochs 8 --folds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from common import (
    ImageDataset,
    build_model,
    build_transforms,
    dataloader_workers,
    pick_device,
    scan_train_dir,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image illumination classification - training")
    p.add_argument("--data-dir", required=True, help="train/ va test/ papkalarini o'z ichiga olgan katalog")
    p.add_argument("--train-subdir", default="train")
    p.add_argument("--out-dir", default="outputs", help="checkpoint va loglar saqlanadigan joy")
    p.add_argument("--model", default="tf_efficientnet_b0", help="timm yoki torchvision model nomi")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--folds", type=int, default=5, help="StratifiedKFold split soni")
    p.add_argument(
        "--train-folds",
        default="0",
        help="O'qitiladigan foldlar: '0' (tez), '0,1,2' yoki 'all' (to'liq ansambl)",
    )
    p.add_argument("--num-workers", type=int, default=-1, help="-1 = avtomatik")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cuda | cpu | mps")
    p.add_argument("--no-amp", action="store_true", help="Mixed precision'ni o'chirish")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--brightness-aug", type=float, default=0.0,
                   help="Brightness/contrast jitter kuchi. Faqat diagnose.py label ekspozitsiyaga "
                        "bog'liq EMASligini ko'rsatgandan keyin yoqing (masalan 0.2).")
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, n = 0.0, 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        n += bs
        all_preds.append(logits.detach().float().argmax(1).cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    preds = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)
    return {
        "loss": total_loss / max(n, 1),
        "acc": accuracy_score(targets_np, preds),
        "f1": f1_score(targets_np, preds, average="macro"),
        "preds": preds,
        "targets": targets_np,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    data_dir = Path(args.data_dir).expanduser()
    train_dir = data_dir / args.train_subdir
    if not train_dir.is_dir():
        raise SystemExit(f"Train papkasi topilmadi: {train_dir}")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    amp = (not args.no_amp) and device.type == "cuda"
    workers = dataloader_workers(args.num_workers)

    items = scan_train_dir(train_dir)
    classes = items.classes
    print(f"Device: {device} | AMP: {amp} | workers: {workers}")
    print(f"Klasslar: {classes}")
    for i, c in enumerate(classes):
        print(f"  {c:<8} {(items.labels == i).sum():>6} ta rasm")
    print(f"Jami: {len(items.paths)} ta rasm\n")

    if args.train_folds.strip().lower() == "all":
        target_folds = set(range(args.folds))
    else:
        target_folds = {int(x) for x in args.train_folds.split(",") if x.strip() != ""}

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    train_tf = build_transforms(args.img_size, train=True, brightness_aug=args.brightness_aug)
    val_tf = build_transforms(args.img_size, train=False)

    # OOF (out-of-fold) bashoratlar — CV natijasini halol baholash uchun
    oof_preds = np.full(len(items.paths), -1, dtype=np.int64)
    fold_scores: dict[str, float] = {}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(items.paths, items.labels)):
        if fold not in target_folds:
            continue

        print(f"===== Fold {fold} | train={len(tr_idx)} val={len(va_idx)} =====")
        tr_paths = [items.paths[i] for i in tr_idx]
        va_paths = [items.paths[i] for i in va_idx]

        tr_loader = DataLoader(
            ImageDataset(tr_paths, items.labels[tr_idx], train_tf),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            drop_last=len(tr_idx) > args.batch_size,
            persistent_workers=workers > 0,
        )
        va_loader = DataLoader(
            ImageDataset(va_paths, items.labels[va_idx], val_tf),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )

        model = build_model(args.model, len(classes), pretrained=not args.no_pretrained).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01
        )
        scaler = torch.amp.GradScaler(device.type, enabled=amp)

        best_f1 = -1.0
        best_preds: np.ndarray | None = None
        ckpt_path = out_dir / f"fold{fold}_best.pt"

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr = run_epoch(model, tr_loader, criterion, device, optimizer, scaler, amp)
            va = run_epoch(model, va_loader, criterion, device, amp=amp)
            scheduler.step()

            marker = ""
            if va["f1"] > best_f1:
                best_f1 = va["f1"]
                best_preds = va["preds"]
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "classes": classes,
                        "model_name": args.model,
                        "img_size": args.img_size,
                        "fold": fold,
                        "val_f1": best_f1,
                    },
                    ckpt_path,
                )
                marker = "  <- saqlandi"

            print(
                f"  epoch {epoch:>2}/{args.epochs} | "
                f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
                f"val loss {va['loss']:.4f} acc {va['acc']:.4f} f1 {va['f1']:.4f} | "
                f"{time.time() - t0:.1f}s{marker}"
            )

        fold_scores[f"fold{fold}"] = best_f1
        if best_preds is not None:
            oof_preds[va_idx] = best_preds
        print(f"  Fold {fold} eng yaxshi macro-F1: {best_f1:.4f} -> {ckpt_path}\n")

        del model, optimizer, tr_loader, va_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mask = oof_preds >= 0
    if mask.any():
        print("===== OOF natija (o'qitilgan foldlar bo'yicha) =====")
        print(
            classification_report(
                items.labels[mask], oof_preds[mask], target_names=classes, digits=4
            )
        )
        overall_f1 = f1_score(items.labels[mask], oof_preds[mask], average="macro")
        overall_acc = accuracy_score(items.labels[mask], oof_preds[mask])
        print(f"OOF macro-F1: {overall_f1:.4f} | accuracy: {overall_acc:.4f}")
        fold_scores["oof_macro_f1"] = float(overall_f1)
        fold_scores["oof_accuracy"] = float(overall_acc)

    meta = {
        "classes": classes,
        "model_name": args.model,
        "img_size": args.img_size,
        "folds": args.folds,
        "trained_folds": sorted(target_folds),
        "scores": fold_scores,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nMeta saqlandi: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
