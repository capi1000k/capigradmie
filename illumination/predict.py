"""test/ papkasidagi rasmlar uchun inference va submission fayl yaratish.

outputs/ dagi barcha fold checkpointlarini yuklab, softmax ehtimolliklarini
o'rtachalaydi (ansambl) + ixtiyoriy horizontal-flip TTA.

Misol:
    python illumination/predict.py \
        --data-dir "/opt/goinfre/zanerhon/data_освещённость" \
        --ckpt-dir outputs --out sample_submission.csv --tta
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (
    ImageDataset,
    build_model,
    build_transforms,
    dataloader_workers,
    list_images,
    pick_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image illumination classification - inference")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--test-subdir", default="test")
    p.add_argument("--ckpt-dir", default="outputs")
    p.add_argument("--out", default="sample_submission.csv")
    p.add_argument("--sample-submission", default=None,
                   help="Mavjud sample_submission.csv — ustun nomlari va qator tartibi shundan olinadi")
    p.add_argument("--id-col", default="id", help="sample_submission berilmaganda ishlatiladi")
    p.add_argument("--label-col", default="label")
    p.add_argument("--keep-extension", action="store_true",
                   help="ID sifatida '<uuid>.png' o'rniga '<uuid>' kerak bo'lsa, bu flagni bermang")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--device", default="auto")
    p.add_argument("--tta", action="store_true", help="Horizontal flip TTA")
    p.add_argument("--save-probs", default=None, help="Ehtimolliklarni CSV ga saqlash (ixtiyoriy)")
    return p.parse_args()


def load_checkpoints(ckpt_dir: Path, device: torch.device):
    ckpt_files = sorted(ckpt_dir.glob("*_best.pt")) or sorted(ckpt_dir.glob("*.pt"))
    if not ckpt_files:
        raise SystemExit(f"'{ckpt_dir}' ichida checkpoint (*.pt) topilmadi. Avval train.py ni ishga tushiring.")

    models, classes, img_size = [], None, None
    for path in ckpt_files:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if classes is None:
            classes, img_size = ckpt["classes"], ckpt["img_size"]
        elif ckpt["classes"] != classes or ckpt["img_size"] != img_size:
            raise SystemExit(f"'{path}' boshqa klass/o'lcham bilan o'qitilgan — ansambl qilib bo'lmaydi.")

        model = build_model(ckpt["model_name"], len(classes), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(device)
        models.append(model)
        print(f"Yuklandi: {path.name} (val_f1={ckpt.get('val_f1', float('nan')):.4f})")

    return models, classes, img_size


@torch.no_grad()
def predict(models, loader, device, tta: bool, num_classes: int):
    names: list[str] = []
    probs_sum: list[np.ndarray] = []

    for images, batch_names in loader:
        images = images.to(device, non_blocking=True)
        batch_probs = torch.zeros(images.size(0), num_classes, device=device)

        for model in models:
            batch_probs += F.softmax(model(images).float(), dim=1)
            if tta:
                batch_probs += F.softmax(model(torch.flip(images, dims=[3])).float(), dim=1)

        divisor = len(models) * (2 if tta else 1)
        probs_sum.append((batch_probs / divisor).cpu().numpy())
        names.extend(batch_names)

    return names, np.concatenate(probs_sum)


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir).expanduser()
    test_dir = data_dir / args.test_subdir
    if not test_dir.is_dir():
        raise SystemExit(f"Test papkasi topilmadi: {test_dir}")

    device = pick_device(args.device)
    models, classes, img_size = load_checkpoints(Path(args.ckpt_dir).expanduser(), device)

    test_paths = list_images(test_dir)
    if not test_paths:
        raise SystemExit(f"'{test_dir}' ichida rasm topilmadi.")
    print(f"Test rasmlari: {len(test_paths)} ta | device: {device} | TTA: {args.tta}")

    loader = DataLoader(
        ImageDataset(test_paths, None, build_transforms(img_size, train=False)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=dataloader_workers(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    names, probs = predict(models, loader, device, args.tta, len(classes))
    labels = [classes[i] for i in probs.argmax(1)]

    ids = names if args.keep_extension else [Path(n).stem for n in names]
    df = pd.DataFrame({"__id": ids, "__label": labels})

    if args.sample_submission:
        sample = pd.read_csv(args.sample_submission)
        id_col, label_col = sample.columns[0], sample.columns[1]
        # Sample'dagi ID kengaytmali bo'lsa ('.png'), moslashtiramiz
        sample_has_ext = sample[id_col].astype(str).str.contains(r"\.\w+$", regex=True).any()
        df["__id"] = [n if sample_has_ext else Path(n).stem for n in names]
        merged = sample[[id_col]].merge(
            df.rename(columns={"__id": id_col, "__label": label_col}), on=id_col, how="left"
        )
        missing = int(merged[label_col].isna().sum())
        if missing:
            print(f"OGOHLANTIRISH: sample'dagi {missing} ta ID uchun rasm topilmadi -> 'normal' qo'yildi.")
            merged[label_col] = merged[label_col].fillna("normal")
        out_df = merged
    else:
        out_df = df.rename(columns={"__id": args.id_col, "__label": args.label_col})
        out_df = out_df.sort_values(out_df.columns[0]).reset_index(drop=True)

    out_path = Path(args.out).expanduser()
    out_df.to_csv(out_path, index=False)
    print(f"\nSubmission saqlandi: {out_path} ({len(out_df)} qator)")
    print(out_df.head(5).to_string(index=False))
    print("\nBashorat taqsimoti:")
    print(out_df[out_df.columns[1]].value_counts().to_string())

    if args.save_probs:
        prob_df = pd.DataFrame(probs, columns=classes)
        prob_df.insert(0, "id", ids)
        prob_df.to_csv(args.save_probs, index=False)
        print(f"Ehtimolliklar saqlandi: {args.save_probs}")


if __name__ == "__main__":
    main()
