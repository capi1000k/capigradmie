"""Tez sanity-check baseline: yorqinlik gistogrammasi + Logistic Regression.

Yoritilganlik masalasi asosan global statistikaga bog'liq, shuning uchun bu oddiy
baseline ko'pincha 0.95+ CV beradi va bir necha daqiqada ishlaydi (GPU kerak emas).
Undan ikki maqsadda foydalaning:
  1) Datani va labellarni tekshirish (agar baseline juda past bo'lsa — data muammosi).
  2) Deep model natijasini solishtirish uchun mos yozuvlar chizig'i.

Misol:
    python illumination/baseline.py --data-dir "/opt/goinfre/zanerhon/data_освещённость" \
        --out baseline_submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import list_images, scan_train_dir, seed_everything


def extract_features(path: Path) -> np.ndarray:
    """Rasmdan yorug'likka oid ixcham belgi vektorini oladi."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:  # cv2 ba'zi PNG larni o'qiy olmasa — PIL orqali
        from PIL import Image

        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    hist = cv2.calcHist([gray.astype(np.float32)], [0], None, [32], [0.0, 1.0]).ravel()
    hist = hist / (hist.sum() + 1e-8)

    stats = np.array(
        [
            v.mean(), v.std(), np.median(v),
            *np.percentile(v, [1, 5, 25, 75, 95, 99]),
            gray.mean(), gray.std(),
            float((gray < 0.15).mean()),   # juda qorong'u pikselar ulushi
            float((gray > 0.85).mean()),   # to'yingan (overexposed) pikselar ulushi
            float(hsv[:, :, 1].mean()) / 255.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([stats, hist])


def build_matrix(paths: list[Path], desc: str) -> np.ndarray:
    feats = []
    for i, p in enumerate(paths, 1):
        feats.append(extract_features(p))
        if i % 500 == 0 or i == len(paths):
            print(f"  {desc}: {i}/{len(paths)}")
    return np.vstack(feats)


def main() -> None:
    ap = argparse.ArgumentParser(description="Histogram + LogisticRegression baseline")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-subdir", default="train")
    ap.add_argument("--test-subdir", default="test")
    ap.add_argument("--out", default="baseline_submission.csv")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--keep-extension", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    data_dir = Path(args.data_dir).expanduser()

    items = scan_train_dir(data_dir / args.train_subdir)
    print(f"Klasslar: {items.classes} | {len(items.paths)} ta train rasm")
    x_train = build_matrix(items.paths, "train")

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof = cross_val_predict(clf, x_train, items.labels, cv=cv, n_jobs=-1)
    print("\n===== Baseline CV =====")
    print(classification_report(items.labels, oof, target_names=items.classes, digits=4))

    clf.fit(x_train, items.labels)

    test_paths = list_images(data_dir / args.test_subdir)
    print(f"\n{len(test_paths)} ta test rasm")
    x_test = build_matrix(test_paths, "test")
    preds = clf.predict(x_test)

    ids = [p.name if args.keep_extension else p.stem for p in test_paths]
    df = pd.DataFrame({args.id_col: ids, args.label_col: [items.classes[i] for i in preds]})
    df.to_csv(args.out, index=False)
    print(f"\nSaqlandi: {args.out}")
    print(df[args.label_col].value_counts().to_string())


if __name__ == "__main__":
    main()
