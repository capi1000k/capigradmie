"""Kengaytirilgan feature baseline + feature-guruh ablatsiyasi.

baseline.py (global gistogramma) 0.47 berdi. Bu skript global statistikadan
tashqari FAZOVIY va TEKSTURA featurelarini qo'shadi va — eng muhimi — har bir
feature guruhini ALOHIDA baholaydi, shunda qaysi guruh signal olib yurishini
aniq bilamiz, taxmin qilmaymiz.

Ishga tushirish:
    python illumination/baseline_v2.py --data-dir "/opt/goinfre/zanerhon/data_освещённость" \
        --ablation --out baseline_v2_submission.csv

Featurelar keshlanadi (--cache-dir), qayta ishga tushirish tez bo'ladi.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.inspection import permutation_importance

from common import list_images, scan_train_dir, seed_everything

SIZE = 256


def _load(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image

        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def _hist(x: np.ndarray, bins: int, rng: tuple[float, float]) -> np.ndarray:
    h, _ = np.histogram(x, bins=bins, range=rng)
    return (h / (h.sum() + 1e-8)).astype(np.float32)


def extract_groups(path: Path) -> dict[str, np.ndarray]:
    """Featurelarni GURUHLAB qaytaradi — ablatsiya shu guruhlar ustida ishlaydi."""
    img = _load(path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lum = lab[:, :, 0].astype(np.float32) / 255.0
    g = gray.astype(np.float32) / 255.0
    groups: dict[str, np.ndarray] = {}

    # --- G1: global ekspozitsiya (baseline.py dagi bilan bir xil g'oya) ---
    groups["global"] = np.array([
        lum.mean(), lum.std(), np.median(lum),
        *np.percentile(lum, [1, 5, 10, 25, 75, 90, 95, 99]),
        float((g < 0.10).mean()), float((g < 0.20).mean()),
        float((g > 0.80).mean()), float((g > 0.95).mean()),
        np.percentile(lum, 99) - np.percentile(lum, 1),
    ], dtype=np.float32)

    # --- G2: gistogramma shakli ---
    hist = _hist(g, 48, (0.0, 1.0))
    p = hist + 1e-8
    entropy = float(-(p * np.log(p)).sum())
    # Gistogramma momentlari: qiyalik (skew) va cho'qqilik (kurtosis)
    centers = (np.arange(48) + 0.5) / 48
    mu = float((hist * centers).sum())
    var = float((hist * (centers - mu) ** 2).sum()) + 1e-8
    skew = float((hist * (centers - mu) ** 3).sum()) / var**1.5
    kurt = float((hist * (centers - mu) ** 4).sum()) / var**2
    groups["hist"] = np.concatenate([hist, np.array([entropy, skew, kurt], dtype=np.float32)])

    # --- G3: FAZOVIY yoritish (lokal yoritilganlik, soya, backlight) ---
    blocks = lum.reshape(8, SIZE // 8, 8, SIZE // 8).mean(axis=(1, 3))  # 8x8 = 64
    blocks_std = lum.reshape(8, SIZE // 8, 8, SIZE // 8).std(axis=(1, 3))
    h4 = lum.reshape(4, SIZE // 4, 4, SIZE // 4).mean(axis=(1, 3))
    groups["spatial"] = np.concatenate([
        blocks.ravel(),
        np.array([
            blocks.std(), blocks.max() - blocks.min(),
            blocks_std.mean(), blocks_std.std(),
            h4[:2].mean() - h4[2:].mean(),        # yuqori - quyi
            h4[:, :2].mean() - h4[:, 2:].mean(),  # chap - o'ng
            abs(h4[:2].mean() - h4[2:].mean()),
            abs(h4[:, :2].mean() - h4[:, 2:].mean()),
            h4[1:3, 1:3].mean() - h4.mean(),      # markaz - umumiy (vinyetka/spot)
            float(np.percentile(blocks, 90) - np.percentile(blocks, 10)),
        ], dtype=np.float32),
    ]).astype(np.float32)

    # --- G4: tekstura / shovqin (qorong'u rasmlarda shovqin ko'p, blur boshqacha) ---
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2) / 255.0
    # Yuqori chastotali shovqin: median filtrdan keyingi qoldiq
    resid = g - cv2.medianBlur(gray, 3).astype(np.float32) / 255.0
    groups["texture"] = np.array([
        lap.var() / 1e4, float(np.abs(lap).mean()) / 255.0,
        mag.mean(), mag.std(), float(np.percentile(mag, 95)),
        resid.std(), float(np.abs(resid).mean()),
        # Yorqinlikka normallashtirilgan kontrast (Michelson / RMS)
        float(g.std() / (g.mean() + 1e-6)),
    ], dtype=np.float32)

    # --- G5: rang / oq balans (yoritish harorati) ---
    b, gr, r = img[:, :, 0].astype(np.float32), img[:, :, 1].astype(np.float32), img[:, :, 2].astype(np.float32)
    rg, yb = r - gr, 0.5 * (r + gr) - b
    a_ch, b_ch = lab[:, :, 1].astype(np.float32) - 128, lab[:, :, 2].astype(np.float32) - 128
    groups["color"] = np.array([
        r.mean() / 255, gr.mean() / 255, b.mean() / 255,
        r.std() / 255, gr.std() / 255, b.std() / 255,
        (r.mean() + 1) / (b.mean() + 1),          # rang harorati proksisi
        hsv[:, :, 1].mean() / 255, hsv[:, :, 1].std() / 255,
        float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)) / 255,
        a_ch.mean() / 128, b_ch.mean() / 128, a_ch.std() / 128, b_ch.std() / 128,
    ], dtype=np.float32)

    # --- G6: yorqinlikdan MUSTAQIL struktura (per-image standartlashtirilgandan keyin) ---
    # Agar shu guruh yolg'iz o'zi ishlasa -> label kontent/sahnaga bog'liq, yoritishga emas.
    norm = (g - g.mean()) / (g.std() + 1e-6)
    groups["structure"] = np.concatenate([
        _hist(np.clip(norm, -3, 3), 24, (-3.0, 3.0)),
        np.array([float(np.abs(norm).mean()), float(np.percentile(norm, 95) - np.percentile(norm, 5))],
                 dtype=np.float32),
    ]).astype(np.float32)

    return groups


def build_matrix(paths: list[Path], desc: str, cache: Path | None):
    if cache is not None and cache.exists():
        data = np.load(cache, allow_pickle=True)
        print(f"  {desc}: keshdan yuklandi ({cache})")
        return data["x"], list(data["names"]), {k: tuple(v) for k, v in data["slices"].item().items()}

    rows, slices, names = [], {}, []
    for i, p in enumerate(paths, 1):
        groups = extract_groups(p)
        if not slices:
            start = 0
            for gname, vec in groups.items():
                slices[gname] = (start, start + len(vec))
                names.extend(f"{gname}:{j}" for j in range(len(vec)))
                start += len(vec)
        rows.append(np.concatenate(list(groups.values())))
        if i % 500 == 0 or i == len(paths):
            print(f"  {desc}: {i}/{len(paths)}")

    x = np.vstack(rows).astype(np.float32)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, x=x, names=np.array(names), slices=slices)
    return x, names, slices


def make_clf(seed: int) -> HistGradientBoostingClassifier:
    # Gradient boosting: LogisticRegression'dan farqli, chiziqli bo'lmagan
    # va featurelar orasidagi o'zaro ta'sirni tuta oladi.
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=seed,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Kengaytirilgan feature baseline + ablatsiya")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-subdir", default="train")
    ap.add_argument("--test-subdir", default="test")
    ap.add_argument("--out", default="baseline_v2_submission.csv")
    ap.add_argument("--cache-dir", default=".feat_cache")
    ap.add_argument("--ablation", action="store_true", help="Har feature guruhini alohida baholash")
    ap.add_argument("--importance", action="store_true", help="Permutation importance (sekinroq)")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--keep-extension", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    data_dir = Path(args.data_dir).expanduser()
    cache_dir = Path(args.cache_dir)

    items = scan_train_dir(data_dir / args.train_subdir)
    classes, y = items.classes, items.labels
    print(f"Klasslar: {classes} | {len(items.paths)} ta train rasm")
    x, names, slices = build_matrix(items.paths, "train", cache_dir / "train.npz")
    print(f"Feature o'lchami: {x.shape[1]}")
    for gname, (a, b) in slices.items():
        print(f"  {gname:<10} {b - a:>4} ta feature")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    # ---------- Ablatsiya: qaysi guruh signal olib yuradi? ----------
    if args.ablation:
        print("\n===== Feature guruh ablatsiyasi (har biri YOLG'IZ) =====")
        results = []
        for gname, (a, b) in slices.items():
            oof_g = cross_val_predict(make_clf(args.seed), x[:, a:b], y, cv=cv, n_jobs=-1)
            acc = accuracy_score(y, oof_g)
            f1 = f1_score(y, oof_g, average="macro")
            results.append((gname, acc, f1))
            print(f"  {gname:<10} accuracy={acc:.4f}  macro-F1={f1:.4f}")
        print(f"\n  Tasodifiy daraja = {1/len(classes):.4f}")
        best = max(results, key=lambda r: r[2])
        print(f"  Eng kuchli guruh: {best[0]} (F1={best[2]:.4f})")
        print("  Talqin:")
        print("    'global'/'hist' kuchli   -> ekspozitsiya baribir muhim, model sig'imi yetmagan")
        print("    'spatial' kuchli         -> label LOKAL yoritishga bog'liq -> CNN aniq yutadi")
        print("    'texture'/'color' kuchli -> yoritish sharoiti bilvosita (shovqin/rang) sezilyapti")
        print("    'structure' kuchli       -> label KONTENTga bog'liq, yoritishga emas (ogohlantiruvchi!)")
        print("    hammasi ~0.33-0.45       -> labellar shovqinli yoki juda sub'ektiv")

    # ---------- To'liq model ----------
    print("\n===== To'liq feature to'plami (5-fold CV) =====")
    oof = cross_val_predict(make_clf(args.seed), x, y, cv=cv, n_jobs=-1)
    print(classification_report(y, oof, target_names=classes, digits=4))
    print(f"macro-F1: {f1_score(y, oof, average='macro'):.4f}")

    clf = make_clf(args.seed)
    clf.fit(x, y)

    if args.importance:
        print("\n===== Permutation importance (top 15) =====")
        imp = permutation_importance(clf, x, y, n_repeats=3, random_state=args.seed, n_jobs=-1)
        for k in np.argsort(-imp.importances_mean)[:15]:
            print(f"  {names[k]:<16} {imp.importances_mean[k]:.5f}")

    # ---------- Test ----------
    test_paths = list_images(data_dir / args.test_subdir)
    print(f"\n{len(test_paths)} ta test rasm")
    x_test, _, _ = build_matrix(test_paths, "test", cache_dir / "test.npz")
    preds = clf.predict(x_test)

    ids = [p.name if args.keep_extension else p.stem for p in test_paths]
    df = pd.DataFrame({args.id_col: ids, args.label_col: [classes[i] for i in preds]})
    df.to_csv(args.out, index=False)
    print(f"\nSaqlandi: {args.out}")
    print(df[args.label_col].value_counts().to_string())


if __name__ == "__main__":
    main()
