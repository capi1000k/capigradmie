"""Datani diagnostika qilish: label aslida NIMAGA bog'liqligini aniqlash.

Baseline 0.47 bergani — global yorqinlik label emasligini bildiradi.
Bu skript feature o'ylab topishdan OLDIN quyidagi savollarga javob beradi:

  1. Sinflar bo'yicha yorqinlik taqsimoti qanchalik ustma-ust tushadi?
  2. Bitta oddiy feature (o'rtacha yorqinlik, kontrast, ...) qancha ajrata oladi?
  3. Rasm kontenti sinflar bo'ylab bir xilmi (bir xil base rasm turli yoritishda)?
  4. Yorug'lik fazoviy jihatdan notekismi (yuqori/quyi, chap/o'ng)?
  5. Train va test taqsimoti mos keladimi?
  6. Rasmlar aslida qanday ko'rinadi? -> montaj PNG saqlanadi, KO'Z BILAN ko'r.

Ishga tushirish:
    python illumination/diagnose.py --data-dir "/opt/goinfre/zanerhon/data_освещённость"
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from common import list_images, scan_train_dir


def load_bgr(path: Path, size: int | None = 256) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def simple_features(img_bgr: np.ndarray) -> dict[str, float]:
    """Diagnostika uchun bir nechta oddiy, talqin qilinadigan o'lchov."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0].astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = lum.shape
    top, bottom = lum[: h // 2].mean(), lum[h // 2 :].mean()
    left, right = lum[:, : w // 2].mean(), lum[:, w // 2 :].mean()

    # 4x4 blokdagi o'rtacha yorqinliklar -> fazoviy notekislik
    blocks = lum.reshape(4, h // 4, 4, w // 4).mean(axis=(1, 3))

    return {
        "mean_lum": float(lum.mean()),
        "std_lum": float(lum.std()),
        "p05": float(np.percentile(lum, 5)),
        "p95": float(np.percentile(lum, 95)),
        "dyn_range": float(np.percentile(lum, 99) - np.percentile(lum, 1)),
        "frac_dark": float((lum < 0.15).mean()),
        "frac_blown": float((lum > 0.90).mean()),
        "saturation": float(hsv[:, :, 1].mean()) / 255.0,
        "vert_grad": float(top - bottom),          # yuqori vs quyi yoritilganlik
        "horiz_grad": float(left - right),         # chap vs o'ng
        "abs_vert_grad": float(abs(top - bottom)),
        "abs_horiz_grad": float(abs(left - right)),
        "block_std": float(blocks.std()),          # yoritishning fazoviy notekisligi
        "block_range": float(blocks.max() - blocks.min()),
        "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 1e4,
        "edge_mean": float(cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3).__abs__().mean()) / 255.0,
    }


def single_feature_power(values: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Bitta feature bo'yicha eng yaxshi 2 chegarali (3 sinf) accuracy — signal kuchi o'lchovi."""
    order = np.argsort(values)
    v, y = values[order], labels[order]
    n = len(v)
    if n_classes != 3:
        return float("nan")

    # Har bir sinfning kumulyativ sanog'i -> barcha (i, j) kesimlarni tez baholaymiz
    onehot = np.zeros((n, n_classes), dtype=np.int32)
    onehot[np.arange(n), y] = 1
    cum = np.vstack([np.zeros(n_classes, dtype=np.int32), np.cumsum(onehot, axis=0)])

    best = 0
    step = max(1, n // 200)  # 200 ta chegara nuqtasi yetarli
    cuts = list(range(0, n + 1, step))
    for i in cuts:
        left = cum[i]
        for j in cuts:
            if j < i:
                continue
            mid = cum[j] - cum[i]
            right = cum[n] - cum[j]
            best = max(best, int(left.max() + mid.max() + right.max()))
    return best / n


def content_overlap(paths_by_class: dict[str, list[Path]], sample: int = 300) -> None:
    """Sinflar orasida bir xil BAZAVIY rasm ishlatilganmi (kontrast-normallashtirilgan hash)."""
    print("\n===== 3. Kontent ustma-ustligi (sinflar bir xil rasmlardan yasalganmi?) =====")
    hashes: dict[str, set[str]] = {}
    for cls, paths in paths_by_class.items():
        chosen = paths[:: max(1, len(paths) // sample)][:sample]
        hs = set()
        for p in chosen:
            img = load_bgr(p, size=64)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            # Yorqinlik va kontrastni yo'q qilamiz -> faqat STRUKTURA qoladi
            gray = (gray - gray.mean()) / (gray.std() + 1e-6)
            small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
            bits = (small > np.median(small)).astype(np.uint8).tobytes()
            hs.add(hashlib.md5(bits).hexdigest())
        hashes[cls] = hs
        print(f"  {cls:<8} {len(chosen)} ta namunadan {len(hs)} ta noyob struktura-hash")

    classes = list(hashes)
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            a, b = classes[i], classes[j]
            shared = len(hashes[a] & hashes[b])
            print(f"  {a} <-> {b}: {shared} ta umumiy struktura")
    print("  Izoh: umumiy struktura KO'P bo'lsa -> bir xil rasmga turli yoritish qo'llangan")
    print("        (unda farq faqat ekspozitsiyada bo'lishi kerak edi -> qarama-qarshilik!).")
    print("        Umumiy struktura ~0 bo'lsa -> sahnalar turlicha, kontent shovqin qo'shyapti.")


def save_montage(paths_by_class: dict[str, list[Path]], out: Path, per_class: int = 6) -> None:
    tiles = []
    for cls in sorted(paths_by_class):
        paths = paths_by_class[cls]
        idx = np.linspace(0, len(paths) - 1, per_class).astype(int)
        row = [load_bgr(paths[i], size=160) for i in idx]
        # Sinf nomini birinchi tile ustiga yozamiz
        cv2.putText(row[0], cls, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(np.hstack(row))
    cv2.imwrite(str(out), np.vstack(tiles))
    print(f"\nMontaj saqlandi: {out}  <- SHU RASMNI OCHIB KO'RING, bu eng muhim qadam")


def main() -> None:
    ap = argparse.ArgumentParser(description="Illumination dataset diagnostikasi")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-subdir", default="train")
    ap.add_argument("--test-subdir", default="test")
    ap.add_argument("--limit-per-class", type=int, default=0, help="0 = hammasi")
    ap.add_argument("--out-montage", default="diagnose_samples.png")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    items = scan_train_dir(data_dir / args.train_subdir)
    classes = items.classes

    paths_by_class: dict[str, list[Path]] = defaultdict(list)
    for p, y in zip(items.paths, items.labels):
        paths_by_class[classes[y]].append(p)
    if args.limit_per_class:
        paths_by_class = {c: v[: args.limit_per_class] for c, v in paths_by_class.items()}

    # ---------- 0. Fayl metadata ----------
    print("===== 0. Fayl metadata =====")
    sizes, modes = Counter(), Counter()
    for cls, paths in paths_by_class.items():
        for p in paths[:100]:
            with Image.open(p) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
    print(f"  Eng ko'p uchraydigan o'lchamlar: {sizes.most_common(5)}")
    print(f"  Rang rejimlari: {dict(modes)}")

    # ---------- Featurelarni hisoblash ----------
    feats_by_class: dict[str, list[dict[str, float]]] = {}
    for cls, paths in paths_by_class.items():
        print(f"  {cls} hisoblanmoqda ({len(paths)} ta)...")
        feats_by_class[cls] = [simple_features(load_bgr(p)) for p in paths]

    names = list(next(iter(feats_by_class.values()))[0])
    matrices = {c: np.array([[f[n] for n in names] for f in fs]) for c, fs in feats_by_class.items()}
    x_all = np.vstack([matrices[c] for c in classes])
    y_all = np.concatenate([np.full(len(matrices[c]), i) for i, c in enumerate(classes)])

    # ---------- 1. Sinf bo'yicha statistika ----------
    print("\n===== 1. Sinflar bo'yicha o'rtacha qiymatlar =====")
    header = f"{'feature':<16}" + "".join(f"{c:>12}" for c in classes) + f"{'ajratish':>11}"
    print(header)
    print("-" * len(header))

    powers = {}
    for k, name in enumerate(names):
        power = single_feature_power(x_all[:, k], y_all, len(classes))
        powers[name] = power
        means = "".join(f"{matrices[c][:, k].mean():>12.4f}" for c in classes)
        print(f"{name:<16}{means}{power:>11.3f}")

    print("\n  'ajratish' = shu bitta feature bilan erishish mumkin bo'lgan eng yaxshi accuracy")
    print(f"  (tasodifiy = {1/len(classes):.3f}). 0.9+ bo'lsa -> feature label bilan bevosita bog'liq.")
    top = sorted(powers.items(), key=lambda kv: -kv[1])[:5]
    print(f"  Eng kuchli 5 ta: {[(n, round(p, 3)) for n, p in top]}")

    # ---------- 2. Ustma-ustlik ----------
    print("\n===== 2. mean_lum taqsimotining ustma-ustligi =====")
    k = names.index("mean_lum")
    for c in classes:
        v = matrices[c][:, k]
        print(f"  {c:<8} min={v.min():.3f} p10={np.percentile(v,10):.3f} "
              f"median={np.median(v):.3f} p90={np.percentile(v,90):.3f} max={v.max():.3f}")
    print("  Diapazonlar deyarli bir xil bo'lsa -> global yorqinlik label EMAS (tasdiqlanadi).")

    # ---------- 3. Kontent ustma-ustligi ----------
    content_overlap(paths_by_class)

    # ---------- 4. Fazoviy notekislik ----------
    print("\n===== 4. Yoritishning fazoviy notekisligi =====")
    for nm in ("abs_vert_grad", "abs_horiz_grad", "block_std", "block_range"):
        k = names.index(nm)
        vals = "  ".join(f"{c}={matrices[c][:, k].mean():.4f}" for c in classes)
        print(f"  {nm:<16} {vals}   (ajratish={powers[nm]:.3f})")
    print("  Bu qiymatlar sinflar bo'yicha sezilarli farq qilsa -> label LOKAL yoritishga bog'liq,")
    print("  ya'ni fazoviy (patch-wise) featurelar yoki CNN kerak.")

    # ---------- 5. Train vs test ----------
    test_dir = data_dir / args.test_subdir
    if test_dir.is_dir():
        test_paths = list_images(test_dir)
        sample = test_paths[:: max(1, len(test_paths) // 300)][:300]
        tf = np.array([[simple_features(load_bgr(p))[n] for n in names] for p in sample])
        print(f"\n===== 5. Train vs Test taqsimoti ({len(sample)} ta test namunasi) =====")
        for nm in ("mean_lum", "std_lum", "block_std", "saturation"):
            k = names.index(nm)
            print(f"  {nm:<12} train={x_all[:, k].mean():.4f}  test={tf[:, k].mean():.4f}")
        print("  Katta farq bo'lsa -> distribution shift, test boshqa sharoitda yig'ilgan.")

    # ---------- 6. Montaj ----------
    save_montage(paths_by_class, Path(args.out_montage))

    print("\n===== Keyingi qadam =====")
    print("  * Montajga qarang: sinflar orasidagi farqni KO'Z bilan ko'ra olasizmi?")
    print("  * Ko'ra olmasangiz -> labellar shovqinli/sub'ektiv, tavsiya: CNN + kuchli regularizatsiya.")
    print("  * Farq lokal (soya, backlight) bo'lsa -> baseline_v2.py dagi fazoviy featurelar yordam beradi.")
    print("  * Yuqoridagi chiqishni menga yuboring — feature strategiyasini shunga qarab aniqlaymiz.")


if __name__ == "__main__":
    main()
