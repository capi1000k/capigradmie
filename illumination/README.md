# Image Illumination Classification (bright / dark / normal)

3 sinfli rasm klassifikatsiyasi uchun to'liq pipeline: transfer learning (timm/torchvision),
Stratified K-Fold, checkpoint saqlash, ansambl + TTA inference va submission generatsiyasi.

## Fayllar

| Fayl | Vazifasi |
|---|---|
| `common.py` | Dataset, transformlar, model quruvchi (train va inference uchun umumiy) |
| `train.py` | Stratified K-Fold o'qitish, har fold uchun eng yaxshi checkpoint |
| `predict.py` | Test rasmlari bo'yicha ansambl inference + `sample_submission.csv` |
| `baseline.py` | Tez tekshiruv: yorqinlik gistogrammasi + LogisticRegression (GPU kerak emas) |

## 1. O'rnatish

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r illumination/requirements.txt
# CUDA uchun mos torch versiyasini https://pytorch.org/get-started/locally/ dan oling
```

## 2. Data strukturasi

```
/opt/goinfre/zanerhon/data_освещённость/
├── train/
│   ├── bright/*.png
│   ├── dark/*.png
│   └── normal/*.png
└── test/*.png          # UUID nomli fayllar
```

Klass nomlari `train/` ichidagi papka nomlaridan avtomatik olinadi (alifbo tartibida),
shuning uchun papkalar boshqacha nomlansa ham kod ishlaydi.

## 3. Tez sanity-check (ixtiyoriy, ~2 daqiqa)

```bash
python illumination/baseline.py \
  --data-dir "/opt/goinfre/zanerhon/data_освещённость" \
  --out baseline_submission.csv
```

Agar bu baseline CV F1 ~0.95+ bersa — data va labellar joyida. Juda past bo'lsa,
avval datani tekshiring, deep model ham qutqarmaydi.

## 4. O'qitish

Tez variant (bitta fold, ~10-20 daqiqa GPU'da):

```bash
python illumination/train.py \
  --data-dir "/opt/goinfre/zanerhon/data_освещённость" \
  --model tf_efficientnet_b0 --img-size 224 \
  --epochs 8 --batch-size 32 --folds 5 --train-folds 0 \
  --out-dir outputs
```

To'liq ansambl (5 fold, eng yaxshi natija):

```bash
python illumination/train.py \
  --data-dir "/opt/goinfre/zanerhon/data_освещённость" \
  --model tf_efficientnet_b0 --epochs 10 --folds 5 --train-folds all \
  --out-dir outputs
```

Natija: `outputs/fold{N}_best.pt` + `outputs/meta.json` (CV skorlar bilan).

Alternativ modellar: `--model resnet50`, `--model convnext_tiny`,
`--model vit_small_patch16_224 --img-size 224`, `--model tf_efficientnetv2_s --img-size 300`.
timm o'rnatilmagan bo'lsa, kod avtomatik torchvision'ga tushadi (`--model resnet18`).

GPU yo'q bo'lsa: `--device cpu --img-size 160 --batch-size 16 --epochs 4`.

## 5. Inference va submission

```bash
python illumination/predict.py \
  --data-dir "/opt/goinfre/zanerhon/data_освещённость" \
  --ckpt-dir outputs \
  --out sample_submission.csv \
  --tta
```

`outputs/` dagi barcha `*_best.pt` fayllar avtomatik ansambl qilinadi (softmax o'rtachasi).

**Agar tashkilotchilar bergan `sample_submission.csv` bo'lsa** — ustun nomlari va qator
tartibi aynan saqlanishi uchun uni ko'rsating:

```bash
python illumination/predict.py \
  --data-dir "/opt/goinfre/zanerhon/data_освещённость" \
  --ckpt-dir outputs --tta \
  --sample-submission /path/to/sample_submission.csv \
  --out submission.csv
```

ID formatini boshqarish: standart holatda kengaytmasiz (`0021e90d-...`),
`.png` bilan kerak bo'lsa `--keep-extension` bering. `--sample-submission` berilganda
format sample fayldan avtomatik aniqlanadi.

## Muhim texnik izoh: augmentatsiya

Bu masalada **brightness / contrast / gamma / autocontrast augmentatsiyalari ishlatilmaydi**.
Ular aynan bashorat qilinayotgan belgini o'zgartiradi va `dark` rasmni `normal`ga aylantirib,
labelni yolg'onga chiqaradi — model buzilgan signal ustida o'qiydi va val skor tushadi.
Shuning uchun `common.py` da faqat geometrik augmentatsiyalar (flip, kichik affine) bor.
Shu sababdan validatsiyada ham `Resize` ishlatiladi, `CenterCrop` emas: kesish global
yorug'lik statistikasini o'zgartirib yuboradi.

## Tavsiya etilgan ish tartibi

1. `baseline.py` — data to'g'riligini tekshirish (2 daqiqa).
2. `train.py --train-folds 0` — pipeline ishlashiga ishonch hosil qilish.
3. OOF F1 qoniqarli bo'lsa — `--train-folds all` bilan to'liq ansambl.
4. `predict.py --tta` — yakuniy submission.
