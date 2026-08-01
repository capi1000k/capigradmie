# AI IJC Challenge: Lighting Level Classification

## Task Overview
Classify images into 3 lighting levels:
- **0** = dark (low illumination)
- **1** = normal (medium illumination)  
- **2** = bright (high illumination)

**Metric**: Accuracy
**Passing score**: ≥ 0.40 (max = 1.0 point)
**Good solution**: ≥ 0.70
**Excellent**: ≥ 0.90

### Scoring Formula
```
points = max(0, (accuracy - 0.40) / (1 - 0.40))
```

---

## Data Format

### Input Data
Download from: https://cloud.mail.ru/public/GCsv/1BXmZPEBj

Extract to `./data/` with this structure:
```
data/
├── train/
│   ├── 0/          (dark images)
│   ├── 1/          (normal images)
│   └── 2/          (bright images)
├── test/           (unlabeled test images)
├── train.csv       (id, label)
├── test.csv        (id)
└── sample_submission.csv (template)
```

### Files
- **train.csv**: `id`, `label` (0/1/2)
- **test.csv**: `id` (no labels)
- **sample_submission.csv**: Template for submission with columns `id`, `label`

---

## Solution Architecture

### Model: ResNet50 (Transfer Learning)
- **Base**: Pre-trained ResNet50 on ImageNet
- **Head**: Custom linear layer for 3-class classification
- **Image size**: 224×224 (ImageNet standard)

### Training Strategy
1. **Augmentation**:
   - Random horizontal flip (50%)
   - Random rotation (±10°)
   - Color jitter (brightness, contrast, saturation ±20%)
   - Gaussian blur

2. **Optimization**:
   - Optimizer: Adam (lr=0.001)
   - Scheduler: Cosine Annealing
   - Loss: CrossEntropyLoss
   - Epochs: 50

3. **Validation**:
   - 80/20 train/val split (stratified)
   - Early stopping via best model checkpoint

4. **Test Predictions**:
   - **Test-Time Augmentation (TTA)**: 5 augmentations per image
   - Average predictions across augmentations
   - Final label: argmax of averaged probabilities

---

## Quick Start

### 1. Download Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Data
```bash
python download_data.py
# Then manually download from cloud.mail.ru link and extract to ./data/
```

### 3. Train & Predict
```bash
python solution.py
```

**Output**: `submission.csv` (ready for upload)

---

## Expected Results

### Performance Targets
- **Baseline** (no augmentation, 30 epochs): ~0.65 accuracy
- **With augmentation** (current): ~0.75 accuracy
- **With TTA + ensemble**: ~0.85+ accuracy

### Output File Format
```csv
id,label
02137a86-0743-40e0-845b-6d22d1d5cc85,0
025d39a8-7859-4558-9bf9-bbdd475c6100,1
02a2a878-c5a4-490a-8061-6b2f4ac3b6d0,0
...
```

---

## Optimization Tips

### If accuracy < 0.40:
- Increase epochs to 75-100
- Use stronger augmentations (RandomContrast, RandomBrightness from albumentations)
- Check data format (ensure 3-channel RGB images)

### If accuracy 0.40-0.70:
- Add TTA (currently implemented)
- Try ensemble of 2-3 models (ResNet50 + EfficientNet-B0 + ViT)
- Increase batch size to 64
- Use learning rate scheduling

### For 0.90+ accuracy:
- Ensemble 3-4 different architectures
- TTA with 10+ augmentations
- Focal Loss instead of CrossEntropyLoss (handles class imbalance)
- Multi-crop prediction (crop different regions)

---

## File Structure
```
.
├── solution.py           (Main training + inference script)
├── download_data.py      (Data download helper)
├── requirements.txt      (Dependencies)
├── SOLUTION.md          (This file)
├── submission.csv       (Output - ready to submit)
└── best_model.pth       (Saved best model weights)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No GPU available` | Code auto-falls back to CPU (slower but works) |
| `Out of memory` | Reduce batch_size from 32 to 16 |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Data not found` | Ensure data/ folder exists with correct structure |
| `Low accuracy` | Check image format (must be RGB PNG), increase epochs |

---

## References
- Challenge: https://aiijc.com/ru/
- ResNet paper: https://arxiv.org/abs/1512.03385
- PyTorch: https://pytorch.org/
