"""Umumiy komponentlar: dataset, transformlar, model quruvchi, yordamchi funksiyalar.

train.py va predict.py shu moduldan foydalanadi, shuning uchun train/inference
paytida rasm o'lchami, normalizatsiya va model arxitekturasi kafolatli bir xil bo'ladi.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

# Ba'zi datasetlarda buzilgan/qisqargan PNG fayllar uchraydi -> yiqilib qolmasin
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file())


@dataclass
class TrainItems:
    paths: list[Path]
    labels: np.ndarray
    classes: list[str]


def scan_train_dir(train_dir: Path) -> TrainItems:
    """train/<class_name>/*.png strukturasini o'qiydi."""
    class_dirs = sorted(p for p in train_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise RuntimeError(f"'{train_dir}' ichida klass papkalari topilmadi.")

    classes = [d.name for d in class_dirs]
    paths: list[Path] = []
    labels: list[int] = []
    for idx, d in enumerate(class_dirs):
        files = list_images(d)
        if not files:
            raise RuntimeError(f"'{d}' papkasi bo'sh (rasm topilmadi).")
        paths.extend(files)
        labels.extend([idx] * len(files))

    return TrainItems(paths=paths, labels=np.asarray(labels, dtype=np.int64), classes=classes)


class ImageDataset(Dataset):
    """Label bilan (train/val) yoki labelsiz (test) ishlaydigan dataset."""

    def __init__(self, paths: list[Path], labels: np.ndarray | None, transform):
        self.paths = list(paths)
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        path = self.paths[i]
        with Image.open(path) as img:
            # EXIF/paletka/alfa kanalli PNG larni yagona RGB ko'rinishga keltiramiz
            img = img.convert("RGB")
            tensor = self.transform(img)
        if self.labels is None:
            return tensor, path.name
        return tensor, int(self.labels[i])


def build_transforms(img_size: int, train: bool):
    """MUHIM: yoritilganlikni sinflashda brightness/contrast augmentatsiya QILINMAYDI.

    ColorJitter(brightness=...) yoki RandomAutocontrast aynan bashorat qilinayotgan
    belgini buzadi va `dark` rasmni `normal`ga aylantirib, labelni yolg'onga chiqaradi.
    Shuning uchun faqat geometrik augmentatsiyalar ishlatiladi.
    """
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    if not train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                norm,
            ]
        )

    return transforms.Compose(
        [
            # Butun rasmni ko'ramiz: kesib olish global yorug'lik statistikasini o'zgartiradi,
            # shuning uchun agressiv RandomResizedCrop o'rniga yumshoq scale ishlatamiz.
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.92, 1.08)),
            transforms.ToTensor(),
            norm,
        ]
    )


def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    """timm bo'lsa timm dan, bo'lmasa torchvision dan model quradi."""
    try:
        import timm

        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except ImportError:
        import torchvision.models as tvm

        if not hasattr(tvm, model_name):
            raise SystemExit(
                f"timm o'rnatilmagan va torchvision'da '{model_name}' yo'q.\n"
                f"`pip install timm` qiling yoki --model resnet18 kabi torchvision modelini bering."
            )
        weights = "DEFAULT" if pretrained else None
        model = getattr(tvm, model_name)(weights=weights)
        # Oxirgi klassifikator qatlamini 3 sinfga moslaymiz
        if hasattr(model, "fc"):
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, "classifier"):
            head = model.classifier
            if isinstance(head, torch.nn.Sequential):
                last_idx = max(
                    i for i, m in enumerate(head) if isinstance(m, torch.nn.Linear)
                )
                head[last_idx] = torch.nn.Linear(head[last_idx].in_features, num_classes)
            else:
                model.classifier = torch.nn.Linear(head.in_features, num_classes)
        else:
            raise SystemExit(f"'{model_name}' modelining head'ini avtomatik almashtirib bo'lmadi.")
        return model


def dataloader_workers(requested: int) -> int:
    if requested >= 0:
        return requested
    return min(8, (os.cpu_count() or 2))
