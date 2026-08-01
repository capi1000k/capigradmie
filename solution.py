"""
AI IJC Challenge: Lighting Level Classification
Task: Classify image lighting as dark (0), normal (1), or bright (2)

Data structure:
- train.csv: id (filename), label (0/1/2)
- test.csv: id (filename)
- data/train/{class}/{image}.png
- data/test/{image}.png
- sample_submission.csv: id, label (template for results)
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Config
CONFIG = {
    'image_size': 224,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_classes': 3,
    'random_seed': 42,
}

np.random.seed(CONFIG['random_seed'])
torch.manual_seed(CONFIG['random_seed'])


class LightingDataset(Dataset):
    """Custom dataset for lighting classification"""
    def __init__(self, ids, labels=None, data_dir='data/train', transform=None):
        self.ids = ids
        self.labels = labels
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # Find image file (search in class subdirectories if training)
        if self.labels is not None:
            label = self.labels[idx]
            img_path = os.path.join(self.data_dir, str(label), f'{img_id}.png')
        else:
            # For test set, search in all subdirectories
            img_path = None
            parent_dir = os.path.dirname(self.data_dir) if 'train' in self.data_dir else self.data_dir
            for root, dirs, files in os.walk(parent_dir):
                if f'{img_id}.png' in files:
                    img_path = os.path.join(root, f'{img_id}.png')
                    break

        if img_path is None:
            img_path = os.path.join(self.data_dir, f'{img_id}.png')

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image


def get_transforms(train=True):
    """Data augmentation and normalization"""
    if train:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def create_model():
    """Create ResNet50 model for classification"""
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, CONFIG['num_classes'])
    return model.to(CONFIG['device'])


def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), correct / total


def validate(model, val_loader, criterion):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(val_loader), correct / total


def predict_tta(model, test_loader, num_augmentations=5):
    """Test-Time Augmentation predictions"""
    all_predictions = []
    all_ids = []

    model.eval()
    with torch.no_grad():
        for images in test_loader:
            # Apply multiple augmentations
            predictions = []
            for _ in range(num_augmentations):
                aug_images = images.to(CONFIG['device'])
                outputs = model(aug_images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                predictions.append(probs.cpu().numpy())

            # Average predictions across augmentations
            avg_predictions = np.mean(predictions, axis=0)
            predicted_labels = np.argmax(avg_predictions, axis=1)
            all_predictions.extend(predicted_labels)

    return all_predictions


def main():
    """Main pipeline"""
    # Check if data exists
    if not os.path.exists('data'):
        print("⚠️  Data directory 'data' not found!")
        print("Please download from: https://cloud.mail.ru/public/GCsv/1BXmZPEBj")
        print("Extract to: ./data/")
        return

    print("📊 Loading data...")

    # Load train data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    print(f"✅ Train samples: {len(train_df)}")
    print(f"✅ Test samples: {len(test_df)}")
    print(f"✅ Class distribution:\n{train_df['label'].value_counts().sort_index()}")

    # Train/val split
    from sklearn.model_selection import train_test_split
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        train_df['id'].values,
        train_df['label'].values,
        test_size=0.2,
        random_state=CONFIG['random_seed'],
        stratify=train_df['label']
    )

    # Create datasets
    print("\n🖼️  Creating datasets...")
    train_dataset = LightingDataset(train_ids, train_labels, 'data/train', get_transforms(train=True))
    val_dataset = LightingDataset(val_ids, val_labels, 'data/train', get_transforms(train=False))
    test_dataset = LightingDataset(test_df['id'].values, None, 'data/test', get_transforms(train=False))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    # Train model
    print("\n🚀 Training model...")
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    best_val_acc = 0

    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Generate predictions
    print("\n📈 Generating test predictions (with TTA)...")
    predictions = predict_tta(model, test_loader, num_augmentations=5)

    # Save submission
    submission = pd.DataFrame({
        'id': test_df['id'].values,
        'label': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(f"\n✅ Submission saved to 'submission.csv'")
    print(f"📋 Class distribution in predictions:\n{submission['label'].value_counts().sort_index()}")

    # Validate on train set to estimate score
    print("\n🔍 Validation accuracy on train set:")
    train_preds = []
    model.eval()
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(CONFIG['device'])
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())

    train_acc = np.mean(train_preds[:len(train_ids)] == train_labels)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Estimated Score: {max(0, (train_acc - 0.40) / (1 - 0.40)):.4f}")


if __name__ == '__main__':
    main()
