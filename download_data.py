"""
Download and extract training data for AI IJC challenge.
Data source: https://cloud.mail.ru/public/GCsv/1BXmZPEBj
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

def download_data():
    """Download dataset from cloud"""
    # Note: The direct cloud.mail.ru link might need browser access
    # Alternative: manually download and extract to data/ directory

    data_dir = Path('data')

    if data_dir.exists():
        print("✅ Data directory already exists")
        return

    print("📥 Please download data from:")
    print("   https://cloud.mail.ru/public/GCsv/1BXmZPEBj")
    print("\n📂 Extract contents to ./data/ directory")
    print("\nExpected structure:")
    print("   data/")
    print("   ├── train/")
    print("   │   ├── 0/ (dark images)")
    print("   │   ├── 1/ (normal images)")
    print("   │   └── 2/ (bright images)")
    print("   ├── test/ (unlabeled images)")
    print("   ├── train.csv (id, label)")
    print("   ├── test.csv (id)")
    print("   └── sample_submission.csv (template)")


if __name__ == '__main__':
    download_data()
