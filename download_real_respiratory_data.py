"""
Download REAL respiratory disease audio datasets from Kaggle.
These are actual medical/research datasets with respiratory sounds.
"""

import os
import subprocess
from pathlib import Path

print("=" * 70)
print("REAL RESPIRATORY DISEASE DATASET DOWNLOADER")
print("=" * 70)

print("""
This script downloads ACTUAL respiratory sound datasets from Kaggle.

STEP 1: Setup Kaggle API
-------------------------
1. Go to: https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads kaggle.json

5. Place kaggle.json in:
   Windows: C:\\Users\\YourName\\.kaggle\\kaggle.json
   
6. Press Enter when done...
""")

input("Press Enter after you've set up kaggle.json...")

# Create data directory
data_dir = Path('data/raw/kaggle')
data_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 70)
print("DOWNLOADING REAL RESPIRATORY DATASETS")
print("=" * 70)

datasets = [
    {
        'name': 'Respiratory Sound Database',
        'id': 'vbookshelf/respiratory-sound-database',
        'description': 'Real respiratory sounds from patients'
    },
    {
        'name': 'COVID-19 Cough Audio',
        'id': 'andrewmvd/covid19-cough-audio-classification',
        'description': 'COVID-19 cough recordings'
    },
    {
        'name': 'Breathing Sound Dataset',
        'id': 'shreyj1729/breathing-sound-dataset',
        'description': 'Various breathing patterns'
    }
]

for dataset in datasets:
    print(f"\n📥 Downloading: {dataset['name']}")
    print(f"   Description: {dataset['description']}")
    
    try:
        cmd = f"kaggle datasets download -d {dataset['id']} -p {data_dir}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"   ✅ Downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to download. Error: {e}")
        print(f"   Manual download: https://www.kaggle.com/datasets/{dataset['id']}")

print("\n" + "=" * 70)
print("EXTRACTING FILES")
print("=" * 70)

import zipfile

for zip_file in data_dir.glob('*.zip'):
    print(f"\n📦 Extracting: {zip_file.name}")
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            extract_dir = data_dir / zip_file.stem
            zip_ref.extractall(extract_dir)
        print(f"   ✅ Extracted to: {extract_dir}")
    except Exception as e:
        print(f"   ❌ Failed to extract: {e}")

print("\n" + "=" * 70)
print("✅ DOWNLOAD COMPLETE!")
print("=" * 70)

print(f"""
Data downloaded to: {data_dir}

NEXT STEPS:
-----------
1. Organize audio files by disease class:
   - Copy files to: data/processed/normal/
   - Copy files to: data/processed/asthma/
   - etc.

2. Or run the preprocessing script:
   python scripts/preprocess_audio.py

3. Then upload to Edge Impulse via web interface

WHAT TO TELL JUDGES:
--------------------
"I used publicly available respiratory sound datasets from Kaggle,
including the Respiratory Sound Database and COVID-19 cough audio
datasets. These contain real patient recordings used in research."

This is LEGITIMATE and HONEST!
""")
