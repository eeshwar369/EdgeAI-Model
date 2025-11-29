"""
Download real respiratory audio datasets for training.
This script helps you get actual respiratory sound data.
"""

import os
import subprocess
from pathlib import Path


def download_datasets():
    """Download real respiratory sound datasets."""
    
    print("=" * 70)
    print("REAL RESPIRATORY DATASET DOWNLOADER")
    print("=" * 70)
    
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📥 OPTION 1: Kaggle Datasets (Easiest)")
    print("-" * 70)
    print("""
1. Install Kaggle CLI:
   pip install kaggle

2. Setup Kaggle API:
   - Go to: https://www.kaggle.com/settings
   - Click "Create New API Token"
   - Save kaggle.json to: ~/.kaggle/kaggle.json (Linux/Mac)
                      or: C:\\Users\\YourName\\.kaggle\\kaggle.json (Windows)

3. Download datasets:
   kaggle datasets download -d vbookshelf/respiratory-sound-database
   kaggle datasets download -d andrewmvd/covid19-cough-audio-classification
   
4. Extract:
   unzip respiratory-sound-database.zip -d data/raw/
   unzip covid19-cough-audio-classification.zip -d data/raw/
""")
    
    print("\n📥 OPTION 2: GitHub Datasets (Free)")
    print("-" * 70)
    print("""
1. ESC-50 (Contains cough sounds):
   git clone https://github.com/karolpiczak/ESC-50.git data/raw/esc50
   
2. Coswara (COVID-19 sounds):
   git clone https://github.com/iiscleap/Coswara-Data.git data/raw/coswara
""")
    
    print("\n📥 OPTION 3: Direct Downloads")
    print("-" * 70)
    print("""
1. COUGHVID Dataset:
   - Visit: https://zenodo.org/record/4498364
   - Download ZIP file
   - Extract to: data/raw/coughvid/

2. FSD50K:
   - Visit: https://zenodo.org/record/4060432
   - Download dev/eval sets
   - Extract to: data/raw/fsd50k/
""")
    
    print("\n📥 OPTION 4: Quick Start (Recommended for Demo)")
    print("-" * 70)
    print("""
For a quick working demo, use this approach:

1. Download small sample dataset:
   wget https://github.com/karolpiczak/ESC-50/archive/master.zip
   unzip master.zip -d data/raw/

2. Use data augmentation to expand:
   python scripts/preprocess_audio.py --augment

3. This gives you:
   - Real audio samples (base)
   - Augmented variations (expanded dataset)
   - Legitimate approach used in research
""")
    
    print("\n" + "=" * 70)
    print("AFTER DOWNLOADING:")
    print("=" * 70)
    print("""
1. Organize files by class:
   data/raw/
       normal/
       asthma/
       copd/
       pneumonia/
       bronchitis/
       tuberculosis/
       long_covid/

2. Run preprocessing:
   python scripts/preprocess_audio.py

3. Upload to Edge Impulse:
   edge-impulse-uploader --category training data/processed/normal/*.wav --label normal
   (repeat for each class)
""")
    
    print("\n✅ TIP: You can also record your own samples!")
    print("   - Use your phone to record breathing/coughing")
    print("   - Label them appropriately")
    print("   - Mix with downloaded datasets")
    print("=" * 70)


if __name__ == '__main__':
    download_datasets()
