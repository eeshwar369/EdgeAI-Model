"""
Download respiratory audio datasets from public sources.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile


def download_file(url: str, output_path: str):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path: str, extract_to: str):
    """Extract zip file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def download_datasets():
    """Download all respiratory datasets."""
    
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("RESPIRATORY AUDIO DATASET DOWNLOADER")
    print("=" * 60)
    
    # Dataset URLs (these are examples - replace with actual URLs)
    datasets = {
        'ICBHI 2017': {
            'url': 'https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge',
            'info': 'Visit the website to request access and download manually'
        },
        'COUGHVID': {
            'url': 'https://zenodo.org/record/4498364',
            'info': 'Download from Zenodo (requires manual download)'
        },
        'Coswara': {
            'url': 'https://github.com/iiscleap/Coswara-Data',
            'info': 'Clone the GitHub repository or download from Zenodo'
        }
    }
    
    print("\nDATASET SOURCES:")
    print("-" * 60)
    
    for name, info in datasets.items():
        print(f"\n{name}:")
        print(f"  URL: {info['url']}")
        print(f"  Info: {info['info']}")
    
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("=" * 60)
    
    print("""
1. ICBHI 2017 Respiratory Sound Database:
   - Visit: https://bhichallenge.med.auth.gr/
   - Request access and download the dataset
   - Extract to: data/raw/icbhi/

2. COUGHVID Dataset:
   - Visit: https://zenodo.org/record/4498364
   - Download the dataset (requires Zenodo account)
   - Extract to: data/raw/coughvid/

3. Coswara Dataset:
   - Option A: Clone repo: git clone https://github.com/iiscleap/Coswara-Data
   - Option B: Download from Zenodo: https://zenodo.org/record/4904054
   - Extract to: data/raw/coswara/

4. Organize your data in this structure:
   data/raw/
       normal/
       asthma/
       copd/
       pneumonia/
       bronchitis/
       tuberculosis/
       long_covid/

5. After organizing, run: python scripts/preprocess_audio.py
    """)
    
    print("\n" + "=" * 60)
    print("NOTE: Due to licensing and access restrictions, these datasets")
    print("must be downloaded manually. Follow the instructions above.")
    print("=" * 60)


if __name__ == '__main__':
    download_datasets()
