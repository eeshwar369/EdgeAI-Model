"""
Upload processed data to Edge Impulse platform.
"""

import os
import json
import subprocess
from pathlib import Path


def upload_to_edge_impulse():
    """Upload dataset to Edge Impulse."""
    
    print("=" * 60)
    print("EDGE IMPULSE DATA UPLOAD")
    print("=" * 60)
    
    # Check if Edge Impulse CLI is installed
    try:
        result = subprocess.run(['edge-impulse-uploader', '--version'], 
                              capture_output=True, text=True)
        print(f"\nEdge Impulse CLI version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("\nError: Edge Impulse CLI not found!")
        print("Install with: npm install -g edge-impulse-cli")
        return
    
    # Check if logged in
    print("\nChecking Edge Impulse login status...")
    try:
        subprocess.run(['edge-impulse-uploader', '--info'], check=True)
    except subprocess.CalledProcessError:
        print("\nNot logged in. Please login first:")
        print("  edge-impulse-login")
        return
    
    # Upload data
    data_dir = Path('data/raw')
    
    if not data_dir.exists():
        print(f"\nError: Data directory {data_dir} not found!")
        return
    
    print(f"\nUploading data from {data_dir}...")
    print("\nThis will upload audio files to your Edge Impulse project.")
    print("Make sure you have selected the correct project.")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Upload cancelled.")
        return
    
    # Upload each class
    classes = ['normal', 'asthma', 'copd', 'pneumonia', 'bronchitis', 'tuberculosis', 'long_covid']
    
    for cls in classes:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            print(f"\nSkipping {cls} (directory not found)")
            continue
        
        audio_files = list(cls_dir.glob('*.wav'))
        if not audio_files:
            print(f"\nSkipping {cls} (no audio files)")
            continue
        
        print(f"\nUploading {len(audio_files)} files from {cls}...")
        
        for audio_file in audio_files:
            try:
                subprocess.run([
                    'edge-impulse-uploader',
                    '--category', 'training',
                    '--label', cls,
                    str(audio_file)
                ], check=True, capture_output=True)
                print(f"  ✓ {audio_file.name}")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ {audio_file.name}: {e}")
    
    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Go to Edge Impulse Studio: https://studio.edgeimpulse.com")
    print("2. Configure DSP blocks (MFCC, Mel-Spectrogram)")
    print("3. Train your model")
    print("4. Deploy to your edge device")


if __name__ == '__main__':
    upload_to_edge_impulse()
