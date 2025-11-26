"""
Create synthetic sample audio data for testing.
Useful when real datasets are not yet downloaded.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from scipy import signal

def generate_breathing_sound(duration=3.0, sr=16000, breathing_rate=15):
    """Generate synthetic breathing sound."""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Breathing frequency (breaths per minute to Hz)
    freq = breathing_rate / 60
    
    # Generate breathing pattern (sine wave with harmonics)
    audio = np.sin(2 * np.pi * freq * t)
    audio += 0.5 * np.sin(2 * np.pi * 2 * freq * t)
    audio += 0.3 * np.sin(2 * np.pi * 3 * freq * t)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, len(audio))
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio

def generate_wheeze_sound(duration=3.0, sr=16000):
    """Generate synthetic wheeze sound (asthma)."""
    t = np.linspace(0, duration, int(sr * duration))
    
    # High-pitched wheeze (400-800 Hz)
    freq = 600
    audio = np.sin(2 * np.pi * freq * t)
    
    # Modulate amplitude (breathing pattern)
    mod_freq = 0.3
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
    audio = audio * modulation
    
    # Add noise
    noise = np.random.normal(0, 0.15, len(audio))
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio

def generate_cough_sound(duration=3.0, sr=16000, num_coughs=3):
    """Generate synthetic cough sound."""
    audio = np.zeros(int(sr * duration))
    
    # Generate individual coughs
    cough_duration = 0.3
    cough_samples = int(sr * cough_duration)
    
    for i in range(num_coughs):
        # Random position
        pos = int(np.random.uniform(0.5, duration - 0.5) * sr)
        
        # Generate cough (burst of noise)
        cough = np.random.normal(0, 1, cough_samples)
        
        # Apply envelope
        envelope = signal.windows.hann(cough_samples)
        cough = cough * envelope
        
        # Add to audio
        if pos + cough_samples < len(audio):
            audio[pos:pos+cough_samples] += cough
    
    # Add background breathing
    breathing = generate_breathing_sound(duration, sr)
    audio = audio + 0.3 * breathing
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio

def create_sample_dataset():
    """Create synthetic sample dataset."""
    
    print("=" * 60)
    print("CREATING SYNTHETIC SAMPLE DATA")
    print("=" * 60)
    
    # Create directories
    base_dir = Path('samples')
    classes = ['normal', 'asthma', 'copd', 'pneumonia', 'bronchitis', 'tuberculosis', 'long_covid']
    
    for cls in classes:
        cls_dir = base_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
    
    sr = 16000
    duration = 3.0
    
    # Generate normal breathing samples
    print("\nGenerating normal breathing samples...")
    for i in range(5):
        audio = generate_breathing_sound(duration, sr, breathing_rate=15 + i)
        sf.write(base_dir / 'normal' / f'normal_{i+1:03d}.wav', audio, sr)
    
    # Generate asthma (wheeze) samples
    print("Generating asthma samples...")
    for i in range(5):
        audio = generate_wheeze_sound(duration, sr)
        sf.write(base_dir / 'asthma' / f'asthma_{i+1:03d}.wav', audio, sr)
    
    # Generate COPD samples (similar to wheeze but different pattern)
    print("Generating COPD samples...")
    for i in range(5):
        audio = generate_wheeze_sound(duration, sr)
        # Add more noise for COPD
        audio += np.random.normal(0, 0.2, len(audio))
        audio = audio / np.max(np.abs(audio)) * 0.8
        sf.write(base_dir / 'copd' / f'copd_{i+1:03d}.wav', audio, sr)
    
    # Generate pneumonia (wet cough) samples
    print("Generating pneumonia samples...")
    for i in range(5):
        audio = generate_cough_sound(duration, sr, num_coughs=3+i)
        sf.write(base_dir / 'pneumonia' / f'pneumonia_{i+1:03d}.wav', audio, sr)
    
    # Generate bronchitis samples
    print("Generating bronchitis samples...")
    for i in range(5):
        audio = generate_cough_sound(duration, sr, num_coughs=4)
        sf.write(base_dir / 'bronchitis' / f'bronchitis_{i+1:03d}.wav', audio, sr)
    
    # Generate tuberculosis (dry cough) samples
    print("Generating tuberculosis samples...")
    for i in range(5):
        audio = generate_cough_sound(duration, sr, num_coughs=2)
        # Make it drier (less low frequency)
        b, a = signal.butter(4, 500, 'hp', fs=sr)
        audio = signal.filtfilt(b, a, audio)
        audio = audio / np.max(np.abs(audio)) * 0.8
        sf.write(base_dir / 'tuberculosis' / f'tuberculosis_{i+1:03d}.wav', audio, sr)
    
    # Generate long-COVID samples
    print("Generating long-COVID samples...")
    for i in range(5):
        audio = generate_breathing_sound(duration, sr, breathing_rate=20 + i)
        # Add irregular pattern
        audio += 0.3 * np.random.normal(0, 0.1, len(audio))
        audio = audio / np.max(np.abs(audio)) * 0.8
        sf.write(base_dir / 'long_covid' / f'long_covid_{i+1:03d}.wav', audio, sr)
    
    print("\n" + "=" * 60)
    print("SAMPLE DATA CREATION COMPLETE!")
    print("=" * 60)
    print(f"\nCreated 35 synthetic audio samples in {base_dir}/")
    print("\nNote: These are synthetic samples for testing only.")
    print("For real training, download actual respiratory sound datasets.")
    print("\nYou can now test inference:")
    print("  python scripts/test_inference.py --audio samples/asthma/asthma_001.wav")

if __name__ == '__main__':
    create_sample_dataset()
