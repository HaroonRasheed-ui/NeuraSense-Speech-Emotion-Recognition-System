"""
Feature Extraction Utility for Speech Emotion Recognition
Extracts rich feature sets for high accuracy (85-90%+)
"""

import librosa
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def extract_features(file_path=None, audio=None, sr=22050, duration=3):
    """
    Extract a rich set of audio features for emotion recognition.
    Supports both file path and raw audio array input (for real-time).
    
    Features extracted:
    - MFCC (40 coefficients + delta + delta-delta)
    - Chroma features
    - Mel spectrogram
    - Spectral features (centroid, rolloff, bandwidth, contrast)
    - Zero crossing rate
    - RMS energy
    - Tonnetz
    """
    try:
        if file_path is not None:
            audio, sr = librosa.load(file_path, duration=duration, offset=0.5, sr=sr)
        
        if audio is None or len(audio) == 0:
            return None

        # Pad or trim to fixed length
        target_length = int(sr * duration)
        if len(audio) < target_length:
            pad_width = target_length - len(audio)
            audio = np.pad(audio, (0, int(pad_width)), mode='constant')
        else:
            audio = audio[:target_length]

        features = []

        # 1. MFCC (40) + Delta (40) + Delta-Delta (40) = 120
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features.extend(np.mean(mfcc.T, axis=0))
        features.extend(np.std(mfcc.T, axis=0))
        features.extend(np.mean(mfcc_delta.T, axis=0))
        features.extend(np.mean(mfcc_delta2.T, axis=0))

        # 2. Chroma (12)
        stft = np.abs(librosa.stft(audio))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr, n_chroma=12)
        features.extend(np.mean(chroma.T, axis=0))
        features.extend(np.std(chroma.T, axis=0))

        # 3. Mel Spectrogram (128)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel)
        features.extend(np.mean(mel_db.T, axis=0))

        # 4. Spectral Centroid
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(np.mean(spec_centroid))
        features.append(np.std(spec_centroid))

        # 5. Spectral Rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.append(np.mean(spec_rolloff))
        features.append(np.std(spec_rolloff))

        # 6. Spectral Bandwidth
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features.append(np.mean(spec_bandwidth))
        features.append(np.std(spec_bandwidth))

        # 7. Spectral Contrast (7 bands)
        spec_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        features.extend(np.mean(spec_contrast.T, axis=0))

        # 8. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        # 9. RMS Energy
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))
        features.append(np.std(rms))

        # 10. Tonnetz (6)
        harmonic = librosa.effects.harmonic(audio)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        features.extend(np.mean(tonnetz.T, axis=0))

        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None


def augment_audio(audio, sr):
    """Data augmentation to improve model generalization."""
    augmented = [audio]

    # Add noise
    noise = np.random.randn(len(audio)) * 0.005
    augmented.append(audio + noise)

    # Time stretch
    try:
        stretched = librosa.effects.time_stretch(audio, rate=0.9)
        augmented.append(stretched)
        stretched2 = librosa.effects.time_stretch(audio, rate=1.1)
        augmented.append(stretched2)
    except:
        pass

    # Pitch shift
    try:
        pitched_up = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
        augmented.append(pitched_up)
        pitched_down = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
        augmented.append(pitched_down)
    except:
        pass

    return augmented