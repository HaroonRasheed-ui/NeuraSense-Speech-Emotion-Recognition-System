"""
Quick prediction script - test a single audio file
Usage: python predict.py path/to/audio.wav
"""

import sys
import os
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import librosa

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.feature_extractor import extract_features  # type: ignore[import]

EMOTION_EMOJI = {
    'happy': '😄', 'sad': '😢', 'angry': '😠', 'fearful': '😨',
    'disgust': '🤢', 'surprised': '😲', 'neutral': '😐', 'calm': '😌',
}

def predict(audio_path):
    print("\n🔄 Loading model...")
    model = tf.keras.models.load_model('./models/emotion_model.h5')

    with open('./models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('./models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    print(f"🎵 Analyzing: {audio_path}\n")
    features = extract_features(file_path=audio_path)

    if features is None:
        print("❌ Could not extract features from audio")
        return

    features_scaled = scaler.transform([features])
    features_reshaped = features_scaled.reshape(1, features_scaled.shape[1], 1)
    predictions = model.predict(features_reshaped, verbose=0)[0]

    print("=" * 40)
    print("  EMOTION ANALYSIS RESULTS")
    print("=" * 40)

    sorted_idx = np.argsort(predictions)[::-1]
    for i, idx in enumerate(sorted_idx):
        emotion = le.classes_[idx]
        prob = predictions[idx] * 100
        emoji = EMOTION_EMOJI.get(emotion, '🔹')
        bar = '█' * int(prob / 5)
        marker = ' ← TOP' if i == 0 else ''
        print(f"  {emoji} {emotion:<12} {bar:<20} {prob:.1f}%{marker}")

    print("=" * 40)
    top = le.classes_[sorted_idx[0]]
    conf = predictions[sorted_idx[0]] * 100
    print(f"\n  Result: {EMOTION_EMOJI.get(top, '🎤')} {top.upper()} ({conf:.1f}% confidence)\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)

    predict(audio_path)