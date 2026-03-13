# """
# Speech Emotion Recognition - Flask Backend
# Real-time voice analysis via WebSocket
# Run: python app.py
# """

# import os
# import sys
# import pickle
# import numpy as np
# import warnings
# warnings.filterwarnings('ignore')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# from typing import Optional, Any
# from flask import Flask, render_template, request, jsonify
# from flask_socketio import SocketIO, emit
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import librosa
# import base64
# import tempfile

# _ROOT = os.path.dirname(os.path.abspath(__file__))
# if _ROOT not in sys.path:
#     sys.path.insert(0, _ROOT)

# from utils.feature_extractor import extract_features  # type: ignore[import]

# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'emotion_recognition_secret'
# socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# # ─────────────────────────────────────────────
# # LOAD MODEL & ARTIFACTS
# # ─────────────────────────────────────────────
# MODEL_PATH   = './models/emotion_model.h5'
# SCALER_PATH  = './models/scaler.pkl'
# ENCODER_PATH = './models/label_encoder.pkl'
# SVM_PATH     = './models/svm_model.pkl'
# RF_PATH      = './models/rf_model.pkl'

# model: Optional[Any] = None
# scaler: Optional[StandardScaler] = None
# label_encoder: Optional[LabelEncoder] = None
# svm_model: Optional[Any] = None
# rf_model:  Optional[Any] = None

# EMOTION_EMOJI = {
#     'happy': '😄',
#     'sad': '😢',
#     'angry': '😠',
#     'fearful': '😨',
#     'disgust': '🤢',
#     'surprised': '😲',
#     'neutral': '😐',
#     'calm': '😌',
# }

# EMOTION_COLORS = {
#     'happy': '#FFD700',
#     'sad': '#4A90D9',
#     'angry': '#E74C3C',
#     'fearful': '#9B59B6',
#     'disgust': '#27AE60',
#     'surprised': '#E67E22',
#     'neutral': '#95A5A6',
#     'calm': '#1ABC9C',
# }


# def load_model_artifacts():
#     """Load trained model and preprocessing artifacts."""
#     global model, scaler, label_encoder, svm_model, rf_model

#     if not os.path.exists(MODEL_PATH):
#         print("\n⚠️  Model not found! Please train first:")
#         print("   python train_model.py")
#         return False

#     print("🔄 Loading model...")
#     model = tf.keras.models.load_model(MODEL_PATH)
#     print("✅ Model loaded")

#     with open(SCALER_PATH, 'rb') as f:
#         scaler = pickle.load(f)
#     print("✅ Scaler loaded")

#     with open(ENCODER_PATH, 'rb') as f:
#         label_encoder = pickle.load(f)
#     print("✅ Label encoder loaded")

#     # Load ensemble models if available
#     if os.path.exists(SVM_PATH):
#         with open(SVM_PATH, 'rb') as f:
#             svm_model = pickle.load(f)
#         print("✅ SVM model loaded")
#     if os.path.exists(RF_PATH):
#         with open(RF_PATH, 'rb') as f:
#             rf_model = pickle.load(f)
#         print("✅ RF model loaded")

#     # Warm up model
#     assert model is not None, "Model failed to load"
#     input_dim = model.input_shape[1]  # type: ignore[index]
#     dummy = np.zeros((1, input_dim, 1))
#     model.predict(dummy, verbose=0)  # type: ignore[union-attr]
#     print("✅ Model warmed up\n")
#     return True


# def predict_emotion(audio_array: np.ndarray, sample_rate: int = 22050) -> Optional[dict]:
#     """Run emotion prediction on audio array."""
#     if model is None or scaler is None or label_encoder is None:
#         return None

#     features = extract_features(audio=audio_array, sr=sample_rate, duration=3)
#     if features is None:
#         return None

#     features_scaled = scaler.transform([features])
#     features_reshaped = features_scaled.reshape(1, features_scaled.shape[1], 1)

#     cnn_probs = model.predict(features_reshaped, verbose=0)[0]

#     # Ensemble: CNN + SVM + RF (if available)
#     if svm_model is not None and rf_model is not None:
#         classes = list(label_encoder.classes_)
#         svm_p = svm_model.predict_proba(features_scaled)[0]
#         rf_p  = rf_model.predict_proba(features_scaled)[0]

#         def align(probs, src_classes):
#             out = np.zeros(len(classes))
#             for i, c in enumerate(src_classes):
#                 if c in classes:
#                     out[classes.index(c)] = probs[i]
#             return out

#         svm_p = align(svm_p, list(svm_model.classes_))
#         rf_p  = align(rf_p,  list(rf_model.classes_))
#         predictions = 0.50 * cnn_probs + 0.30 * svm_p + 0.20 * rf_p
#     else:
#         predictions = cnn_probs

#     # Build result dict
#     emotion_probs = {}
#     for i, emotion in enumerate(label_encoder.classes_):
#         emotion_probs[emotion] = float(predictions[i])

#     top_idx = int(np.argmax(predictions))
#     top_emotion = label_encoder.classes_[top_idx]
#     confidence = float(predictions[top_idx])

#     return {
#         'emotion': top_emotion,
#         'confidence': round(confidence * 100, 1),
#         'emoji': EMOTION_EMOJI.get(top_emotion, '🎤'),
#         'color': EMOTION_COLORS.get(top_emotion, '#666'),
#         'probabilities': {
#             e: round(p * 100, 1)
#             for e, p in emotion_probs.items()
#         }
#     }


# # ─────────────────────────────────────────────
# # ROUTES
# # ─────────────────────────────────────────────
# @app.route('/')
# def index():
#     classes = list(label_encoder.classes_) if label_encoder else []
#     return render_template('index.html', emotions=classes)


# @app.route('/predict_file', methods=['POST'])
# def predict_file():
#     """Handle file upload — supports wav, mp3, ogg, flac, m4a."""
#     if 'audio' not in request.files:
#         return jsonify({'error': 'No audio file received'}), 400

#     file = request.files['audio']
#     if file.filename == '':
#         return jsonify({'error': 'Empty filename'}), 400

#     import pathlib
#     ext = pathlib.Path(file.filename).suffix.lower() or '.wav'

#     tmp_src = None
#     tmp_wav = None
#     try:
#         # Step 1: save uploaded file with correct extension
#         fd, tmp_src = tempfile.mkstemp(suffix=ext)
#         os.close(fd)
#         file.save(tmp_src)
#         print(f"[upload] saved {tmp_src}  size={os.path.getsize(tmp_src)} bytes  ext={ext}")

#         # Step 2: convert non-WAV formats using imageio-ffmpeg (bundled, no system install)
#         if ext != '.wav':
#             try:
#                 import imageio_ffmpeg
#                 ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
#                 fd2, tmp_wav = tempfile.mkstemp(suffix='.wav')
#                 os.close(fd2)
#                 import subprocess
#                 result = subprocess.run(
#                     [ffmpeg_exe, '-y', '-i', tmp_src,
#                      '-ar', '22050', '-ac', '1', '-f', 'wav', tmp_wav],
#                     capture_output=True, text=True
#                 )
#                 if result.returncode == 0:
#                     load_path = tmp_wav
#                     print(f"[upload] converted to wav: {tmp_wav}")
#                 else:
#                     print(f"[upload] ffmpeg error: {result.stderr[-300:]}")
#                     load_path = tmp_src
#             except Exception as conv_err:
#                 print(f"[upload] conversion failed: {conv_err}")
#                 load_path = tmp_src
#         else:
#             load_path = tmp_src

#         # Step 3: load with librosa
#         audio, sr = librosa.load(load_path, sr=22050, duration=3, mono=True)
#         print(f"[upload] audio shape={audio.shape}  sr={sr}")

#         if len(audio) < 1000:
#             return jsonify({'error': 'Audio too short — upload at least 1 second'}), 422

#         result = predict_emotion(audio, int(sr))
#         if result:
#             print(f"[upload] {result['emotion']} ({result['confidence']}%)")
#             return jsonify({'success': True, 'result': result})
#         else:
#             return jsonify({'error': 'Feature extraction failed'}), 422

#     except Exception as e:
#         import traceback
#         print(f"[upload] ERROR:\n{traceback.format_exc()}")
#         return jsonify({'error': f'Processing error: {str(e)}'}), 500

#     finally:
#         for p in [tmp_src, tmp_wav]:
#             if p and os.path.exists(p):
#                 try:
#                     os.unlink(p)
#                 except Exception:
#                     pass


# # ─────────────────────────────────────────────
# # WEBSOCKET - REAL-TIME AUDIO
# # ─────────────────────────────────────────────
# audio_buffer = {}

# @socketio.on('connect')
# def handle_connect():
#     sid = request.args.get('sid', getattr(request, 'sid', 'unknown'))  # type: ignore[attr-defined]
#     print(f"🔌 Client connected: {sid}")
#     conn_sid: str = getattr(request, 'sid', 'default')  # type: ignore[attr-defined]
#     audio_buffer[conn_sid] = []


# @socketio.on('disconnect')
# def handle_disconnect():
#     conn_sid: str = getattr(request, 'sid', 'default')  # type: ignore[attr-defined]
#     print(f"🔌 Client disconnected: {conn_sid}")
#     audio_buffer.pop(conn_sid, None)


# @socketio.on('audio_chunk')
# def handle_audio_chunk(data):
#     """Receive audio chunks from browser and accumulate."""
#     sid: str = getattr(request, 'sid', 'default')  # type: ignore[attr-defined]
#     if sid not in audio_buffer:
#         audio_buffer[sid] = []

#     # Decode base64 audio chunk
#     try:
#         audio_bytes = base64.b64decode(data['audio'])
#         audio_buffer[sid].append(audio_bytes)

#         # Analyze every ~2 seconds of audio (at 44100Hz, 16-bit = 176400 bytes)
#         total_bytes = sum(len(c) for c in audio_buffer[sid])
#         if total_bytes >= 88200:  # ~1 second at 44100Hz 16-bit mono
#             # Combine and convert
#             combined = b''.join(audio_buffer[sid])
#             audio_buffer[sid] = []  # Reset buffer

#             # Convert PCM bytes to numpy array
#             audio_int = np.frombuffer(combined, dtype=np.int16)
#             audio_float = audio_int.astype(np.float32) / 32768.0

#             # Resample from 44100 to 22050
#             audio_resampled = librosa.resample(audio_float, orig_sr=44100, target_sr=22050)

#             result = predict_emotion(audio_resampled, 22050)
#             if result:
#                 emit('emotion_result', result)

#     except Exception as e:
#         print(f"Audio chunk error: {e}")


# @socketio.on('predict_recorded')
# def handle_predict_recorded(data):
#     """Handle a complete recorded audio for prediction."""
#     try:
#         audio_bytes = base64.b64decode(data['audio'])
#         sample_rate = data.get('sampleRate', 44100)

#         audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
#         audio_float = audio_int.astype(np.float32) / 32768.0

#         if sample_rate != 22050:
#             audio_float = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=22050)

#         result = predict_emotion(audio_float, 22050)
#         if result:
#             emit('emotion_result', result)
#         else:
#             emit('error', {'message': 'Could not analyze audio'})

#     except Exception as e:
#         emit('error', {'message': str(e)})


# # ─────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────
# if __name__ == '__main__':
#     print("=" * 60)
#     print("  SPEECH EMOTION RECOGNITION - WEB APP")
#     print("=" * 60)

#     if load_model_artifacts():
#         print("🚀 Starting server at http://localhost:5000")
#         print("   Press Ctrl+C to stop\n")
#         socketio.run(app, host='0.0.0.0', port=5000, debug=False)
#     else:
#         sys.exit(1)

"""
Speech Emotion Recognition - Flask Backend
Real-time voice analysis via WebSocket
Run: python app.py
"""

import os
import sys
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import Optional, Any
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import librosa
import base64
import tempfile
import csv
import threading
import time
from datetime import datetime

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.feature_extractor import extract_features  # type: ignore[import]

app = Flask(__name__)
app.config['SECRET_KEY'] = 'emotion_recognition_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─────────────────────────────────────────────
# LOAD MODEL & ARTIFACTS
# ─────────────────────────────────────────────
MODEL_PATH   = './models/emotion_model.h5'
SCALER_PATH  = './models/scaler.pkl'
ENCODER_PATH = './models/label_encoder.pkl'
SVM_PATH     = './models/svm_model.pkl'
RF_PATH      = './models/rf_model.pkl'

model: Optional[Any] = None
scaler: Optional[StandardScaler] = None
label_encoder: Optional[LabelEncoder] = None
svm_model: Optional[Any] = None
rf_model:  Optional[Any] = None

# ─────────────────────────────────────────────
# ONLINE LEARNING — FEEDBACK SYSTEM
# ─────────────────────────────────────────────
FEEDBACK_CSV     = './models/feedback_data.csv'
RETRAIN_EVERY    = 20       # retrain after every N confirmed feedbacks
MIN_SAMPLES      = 50       # minimum feedback samples before retraining starts
feedback_lock    = threading.Lock()
retrain_lock     = threading.Lock()
is_retraining    = False
feedback_count   = 0        # total confirmed feedbacks collected
last_prediction_features: Optional[np.ndarray] = None   # store last features for feedback

def _count_feedback():
    """Count existing feedback rows in CSV."""
    if not os.path.exists(FEEDBACK_CSV):
        return 0
    with open(FEEDBACK_CSV, 'r') as f:
        return max(0, sum(1 for _ in f) - 1)  # subtract header

def save_feedback(features: np.ndarray, correct_label: str):
    """Save confirmed feature vector + label to feedback CSV."""
    global feedback_count
    file_exists = os.path.exists(FEEDBACK_CSV)
    with feedback_lock:
        with open(FEEDBACK_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Write header: feature_0, feature_1, ..., label, timestamp
                header = [f'f{i}' for i in range(len(features))] + ['label', 'timestamp']
                writer.writerow(header)
            row = list(features) + [correct_label, datetime.now().isoformat()]
            writer.writerow(row)
        feedback_count = _count_feedback()
    print(f"[feedback] saved: {correct_label} | total={feedback_count}")
    return feedback_count

def retrain_model_background():
    """
    Background thread: fine-tune CNN on feedback data + original cached data.
    Uses a small learning rate so the model improves without forgetting.
    """
    global model, scaler, is_retraining
    is_retraining = True
    print("\n[retrain] Starting background retraining...")

    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.optimizers import Adam

        # Load feedback data
        df = pd.read_csv(FEEDBACK_CSV)
        if len(df) < MIN_SAMPLES:
            print(f"[retrain] Only {len(df)} samples, need {MIN_SAMPLES}. Skipping.")
            return

        feature_cols = [c for c in df.columns if c.startswith('f')]
        X_fb = df[feature_cols].values.astype(np.float32)
        y_fb = df['label'].values

        # Also load cached training data for replay (prevents forgetting)
        X_all, y_all = X_fb.copy(), y_fb.copy()
        for cache_name in ['ravdess', 'tess']:
            cache_path = f'./cache/{cache_name}_features.pkl'
            if os.path.exists(cache_path):
                import pickle as pkl
                with open(cache_path, 'rb') as f:
                    Xc, yc = pkl.load(f)
                # Sample 10% of cached data for replay (memory efficiency)
                n_replay = max(100, len(Xc) // 10)
                idx = np.random.choice(len(Xc), min(n_replay, len(Xc)), replace=False)
                X_all = np.vstack([X_all, np.array(Xc)[idx]])
                y_all = np.concatenate([y_all, np.array(yc)[idx]])

        print(f"[retrain] Total samples: {len(X_all)} (feedback={len(X_fb)}, replay={len(X_all)-len(X_fb)})")

        # Scale using existing scaler
        X_scaled = scaler.transform(X_all)

        # Encode labels
        y_enc = label_encoder.transform(y_all)
        num_classes = len(label_encoder.classes_)
        y_cat = to_categorical(y_enc, num_classes)

        X_reshaped = X_scaled.reshape(len(X_scaled), X_scaled.shape[1], 1)

        # Fine-tune with very small LR to avoid catastrophic forgetting
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            X_reshaped, y_cat,
            epochs=5,
            batch_size=32,
            validation_split=0.15,
            verbose=1
        )

        val_acc = max(history.history.get('val_accuracy', [0]))
        print(f"[retrain] Fine-tune complete. Best val_acc: {val_acc*100:.1f}%")

        # Save updated model
        model.save(MODEL_PATH)
        print(f"[retrain] Model saved to {MODEL_PATH}")

        # Notify all connected clients
        socketio.emit('retrain_complete', {
            'val_accuracy': round(val_acc * 100, 1),
            'feedback_samples': len(X_fb),
            'message': f'Model updated with {len(X_fb)} feedback samples!'
        })

    except Exception as e:
        import traceback
        print(f"[retrain] ERROR: {traceback.format_exc()}")
    finally:
        is_retraining = False
        print("[retrain] Done.\n")

EMOTION_EMOJI = {
    'happy': '😄',
    'sad': '😢',
    'angry': '😠',
    'fearful': '😨',
    'disgust': '🤢',
    'surprised': '😲',
    'neutral': '😐',
    'calm': '😌',
}

EMOTION_COLORS = {
    'happy': '#FFD700',
    'sad': '#4A90D9',
    'angry': '#E74C3C',
    'fearful': '#9B59B6',
    'disgust': '#27AE60',
    'surprised': '#E67E22',
    'neutral': '#95A5A6',
    'calm': '#1ABC9C',
}


def load_model_artifacts():
    """Load trained model and preprocessing artifacts."""
    global model, scaler, label_encoder, svm_model, rf_model

    if not os.path.exists(MODEL_PATH):
        print("\n⚠️  Model not found! Please train first:")
        print("   python train_model.py")
        return False

    print("🔄 Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded")

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Scaler loaded")

    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print("✅ Label encoder loaded")

    # Load ensemble models if available
    if os.path.exists(SVM_PATH):
        with open(SVM_PATH, 'rb') as f:
            svm_model = pickle.load(f)
        print("✅ SVM model loaded")
    if os.path.exists(RF_PATH):
        with open(RF_PATH, 'rb') as f:
            rf_model = pickle.load(f)
        print("✅ RF model loaded")

    # Warm up model
    assert model is not None, "Model failed to load"
    input_dim = model.input_shape[1]  # type: ignore[index]
    dummy = np.zeros((1, input_dim, 1))
    model.predict(dummy, verbose=0)  # type: ignore[union-attr]
    print("✅ Model warmed up\n")
    return True


def predict_emotion(audio_array: np.ndarray, sample_rate: int = 22050) -> Optional[dict]:
    """Run emotion prediction on audio array."""
    if model is None or scaler is None or label_encoder is None:
        return None

    features = extract_features(audio=audio_array, sr=sample_rate, duration=3)
    if features is None:
        return None

    features_scaled = scaler.transform([features])
    features_reshaped = features_scaled.reshape(1, features_scaled.shape[1], 1)

    cnn_probs = model.predict(features_reshaped, verbose=0)[0]

    # Ensemble: CNN + SVM + RF (if available)
    if svm_model is not None and rf_model is not None:
        classes = list(label_encoder.classes_)
        svm_p = svm_model.predict_proba(features_scaled)[0]
        rf_p  = rf_model.predict_proba(features_scaled)[0]

        def align(probs, src_classes):
            out = np.zeros(len(classes))
            for i, c in enumerate(src_classes):
                if c in classes:
                    out[classes.index(c)] = probs[i]
            return out

        svm_p = align(svm_p, list(svm_model.classes_))
        rf_p  = align(rf_p,  list(rf_model.classes_))
        predictions = 0.50 * cnn_probs + 0.30 * svm_p + 0.20 * rf_p
    else:
        predictions = cnn_probs

    # Build result dict
    emotion_probs = {}
    for i, emotion in enumerate(label_encoder.classes_):
        emotion_probs[emotion] = float(predictions[i])

    top_idx = int(np.argmax(predictions))
    top_emotion = label_encoder.classes_[top_idx]
    confidence = float(predictions[top_idx])

    result = {
        'emotion': top_emotion,
        'confidence': round(confidence * 100, 1),
        'emoji': EMOTION_EMOJI.get(top_emotion, '🎤'),
        'color': EMOTION_COLORS.get(top_emotion, '#666'),
        'probabilities': {
            e: round(p * 100, 1)
            for e, p in emotion_probs.items()
        }
    }

    # Store raw (unscaled) features for potential feedback saving
    global last_prediction_features
    last_prediction_features = features

    return result


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    classes = list(label_encoder.classes_) if label_encoder else []
    return render_template('index.html', emotions=classes)


@app.route('/predict_file', methods=['POST'])
def predict_file():
    """Handle file upload — supports wav, mp3, ogg, flac, m4a."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file received'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    import pathlib
    ext = pathlib.Path(file.filename).suffix.lower() or '.wav'

    tmp_src = None
    tmp_wav = None
    try:
        # Step 1: save uploaded file with correct extension
        fd, tmp_src = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        file.save(tmp_src)
        print(f"[upload] saved {tmp_src}  size={os.path.getsize(tmp_src)} bytes  ext={ext}")

        # Step 2: convert non-WAV formats using imageio-ffmpeg (bundled, no system install)
        if ext != '.wav':
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                fd2, tmp_wav = tempfile.mkstemp(suffix='.wav')
                os.close(fd2)
                import subprocess
                result = subprocess.run(
                    [ffmpeg_exe, '-y', '-i', tmp_src,
                     '-ar', '22050', '-ac', '1', '-f', 'wav', tmp_wav],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    load_path = tmp_wav
                    print(f"[upload] converted to wav: {tmp_wav}")
                else:
                    print(f"[upload] ffmpeg error: {result.stderr[-300:]}")
                    load_path = tmp_src
            except Exception as conv_err:
                print(f"[upload] conversion failed: {conv_err}")
                load_path = tmp_src
        else:
            load_path = tmp_src

        # Step 3: load with librosa
        audio, sr = librosa.load(load_path, sr=22050, duration=3, mono=True)
        print(f"[upload] audio shape={audio.shape}  sr={sr}")

        if len(audio) < 1000:
            return jsonify({'error': 'Audio too short — upload at least 1 second'}), 422

        result = predict_emotion(audio, int(sr))
        if result:
            print(f"[upload] {result['emotion']} ({result['confidence']}%)")
            return jsonify({'success': True, 'result': result})
        else:
            return jsonify({'error': 'Feature extraction failed'}), 422

    except Exception as e:
        import traceback
        print(f"[upload] ERROR:\n{traceback.format_exc()}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

    finally:
        for p in [tmp_src, tmp_wav]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


# ─────────────────────────────────────────────
# WEBSOCKET - REAL-TIME AUDIO
# ─────────────────────────────────────────────
audio_buffer = {}

@socketio.on('connect')
def handle_connect():
    sid = request.args.get('sid', getattr(request, 'sid', 'unknown'))  # type: ignore[attr-defined]
    print(f"🔌 Client connected: {sid}")
    conn_sid: str = getattr(request, 'sid', 'default')  # type: ignore[attr-defined]
    audio_buffer[conn_sid] = []


@socketio.on('disconnect')
def handle_disconnect():
    conn_sid: str = getattr(request, 'sid', 'default')  # type: ignore[attr-defined]
    print(f"🔌 Client disconnected: {conn_sid}")
    audio_buffer.pop(conn_sid, None)


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Receive audio chunks from browser and accumulate."""
    sid: str = getattr(request, 'sid', 'default')  # type: ignore[attr-defined]
    if sid not in audio_buffer:
        audio_buffer[sid] = []

    # Decode base64 audio chunk
    try:
        audio_bytes = base64.b64decode(data['audio'])
        audio_buffer[sid].append(audio_bytes)

        # Analyze every ~2 seconds of audio (at 44100Hz, 16-bit = 176400 bytes)
        total_bytes = sum(len(c) for c in audio_buffer[sid])
        if total_bytes >= 88200:  # ~1 second at 44100Hz 16-bit mono
            # Combine and convert
            combined = b''.join(audio_buffer[sid])
            audio_buffer[sid] = []  # Reset buffer

            # Convert PCM bytes to numpy array
            audio_int = np.frombuffer(combined, dtype=np.int16)
            audio_float = audio_int.astype(np.float32) / 32768.0

            # Resample from 44100 to 22050
            audio_resampled = librosa.resample(audio_float, orig_sr=44100, target_sr=22050)

            result = predict_emotion(audio_resampled, 22050)
            if result:
                emit('emotion_result', result)

    except Exception as e:
        print(f"Audio chunk error: {e}")


@socketio.on('predict_recorded')
def handle_predict_recorded(data):
    """Handle a complete recorded audio for prediction."""
    try:
        audio_bytes = base64.b64decode(data['audio'])
        sample_rate = data.get('sampleRate', 44100)

        audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int.astype(np.float32) / 32768.0

        if sample_rate != 22050:
            audio_float = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=22050)

        result = predict_emotion(audio_float, 22050)
        if result:
            emit('emotion_result', result)
        else:
            emit('error', {'message': 'Could not analyze audio'})

    except Exception as e:
        emit('error', {'message': str(e)})


# ─────────────────────────────────────────────
# FEEDBACK & ONLINE LEARNING ENDPOINTS
# ─────────────────────────────────────────────

@app.route('/feedback', methods=['POST'])
def receive_feedback():
    """
    Receive user feedback on a prediction.
    Body: { predicted: 'happy', correct: 'sad', confirmed: true/false }
    """
    global is_retraining

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data'}), 400

    correct_label  = data.get('correct')
    confirmed      = data.get('confirmed', False)   # True = prediction was right

    if not correct_label:
        return jsonify({'error': 'Missing correct label'}), 400

    # Validate label
    valid_labels = list(label_encoder.classes_) if label_encoder else []
    if correct_label not in valid_labels:
        return jsonify({'error': f'Invalid label. Valid: {valid_labels}'}), 400

    # Use stored features from last prediction
    if last_prediction_features is None:
        return jsonify({'error': 'No recent prediction found'}), 400

    # Scale features (same as training)
    features_scaled = scaler.transform([last_prediction_features])[0]

    # Save to feedback CSV
    count = save_feedback(features_scaled, correct_label)

    response = {
        'success': True,
        'message': f'Feedback saved! Total: {count} samples',
        'feedback_count': count,
        'retrain_triggered': False,
        'is_retraining': is_retraining
    }

    # Trigger background retraining if enough samples and not already retraining
    if count >= MIN_SAMPLES and count % RETRAIN_EVERY == 0 and not is_retraining:
        print(f"[feedback] Triggering background retrain at {count} samples...")
        t = threading.Thread(target=retrain_model_background, daemon=True)
        t.start()
        response['retrain_triggered'] = True
        response['message'] = f'Feedback saved! Retraining started with {count} samples...'

    return jsonify(response)


@app.route('/feedback/status', methods=['GET'])
def feedback_status():
    """Return current feedback count and retraining status."""
    count = _count_feedback()
    return jsonify({
        'feedback_count': count,
        'is_retraining': is_retraining,
        'retrain_threshold': MIN_SAMPLES,
        'next_retrain_in': max(0, MIN_SAMPLES - count) if count < MIN_SAMPLES else RETRAIN_EVERY - (count % RETRAIN_EVERY),
        'valid_emotions': list(label_encoder.classes_) if label_encoder else []
    })


@app.route('/feedback/trigger_retrain', methods=['POST'])
def trigger_retrain():
    """Manually trigger retraining (for testing)."""
    global is_retraining
    if is_retraining:
        return jsonify({'error': 'Already retraining'}), 400
    count = _count_feedback()
    if count < 10:
        return jsonify({'error': f'Need at least 10 samples, have {count}'}), 400
    t = threading.Thread(target=retrain_model_background, daemon=True)
    t.start()
    return jsonify({'success': True, 'message': f'Retraining started with {count} samples'})


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  SPEECH EMOTION RECOGNITION - WEB APP")
    print("=" * 60)

    if load_model_artifacts():
        print("🚀 Starting server at http://localhost:5000")
        print("   Press Ctrl+C to stop\n")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    else:
        sys.exit(1)