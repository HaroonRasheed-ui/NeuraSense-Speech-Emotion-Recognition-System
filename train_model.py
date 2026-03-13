"""
Speech Emotion Recognition - FINAL Training Script
===================================================
Key improvements:
  1. FEATURE CACHING  - extracts once, saves to disk, reloads in seconds next run
  2. FIXED ENSEMBLE   - SVM class alignment bug fixed (str conversion)
  3. OVERFITTING FIX  - stronger dropout + L2 reg + lower augmentation ratio
  4. LSTM model       - as required by the internship instructions (CNN + LSTM)
  5. FAST             - cached run takes < 2 min total

Run: python train_model.py
"""

import os, sys, pickle, hashlib, warnings
import numpy as np
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tqdm import tqdm
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from utils.feature_extractor import extract_features  # type: ignore[import]

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Input, LSTM, Bidirectional, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

print("=" * 60)
print("  SPEECH EMOTION RECOGNITION  —  FINAL TRAINING")
print("=" * 60)

# ── CONFIG ────────────────────────────────────────────────
RAVDESS_PATH  = "./dataset/ravdess"
TESS_PATH     = "./dataset/tess"
MODEL_DIR     = "./models"
CACHE_DIR     = "./cache"        # features saved here after first run
SR            = 22050
DURATION      = 3
BATCH_SIZE    = 64
EPOCHS        = 100
AUGMENT       = True

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

RAVDESS_MAP = {
    '01': 'neutral', '02': 'neutral',  '03': 'happy',
    '04': 'sad',     '05': 'angry',    '06': 'fearful',
    '07': 'disgust', '08': 'surprised'
}
TESS_MAP = {
    'angry':'angry','disgust':'disgust','fear':'fearful',
    'happy':'happy','neutral':'neutral','ps':'surprised','sad':'sad'
}


# ══════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING WITH CACHING
# ══════════════════════════════════════════════════════════
def _cache_path(name):
    return os.path.join(CACHE_DIR, f"{name}_features.pkl")


def load_ravdess(path):
    cache = _cache_path("ravdess")
    if os.path.exists(cache):
        print("  ✅ RAVDESS cache found — loading instantly...")
        with open(cache, 'rb') as f:
            return pickle.load(f)

    X, y = [], []
    if not os.path.exists(path):
        print(f"  ⚠  RAVDESS not found at {path}"); return X, y

    files = [
        os.path.join(r, f)
        for r, _, fs in os.walk(path) for f in fs if f.endswith('.wav')
    ]
    print(f"\n  RAVDESS: {len(files)} files  (first-time extraction, will cache)")

    for fp in tqdm(files, desc="  RAVDESS", ncols=80):
        parts = os.path.basename(fp).split('-')
        if len(parts) < 3: continue
        emotion = RAVDESS_MAP.get(parts[2])
        if not emotion: continue

        feats = extract_features(file_path=fp, sr=SR, duration=DURATION)
        if feats is None: continue
        X.append(feats); y.append(emotion)

        if AUGMENT:
            try:
                audio, _ = librosa.load(fp, sr=SR, duration=DURATION, offset=0.5)
                for steps in [2, -2]:
                    aug = librosa.effects.pitch_shift(audio, sr=SR, n_steps=steps)
                    f2  = extract_features(audio=aug, sr=SR, duration=DURATION)
                    if f2 is not None:
                        X.append(f2); y.append(emotion)
            except Exception:
                pass

    with open(cache, 'wb') as f:
        pickle.dump((X, y), f)
    print(f"  ✅ RAVDESS: {len(X)} samples cached → {cache}")
    return X, y


def load_tess(path):
    cache = _cache_path("tess")
    if os.path.exists(cache):
        print("  ✅ TESS cache found — loading instantly...")
        with open(cache, 'rb') as f:
            return pickle.load(f)

    X, y = [], []
    if not os.path.exists(path):
        print(f"  ⚠  TESS not found at {path}"); return X, y

    files = [
        (os.path.join(r, f), f)
        for r, _, fs in os.walk(path) for f in fs if f.endswith('.wav')
    ]
    print(f"\n  TESS: {len(files)} files  (first-time extraction, will cache)")

    for fp, fname in tqdm(files, desc="  TESS", ncols=80):
        parts  = fname.lower().replace('.wav','').split('_')
        emotion = next((TESS_MAP[p] for p in parts if p in TESS_MAP), None)
        if not emotion: continue
        feats = extract_features(file_path=fp, sr=SR, duration=DURATION)
        if feats is None: continue
        X.append(feats); y.append(emotion)

    with open(cache, 'wb') as f:
        pickle.dump((X, y), f)
    print(f"  ✅ TESS: {len(X)} samples cached → {cache}")
    return X, y


# ══════════════════════════════════════════════════════════
# STEP 2 — MODEL: CNN + LSTM (as per internship requirement)
#   Anti-overfitting measures:
#     - L2 regularisation on all Conv/Dense
#     - Spatial dropout after CNN blocks
#     - Dropout 0.4-0.5 in classifier
#     - BatchNorm throughout
#     - EarlyStopping on val_accuracy
# ══════════════════════════════════════════════════════════
def build_cnn_lstm(input_dim, num_classes):
    """
    CNN + LSTM hybrid — required by internship spec.
    CNN extracts local patterns, LSTM captures temporal sequence.
    L2 regularisation + dropout prevents overfitting.
    """
    reg = regularizers.l2(1e-4)
    inp = Input(shape=(input_dim, 1))

    # ── CNN Block 1 ──
    x = Conv1D(64, 5, padding='same', activation='relu',
               kernel_regularizer=reg)(inp)
    x = BatchNormalization()(x)
    x = Conv1D(64, 5, padding='same', activation='relu',
               kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(4)(x)
    x = Dropout(0.3)(x)

    # ── CNN Block 2 ──
    x = Conv1D(128, 3, padding='same', activation='relu',
               kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, padding='same', activation='relu',
               kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(4)(x)
    x = Dropout(0.3)(x)

    # ── LSTM Block ── (temporal modelling)
    x = LSTM(128, return_sequences=True,
             dropout=0.3, recurrent_dropout=0.2,
             kernel_regularizer=reg)(x)
    x = LSTM(64, return_sequences=False,
             dropout=0.3, recurrent_dropout=0.2,
             kernel_regularizer=reg)(x)

    # ── Classifier ──
    x = Dense(128, activation='relu', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ══════════════════════════════════════════════════════════
# STEP 3 — ENSEMBLE  (bug-fixed: str() cast on classes)
# ══════════════════════════════════════════════════════════
def train_ensemble(X_tr, y_tr, X_val, y_val, le):
    print("\n  Training SVM (RBF kernel)...")
    svm = SVC(kernel='rbf', C=10, gamma='scale',
              probability=True, random_state=42)
    svm.fit(X_tr, y_tr)
    svm_acc = svm.score(X_val, y_val)
    print(f"  SVM val accuracy : {svm_acc*100:.1f}%")

    print("\n  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    rf_acc = rf.score(X_val, y_val)
    print(f"  RF  val accuracy : {rf_acc*100:.1f}%")

    return svm, rf


def ensemble_predict(cnn, svm, rf, X_scaled, X_cnn, le):
    """Soft vote: CNN 60% + SVM 25% + RF 15%"""
    cnn_probs = cnn.predict(X_cnn, verbose=0)
    svm_probs = svm.predict_proba(X_scaled)
    rf_probs  = rf.predict_proba(X_scaled)

    # ── FIX: convert all class labels to str for consistent lookup ──
    classes     = [str(c) for c in le.classes_]
    svm_classes = [str(c) for c in svm.classes_]
    rf_classes  = [str(c) for c in rf.classes_]

    def align(probs, src_classes):
        out = np.zeros((probs.shape[0], len(classes)))
        for i, c in enumerate(src_classes):
            c = str(c)
            if c in classes:
                out[:, classes.index(c)] = probs[:, i]
        return out

    svm_probs = align(svm_probs, svm_classes)
    rf_probs  = align(rf_probs,  rf_classes)

    combined = 0.60 * cnn_probs + 0.25 * svm_probs + 0.15 * rf_probs
    return np.argmax(combined, axis=1)


# ══════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════
def save_plots(history, y_val, y_pred, classes):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(history.history['accuracy'],     label='Train', lw=2, color='#3b82f6')
    a1.plot(history.history['val_accuracy'], label='Val',   lw=2, color='#10b981')
    a1.set_title('Accuracy', fontweight='bold'); a1.legend(); a1.grid(alpha=.3)
    a1.set_xlabel('Epoch'); a1.set_ylabel('Accuracy')

    a2.plot(history.history['loss'],     label='Train', lw=2, color='#3b82f6')
    a2.plot(history.history['val_loss'], label='Val',   lw=2, color='#10b981')
    a2.set_title('Loss', fontweight='bold'); a2.legend(); a2.grid(alpha=.3)
    a2.set_xlabel('Epoch'); a2.set_ylabel('Loss')

    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/training_history.png', dpi=150)
    plt.close()

    cm  = confusion_matrix(y_val, y_pred)
    pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (%)', fontweight='bold')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/confusion_matrix.png', dpi=150)
    plt.close()
    print(f"  Plots saved to {MODEL_DIR}/")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    # ── Load (cached after first run) ──
    print("\nSTEP 1: Loading datasets")
    Xr, yr = load_ravdess(RAVDESS_PATH)
    Xt, yt = load_tess(TESS_PATH)

    if not Xr and not Xt:
        print("\nERROR: No data. Run: python download_dataset.py"); return

    X_raw = np.array(Xr + Xt, dtype=np.float32)
    y_raw = np.array(yr + yt)

    print(f"\n  Total: {len(X_raw)} samples | Feature dim: {X_raw.shape[1]}")
    for em, cnt in zip(*np.unique(y_raw, return_counts=True)):
        print(f"    {em:<12}: {cnt}")

    # ── Encode + scale ──
    print("\nSTEP 2: Preprocessing")
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"  Classes ({num_classes}): {list(le.classes_)}")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_scaled, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )
    print(f"  Train: {len(X_tr)} | Val: {len(X_val)}")

    X_tr_cnn  = X_tr.reshape( len(X_tr),  X_tr.shape[1],  1)
    X_val_cnn = X_val.reshape(len(X_val), X_val.shape[1], 1)
    y_tr_cat  = to_categorical(y_tr,  num_classes)
    y_val_cat = to_categorical(y_val, num_classes)

    # ── Build CNN + LSTM ──
    print("\nSTEP 3: Building CNN + LSTM model")
    cnn = build_cnn_lstm(X_tr.shape[1], num_classes)
    cnn.summary()

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', patience=15,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=7, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            f'{MODEL_DIR}/emotion_model.h5',
            monitor='val_accuracy', save_best_only=True, verbose=1
        )
    ]

    # ── Train ──
    print(f"\nSTEP 4: Training  (batch={BATCH_SIZE}, max_epochs={EPOCHS})")
    print("  (dataset loading is now instant from cache)\n")

    history = cnn.fit(
        X_tr_cnn, y_tr_cat,
        validation_data=(X_val_cnn, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    _, cnn_acc = cnn.evaluate(X_val_cnn, y_val_cat, verbose=0)
    print(f"\n  CNN val accuracy: {cnn_acc*100:.2f}%")

    # ── Ensemble ──
    print("\nSTEP 5: Training ensemble")
    svm, rf = train_ensemble(X_tr, y_tr, X_val, y_val, le)

    y_pred   = ensemble_predict(cnn, svm, rf, X_val, X_val_cnn, le)
    ens_acc  = (y_pred == y_val).mean()

    print(f"\n  CNN-only  accuracy : {cnn_acc*100:.2f}%")
    print(f"  ENSEMBLE  accuracy : {ens_acc*100:.2f}%")
    print("\n  Per-class report (Ensemble):")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    # ── Save ──
    print("\nSTEP 6: Saving artifacts")
    cnn.save(f'{MODEL_DIR}/emotion_model.h5')

    for fname, obj in [
        ('scaler.pkl',        scaler),
        ('label_encoder.pkl', le),
        ('svm_model.pkl',     svm),
        ('rf_model.pkl',      rf),
        ('feature_dim.pkl',   int(X_scaled.shape[1])),
    ]:
        with open(f'{MODEL_DIR}/{fname}', 'wb') as f:
            pickle.dump(obj, f)

    print("  ✅ All artifacts saved to ./models/")

    # ── Plots ──
    print("\nSTEP 7: Saving plots")
    save_plots(history, y_val, y_pred, le.classes_)

    print("\n" + "=" * 60)
    print(f"  ✅  TRAINING COMPLETE")
    print(f"  CNN   accuracy : {cnn_acc*100:.2f}%")
    print(f"  Ensemble acc   : {ens_acc*100:.2f}%")
    print("=" * 60)
    print("\n  Next step: python app.py\n")


if __name__ == '__main__':
    main()