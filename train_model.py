"""
NeuraSense: Speech Emotion Recognition training script
=======================================================

Evaluation protocol (speaker independent):
  1. Every sample is tagged with its speaker (RAVDESS actor, TESS speaker).
  2. Duplicate files in the Kaggle copies are removed by file name.
  3. Speakers are split into folds with StratifiedGroupKFold, so no speaker
     ever appears in both training and test data.
  4. Augmented (pitch shifted) copies are only used for training. Test and
     validation sets contain original recordings only.
  5. Early stopping uses a validation set of separate held out speakers,
     never the test set.
  6. The feature scaler is fit on training data only, in every fold.

Optionally the old random split protocol is also run, so the gap between
the two numbers can be reported.

Run: python train_model.py
Results: models/results.json, models/confusion_matrix.png
"""

import os, sys, json, pickle, random, warnings
import numpy as np
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tqdm import tqdm
import librosa
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from utils.feature_extractor import extract_features  # type: ignore[import]

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Input, LSTM
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

# ── CONFIG ────────────────────────────────────────────────
RAVDESS_PATH = "./dataset/ravdess"
TESS_PATH    = "./dataset/tess"
MODEL_DIR    = "./models"
CACHE_DIR    = "./cache"
SR           = 22050
DURATION     = 3
BATCH_SIZE   = 64
EPOCHS       = 100
AUGMENT      = True
SEED         = 42

N_FOLDS               = 5      # speaker grouped folds
RUN_ALL_FOLDS         = True   # False = only fold 1 (faster)
VAL_SPEAKER_FRACTION  = 0.15   # share of training speakers held out for early stopping
COMPARE_RANDOM_SPLIT  = True   # also run the old random split protocol
VERBOSE               = 0      # Keras training output (0, 1 or 2)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

RAVDESS_MAP = {
    '01': 'neutral', '02': 'neutral', '03': 'happy',
    '04': 'sad',     '05': 'angry',   '06': 'fearful',
    '07': 'disgust', '08': 'surprised'
}
TESS_MAP = {
    'angry': 'angry', 'disgust': 'disgust', 'fear': 'fearful',
    'happy': 'happy', 'neutral': 'neutral', 'ps': 'surprised', 'sad': 'sad'
}


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ══════════════════════════════════════════════════════════
# STEP 1: DATA LOADING (speaker tagged, deduplicated, cached)
# ══════════════════════════════════════════════════════════
def _cache_path(name):
    # v2 cache stores speaker and augmentation info; old caches are ignored
    return os.path.join(CACHE_DIR, f"{name}_v2_records.pkl")


def _wav_files_dedup(path):
    """Walk a folder and keep one copy of each file name."""
    seen, files, dupes = set(), [], 0
    for r, _, fs in os.walk(path):
        for f in sorted(fs):
            if not f.lower().endswith('.wav'):
                continue
            if f in seen:
                dupes += 1
                continue
            seen.add(f)
            files.append(os.path.join(r, f))
    return files, dupes


def load_ravdess(path):
    cache = _cache_path("ravdess")
    if os.path.exists(cache):
        print("  RAVDESS cache found, loading...")
        with open(cache, 'rb') as f:
            return pickle.load(f)

    if not os.path.exists(path):
        print(f"  RAVDESS not found at {path}")
        return []

    files, dupes = _wav_files_dedup(path)
    print(f"\n  RAVDESS: {len(files)} unique files ({dupes} duplicates skipped)")

    records = []
    for fp in tqdm(files, desc="  RAVDESS", ncols=80):
        parts = os.path.basename(fp).replace('.wav', '').split('-')
        if len(parts) < 7:
            continue
        emotion = RAVDESS_MAP.get(parts[2])
        if not emotion:
            continue
        speaker = f"ravdess_actor_{parts[6]}"

        feats = extract_features(file_path=fp, sr=SR, duration=DURATION)
        if feats is None:
            continue
        records.append(dict(x=feats, y=emotion, speaker=speaker, aug=False))

        if AUGMENT:
            try:
                audio, _ = librosa.load(fp, sr=SR, duration=DURATION, offset=0.5)
                for steps in [2, -2]:
                    aug = librosa.effects.pitch_shift(audio, sr=SR, n_steps=steps)
                    f2 = extract_features(audio=aug, sr=SR, duration=DURATION)
                    if f2 is not None:
                        records.append(dict(x=f2, y=emotion, speaker=speaker, aug=True))
            except Exception:
                pass

    with open(cache, 'wb') as f:
        pickle.dump(records, f)
    print(f"  RAVDESS: {len(records)} samples cached")
    return records


def _tess_speaker(fp):
    prefix = os.path.basename(fp).split('_')[0].upper()
    if prefix in ('OAF', 'YAF'):
        return f"tess_{prefix}"
    folder = os.path.basename(os.path.dirname(fp)).split('_')[0].upper()
    return f"tess_{folder}" if folder in ('OAF', 'YAF') else None


def load_tess(path):
    cache = _cache_path("tess")
    if os.path.exists(cache):
        print("  TESS cache found, loading...")
        with open(cache, 'rb') as f:
            return pickle.load(f)

    if not os.path.exists(path):
        print(f"  TESS not found at {path}")
        return []

    files, dupes = _wav_files_dedup(path)
    print(f"\n  TESS: {len(files)} unique files ({dupes} duplicates skipped)")

    records = []
    for fp in tqdm(files, desc="  TESS", ncols=80):
        parts = os.path.basename(fp).lower().replace('.wav', '').split('_')
        emotion = next((TESS_MAP[p] for p in parts if p in TESS_MAP), None)
        speaker = _tess_speaker(fp)
        if not emotion or not speaker:
            continue
        feats = extract_features(file_path=fp, sr=SR, duration=DURATION)
        if feats is None:
            continue
        records.append(dict(x=feats, y=emotion, speaker=speaker, aug=False))

    with open(cache, 'wb') as f:
        pickle.dump(records, f)
    print(f"  TESS: {len(records)} samples cached")
    return records


# ══════════════════════════════════════════════════════════
# STEP 2: MODEL (CNN + stacked LSTM)
# ══════════════════════════════════════════════════════════
def build_cnn_lstm(input_dim, num_classes):
    reg = regularizers.l2(1e-4)
    inp = Input(shape=(input_dim, 1))

    x = Conv1D(64, 5, padding='same', activation='relu', kernel_regularizer=reg)(inp)
    x = BatchNormalization()(x)
    x = Conv1D(64, 5, padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(4)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(4)(x)
    x = Dropout(0.3)(x)

    x = LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2,
             kernel_regularizer=reg)(x)
    x = LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2,
             kernel_regularizer=reg)(x)

    x = Dense(128, activation='relu', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inp, out)
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ══════════════════════════════════════════════════════════
# STEP 3: ENSEMBLE
# Class columns are aligned by integer label. (The previous version
# compared string class names with integer labels, so SVM and RF
# probabilities were always zero and the "ensemble" was the CNN alone.)
# ══════════════════════════════════════════════════════════
def train_classical(X_fit, y_fit):
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=SEED)
    svm.fit(X_fit, y_fit)
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=SEED)
    rf.fit(X_fit, y_fit)
    return svm, rf


def ensemble_predict(cnn, svm, rf, X_flat, n_classes):
    cnn_p = cnn.predict(X_flat[..., None], verbose=0)

    def aligned(model):
        p = model.predict_proba(X_flat)
        out = np.zeros((len(X_flat), n_classes))
        for i, c in enumerate(model.classes_):
            out[:, int(c)] = p[:, i]
        return out

    combined = 0.60 * cnn_p + 0.25 * aligned(svm) + 0.15 * aligned(rf)
    return np.argmax(combined, axis=1), np.argmax(cnn_p, axis=1)


# ══════════════════════════════════════════════════════════
# STEP 4: TRAIN AND EVALUATE ONE SPLIT
# ══════════════════════════════════════════════════════════
def train_and_evaluate(X, y_enc, fit_idx, val_idx, test_idx, n_classes, tag):
    set_seeds()
    scaler = StandardScaler().fit(X[fit_idx])
    Xf, Xv, Xt = (scaler.transform(X[i]) for i in (fit_idx, val_idx, test_idx))
    yf, yv, yt = y_enc[fit_idx], y_enc[val_idx], y_enc[test_idx]

    cnn = build_cnn_lstm(Xf.shape[1], n_classes)
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
    ]
    history = cnn.fit(
        Xf[..., None], to_categorical(yf, n_classes),
        validation_data=(Xv[..., None], to_categorical(yv, n_classes)),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=VERBOSE
    )

    svm, rf = train_classical(Xf, yf)
    ens_pred, cnn_pred = ensemble_predict(cnn, svm, rf, Xt, n_classes)

    metrics = dict(
        protocol=tag,
        n_train=int(len(fit_idx)), n_val=int(len(val_idx)), n_test=int(len(test_idx)),
        cnn_accuracy=float(accuracy_score(yt, cnn_pred)),
        cnn_macro_f1=float(f1_score(yt, cnn_pred, average='macro')),
        ensemble_accuracy=float(accuracy_score(yt, ens_pred)),
        ensemble_macro_f1=float(f1_score(yt, ens_pred, average='macro')),
    )
    artifacts = dict(cnn=cnn, svm=svm, rf=rf, scaler=scaler, history=history)
    return metrics, artifacts, yt, ens_pred


# ══════════════════════════════════════════════════════════
# STEP 5: PROTOCOLS
# ══════════════════════════════════════════════════════════
def speaker_independent_cv(X, y_enc, groups, is_aug, n_classes):
    real_idx = np.where(~is_aug)[0]
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_speakers = sorted(set(groups))

    fold_metrics, y_true_all, y_pred_all, first_artifacts = [], [], [], None
    for k, (_, te_r) in enumerate(sgkf.split(X[real_idx], y_enc[real_idx], groups[real_idx])):
        test_spk = set(groups[real_idx][te_r])
        train_spk = [s for s in all_speakers if s not in test_spk]

        rng = np.random.RandomState(SEED + k)
        n_val = max(1, int(round(len(train_spk) * VAL_SPEAKER_FRACTION)))
        val_spk = set(rng.choice(train_spk, n_val, replace=False))
        fit_spk = set(train_spk) - val_spk

        test_idx = np.where(np.isin(groups, list(test_spk)) & ~is_aug)[0]
        val_idx  = np.where(np.isin(groups, list(val_spk)) & ~is_aug)[0]
        fit_idx  = np.where(np.isin(groups, list(fit_spk)))[0]  # includes augmented copies

        # Hard check: no speaker overlap between any two sets
        assert not (set(groups[fit_idx]) & test_spk)
        assert not (set(groups[val_idx]) & test_spk)
        assert not (set(groups[fit_idx]) & val_spk)

        print(f"\n  Fold {k + 1}/{N_FOLDS}: {len(fit_spk)} train, {len(val_spk)} val, "
              f"{len(test_spk)} test speakers")
        m, art, yt, yp = train_and_evaluate(X, y_enc, fit_idx, val_idx, test_idx,
                                            n_classes, f"speaker_independent_fold_{k + 1}")
        m['test_speakers'] = sorted(test_spk)
        print(f"    CNN       acc {m['cnn_accuracy']*100:.2f}%  macro F1 {m['cnn_macro_f1']*100:.2f}%")
        print(f"    Ensemble  acc {m['ensemble_accuracy']*100:.2f}%  macro F1 {m['ensemble_macro_f1']*100:.2f}%")

        fold_metrics.append(m)
        y_true_all.append(yt)
        y_pred_all.append(yp)
        if first_artifacts is None:
            first_artifacts = art
        if not RUN_ALL_FOLDS:
            break

    return fold_metrics, np.concatenate(y_true_all), np.concatenate(y_pred_all), first_artifacts


def random_split_baseline(X, y_enc, n_classes):
    """The old protocol: random split over all samples, augmented copies of the
    same recording can land on both sides, and the test set is also used for
    early stopping. Kept only to measure how much it inflates results."""
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=y_enc)
    m, _, _, _ = train_and_evaluate(X, y_enc, tr, te, te, n_classes, "random_split_old_protocol")
    return m


def summarise(fold_metrics, key):
    vals = np.array([m[key] for m in fold_metrics])
    return dict(mean=float(vals.mean()), std=float(vals.std()))


def save_plots(history, y_true, y_pred, classes):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in [(a1, 'accuracy', 'Accuracy'), (a2, 'loss', 'Loss')]:
        ax.plot(history.history[key], label='Train', lw=2, color='#3b82f6')
        ax.plot(history.history[f'val_{key}'], label='Val (held out speakers)', lw=2, color='#10b981')
        ax.set_title(f'{title} (fold 1)', fontweight='bold'); ax.legend(); ax.grid(alpha=.3)
        ax.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/training_history.png', dpi=150)
    plt.close()

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    pct = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(pct, annot=True, fmt='.1f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (%), speaker independent, all test folds', fontweight='bold')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/confusion_matrix.png', dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    set_seeds()
    print("=" * 60)
    print("  NEURASENSE TRAINING (speaker independent evaluation)")
    print("=" * 60)

    print("\nSTEP 1: Loading datasets")
    records = load_ravdess(RAVDESS_PATH) + load_tess(TESS_PATH)
    if not records:
        print("\nERROR: No data. Run: python download_dataset.py")
        return

    X = np.array([r['x'] for r in records], dtype=np.float32)
    y_raw = np.array([r['y'] for r in records])
    groups = np.array([r['speaker'] for r in records])
    is_aug = np.array([r['aug'] for r in records], dtype=bool)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    print(f"\n  {len(X)} samples ({(~is_aug).sum()} original, {is_aug.sum()} augmented)")
    print(f"  {len(set(groups))} speakers | {n_classes} classes: {list(le.classes_)}")

    print("\nSTEP 2: Speaker independent cross validation")
    folds, y_true, y_pred, art = speaker_independent_cv(X, y_enc, groups, is_aug, n_classes)

    results = dict(
        speaker_independent=dict(
            n_folds_run=len(folds),
            cnn_accuracy=summarise(folds, 'cnn_accuracy'),
            cnn_macro_f1=summarise(folds, 'cnn_macro_f1'),
            ensemble_accuracy=summarise(folds, 'ensemble_accuracy'),
            ensemble_macro_f1=summarise(folds, 'ensemble_macro_f1'),
            folds=folds,
            classification_report=classification_report(
                y_true, y_pred, target_names=le.classes_, output_dict=True),
        )
    )

    print("\n  Per class report (ensemble, all test folds pooled):")
    print(classification_report(y_true, y_pred, target_names=le.classes_, digits=3))

    if COMPARE_RANDOM_SPLIT:
        print("\nSTEP 3: Old random split protocol (for comparison only)")
        results['random_split_old_protocol'] = random_split_baseline(X, y_enc, n_classes)

    print("\nSTEP 4: Saving artifacts (model from fold 1)")
    art['cnn'].save(f'{MODEL_DIR}/emotion_model.h5')
    for fname, obj in [
        ('scaler.pkl', art['scaler']), ('label_encoder.pkl', le),
        ('svm_model.pkl', art['svm']), ('rf_model.pkl', art['rf']),
        ('feature_dim.pkl', int(X.shape[1])),
    ]:
        with open(f'{MODEL_DIR}/{fname}', 'wb') as f:
            pickle.dump(obj, f)
    with open(f'{MODEL_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    save_plots(art['history'], y_true, y_pred, le.classes_)

    si = results['speaker_independent']
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Speaker independent ({si['n_folds_run']} fold(s)), mean +/- std:")
    for k in ['cnn_accuracy', 'cnn_macro_f1', 'ensemble_accuracy', 'ensemble_macro_f1']:
        print(f"    {k:<20} {si[k]['mean']*100:6.2f}% +/- {si[k]['std']*100:.2f}")
    if COMPARE_RANDOM_SPLIT:
        rs = results['random_split_old_protocol']
        print("  Old random split protocol:")
        print(f"    ensemble_accuracy    {rs['ensemble_accuracy']*100:6.2f}%")
        print(f"    ensemble_macro_f1    {rs['ensemble_macro_f1']*100:6.2f}%")
    print("\n  Full results saved to models/results.json")


if __name__ == '__main__':
    main()
