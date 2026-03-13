# 🧠 NeuraSense — Speech Emotion Recognition System

A deep learning system that recognizes human emotions from speech in **real-time**.

## 🎯 Architecture
- **Model**: CNN + Bidirectional LSTM + Multi-Head Attention
- **Features**: MFCC (40) + Delta + Delta-Delta + Chroma + Mel + Spectral + Tonnetz
- **Target Accuracy**: 85–90%+
- **Datasets**: RAVDESS + TESS (combined ~4,000+ samples with augmentation)
- **Emotions**: Happy, Sad, Angry, Fearful, Disgust, Surprised, Neutral

---

## ⚙️ Setup — Step by Step

### Step 1: Install Python Requirements
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Windows users**: If PyAudio fails, install it with:
> ```
> pip install pipwin
> pipwin install pyaudio
> ```

---

### Step 2: Get Kaggle API Key
1. Go to [kaggle.com](https://www.kaggle.com) → Profile → Settings
2. Scroll to **API** → Click **Create New Token**
3. This downloads `kaggle.json`
4. Place it at `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\YourName\.kaggle\kaggle.json`
   - Mac/Linux: `~/.kaggle/kaggle.json`

---

### Step 3: Download Datasets
```bash
python download_dataset.py
```
This downloads RAVDESS (~500MB) and TESS (~200MB) automatically.

**OR** download manually from Kaggle:
- [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) → Extract to `./dataset/ravdess/`
- [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) → Extract to `./dataset/tess/`

---

### Step 4: Train the Model
```bash
python train_model.py
```

**Training time**: ~30–90 minutes (depends on your GPU/CPU)

After training you'll see:
- `models/emotion_model.h5` — trained model
- `models/scaler.pkl` — feature scaler
- `models/label_encoder.pkl` — class labels
- `models/training_history.png` — accuracy/loss curves
- `models/confusion_matrix.png` — per-class accuracy

---

### Step 5: Run the Web App
```bash
python app.py
```

Open your browser: **http://localhost:5000**

---

## 🖥️ Features

| Feature | Description |
|---------|-------------|
| ⚡ Real-time | Analyze emotion as you speak, live |
| 🎙️ Record | Record audio, then analyze |
| 📁 Upload | Upload any audio file (.wav, .mp3, .ogg, etc.) |
| 📊 Chart | See probability distribution for all emotions |
| 🕒 History | See last 20 predictions with timestamps |

---

## 🧪 Quick Test (without web app)
```bash
python predict.py path/to/audio.wav
```

---

## 📁 Project Structure
```
emotion_recognition/
├── app.py                  # Flask + SocketIO backend
├── train_model.py          # Model training script
├── predict.py              # CLI prediction tool
├── download_dataset.py     # Dataset downloader
├── requirements.txt        # Python dependencies
├── utils/
│   └── feature_extractor.py # Audio feature extraction
├── templates/
│   └── index.html          # Web UI
├── models/                 # Saved after training
│   ├── emotion_model.h5
│   ├── scaler.pkl
│   └── label_encoder.pkl
└── dataset/                # Created after download
    ├── ravdess/
    └── tess/
```

---

## 📈 Accuracy Tips

To maximize accuracy:
1. **Use both RAVDESS + TESS** — more data = better model
2. **Keep augmentation ON** — `USE_AUGMENTATION = True` in `train_model.py`
3. **Let training complete** — early stopping handles it
4. **GPU recommended** — install `tensorflow-gpu` for faster training

Expected results:
- RAVDESS only: ~75–82%
- RAVDESS + TESS: **~85–92%**

---

## 🛠️ VS Code Tips
- Install the **Python** extension
- Set your interpreter to the virtual environment
- Use the integrated terminal to run commands
- Install **Pylance** for IntelliSense

---

## ⚠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: librosa` | Run `pip install -r requirements.txt` |
| `PyAudio not found` | Windows: `pipwin install pyaudio` |
| `Model not found` | Run `python train_model.py` first |
| `Kaggle download fails` | Place `kaggle.json` at `~/.kaggle/kaggle.json` |
| `Microphone denied` | Allow mic in browser → Settings → Privacy |