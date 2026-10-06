# NeuraSense: Speech Emotion Recognition

NeuraSense classifies emotion from speech into seven classes (angry, disgust, fearful, happy, neutral, sad, surprised) and runs in real time through a small Flask web app. It started as an internship project and I later reworked the evaluation, which is the part of this repo I care most about.

## Results

<!-- RESULTS START -->
Results not generated yet. Run `python train_model.py`, then `python fill_readme.py`.
<!-- RESULTS END -->

## Why the evaluation changed

The first version of this project used a random split over all samples. That looked fine and reported high accuracy, but it had four leaks:

1. The same speakers appeared in both training and test sets, so the model could partly learn voices instead of emotions.
2. Pitch shifted copies of a recording were created before the split, so a test sample often had a near copy in the training data.
3. The test set was also used for early stopping, so it influenced which model was kept.
4. The Kaggle copies of the datasets contain duplicated folders, so some identical files could end up on both sides.

There was also a bug in the ensemble: class labels were compared as strings against integers, so the SVM and Random Forest probabilities were always zero and the "ensemble" was really the CNN alone.

The current `train_model.py` fixes all of these. Each sample is tagged with its speaker, duplicates are removed by file name, folds are built with `StratifiedGroupKFold` on speakers, augmentation is only applied to training data, early stopping uses separate held out speakers, and the scaler is fit on training data only. The script still runs the old protocol once so the gap between the two numbers is visible.

## Data

| Dataset | Speakers | Notes |
| --- | --- | --- |
| [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) | 24 actors | "calm" is merged into "neutral" |
| [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) | 2 speakers | "pleasant surprise" mapped to "surprised" |

RAVDESS training recordings are augmented with pitch shifts of +2 and -2 semitones. Validation and test sets only contain original recordings.

## Model

Each clip is trimmed or padded to 3 seconds and turned into a feature vector: 40 MFCCs (mean and standard deviation), MFCC deltas and delta deltas, chroma, mel spectrogram, spectral descriptors, zero crossing rate, RMS energy and tonnetz.

The main model is a 1D CNN (two convolutional blocks with batch norm and dropout) followed by two stacked LSTM layers and a dense classifier, with L2 regularisation throughout. Its output is combined with an RBF SVM and a Random Forest by weighted soft voting (0.60, 0.25, 0.15).

## How to reproduce

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Kaggle API key at ~/.kaggle/kaggle.json, then:
python download_dataset.py

python train_model.py             # speaker independent CV + old protocol comparison
python fill_readme.py             # writes the results table into this README
```

Set `RUN_ALL_FOLDS = False` in `train_model.py` for a quick single fold run. Seeds are fixed, but TensorFlow on GPU is not fully deterministic, so expect small differences between runs.

Run the web app or a single prediction:

```bash
python app.py                     # http://localhost:5000
python predict.py path/to/audio.wav
```

## Limitations

- **Acted speech.** Both datasets are actors reading fixed sentences with exaggerated emotion. Performance on natural, spontaneous speech will be lower.
- **Few speakers.** 26 speakers in total, and TESS contributes only two. Two of the five folds test on a single TESS speaker, so per fold results vary a lot. The standard deviation in the table matters as much as the mean.
- **English only**, and no control for recording conditions or microphone.
- **Clip level features.** Features are averaged over the whole clip, so the LSTM sees a sequence of pooled feature positions, not frames over time.
- The deployed model in `models/` is the one trained in fold 1, not a model trained on all speakers.

## Project structure

```
app.py                    Flask + SocketIO web app (record, upload, real time)
train_model.py            Training and speaker independent evaluation
predict.py                Command line prediction
fill_readme.py            Writes results from models/results.json into the README
download_dataset.py       Kaggle dataset downloader
utils/feature_extractor.py
templates/index.html
models/                   Created after training (model, scaler, results.json, plots)
```
