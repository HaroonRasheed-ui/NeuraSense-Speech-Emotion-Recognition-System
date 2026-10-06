"""
Writes the results from models/results.json into README.md.
Run after: python train_model.py
"""
import json, re

with open("models/results.json") as f:
    r = json.load(f)

si = r["speaker_independent"]
pct = lambda d: f"{d['mean']*100:.1f}% ± {d['std']*100:.1f}"

lines = [
    "<!-- RESULTS START -->",
    f"**Speaker independent evaluation** ({si['n_folds_run']} fold, no speaker appears in both training and test data), mean ± standard deviation across folds:",
    "",
    "| Model | Accuracy | Macro F1 |",
    "| --- | --- | --- |",
    f"| CNN + LSTM | {pct(si['cnn_accuracy'])} | {pct(si['cnn_macro_f1'])} |",
    f"| Ensemble (CNN + SVM + RF) | {pct(si['ensemble_accuracy'])} | {pct(si['ensemble_macro_f1'])} |",
    "",
    "Per class results (ensemble, all test folds pooled):",
    "",
    "| Emotion | Precision | Recall | F1 |",
    "| --- | --- | --- | --- |",
]
rep = si["classification_report"]
for cls, v in rep.items():
    if isinstance(v, dict) and cls not in ("macro avg", "weighted avg"):
        lines.append(f"| {cls} | {v['precision']*100:.1f}% | {v['recall']*100:.1f}% | {v['f1-score']*100:.1f}% |")

if "random_split_old_protocol" in r:
    rs = r["random_split_old_protocol"]
    lines += ["",
        f"**Old random split protocol**, for comparison: {rs['ensemble_accuracy']*100:.1f}% accuracy, "
        f"{rs['ensemble_macro_f1']*100:.1f}% macro F1. The difference between this and the speaker "
        "independent result is how much the leaky protocol inflated performance."]
lines.append("<!-- RESULTS END -->")

with open("README.md", encoding="utf-8") as f:
    readme = f.read()
readme = re.sub(r"<!-- RESULTS START -->.*?<!-- RESULTS END -->", "\n".join(lines), readme, flags=re.S)
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
print("README.md updated with results.")
