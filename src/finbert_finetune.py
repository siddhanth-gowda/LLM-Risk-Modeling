import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


TRAIN_DATA = "data/processed/finetune_train.csv"
POLICY_TEXT_DIR = "data/text/rbi_policy/"
MODEL_DIR = "models/finbert_results/"
SCORES_OUTPUT = "data/processed/finbert_scores.csv"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("\nLoading training data...\n")
df = pd.read_csv(TRAIN_DATA)

label2id = {"hawkish": 0, "dovish": 1, "neutral": 2}
id2label = {v: k for k, v in label2id.items()}

df["y"] = df["label"].map(label2id)  # assigns numerical values to the labels in the dataframe

print("Building improved TF–IDF features...\n")

vectorizer = TfidfVectorizer(
    max_features=8000,     # more expressive than before
    ngram_range=(1, 3),    # uni + bi + trigrams
    min_df=3,              # remove very rare terms
    max_df=0.85,           # remove overly common terms
    stop_words="english")

X = vectorizer.fit_transform(df["text"])
y = df["y"].values

# Train-test split randomly as time-based chronology not important
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Training XGBoost classifier...\n")

clf = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob")

clf.fit(X_train, y_train)

# Evaluate
preds = clf.predict(X_test)
print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        preds,
        target_names=["hawkish", "dovish", "neutral"]))

# Save model + vectorizer
joblib.dump(clf, MODEL_DIR + "tfidf_xgb_classifier.pkl")
joblib.dump(vectorizer, MODEL_DIR + "tfidf_vectorizer.pkl")
print(f"\nModels saved in: {MODEL_DIR}")

print("\nGenerating document-level scores...\n")

rows = []

for txt_file in Path(POLICY_TEXT_DIR).glob("*.txt"):
    date = txt_file.stem.replace("_policy", "")
    content = txt_file.read_text(encoding="utf-8", errors="ignore")

    # Split into sentences (basic but workable)
    sentences = [s.strip() for s in content.split(". ") if len(s.split()) >= 10]

    if not sentences:
        continue

    # Convert sentences to TF–IDF
    X_doc = vectorizer.transform(sentences)

    # Get class probabilities per sentence
    probs = clf.predict_proba(X_doc)

    # Convert probabilities → hard predictions
    pred_labels = np.argmax(probs, axis=1)  # 0=h, 1=d, 2=n

    # Compute SHARE of each tone in the document
    hawk_share = np.mean(pred_labels == 0)
    dov_share = np.mean(pred_labels == 1)
    neu_share = np.mean(pred_labels == 2)

    rows.append({"Date": date,
        "hawkish_share": float(hawk_share),
        "dovish_share": float(dov_share),
        "neutral_share": float(neu_share),})

score_df = pd.DataFrame(rows).sort_values("Date")
# macro tone signal
score_df["macro_tone_signal"] = (
    score_df["hawkish_share"] - score_df["dovish_share"])

# Signal smoothing (ema)
score_df["macro_tone_signal_ema"] = (
    score_df["macro_tone_signal"].ewm(span=3).mean())
score_df.to_csv(SCORES_OUTPUT, index=False)

print(f"Saved document-level scores to: {SCORES_OUTPUT}")