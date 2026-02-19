import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

plt.style.use("seaborn-v0_8")

DATA_PATH = "data/processed/final_dataset.csv"
FIGURE_PATH = "data/processed/figures/"

os.makedirs(FIGURE_PATH, exist_ok=True)

print("\nLoading final dataset (price + NLP features)...\n")

df = pd.read_csv(DATA_PATH)

'''
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
'''

PRICE_FEATURES = [
    "ret_1d",
    "ret_5d",
    "vol_5d",
    "vol_10d",
    "mom_10d",
    "mom_20d",
    "drawdown_20d"]

NLP_FEATURES = ["macro_tone_signal_ema"]

FEATURES = PRICE_FEATURES + NLP_FEATURES
TARGET = "Risk_Off"

print("Features used:")
print(FEATURES)

df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]

TRAIN_END_DATE = "2022-01-01"

df = df.set_index("Date")

train = df.loc[:TRAIN_END_DATE]   # time-based train-test split
test  = df.loc[TRAIN_END_DATE:]

X_train = train[FEATURES]
y_train = train[TARGET]

X_test  = test[FEATURES]
y_test  = test[TARGET]

print("\nTime split:")
print("Train:", train.index.min(), "→", train.index.max())
print("Test: ", test.index.min(), "→", test.index.max())
print("Train size:", len(train))
print("Test size:", len(test))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        random_state=42),

    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42),

    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42)}

trained_gb_model = None

print("\nTraining models and generating ROC curves...\n")

for name, model in models.items():

    print(f"Training {name.replace('_', ' ').title()}...")

    # Logistic regression uses scaled features
    if name == "logistic_regression":
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:, 1]

    else:
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        if name == "gradient_boosting":
            trained_gb_model = model

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f}")

    plt.figure(figsize=(6, 5))
    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{name.replace('_', ' ').title()} ROC Curve\n(Price + NLP Features)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    save_path = f"{FIGURE_PATH}{name}_roc_price_nlp.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

importance = trained_gb_model.feature_importances_

feat_imp = pd.DataFrame({
    "feature": FEATURES,
    "importance": importance
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feat_imp,
    x="importance",
    y="feature")

plt.title("Feature Importance — Gradient Boosting (Price + NLP)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

importance_path = f"{FIGURE_PATH}gradient_boosting_feature_importance.png"

plt.savefig(importance_path, dpi=300, bbox_inches="tight")
plt.close()