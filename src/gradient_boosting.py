import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

data_path = "data/processed/price_features.csv"
figure_path = "data/processed/figures/"

df = pd.read_csv(data_path, parse_dates=["Date"])
df = df.set_index("Date").sort_index()

features = [
    "ret_1d",
    "ret_5d",
    "vol_5d",
    "vol_10d",
    "mom_10d",
    "mom_20d",
    "drawdown_20d"]

target = "Risk_Off"

# train-test time series split
train = df.loc[:"2019-12-31"]  # creates a new dataframe
test = df.loc["2020-01-01":]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


gb_model = GradientBoostingClassifier(
    n_estimators=200,      # number of boosting stages
    learning_rate=0.05,    # small steps for stability
    max_depth=3,           # shallow trees (prevents overfitting)
    subsample=0.8,         # adds randomness like RF
    random_state=42)

gb_model.fit(X_train, y_train)

# Probabilities
y_train_prob = gb_model.predict_proba(X_train)[:, 1]
y_test_prob = gb_model.predict_proba(X_test)[:, 1]

# ROC AUC
train_auc = roc_auc_score(y_train, y_train_prob)
test_auc = roc_auc_score(y_test, y_test_prob)

print(f"Train ROC AUC: {train_auc:.3f}")
print(f"Test  ROC AUC: {test_auc:.3f}")

# KS Statistic
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
ks_stat = np.max(tpr - fpr)

print(f"Test KS Statistic: {ks_stat:.3f}")

# Confusion Matrix (threshold = 0.5)
y_test_pred = (y_test_prob > 0.5).astype(int)
cm = confusion_matrix(y_test, y_test_pred)

print("Confusion Matrix (Test):")
print(cm)

feat_imp = pd.DataFrame({
    "Feature": features,
    "Importance": gb_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nGradient Boosting Feature Importance:")
print(feat_imp)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"GB ROC (AUC = {test_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Gradient Boosting Model")
plt.legend()
plt.tight_layout()
plt.savefig(f"{figure_path}ROC_GradientBoosting.png", dpi=300, bbox_inches="tight")
plt.show()