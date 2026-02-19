import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

# train-test time-series split
train = df.loc[:"2019-12-31"]   # creates a new dataframe
test = df.loc["2020-01-01":]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000)
model.fit(X_train_scaled, y_train)

# Probabilities
y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

# ROC AUC
train_auc = roc_auc_score(y_train, y_train_prob)
test_auc = roc_auc_score(y_test, y_test_prob)

print(f"Train ROC AUC: {train_auc:.3f}")
print(f"Test  ROC AUC: {test_auc:.3f}")

# KS Statistic (max separation)
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
ks_stat = np.max(tpr - fpr)

print(f"Test KS Statistic: {ks_stat:.3f}")

# Confusion Matrix (threshold = 0.5)
y_test_pred = (y_test_prob > 0.5).astype(int)
cm = confusion_matrix(y_test, y_test_pred)

print("Confusion Matrix (Test):")
print(cm)

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
}).sort_values("Coefficient", ascending=False)

print("\nLogistic Regression Coefficients:")
print(coef_df)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {test_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Logistic Regression Model")
plt.legend()
plt.tight_layout()
plt.savefig(f"{figure_path}ROC_LogisticRegression.png", dpi=300, bbox_inches="tight")
plt.show()