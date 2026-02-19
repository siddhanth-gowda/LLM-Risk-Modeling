import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")


raw_path = "data/raw/"
processed_path = "data/processed/"
figure_path = f"{processed_path}figures/"

os.makedirs(processed_path, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)

nifty_data = f"{raw_path}nifty50.csv"
vix_data = f"{raw_path}indiavix.csv"

forward_window = 5
risk_percentile = 0.75

# Historical calibration window (to avoid leakage)
calibration_start = "2014-01-01"
calibration_end = "2019-12-31"


nifty = pd.read_csv(nifty_data) # Reading Nifty data

date_col = [c for c in nifty.columns if "Date" in c][0]
close_col = [c for c in nifty.columns if "Close" in c][0]

nifty = nifty[[date_col, close_col]].copy()
nifty.columns = ["Date", "NIFTY_Close"]
nifty["Date"] = pd.to_datetime(nifty["Date"], format="mixed", dayfirst=True)
nifty = nifty.set_index("Date").sort_index().dropna()


vix = pd.read_csv(vix_data) # Reading VIX data

date_col = [c for c in vix.columns if "Date" in c][0]
close_col = [c for c in vix.columns if "Close" in c][0]

vix = vix[[date_col, close_col]].copy()
vix.columns = ["Date", "IndiaVIX"]
vix["Date"] = pd.to_datetime(vix["Date"], format="mixed", dayfirst=True)
vix = vix.set_index("Date").sort_index().dropna()


# Aligning NIFTY and India VIX data
df = nifty.join(vix, how="inner").dropna()  # inner join returns new df containing rows with matching index
                                            # this aligns the dates

# Constructing risk-off target = mean of NEXT 5 days of IndiaVix
df["VIX_Fwd_Avg_5D"] = (df["IndiaVIX"].shift(-1).rolling(forward_window).mean())

historical_vix = df.loc[calibration_start:calibration_end,"IndiaVIX"]
vix_threshold = historical_vix.quantile(risk_percentile) 
# quantile() finds the threshold below which 75% of the values lie

df["Risk_Off"] = (df["VIX_Fwd_Avg_5D"] > vix_threshold).astype(int)
# astype(int) converts True and False to 1 and 0
df = df.dropna()

print(f"India VIX {int(risk_percentile * 100)}th percentile threshold: {vix_threshold:.2f}")
print(f"Risk-Off frequency: {df['Risk_Off'].mean() * 100:.2f}%")

df_feat = df.copy()

# creating price features (ML inputs)
# Returns
df_feat["ret_1d"] = df_feat["NIFTY_Close"].pct_change()  # % change between current and prev value
df_feat["ret_5d"] = df_feat["NIFTY_Close"].pct_change(5)

# Volatility
df_feat["vol_5d"] = df_feat["ret_1d"].rolling(5).std()   # volatility of the RETURNS, not Nifty_Close
df_feat["vol_10d"] = df_feat["ret_1d"].rolling(10).std()

# Momentum
df_feat["mom_10d"] = df_feat["NIFTY_Close"] / df_feat["NIFTY_Close"].shift(10) - 1
df_feat["mom_20d"] = df_feat["NIFTY_Close"] / df_feat["NIFTY_Close"].shift(20) - 1

# Drawdown
rolling_max = df_feat["NIFTY_Close"].rolling(20).max()
df_feat["drawdown_20d"] = df_feat["NIFTY_Close"] / rolling_max - 1

price_features = df_feat[[
    "ret_1d",
    "ret_5d",
    "vol_5d",
    "vol_10d",
    "mom_10d",
    "mom_20d",
    "drawdown_20d",
    "Risk_Off"
]].copy()

price_features = price_features.reset_index()

price_features = price_features.dropna()

price_features.to_csv("data/processed/price_features.csv", index=False)
# this file will be used as input for ML models
print("Saved price_features.csv")


# Plot 1: India VIX with Risk-Off regimes 
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["IndiaVIX"], label="India VIX", linewidth=1)

risk_dates = df[df["Risk_Off"] == 1].index

plt.scatter(risk_dates,df.loc[risk_dates, "IndiaVIX"], color="red", s=10, label="Risk-Off")

plt.title("India VIX with Risk-Off Regimes")
plt.legend()
plt.tight_layout()

plt.savefig(f"{figure_path}indiavix_risk_regimes.png", dpi=300, bbox_inches="tight")
plt.show()


# Plot 2: NIFTY Returns during Risk-Off
df["Returns"] = df["NIFTY_Close"].pct_change()
df = df.dropna()

plt.figure(figsize=(12, 5))
plt.plot(df.index, df["Returns"], label="Daily NIFTY Returns", alpha=0.7)

plt.scatter(risk_dates,df.loc[df["Risk_Off"] == 1, "Returns"], color="red", s=10, label="Risk-Off")

plt.title("NIFTY Returns During Risk-Off Periods")
plt.legend()
plt.tight_layout()

plt.savefig(f"{figure_path}nifty_returns_risk_periods.png", dpi=300, bbox_inches="tight")
plt.show()


FINAL_COLS = ["NIFTY_Close", "IndiaVIX", "VIX_Fwd_Avg_5D", "Risk_Off"]

df[FINAL_COLS].to_csv(f"{processed_path}market_data.csv")