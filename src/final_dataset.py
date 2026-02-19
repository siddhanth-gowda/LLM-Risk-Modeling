import os
import pandas as pd

PRICE_FEATURES_PATH = "data/processed/price_features.csv"
MACRO_SIGNAL_PATH   = "data/processed/finbert_scores.csv"
OUTPUT_PATH         = "data/processed/final_dataset.csv"

os.makedirs("data/processed", exist_ok=True)

price_df = pd.read_csv(PRICE_FEATURES_PATH)
price_df["Date"] = pd.to_datetime(price_df["Date"])
price_df = price_df.sort_values("Date").reset_index(drop=True)

macro_df = pd.read_csv(MACRO_SIGNAL_PATH)
macro_df["Date"] = pd.to_datetime(macro_df["Date"])
macro_df = macro_df.sort_values("Date").reset_index(drop=True)

print("\nMerging price + macro signal...")

final_df = pd.merge_asof(
    price_df,
    macro_df,
    on="Date",
    direction="backward")

# Forward fill macro signal (important because RBI meetings are sparse)
final_df["macro_tone_signal_ema"] = final_df["macro_tone_signal_ema"].ffill()
# Drop rows still missing macro signal
final_df = final_df.dropna(subset=["macro_tone_signal_ema"])

final_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nFinal dataset created and save to: {OUTPUT_PATH}")