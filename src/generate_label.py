import os
import re
import pandas as pd
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize


FOLDER_PATH = Path("data/text/rbi_policy/")
OUTPUT_FILE = Path("data/processed/finetune_train.csv")


HAWKISH_KEYWORDS = [
    'inflation', 'rise', 'increase', 'higher', 'elevated', 'upside', 'risk',
    'pressure', 'tightening', 'hike', 'concern', 'vigilant', 'uncertainty',
    'hardening', 'surpass', 'persist', 'second-round', 'anchor', 'monitor',
    'volatility', 'shock', 'constrain', 'high']

DOVISH_KEYWORDS = [
    'growth', 'slowdown', 'weak', 'decline', 'decrease', 'lower', 'downside',
    'support', 'revive', 'recovery', 'mitigate', 'ease', 'accommodative',
    'liquidity', 'surplus', 'soften', 'moderate', 'below', 'stable', 'resilient',
    'traction', 'revival', 'cushion', 'buffer', 'rate cut']

NEUTRAL_KEYWORDS = [
    'unchanged', 'remain', 'decided', 'meeting', 'minutes', 'vote', 'member',
    'statement', 'projected', 'stood', 'account', 'range', 'target', 'policy',
    'repo', 'reverse', 'bank', 'mpc', 'quarter', 'half', 'basis', 'points']


def clean_text(text: str) -> str:
    """Basic text cleaning for RBI policy language."""
    text = re.sub(r"\s+", " ", text)      # normalize whitespace
    text = text.replace("\\", "")         # remove stray backslashes
    text = text.strip()
    return text

def get_label(sentence: str) -> str | None:
    """
    Heuristic labeling based on normalized keyword scores.
    Returns 'hawkish', 'dovish', 'neutral', or None (skip).
    """
    s = sentence.lower()
    words = s.split()
    length = max(1, len(words))

    h_count = sum(1 for w in HAWKISH_KEYWORDS if w in s)
    d_count = sum(1 for w in DOVISH_KEYWORDS if w in s)
    n_count = sum(1 for w in NEUTRAL_KEYWORDS if w in s)

    # Normalize by sentence length (reduces bias toward long sentences)
    h_score = h_count / length
    d_score = d_count / length
    n_score = n_count / length

    # Thresholds (tunable)
    MIN_SCORE = 0.01

    if h_score > d_score and h_score >= MIN_SCORE:
        return "hawkish"
    elif d_score > h_score and d_score >= MIN_SCORE:
        return "dovish"
    elif n_score >= MIN_SCORE:
        return "neutral"
    else:
        return None  # skip unclear / low-signal sentences


def main():
    txt_files = list(FOLDER_PATH.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {FOLDER_PATH}")

    print(f"Found {len(txt_files)} policy text files. Processing...\n")

    rows = []

    for fp in txt_files:
        print(f"Reading: {fp.name}")

        try:
            content = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"⚠️ Could not read {fp}: {e}")
            continue

        # Better sentence tokenizer than regex
        sentences = sent_tokenize(content)

        for s in sentences:
            clean_s = clean_text(s)

            # Filter by length to avoid headers / fragments
            word_count = len(clean_s.split())
            if 10 <= word_count <= 60:
                label = get_label(clean_s)
                if label:
                    rows.append({"text": clean_s, "label": label})

    # Create DataFrame
    df = pd.DataFrame(rows)
    print(f"\nInitial labeled sentences: {len(df)}")

    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"After deduplication: {len(df)}")

    print("\nClass distribution BEFORE balancing:")
    print(df["label"].value_counts())

    # Downsample majority classes to match minority class
    min_class_size = df["label"].value_counts().min()

    df_balanced = (
        df
        .groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min_class_size, random_state=42))
        .reset_index(drop=True)
    )

    print("\nClass distribution AFTER balancing:")
    print(df_balanced["label"].value_counts())
    print(f"\nFinal dataset size: {len(df_balanced)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n✅ Saved cleaned, balanced dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()