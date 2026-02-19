# LLMRisk: Market Risk Prediction using Machine Learning and NLP

## Overview

This project predicts market risk regimes using a hybrid approach combining:

• Price-based technical features  
• NLP-based macroeconomic signals extracted from RBI monetary policy statements  

The objective is to improve risk prediction by integrating forward-looking macro sentiment with market data.

---

## Methodology

### 1. Price Feature Engineering

Using NIFTY50 and India VIX data:

- Returns
- Volatility
- Momentum
- Drawdowns

Risk-off regime defined as:

Forward 5-day average VIX > 75th percentile threshold

---

### 2. NLP Pipeline

RBI policy statements processed using:

- PDF to text conversion
- Sentence extraction and labeling
- TF-IDF + XGBoost classifier (FinBERT-inspired)
- Tone signal generation and EMA smoothing

Macro tone signal reflects hawkish vs dovish central bank stance.

---

### 3. Final Dataset

Combined:

- Price features
- Macro tone signal

Target:

Risk-Off regime classification

---

### 4. Machine Learning Models

Models trained:

• Logistic Regression (baseline)  
• Random Forest  
• Gradient Boosting  

Time-series split used to prevent lookahead bias.

---

## Results

Gradient Boosting achieved highest performance:

ROC-AUC ≈ 0.73

Feature importance analysis shows macro tone signal contributes significantly to risk prediction.

---

## Project Structure

src/ → source code
data/raw/ → raw market and RBI data
data/processed/ → processed datasets
models/ → trained NLP models
figures/ → ROC curves and feature importance


---

## Technologies Used

Python  
Machine Learning  
Logistic Regression  
Random Forest  
Gradient Boosting  
Natural Language Processing  
TF-IDF  
XGBoost  
Time Series Analysis  

---

## How to Run

Step 1: Generate price features

Step 2: Generate NLP signal

Step 3: Build final dataset

Step 4: Train models

---

Author: Siddhanth Gowda

---
## Author

Siddhanth Gowda
