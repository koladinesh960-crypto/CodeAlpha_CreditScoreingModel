# 🏦 Credit Scoring Model

> **CodeAlpha Machine Learning Internship — Task 1**

## Overview
A machine-learning pipeline that predicts an individual's **creditworthiness** using financial features such as income, debt ratio, payment history, and more.

Two classification models are trained and compared:
- **Logistic Regression** — interpretable baseline
- **Random Forest** — higher-accuracy ensemble method

The project includes an interactive **Flask web application** for real-time predictions.

## Tech Stack
`Python 3.10+` · `Scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Flask` · `Joblib`

## SETUP_INSTRUCTIONS

### Prerequisites
- Python 3.10 or later
- pip

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/koladinesh960-crypto/CodeAlpha_CreditScoreingModel.git
cd CodeAlpha_CreditScoreingModel/CodeAlpha_CreditScoring

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline
```bash
python main.py
```
This will:
1. Generate a synthetic credit-scoring dataset (1,000 samples)
2. Preprocess and split into train/test sets
3. Train Logistic Regression and Random Forest models
4. Evaluate both models and save metrics + plots to `outputs/`
5. Save trained models to `models/`

### Launch the Web App
```bash
python app/app.py
```
Open **http://localhost:5000** in your browser.

## Implementation Notes

### How the Task Requirements Were Met
| Requirement | Implementation |
|---|---|
| Feature engineering from financial history | 10 realistic financial features generated with correlated target variable |
| Classification algorithms (LR, DT, RF) | Logistic Regression + Random Forest with 5-fold cross-validation |
| Precision, Recall, F1-Score, ROC-AUC | All metrics computed and printed; ROC curves plotted |
| Dataset: income, debts, payment history | Synthetic dataset with income, debt_ratio, payment_history, loan_amount, etc. |

### Project Structure
```
CodeAlpha_CreditScoring/
├── data/
│   ├── generate_data.py       # Synthetic dataset generator
│   └── credit_data.csv        # Generated dataset (after running pipeline)
├── src/
│   ├── preprocess.py          # Data loading, scaling, train/test split
│   ├── train.py               # Model training with cross-validation
│   ├── evaluate.py            # Metrics computation & plot generation
│   └── predict.py             # Single-sample prediction utility
├── app/
│   ├── app.py                 # Flask web application
│   ├── templates/index.html   # Web form UI
│   └── static/style.css       # Styling
├── models/                    # Saved .pkl models & scaler
├── outputs/                   # Generated plots
├── main.py                    # End-to-end pipeline
├── requirements.txt
└── README.md
```

## EXECUTION_DEMO

### Expected Pipeline Output
```
============================================================
  Credit Scoring Model — Full Pipeline
============================================================

▶ Step 1 — Generating synthetic credit dataset ...
  Saved to data/credit_data.csv

▶ Step 2 — Preprocessing ...
[✓] Preprocessing complete — Train: 800, Test: 200

▶ Step 3 — Training models ...
[✓] Logistic Regression  — CV Accuracy: 0.88xx (±0.0xxx)
[✓] Random Forest        — CV Accuracy: 0.91xx (±0.0xxx)

▶ Step 4 — Evaluating models ...
  (Classification reports for each model)

  ✅  Pipeline complete!
```

### Generated Plots
After running the pipeline, the `outputs/` folder will contain:
- `confusion_matrix_logistic_regression.png`
- `confusion_matrix_random_forest.png`
- `roc_curves.png`
- `feature_importance.png`

## Author
**Kola Dinesh** — [CodeAlpha](https://www.codealpha.tech/) Machine Learning Internship Program
