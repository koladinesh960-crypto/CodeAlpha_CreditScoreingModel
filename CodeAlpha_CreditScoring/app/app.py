"""
app.py
------
Flask web application for real-time credit-scoring predictions.

Usage:
    python app/app.py
    Then open http://localhost:5000 in a browser.
"""

import os
import sys

# Add project root to path so we can import src.*
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, render_template, request, jsonify
from src.predict import load_artifacts, predict_single, FEATURE_NAMES

app = Flask(__name__)

# Load model & scaler once at startup
MODEL, SCALER = load_artifacts(os.path.join(ROOT, "models"))


@app.route("/", methods=["GET"])
def index():
    """Render the prediction form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission and return prediction."""
    try:
        features = {
            "income":             float(request.form["income"]),
            "age":                int(request.form["age"]),
            "debt_ratio":         float(request.form["debt_ratio"]),
            "num_credit_lines":   int(request.form["num_credit_lines"]),
            "payment_history":    float(request.form["payment_history"]),
            "employment_years":   int(request.form["employment_years"]),
            "loan_amount":        float(request.form["loan_amount"]),
            "credit_utilization": float(request.form["credit_utilization"]),
            "num_late_payments":  int(request.form["num_late_payments"]),
            "has_mortgage":       int(request.form.get("has_mortgage", 0)),
        }

        prediction, probability = predict_single(features, MODEL, SCALER)
        label = "✅ Creditworthy" if prediction == 1 else "❌ Not Creditworthy"

        return render_template(
            "index.html",
            prediction=label,
            probability=f"{probability:.1%}",
            features=features,
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    print("[Credit Scoring App] Starting on http://localhost:5000")
    app.run(debug=True, port=5000)
