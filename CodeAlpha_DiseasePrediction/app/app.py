"""
app.py
------
Flask web application for heart-disease risk prediction.

Usage:
    python app/app.py
    Open http://localhost:5002
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, render_template, request
from src.predict import load_artifacts, predict_patient, FEATURE_NAMES

app = Flask(__name__)

MODEL, SCALER = load_artifacts(os.path.join(ROOT, "models"))


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = {
            "age":      int(request.form["age"]),
            "sex":      int(request.form["sex"]),
            "cp":       int(request.form["cp"]),
            "trestbps": int(request.form["trestbps"]),
            "chol":     int(request.form["chol"]),
            "fbs":      int(request.form["fbs"]),
            "restecg":  int(request.form["restecg"]),
            "thalach":  int(request.form["thalach"]),
            "exang":    int(request.form["exang"]),
            "oldpeak":  float(request.form["oldpeak"]),
            "slope":    int(request.form["slope"]),
            "ca":       int(request.form["ca"]),
            "thal":     int(request.form["thal"]),
        }

        prediction, probability = predict_patient(features, MODEL, SCALER)
        if prediction == 1:
            label = "⚠️ Heart Disease Risk Detected"
        else:
            label = "✅ No Heart Disease Detected"

        return render_template(
            "index.html",
            prediction=label,
            probability=f"{probability:.1%}",
            risk_level=prediction,
            features=features,
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    print("[Disease Prediction App] Starting on http://localhost:5002")
    app.run(debug=True, port=5002)
