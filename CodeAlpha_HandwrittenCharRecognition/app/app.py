"""
app.py
------
Flask web application with an HTML5 Canvas for drawing handwritten
characters and getting real-time CNN predictions.

Usage:
    python app/app.py
    Open http://localhost:5001
"""

import os
import sys
import io
import base64
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, render_template, request, jsonify
from src.predict import load_model, predict_image, get_label_map

app = Flask(__name__)

# Load model once at startup
MODEL = load_model(os.path.join(ROOT, "models", "char_recognition_cnn.h5"))
LABEL_MAP = get_label_map()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive a base64-encoded image from the canvas,
    resize to 28×28, and return the predicted character.
    """
    try:
        data = request.json
        img_data = data["image"]

        # Decode base64 image
        # The canvas sends "data:image/png;base64,..." — strip the header
        if "," in img_data:
            img_data = img_data.split(",")[1]

        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale

        # Resize to 28×28
        img = img.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img)

        # Invert if background is white (canvas default)
        # EMNIST expects white-on-black
        if img_array.mean() > 128:
            img_array = 255 - img_array

        # Predict
        predicted_char, confidence, probs = predict_image(img_array, MODEL)

        # Top 5 predictions
        top_indices = np.argsort(probs)[::-1][:5]
        top5 = [
            {"char": LABEL_MAP.get(int(i), "?"), "prob": f"{probs[i]:.1%}"}
            for i in top_indices
        ]

        return jsonify({
            "success": True,
            "prediction": predicted_char,
            "confidence": f"{confidence:.1%}",
            "top5": top5,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    print("[Handwritten Char Recognition App] Starting on http://localhost:5001")
    app.run(debug=True, port=5001)
