"""
predict.py
----------
Predict a single handwritten character from an image file or a raw numpy array.
"""

import os
import sys
import numpy as np

# Add project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_model(model_path: str = None):
    """Load the trained Keras model."""
    from tensorflow.keras.models import load_model as keras_load
    if model_path is None:
        model_path = os.path.join(ROOT, "models", "char_recognition_cnn.h5")
    model = keras_load(model_path)
    return model


def get_label_map():
    """Return the label → character mapping."""
    # Try EMNIST (26 letters) first, fall back to MNIST (10 digits)
    try:
        from emnist import extract_training_samples
        extract_training_samples('letters')  # will raise if not installed
        return {i: chr(65 + i) for i in range(26)}
    except Exception:
        return {i: str(i) for i in range(10)}


def predict_image(img_array: np.ndarray, model=None):
    """
    Predict a handwritten character from a 28×28 grayscale image.

    Parameters
    ----------
    img_array : np.ndarray – shape (28, 28) or (28, 28, 1), values 0–255 or 0–1
    model     : keras Model (optional; loaded from disk if None)

    Returns
    -------
    predicted_char : str
    confidence     : float
    all_probs      : np.ndarray
    """
    if model is None:
        model = load_model()

    label_map = get_label_map()

    # Preprocess
    img = img_array.astype("float32")
    if img.max() > 1.0:
        img /= 255.0
    img = img.reshape(1, 28, 28, 1)

    probs = model.predict(img, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    predicted_char = label_map.get(pred_idx, "?")

    return predicted_char, confidence, probs


if __name__ == "__main__":
    # Quick test with a random image
    dummy = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    char, conf, _ = predict_image(dummy)
    print(f"Predicted: {char}  Confidence: {conf:.2%}")
