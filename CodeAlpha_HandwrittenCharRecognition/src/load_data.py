"""
load_data.py
------------
Downloads and preprocesses the EMNIST Letters dataset for training.
Falls back to MNIST digits if EMNIST is unavailable.
"""

import numpy as np
import os


def load_emnist_letters():
    """
    Load EMNIST Letters dataset.

    Tries the `emnist` package first, then falls back to
    keras MNIST digits with a warning.

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
        Images are 28×28, uint8.  Labels are 0-indexed integers.
    num_classes : int
    label_map   : dict   – {int_label: str_character}
    """
    try:
        # Preferred: use the `emnist` package
        from emnist import extract_training_samples, extract_test_samples

        X_train, y_train = extract_training_samples('letters')
        X_test, y_test   = extract_test_samples('letters')

        # EMNIST Letters labels are 1–26 (A–Z); shift to 0–25
        y_train = y_train - 1
        y_test  = y_test  - 1
        num_classes = 26
        label_map = {i: chr(65 + i) for i in range(26)}   # 0→A, 1→B, …

        print(f"[✓] Loaded EMNIST Letters — {X_train.shape[0]} train, {X_test.shape[0]} test")

    except ImportError:
        print("[!] 'emnist' package not found. Falling back to MNIST digits.")
        print("    Install it with:  pip install emnist")
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        num_classes = 10
        label_map = {i: str(i) for i in range(10)}
        print(f"[✓] Loaded MNIST Digits — {X_train.shape[0]} train, {X_test.shape[0]} test")

    return (X_train, y_train), (X_test, y_test), num_classes, label_map


def preprocess_images(X_train, X_test):
    """
    Normalize pixel values to [0, 1] and reshape to (N, 28, 28, 1) for CNN.
    """
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0

    # Add channel dimension for Conv2D
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test  = X_test.reshape(-1, 28, 28, 1)

    print(f"[✓] Preprocessed images — shape: {X_train.shape}")
    return X_train, X_test
