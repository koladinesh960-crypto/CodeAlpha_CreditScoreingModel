"""
model.py
--------
Defines the CNN architecture for handwritten character recognition.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout,
)


def build_cnn(input_shape=(28, 28, 1), num_classes: int = 26):
    """
    Build a CNN with the following architecture:

        Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
        Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
        Flatten → Dense(256) → BN → Dropout(0.5) → Dense(num_classes, softmax)

    Parameters
    ----------
    input_shape : tuple  – (height, width, channels)
    num_classes : int    – number of output classes

    Returns
    -------
    model : keras.Model
    """
    model = Sequential([
        # --- Block 1 ---
        Conv2D(32, (3, 3), activation="relu", padding="same",
               input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # --- Block 2 ---
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # --- Classifier ---
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("[✓] CNN model built")
    model.summary()
    return model
