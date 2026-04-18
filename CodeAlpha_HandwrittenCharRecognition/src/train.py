"""
train.py
--------
Trains the CNN on the EMNIST Letters dataset with data augmentation
and early stopping.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data_augmenter():
    """
    Create an ImageDataGenerator for on-the-fly data augmentation.
    Applies small rotations, shifts, and zoom to improve generalization.
    """
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )


def train_model(model, X_train, y_train, X_test, y_test,
                model_save_path: str, epochs: int = 20, batch_size: int = 128):
    """
    Train the CNN with data augmentation and early stopping.

    Parameters
    ----------
    model           : keras.Model
    X_train, y_train: Training data
    X_test, y_test  : Validation data
    model_save_path : str – path to save best model (.h5)
    epochs          : int
    batch_size      : int

    Returns
    -------
    history : keras History object
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_accuracy", patience=5,
        restore_best_weights=True, verbose=1,
    )
    checkpoint = ModelCheckpoint(
        model_save_path, monitor="val_accuracy",
        save_best_only=True, verbose=1,
    )

    # Data augmentation
    datagen = get_data_augmenter()
    datagen.fit(X_train)

    print(f"\n[▶] Training for up to {epochs} epochs (early stopping patience=5) ...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )

    print(f"[✓] Training complete — best val_accuracy: "
          f"{max(history.history['val_accuracy']):.4f}")
    return history


def plot_training_curves(history, output_dir: str):
    """Save accuracy and loss curves."""
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[✓] Saved training curves → {path}")
