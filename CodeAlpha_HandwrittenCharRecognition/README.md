# Handwritten Character Recognition

## Overview
A deep-learning pipeline that recognises **handwritten characters (A–Z)** using a Convolutional Neural Network (CNN). Built as part of the **CodeAlpha Machine Learning Internship**.

The model is trained on the **EMNIST Letters** dataset and served through an interactive **Flask web app** with an HTML5 Canvas for drawing.

## Tech Stack
`Python 3.10+` · `TensorFlow / Keras` · `EMNIST` · `Scikit-learn` · `Matplotlib` · `Flask` · `Pillow`

## SETUP_INSTRUCTIONS

### Prerequisites
- Python 3.10 or later
- pip
- (Optional) GPU with CUDA for faster training

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/CodeAlpha_HandwrittenCharRecognition.git
cd CodeAlpha_HandwrittenCharRecognition

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
1. Download and preprocess the EMNIST Letters dataset
2. Build a CNN architecture
3. Train for up to 20 epochs (with early stopping)
4. Evaluate on test set and save plots to `outputs/`
5. Save the best model to `models/char_recognition_cnn.h5`

> **Note:** First run may take a few minutes to download the EMNIST dataset.

### Launch the Web App
```bash
python app/app.py
```
Open **http://localhost:5001** in your browser. Draw a letter on the canvas and click **Predict**!

## Implementation Notes

### How the Task Requirements Were Met
| Requirement | Implementation |
|---|---|
| Dataset: EMNIST (characters) | EMNIST Letters (26 classes A–Z), auto-downloaded |
| Model: CNN | 4-layer CNN with BatchNorm, MaxPool, Dropout |
| Image processing | Pixel normalization, data augmentation (rotation, shift, zoom) |
| Extendable to recognition | Architecture supports any number of classes |

### CNN Architecture
```
Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
Flatten → Dense(256) → BN → Dropout(0.5) → Dense(26, softmax)
```

### Project Structure
```
CodeAlpha_HandwrittenCharRecognition/
├── src/
│   ├── load_data.py           # EMNIST data loading & preprocessing
│   ├── model.py               # CNN architecture definition
│   ├── train.py               # Training with augmentation & early stopping
│   ├── evaluate.py            # Test evaluation & confusion matrix
│   └── predict.py             # Single-image prediction utility
├── app/
│   ├── app.py                 # Flask web application
│   ├── templates/index.html   # Drawing canvas UI
│   └── static/style.css       # Styling
├── models/                    # Saved Keras model (.h5)
├── outputs/                   # Training curves, confusion matrix
├── main.py                    # End-to-end pipeline
├── requirements.txt
└── README.md
```

## EXECUTION_DEMO

### Expected Pipeline Output
```
============================================================
  Handwritten Character Recognition — Full Pipeline
============================================================

▶ Step 1 — Loading dataset ...
[✓] Loaded EMNIST Letters — 88800 train, 14800 test

▶ Step 3 — Building CNN ...
Model: "sequential"
(model summary printed)

▶ Step 4 — Training ...
Epoch 1/20 — accuracy: 0.7xxx — val_accuracy: 0.85xx
...
[✓] Training complete — best val_accuracy: 0.91xx

▶ Step 5 — Evaluating ...
  Test Accuracy: 0.91xx
```

### Generated Files
- `models/char_recognition_cnn.h5` — Trained CNN model
- `outputs/training_curves.png` — Accuracy & loss plots
- `outputs/confusion_matrix.png` — 26×26 confusion matrix
- `outputs/classification_report.txt` — Per-class metrics

## Author
**CodeAlpha Intern** — Machine Learning Internship Program
