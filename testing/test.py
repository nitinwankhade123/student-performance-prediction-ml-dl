"""Independently test the saved best ANN on the held-out test split.

This script performs inference only: it does not train the model, alter model
weights, or choose/tune the classification threshold from test results.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from models.ann_model import StudentPerformanceANN


TEST_PATH = PROJECT_ROOT / "artifacts" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_model.pth"
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"
METRICS_PATH = RESULTS_DIR / "test_metrics.csv"
CONFUSION_PLOT_PATH = RESULTS_DIR / "confusion_matrix.png"
ROC_PLOT_PATH = RESULTS_DIR / "roc_curve.png"
FEATURE_COLUMNS = ["study_hours", "attendance", "previous_marks"]
TARGET_COLUMN = "result"
CLASS_NAMES = {0: "fail", 1: "pass"}
THRESHOLD = 0.5


def run_testing() -> dict[str, object]:
    """Load checkpoint and test split, calculate metrics, and save outputs."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Best model weights not found: {MODEL_PATH}")
    if not TEST_PATH.is_file():
        raise FileNotFoundError(f"Processed test dataset not found: {TEST_PATH}")

    frame = pd.read_csv(TEST_PATH)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Test dataset is missing columns: {missing}")
    if frame[required_columns].isna().any().any():
        raise ValueError("Test dataset contains missing values")

    actual_labels = frame[TARGET_COLUMN].to_numpy(dtype=np.int64)
    if not set(np.unique(actual_labels)).issubset({0, 1}):
        raise ValueError("Test target must contain only encoded labels 0 and 1")
    features = torch.tensor(
        frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32
    )

    model = StudentPerformanceANN()
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Inference only: use the model's raw logits, then sigmoid for probabilities.
    with torch.no_grad():
        logits = model(features)
        probabilities = torch.sigmoid(logits).squeeze(1).cpu().numpy()
    predicted_labels = (probabilities >= THRESHOLD).astype(np.int64)

    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels, zero_division=0)
    recall = recall_score(actual_labels, predicted_labels, zero_division=0)
    f1 = f1_score(actual_labels, predicted_labels, zero_division=0)
    matrix = confusion_matrix(actual_labels, predicted_labels, labels=[0, 1])
    if len(np.unique(actual_labels)) == 2:
        fpr, tpr, _ = roc_curve(actual_labels, probabilities)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, roc_auc = None, None, float("nan")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "actual_label": actual_labels,
            "predicted_label": predicted_labels,
            "predicted_probability": probabilities,
        }
    )
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    metrics = pd.DataFrame(
        [
            ("test_records", int(len(actual_labels))),
            ("threshold", THRESHOLD),
            ("accuracy", float(accuracy)),
            ("precision", float(precision)),
            ("recall", float(recall)),
            ("f1_score", float(f1)),
            ("roc_auc", float(roc_auc)),
            ("true_negative", int(matrix[0, 0])),
            ("false_positive", int(matrix[0, 1])),
            ("false_negative", int(matrix[1, 0])),
            ("true_positive", int(matrix[1, 1])),
        ],
        columns=["metric", "value"],
    )
    metrics.to_csv(METRICS_PATH, index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        yticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        xlabel="Predicted label",
        ylabel="Actual label",
        title="Confusion matrix (test set)",
    )
    midpoint = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{matrix[row, column]}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > midpoint else "black",
            )
    fig.tight_layout()
    fig.savefig(CONFUSION_PLOT_PATH, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    if fpr is not None and tpr is not None:
        ax.plot(fpr, tpr, label=f"ANN ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curve (test set)",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROC_PLOT_PATH, dpi=150)
    plt.close(fig)

    counts = {
        CLASS_NAMES[label]: int(count)
        for label, count in zip(*np.unique(actual_labels, return_counts=True))
    }
    result = {
        "test_records": int(len(actual_labels)),
        "class_distribution": counts,
        "threshold": THRESHOLD,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix.tolist(),
        "roc_auc": float(roc_auc),
        "predictions_path": str(PREDICTIONS_PATH),
        "metrics_path": str(METRICS_PATH),
        "confusion_plot_path": str(CONFUSION_PLOT_PATH),
        "roc_plot_path": str(ROC_PLOT_PATH),
    }
    return result


if __name__ == "__main__":
    result = run_testing()
    print(f"Test records: {result['test_records']}")
    print(f"Actual class distribution: {result['class_distribution']}")
    print(f"Classification threshold: {result['threshold']}")
    print(f"Accuracy: {result['accuracy']:.6f}")
    print(f"Precision: {result['precision']:.6f}")
    print(f"Recall: {result['recall']:.6f}")
    print(f"F1-score: {result['f1_score']:.6f}")
    print("Confusion matrix (rows=actual [fail, pass], columns=predicted [fail, pass]):")
    print(np.asarray(result["confusion_matrix"]))
    print(f"ROC-AUC: {result['roc_auc']:.6f}")
    print(f"Predictions saved to: {result['predictions_path']}")
    print(f"Metrics saved to: {result['metrics_path']}")
    print(f"Confusion matrix plot saved to: {result['confusion_plot_path']}")
    print(f"ROC curve saved to: {result['roc_plot_path']}")
