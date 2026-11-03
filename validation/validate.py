"""Run validation-only diagnostics and compare classification thresholds.

This script loads the existing checkpoint and validation split for inference.
It does not train, update model weights, load the test split, or tune on test data.
Threshold convention: probability >= threshold predicts pass (class 1).
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
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from models.ann_model import StudentPerformanceANN


VALIDATION_PATH = PROJECT_ROOT / "artifacts" / "processed" / "validation.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_model.pth"
GENERATOR_PATH = PROJECT_ROOT / "datasets" / "generate_synthetic_data.py"
RESULTS_DIR = PROJECT_ROOT / "results"
DIAGNOSTICS_PATH = RESULTS_DIR / "validation_diagnostics.csv"
ROC_PLOT_PATH = RESULTS_DIR / "validation_roc_curve.png"
FEATURE_COLUMNS = ["study_hours", "attendance", "previous_marks"]
TARGET_COLUMN = "result"
DEFAULT_THRESHOLD = 0.5


def _probability_summary(probabilities: np.ndarray) -> dict[str, float]:
    """Calculate distribution statistics using sample standard deviation."""
    return {
        "probability_min": float(np.min(probabilities)),
        "probability_max": float(np.max(probabilities)),
        "probability_mean": float(np.mean(probabilities)),
        "probability_std_sample": float(np.std(probabilities, ddof=1)),
    }


def _threshold_metrics(
    actual: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Calculate both class-specific and balanced metrics at one threshold."""
    predicted = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "fail_f1": float(f1_score(actual, predicted, pos_label=0, zero_division=0)),
        "fail_recall": float(recall_score(actual, predicted, pos_label=0, zero_division=0)),
        "fail_precision": float(precision_score(actual, predicted, pos_label=0, zero_division=0)),
        "pass_precision": float(precision_score(actual, predicted, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "accuracy": float(np.mean(actual == predicted)),
        "true_negative": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_positive": int(matrix[1, 1]),
    }


def run_validation_diagnostics() -> dict[str, object]:
    """Generate validation probability, ROC, and threshold diagnostics."""
    if not VALIDATION_PATH.is_file():
        raise FileNotFoundError(f"Validation split not found: {VALIDATION_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Best model weights not found: {MODEL_PATH}")

    frame = pd.read_csv(VALIDATION_PATH)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Validation split is missing columns: {missing}")
    if frame[required].isna().any().any():
        raise ValueError("Validation split contains missing values")

    actual = frame[TARGET_COLUMN].to_numpy(dtype=np.int64)
    if not set(np.unique(actual)).issubset({0, 1}):
        raise ValueError("Validation target must contain only encoded labels 0 and 1")
    if len(np.unique(actual)) != 2:
        raise ValueError("Validation ROC-AUC requires both classes")
    features = torch.tensor(
        frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32
    )

    model = StudentPerformanceANN()
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(features)).squeeze(1).cpu().numpy()

    fpr, tpr, roc_thresholds = roc_curve(actual, probabilities)
    validation_auc = float(auc(fpr, tpr))

    # Evaluate every distinct classification partition induced by probabilities,
    # along with the project's existing 0.5 threshold. This uses validation only.
    unique_probabilities = np.unique(probabilities)
    midpoints = (unique_probabilities[:-1] + unique_probabilities[1:]) / 2.0
    candidates = np.unique(
        np.concatenate(([0.0, DEFAULT_THRESHOLD], midpoints, [1.0]))
    )
    threshold_rows = [
        _threshold_metrics(actual, probabilities, float(threshold))
        for threshold in candidates
    ]
    threshold_frame = pd.DataFrame(threshold_rows)
    threshold_frame["is_default_threshold"] = np.isclose(
        threshold_frame["threshold"], DEFAULT_THRESHOLD
    )

    # Selection criterion: maximize balanced accuracy (equal weight to fail and
    # pass recall); ties prefer fail precision, then proximity to threshold 0.5.
    ranked = threshold_frame.assign(
        distance_from_default=(threshold_frame["threshold"] - DEFAULT_THRESHOLD).abs()
    ).sort_values(
        ["balanced_accuracy", "fail_precision", "distance_from_default"],
        ascending=[False, False, True],
        kind="stable",
    )
    selected_threshold = float(ranked.iloc[0]["threshold"])
    threshold_frame["selected_by_balanced_accuracy"] = np.isclose(
        threshold_frame["threshold"], selected_threshold
    )

    default_metrics = _threshold_metrics(actual, probabilities, DEFAULT_THRESHOLD)
    selected_metrics = _threshold_metrics(actual, probabilities, selected_threshold)
    distribution_rows: list[dict[str, object]] = []
    for label, name in ((None, "all"), (0, "fail"), (1, "pass")):
        values = probabilities if label is None else probabilities[actual == label]
        row: dict[str, object] = {
            "record_type": "probability_summary",
            "group": name,
            "n_records": int(values.size),
            **_probability_summary(values),
        }
        distribution_rows.append(row)

    metadata_rows: list[dict[str, object]] = [
        {
            "record_type": "roc_summary",
            "group": "validation",
            "n_records": int(len(actual)),
            "roc_auc": validation_auc,
            "actual_fail_count": int(np.sum(actual == 0)),
            "actual_pass_count": int(np.sum(actual == 1)),
        },
        {
            "record_type": "threshold_summary",
            "group": "default_0.5",
            "n_records": int(len(actual)),
            **default_metrics,
        },
        {
            "record_type": "threshold_summary",
            "group": "selected_max_balanced_accuracy",
            "n_records": int(len(actual)),
            **selected_metrics,
        },
    ]

    RESULT_SCHEMA = [
        "record_type", "group", "n_records", "probability_min", "probability_max",
        "probability_mean", "probability_std_sample", "roc_auc", "threshold",
        "fail_f1", "fail_recall", "fail_precision", "pass_precision",
        "balanced_accuracy", "accuracy", "true_negative", "false_positive",
        "false_negative", "true_positive", "is_default_threshold",
        "selected_by_balanced_accuracy", "actual_fail_count", "actual_pass_count",
    ]
    summary_frame = pd.DataFrame(distribution_rows + metadata_rows).reindex(
        columns=RESULT_SCHEMA
    )
    threshold_export = threshold_frame.copy()
    threshold_export.insert(0, "record_type", "threshold_candidate")
    threshold_export.insert(1, "group", "validation")
    threshold_export.insert(2, "n_records", int(len(actual)))
    threshold_export = threshold_export.reindex(columns=RESULT_SCHEMA)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([summary_frame, threshold_export], ignore_index=True).to_csv(
        DIAGNOSTICS_PATH, index=False
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Validation ROC (AUC = {validation_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set(
        xlabel="False positive rate (fail incorrectly predicted pass)",
        ylabel="True positive rate (pass correctly detected)",
        title="ROC curve (validation set)",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROC_PLOT_PATH, dpi=150)
    plt.close(fig)

    return {
        "n_records": int(len(actual)),
        "class_counts": {"fail": int(np.sum(actual == 0)), "pass": int(np.sum(actual == 1))},
        "probability_summaries": distribution_rows,
        "roc_auc": validation_auc,
        "default_metrics": default_metrics,
        "selected_threshold": selected_threshold,
        "selected_metrics": selected_metrics,
        "threshold_candidates": int(len(threshold_frame)),
        "diagnostics_path": str(DIAGNOSTICS_PATH),
        "roc_plot_path": str(ROC_PLOT_PATH),
        "generator_path": str(GENERATOR_PATH),
    }


if __name__ == "__main__":
    result = run_validation_diagnostics()
    print(f"Validation records: {result['n_records']}")
    print(f"Validation class distribution: {result['class_counts']}")
    for summary in result["probability_summaries"]:
        print(f"Probability summary ({summary['group']}): {summary}")
    print(f"Validation ROC-AUC: {result['roc_auc']:.6f}")
    print(f"At threshold 0.5 (predict pass when probability >= threshold): {result['default_metrics']}")
    print(
        "Selected threshold by maximum balanced accuracy "
        f"(ties: fail precision, then closest to 0.5): {result['selected_threshold']:.6f}"
    )
    print(f"Selected-threshold metrics: {result['selected_metrics']}")
    print(f"Threshold candidates evaluated: {result['threshold_candidates']}")
    print(f"Validation diagnostics saved to: {result['diagnostics_path']}")
    print(f"Validation ROC curve saved to: {result['roc_plot_path']}")
