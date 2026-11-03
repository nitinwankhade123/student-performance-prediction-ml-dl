"""Independent final test of the frozen Logistic Regression candidate.

The estimator was fitted previously on the training split. This script loads
that artifact, obtains its already selected threshold from validation records,
checks the saved provenance, then evaluates once on the held-out test split.
It never calls fit() and never changes the model or threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "logistic_regression"
COMPARISON_PATH = PROJECT_ROOT / "experiments" / "results" / "model_comparison.csv"
THRESHOLD_PATH = PROJECT_ROOT / "experiments" / "results" / "threshold_comparison.csv"
CONFIG_PATH = EXPERIMENT_DIR / "config.json"
MODEL_PATH = EXPERIMENT_DIR / "model.joblib"
TEST_PATH = PROJECT_ROOT / "artifacts" / "processed" / "test.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_PATH = RESULTS_DIR / "final_test_predictions.csv"
METRICS_PATH = RESULTS_DIR / "final_test_metrics.csv"
CONFUSION_PATH = RESULTS_DIR / "final_test_confusion_matrix.png"
ROC_PATH = RESULTS_DIR / "final_test_roc_curve.png"
FEATURE_COLUMNS = ["study_hours", "attendance", "previous_marks"]
TARGET_COLUMN = "result"


def run_final_test() -> dict[str, object]:
    """Freeze the trained estimator/validation threshold and test once."""
    # Load and verify provenance before opening the test dataset.
    for path in (CONFIG_PATH, MODEL_PATH, COMPARISON_PATH, THRESHOLD_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment artifact not found: {path}")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("fit_split") != "artifacts\\processed\\train.csv":
        raise ValueError("Saved Logistic Regression config does not identify the training split")
    if config.get("threshold_selection_split") != "artifacts\\processed\\validation.csv":
        raise ValueError("Saved threshold provenance is not the validation split")

    comparison = pd.read_csv(COMPARISON_PATH)
    model_rows = comparison.loc[comparison["model"] == "Logistic Regression"]
    if len(model_rows) != 1:
        raise ValueError("Expected exactly one Logistic Regression comparison row")
    threshold_table = pd.read_csv(THRESHOLD_PATH)
    selected_rows = threshold_table.loc[
        (threshold_table["model"] == "Logistic Regression")
        & threshold_table["selected_threshold"].astype(bool)
    ]
    if len(selected_rows) != 1:
        raise ValueError("Expected exactly one validation-selected Logistic Regression threshold")
    threshold = float(selected_rows.iloc[0]["threshold"])
    recorded_threshold = float(model_rows.iloc[0]["threshold"])
    if threshold != recorded_threshold:
        raise ValueError("Selected threshold does not match the model comparison record")
    criterion = str(model_rows.iloc[0]["threshold_selection_criterion"])
    if "validation" not in criterion.lower():
        raise ValueError("Threshold selection criterion is not recorded as validation-based")

    # Load the previously fitted estimator. No fit/refit is called; threshold is
    # now frozen before any test rows or labels are read.
    estimator = joblib.load(MODEL_PATH)
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("Saved Logistic Regression estimator has no predict_proba method")
    frozen_threshold = threshold
    if not TEST_PATH.is_file():
        raise FileNotFoundError(f"Processed test split not found: {TEST_PATH}")

    test_frame = pd.read_csv(TEST_PATH)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in test_frame.columns]
    if missing:
        raise ValueError(f"Test dataset is missing columns: {missing}")
    if test_frame[required].isna().any().any():
        raise ValueError("Test dataset contains missing values")
    y_true = test_frame[TARGET_COLUMN].to_numpy(dtype=np.int64)
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Test target must contain only encoded labels 0 and 1")

    X_test = test_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    pass_probabilities = estimator.predict_proba(X_test)[:, 1]
    y_pred = (pass_probabilities >= frozen_threshold).astype(np.int64)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, pass_probabilities, pos_label=1)
    roc_auc = float(auc(fpr, tpr))
    metrics = {
        "test_records": int(len(y_true)),
        "actual_fail_count": int(np.sum(y_true == 0)),
        "actual_pass_count": int(np.sum(y_true == 1)),
        "classification_threshold": frozen_threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fail": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_fail": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_fail": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_pass": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_pass": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_pass": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": roc_auc,
        "true_negative": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_positive": int(matrix[1, 1]),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "actual_label": y_true,
            "predicted_label": y_pred,
            "predicted_probability_pass": pass_probabilities,
            "classification_threshold": np.full(len(y_true), frozen_threshold),
        }
    ).to_csv(PREDICTIONS_PATH, index=False, float_format="%.17g")
    pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()]).to_csv(
        METRICS_PATH, index=False, float_format="%.17g"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["fail", "pass"],
        yticklabels=["fail", "pass"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title="Logistic Regression confusion matrix (test set)",
    )
    midpoint = matrix.max() / 2.0
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > midpoint else "black",
            )
    fig.tight_layout()
    fig.savefig(CONFUSION_PATH, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set(
        xlabel="False positive rate (fail incorrectly predicted pass)",
        ylabel="True positive rate (pass correctly predicted)",
        title="Logistic Regression ROC curve (test set)",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROC_PATH, dpi=150)
    plt.close(fig)

    return {
        **metrics,
        "threshold_selection_split": config["threshold_selection_split"],
        "threshold_selection_criterion": criterion,
        "predictions_path": str(PREDICTIONS_PATH),
        "metrics_path": str(METRICS_PATH),
        "confusion_plot_path": str(CONFUSION_PATH),
        "roc_plot_path": str(ROC_PATH),
        "confusion_matrix": matrix.tolist(),
    }


if __name__ == "__main__":
    result = run_final_test()
    print("Frozen model: experiments/logistic_regression/model.joblib")
    print(f"Threshold source: {THRESHOLD_PATH}")
    print(f"Threshold selection split: {result['threshold_selection_split']}")
    print(f"Threshold selection criterion: {result['threshold_selection_criterion']}")
    print(f"Frozen threshold: {result['classification_threshold']:.17g}")
    print(f"Test records: {result['test_records']}")
    print(f"Actual class counts: fail={result['actual_fail_count']}, pass={result['actual_pass_count']}")
    for metric in (
        "accuracy", "precision_fail", "recall_fail", "f1_fail", "precision_pass",
        "recall_pass", "f1_pass", "precision_macro", "recall_macro", "f1_macro",
        "balanced_accuracy", "roc_auc",
    ):
        print(f"{metric}: {result[metric]:.6f}")
    print("Confusion matrix (rows actual [fail, pass], columns predicted [fail, pass]):")
    print(np.asarray(result["confusion_matrix"]))
    print(f"Predictions saved to: {result['predictions_path']}")
    print(f"Metrics saved to: {result['metrics_path']}")
    print(f"Confusion matrix saved to: {result['confusion_plot_path']}")
    print(f"ROC curve saved to: {result['roc_plot_path']}")
