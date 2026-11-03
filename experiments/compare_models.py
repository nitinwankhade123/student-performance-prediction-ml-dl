"""Controlled, validation-only comparison of four student-result models.

All models are fitted on the same preprocessed training split. The validation
split is used for threshold selection and comparison. This script never loads
the test split and does not alter the project's original model or results.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from torch import nn
import torch.nn.functional as F

from models.ann_model import StudentPerformanceANN


TRAIN_PATH = PROJECT_ROOT / "artifacts" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "artifacts" / "processed" / "validation.csv"
BASELINE_WEIGHTS_PATH = PROJECT_ROOT / "artifacts" / "best_model.pth"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
COMPARISON_DIR = EXPERIMENTS_DIR / "results"
COMPARISON_CSV = COMPARISON_DIR / "model_comparison.csv"
ROC_PLOT = COMPARISON_DIR / "validation_roc_comparison.png"
FEATURE_COLUMNS = ["study_hours", "attendance", "previous_marks"]
TARGET_COLUMN = "result"
SEED = 42
BASELINE_THRESHOLD = 0.5
MAX_EPOCHS = 100
LEARNING_RATE = 0.001


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if frame[required].isna().any().any():
        raise ValueError(f"{path.name} contains missing values")
    X = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = frame[TARGET_COLUMN].to_numpy(dtype=np.int64)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"{path.name} targets must be encoded as 0/1")
    return X, y


def select_threshold(y_true: np.ndarray, pass_probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Choose threshold by maximum balanced accuracy using validation only.

    Ties are resolved by higher fail precision, then nearest to 0.5.
    ``pass_probabilities >= threshold`` predicts pass (1); lower predicts fail (0).
    """
    unique_probabilities = np.unique(pass_probabilities)
    midpoints = (unique_probabilities[:-1] + unique_probabilities[1:]) / 2.0
    candidates = np.unique(
        np.concatenate(([0.0, BASELINE_THRESHOLD], midpoints, [1.0]))
    )
    rows = []
    for threshold in candidates:
        predicted = (pass_probabilities >= threshold).astype(np.int64)
        matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
        rows.append(
            {
                "threshold": float(threshold),
                "fail_precision": float(precision_score(y_true, predicted, pos_label=0, zero_division=0)),
                "fail_recall": float(recall_score(y_true, predicted, pos_label=0, zero_division=0)),
                "fail_f1": float(f1_score(y_true, predicted, pos_label=0, zero_division=0)),
                "pass_precision": float(precision_score(y_true, predicted, pos_label=1, zero_division=0)),
                "pass_recall": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
                "pass_f1": float(f1_score(y_true, predicted, pos_label=1, zero_division=0)),
                "macro_precision": float(precision_score(y_true, predicted, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(y_true, predicted, average="macro", zero_division=0)),
                "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
                "accuracy": float(accuracy_score(y_true, predicted)),
                "true_negative": int(matrix[0, 0]),
                "false_positive": int(matrix[0, 1]),
                "false_negative": int(matrix[1, 0]),
                "true_positive": int(matrix[1, 1]),
            }
        )
    candidates_frame = pd.DataFrame(rows)
    ranked = candidates_frame.assign(
        _distance=(candidates_frame["threshold"] - BASELINE_THRESHOLD).abs()
    ).sort_values(
        ["balanced_accuracy", "fail_precision", "_distance"],
        ascending=[False, False, True],
        kind="stable",
    )
    return float(ranked.iloc[0]["threshold"]), candidates_frame


def score_predictions(
    name: str,
    y_true: np.ndarray,
    pass_probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    predicted = (pass_probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, pass_probabilities, pos_label=1)
    return {
        "model": name,
        "threshold": threshold,
        "roc_auc": float(auc(fpr, tpr)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "fail_precision": float(precision_score(y_true, predicted, pos_label=0, zero_division=0)),
        "fail_recall": float(recall_score(y_true, predicted, pos_label=0, zero_division=0)),
        "fail_f1": float(f1_score(y_true, predicted, pos_label=0, zero_division=0)),
        "pass_precision": float(precision_score(y_true, predicted, pos_label=1, zero_division=0)),
        "pass_recall": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "pass_f1": float(f1_score(y_true, predicted, pos_label=1, zero_division=0)),
        "macro_precision": float(precision_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "true_negative": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_positive": int(matrix[1, 1]),
    }


def save_predictions(directory: Path, y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> None:
    predicted = (probabilities >= threshold).astype(np.int64)
    pd.DataFrame(
        {
            "actual_label": y_true,
            "predicted_label": predicted,
            "predicted_probability_pass": probabilities,
        }
    ).to_csv(directory / "validation_predictions.csv", index=False)


def save_config(directory: Path, config: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )


def run_experiments() -> pd.DataFrame:
    seed_everything()
    X_train, y_train = load_split(TRAIN_PATH)
    X_validation, y_validation = load_split(VALIDATION_PATH)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    x_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)

    for directory in [
        EXPERIMENTS_DIR / "ann_baseline",
        EXPERIMENTS_DIR / "logistic_regression",
        EXPERIMENTS_DIR / "random_forest",
        EXPERIMENTS_DIR / "weighted_ann",
        COMPARISON_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # 1. Existing ANN: inference from the saved baseline checkpoint only.
    baseline_state = torch.load(BASELINE_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    baseline_model = StudentPerformanceANN()
    baseline_model.load_state_dict(baseline_state, strict=True)
    baseline_model.eval()
    with torch.inference_mode():
        baseline_probabilities = torch.sigmoid(
            baseline_model(x_validation_tensor)
        ).squeeze(1).cpu().numpy()

    # 2. Unweighted Logistic Regression baseline.
    logreg_config = {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight": None,
        "random_state": SEED,
        "fit_split": str(TRAIN_PATH.relative_to(PROJECT_ROOT)),
        "threshold_selection_split": str(VALIDATION_PATH.relative_to(PROJECT_ROOT)),
        "features": FEATURE_COLUMNS,
    }
    logreg = LogisticRegression(**{k: logreg_config[k] for k in ["C", "solver", "max_iter", "class_weight", "random_state"]})
    logreg.fit(X_train, y_train)
    logreg_probabilities = logreg.predict_proba(X_validation)[:, 1]
    joblib.dump(logreg, EXPERIMENTS_DIR / "logistic_regression" / "model.joblib")
    save_config(EXPERIMENTS_DIR / "logistic_regression", logreg_config)

    # 3. Unweighted Random Forest baseline. Inputs remain the identical
    # training-fitted standardized features used by the other experiments.
    rf_config = {
        "estimator": "sklearn.ensemble.RandomForestClassifier",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "class_weight": None,
        "random_state": SEED,
        "n_jobs": -1,
        "fit_split": str(TRAIN_PATH.relative_to(PROJECT_ROOT)),
        "threshold_selection_split": str(VALIDATION_PATH.relative_to(PROJECT_ROOT)),
        "features": FEATURE_COLUMNS,
    }
    rf = RandomForestClassifier(**{k: rf_config[k] for k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "class_weight", "random_state", "n_jobs"]})
    rf.fit(X_train, y_train)
    rf_probabilities = rf.predict_proba(X_validation)[:, 1]
    joblib.dump(rf, EXPERIMENTS_DIR / "random_forest" / "model.joblib")
    save_config(EXPERIMENTS_DIR / "random_forest", rf_config)

    # 4. Same ANN architecture and full-batch Adam setup, using training-only
    # class weights to give the minority fail class greater loss weight.
    seed_everything()
    weighted_ann = StudentPerformanceANN()
    optimizer = torch.optim.Adam(weighted_ann.parameters(), lr=LEARNING_RATE)
    fail_weight = float(np.sum(y_train == 1) / np.sum(y_train == 0))
    sample_weights = torch.where(
        y_train_tensor == 0,
        torch.tensor(fail_weight, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    weighted_history: list[dict[str, float | int]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        weighted_ann.train()
        optimizer.zero_grad(set_to_none=True)
        logits = weighted_ann(x_train_tensor)
        per_sample_loss = F.binary_cross_entropy_with_logits(
            logits, y_train_tensor, reduction="none"
        )
        loss = (per_sample_loss * sample_weights).mean()
        loss.backward()
        optimizer.step()
        weighted_history.append({"epoch": epoch, "weighted_training_loss": float(loss.detach().item())})

    weighted_ann.eval()
    with torch.inference_mode():
        weighted_probabilities = torch.sigmoid(
            weighted_ann(x_validation_tensor)
        ).squeeze(1).cpu().numpy()

    weighted_dir = EXPERIMENTS_DIR / "weighted_ann"
    weighted_weights_path = weighted_dir / "model_weights.pth"
    torch.save(weighted_ann.state_dict(), weighted_weights_path)
    pd.DataFrame(weighted_history).to_csv(weighted_dir / "training_history.csv", index=False)
    weighted_config = {
        "estimator": "StudentPerformanceANN (same architecture as baseline)",
        "architecture": "Linear(3,8) -> ReLU -> Linear(8,1 raw logit)",
        "loss": "per-sample BCEWithLogitsLoss weighted by class",
        "fail_class_weight": fail_weight,
        "pass_class_weight": 1.0,
        "class_weight_formula": "training_pass_count / training_fail_count",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "epochs": MAX_EPOCHS,
        "batching": "full batch",
        "random_seed": SEED,
        "fit_split": str(TRAIN_PATH.relative_to(PROJECT_ROOT)),
        "threshold_selection_split": str(VALIDATION_PATH.relative_to(PROJECT_ROOT)),
        "features": FEATURE_COLUMNS,
        "weights_path": str(weighted_weights_path.relative_to(PROJECT_ROOT)),
    }
    save_config(weighted_dir, weighted_config)

    model_probabilities = {
        "ANN baseline": baseline_probabilities,
        "Logistic Regression": logreg_probabilities,
        "Random Forest": rf_probabilities,
        "Class-weighted ANN": weighted_probabilities,
    }
    experiment_dirs = {
        "ANN baseline": EXPERIMENTS_DIR / "ann_baseline",
        "Logistic Regression": EXPERIMENTS_DIR / "logistic_regression",
        "Random Forest": EXPERIMENTS_DIR / "random_forest",
        "Class-weighted ANN": weighted_dir,
    }
    configs: dict[str, dict[str, object]] = {
        "ANN baseline": {
            "estimator": "StudentPerformanceANN existing checkpoint",
            "architecture": "Linear(3,8) -> ReLU -> Linear(8,1 raw logit)",
            "weights_source": str(BASELINE_WEIGHTS_PATH.relative_to(PROJECT_ROOT)),
            "weights_sha256": hashlib.sha256(BASELINE_WEIGHTS_PATH.read_bytes()).hexdigest(),
            "training_history_source": "results/training_history.csv (100 epochs)",
        "loss": "BCEWithLogitsLoss",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "epochs_completed": 100,
        "configured_max_epochs": 100,
        "batching": "full batch",
        "early_stopping_patience": 10,
        "early_stopping_triggered": False,
        "random_seed": SEED,
        "class_weight": None,
            "threshold_selection_split": str(VALIDATION_PATH.relative_to(PROJECT_ROOT)),
        "threshold_selection_criterion": "maximum validation balanced accuracy; ties by fail precision then distance from 0.5",
            "features": FEATURE_COLUMNS,
        },
        "Logistic Regression": logreg_config,
        "Random Forest": rf_config,
        "Class-weighted ANN": weighted_config,
    }

    comparison_rows: list[dict[str, object]] = []
    roc_curves: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    all_threshold_rows: list[pd.DataFrame] = []
    for name, probabilities in model_probabilities.items():
        chosen_threshold, threshold_frame = select_threshold(y_validation, probabilities)
        result = score_predictions(name, y_validation, probabilities, chosen_threshold)
        result["threshold_selection_criterion"] = "max validation balanced accuracy; ties: fail precision, then distance to 0.5"
        result["validation_records"] = int(len(y_validation))
        result["validation_fail_count"] = int(np.sum(y_validation == 0))
        result["validation_pass_count"] = int(np.sum(y_validation == 1))
        comparison_rows.append(result)

        save_predictions(experiment_dirs[name], y_validation, probabilities, chosen_threshold)
        save_config(experiment_dirs[name], configs[name])

        threshold_frame.insert(0, "model", name)
        threshold_frame["selected_threshold"] = np.isclose(
            threshold_frame["threshold"], chosen_threshold
        )
        threshold_frame["is_default_0_5"] = np.isclose(
            threshold_frame["threshold"], BASELINE_THRESHOLD
        )
        all_threshold_rows.append(threshold_frame)

        fpr, tpr, _ = roc_curve(y_validation, probabilities, pos_label=1)
        roc_curves.append((name, fpr, tpr, result["roc_auc"]))

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(COMPARISON_CSV, index=False)
    pd.concat(all_threshold_rows, ignore_index=True).to_csv(
        COMPARISON_DIR / "threshold_comparison.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, fpr, tpr, roc_auc in roc_curves:
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set(
        xlabel="False positive rate (fail incorrectly predicted pass)",
        ylabel="True positive rate (pass correctly predicted)",
        title="Model comparison: validation ROC curves",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROC_PLOT, dpi=160)
    plt.close(fig)

    return comparison


if __name__ == "__main__":
    results = run_experiments()
    print("Validation-only model comparison (threshold selected by validation balanced accuracy):")
    columns = [
        "model", "threshold", "roc_auc", "balanced_accuracy", "accuracy",
        "fail_precision", "fail_recall", "fail_f1", "pass_precision",
        "pass_recall", "pass_f1", "confusion_matrix",
    ]
    print(results[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Comparison table: {COMPARISON_CSV}")
    print(f"Full threshold table: {COMPARISON_DIR / 'threshold_comparison.csv'}")
    print(f"Validation ROC plot: {ROC_PLOT}")
