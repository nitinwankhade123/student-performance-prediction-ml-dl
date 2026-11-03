"""Reusable binary-classification metrics for ANN and sklearn estimators.

The project encodes fail as 0 and pass as 1. Precision, recall, and F1 are
reported with pass as the positive class; fail-specific metrics are included
to make the minority-class behavior visible. ``y_score`` may contain Logistic
Regression pass probabilities or ANN raw logits (set ``score_type='logits'``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _to_numpy(values: Any, name: str) -> np.ndarray:
    """Convert NumPy-like or Torch-like values into a flat NumPy array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional or have shape [n, 1]")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _to_scores(values: Any, score_type: str) -> np.ndarray:
    """Extract the pass score from vectors, ANN [n,1], or LR [n,2] output."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    scores = np.asarray(values)
    if scores.ndim == 2 and scores.shape[1] == 2:
        if score_type.strip().lower() not in {"probability", "probabilities", "proba"}:
            raise ValueError("two-column scores are supported only as [fail, pass] probabilities")
        scores = scores[:, 1]
    elif scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores[:, 0]
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("y_score must be a vector, [n, 1] logits, or [n, 2] probabilities")
    return scores


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Apply sigmoid stably to ANN raw logits without importing PyTorch."""
    values = logits.astype(np.float64, copy=False)
    probabilities = np.empty_like(values)
    nonnegative = values >= 0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    probabilities[~nonnegative] = exp_values / (1.0 + exp_values)
    return probabilities


def evaluate_binary_predictions(
    y_true: Any,
    y_pred: Any | None = None,
    y_score: Any | None = None,
    *,
    threshold: float = 0.5,
    score_type: str = "probabilities",
    zero_division: int = 0,
) -> dict[str, Any]:
    """Calculate binary metrics for pass=1/fail=0 predictions.

    Args:
        y_true: Ground-truth labels containing only 0 (fail) and 1 (pass).
        y_pred: Optional already-thresholded labels. If omitted, labels are
            derived from ``y_score`` and ``threshold``.
        y_score: Optional pass probabilities or ANN logits. Supplying scores
            enables ROC-AUC; without scores, ``roc_auc`` is ``None``.
        threshold: Probability cutoff used when ``y_pred`` is not supplied.
        score_type: ``"probabilities"`` or ``"logits"``. Logits are converted
            with sigmoid before thresholding and ROC-AUC calculation.
        zero_division: Value used by scikit-learn for undefined precision/F1.

    Returns:
        A dictionary of scalar metrics, a 2x2 confusion matrix ordered as
        rows/columns ``[fail, pass]``, and the threshold used.
    """
    actual_values = _to_numpy(y_true, "y_true")
    try:
        actual_numeric = actual_values.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("y_true must contain numeric binary labels 0 and 1") from error
    if not np.isfinite(actual_numeric).all() or not np.isin(actual_numeric, [0, 1]).all():
        raise ValueError("y_true must contain only encoded labels 0 (fail) and 1 (pass)")
    actual = actual_numeric.astype(np.int64)
    if not np.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be a finite probability between 0 and 1")

    probabilities: np.ndarray | None = None
    if y_score is not None:
        scores = _to_scores(y_score, score_type).astype(np.float64, copy=False)
        if len(scores) != len(actual):
            raise ValueError("y_score and y_true must contain the same number of records")
        if not np.isfinite(scores).all():
            raise ValueError("y_score must contain only finite values")
        normalized_score_type = score_type.strip().lower()
        if normalized_score_type in {"logit", "logits", "raw_logit", "raw_logits"}:
            probabilities = _sigmoid(scores)
        elif normalized_score_type in {"probability", "probabilities", "proba"}:
            probabilities = scores
            if ((probabilities < 0.0) | (probabilities > 1.0)).any():
                raise ValueError("probability scores must be between 0 and 1")
        else:
            raise ValueError("score_type must be 'probabilities' or 'logits'")
    elif score_type.strip().lower() not in {
        "probability", "probabilities", "proba", "logit", "logits", "raw_logit", "raw_logits"
    }:
        raise ValueError("score_type must be 'probabilities' or 'logits'")

    if y_pred is None:
        if probabilities is None:
            raise ValueError("provide y_pred or y_score")
        predicted = (probabilities >= float(threshold)).astype(np.int64)
    else:
        predicted_values = _to_numpy(y_pred, "y_pred")
        try:
            predicted_numeric = predicted_values.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("y_pred must contain numeric binary labels 0 and 1") from error
        if not np.isfinite(predicted_numeric).all() or not np.isin(predicted_numeric, [0, 1]).all():
            raise ValueError("y_pred must contain only encoded labels 0 and 1")
        predicted = predicted_numeric.astype(np.int64)
        if len(predicted) != len(actual):
            raise ValueError("y_pred and y_true must contain the same number of records")

    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    has_both_classes = len(np.unique(actual)) == 2
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, pos_label=1, zero_division=zero_division)),
        "recall": float(recall_score(actual, predicted, pos_label=1, zero_division=zero_division)),
        "f1_score": float(f1_score(actual, predicted, pos_label=1, zero_division=zero_division)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "roc_auc": (
            float(roc_auc_score(actual, probabilities))
            if probabilities is not None and has_both_classes
            else None
        ),
        "precision_fail": float(precision_score(actual, predicted, pos_label=0, zero_division=zero_division)),
        "recall_fail": float(recall_score(actual, predicted, pos_label=0, zero_division=zero_division)),
        "f1_fail": float(f1_score(actual, predicted, pos_label=0, zero_division=zero_division)),
        "confusion_matrix": matrix.tolist(),
        "threshold": float(threshold),
        "n_records": int(len(actual)),
    }
