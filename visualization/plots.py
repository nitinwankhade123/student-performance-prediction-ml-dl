"""Reusable, non-destructive plots for the student-result project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix as sklearn_confusion_matrix, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def _output_path(path: str | Path | None, default_name: str, overwrite: bool) -> Path:
    """Resolve a destination inside results/ and protect existing files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    requested = Path(default_name) if path is None else Path(path)
    if requested.is_absolute():
        destination = requested
    elif requested.parts and requested.parts[0] == "results":
        destination = PROJECT_ROOT / requested
    else:
        destination = RESULTS_DIR / requested
    destination = destination.resolve()
    try:
        destination.relative_to(RESULTS_DIR.resolve())
    except ValueError as error:
        raise ValueError("Plot output must be saved inside the project results/ directory") from error
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing plot: {destination}. "
            "Pass overwrite=True to replace it explicitly."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _save_figure(fig: plt.Figure, path: str | Path | None, default_name: str, overwrite: bool) -> Path:
    """Save and close a figure while preserving existing plots by default."""
    try:
        destination = _output_path(path, default_name, overwrite)
        fig.savefig(destination, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    return destination


def _as_numpy(values: Any) -> np.ndarray:
    """Convert NumPy-like and Torch-like inference outputs for plotting."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def plot_loss_curves(
    history: pd.DataFrame | list[dict[str, Any]] | str | Path,
    output_path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Plot epoch-wise training and validation losses to ``results/``.

    ``history`` may be a DataFrame, a list of row dictionaries, or a CSV path
    containing ``epoch``, ``training_loss``, and ``validation_loss`` columns.
    """
    if isinstance(history, (str, Path)):
        frame = pd.read_csv(history)
    else:
        frame = pd.DataFrame(history)
    required = {"epoch", "training_loss", "validation_loss"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Loss history is missing required columns: {missing}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Loss history must contain non-empty, complete values")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame["epoch"], frame["training_loss"], label="Training loss")
    ax.plot(frame["epoch"], frame["validation_loss"], label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, output_path, "loss_curve.png", overwrite)


def plot_confusion_matrix(
    y_true: Any | None = None,
    y_pred: Any | None = None,
    *,
    matrix: Any | None = None,
    class_names: tuple[str, str] = ("fail", "pass"),
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Draw a 2×2 confusion matrix ordered as fail=0, pass=1."""
    if matrix is None:
        if y_true is None or y_pred is None:
            raise ValueError("provide either matrix or both y_true and y_pred")
        actual = _as_numpy(y_true).reshape(-1)
        predicted = _as_numpy(y_pred).reshape(-1)
        if len(actual) != len(predicted):
            raise ValueError("y_true and y_pred must contain the same number of records")
        values = sklearn_confusion_matrix(actual, predicted, labels=[0, 1])
    else:
        values = np.asarray(matrix)
    if values.shape != (2, 2):
        raise ValueError("confusion matrix must have shape [2, 2]")
    if len(class_names) != 2:
        raise ValueError("class_names must contain exactly two labels")

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(values, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="Actual label",
        title="Confusion matrix",
    )
    midpoint = float(values.max()) / 2.0 if values.size else 0.0
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > midpoint else "black",
            )
    fig.tight_layout()
    return _save_figure(fig, output_path, "confusion_matrix.png", overwrite)


def plot_roc_curve(
    y_true: Any,
    y_score: Any,
    *,
    model_name: str = "Model",
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, float]:
    """Plot an ROC curve from probabilities or continuous model scores.

    ANN logits and Logistic Regression probabilities are both valid scores;
    ROC ranking is invariant under the ANN sigmoid transformation.
    """
    actual = _as_numpy(y_true).reshape(-1)
    scores = _as_numpy(y_score)
    if scores.ndim == 2 and scores.shape[1] == 2:
        scores = scores[:, 1]
    elif scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores[:, 0]
    if actual.ndim != 1 or scores.ndim != 1 or len(actual) != len(scores):
        raise ValueError("y_true and y_score must be equal-length one-dimensional arrays")
    if len(actual) == 0 or not np.isfinite(scores).all():
        raise ValueError("ROC inputs must be non-empty and y_score must be finite")
    if not np.isin(actual, [0, 1]).all() or len(np.unique(actual)) != 2:
        raise ValueError("ROC curve requires both binary classes encoded as 0 and 1")

    fpr, tpr, _ = roc_curve(actual, scores, pos_label=1)
    roc_auc = float(auc(fpr, tpr))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = _save_figure(fig, output_path, "roc_curve.png", overwrite)
    return path, roc_auc
