"""Train the ANN using training data and validation-loss early stopping.

The testing split is deliberately not loaded or accessed in this module.
Run from any working directory with the project's configured Python:
    python training/train.py
"""

from __future__ import annotations

import random
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
from torch import nn

from models.ann_model import StudentPerformanceANN


TRAIN_PATH = PROJECT_ROOT / "artifacts" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "artifacts" / "processed" / "validation.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.pth"
HISTORY_PATH = RESULTS_DIR / "training_history.csv"
LOSS_CURVE_PATH = RESULTS_DIR / "loss_curve.png"
FEATURE_COLUMNS = ["study_hours", "attendance", "previous_marks"]
TARGET_COLUMN = "result"
RANDOM_SEED = 42
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 10


def set_reproducible_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _load_xy(csv_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one processed split and return feature and label tensors."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {missing}")
    if frame[required].isna().any().any():
        raise ValueError(f"{csv_path.name} contains missing values")
    labels = set(frame[TARGET_COLUMN].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{csv_path.name} target must contain only 0 and 1")

    features = torch.tensor(
        frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32
    )
    targets = torch.tensor(
        frame[TARGET_COLUMN].to_numpy(dtype=np.float32).reshape(-1, 1),
        dtype=torch.float32,
    )
    return features, targets


def train_model() -> dict[str, object]:
    """Train, early-stop on validation loss, and save model/history/plot."""
    set_reproducible_seed()
    X_train, y_train = _load_xy(TRAIN_PATH)
    X_validation, y_validation = _load_xy(VALIDATION_PATH)

    model = StudentPerformanceANN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(X_train)
        train_loss = criterion(train_logits, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(X_validation)
            validation_loss = criterion(validation_logits, y_validation)

        train_loss_value = float(train_loss.detach().item())
        validation_loss_value = float(validation_loss.item())
        history.append(
            {
                "epoch": epoch,
                "training_loss": train_loss_value,
                "validation_loss": validation_loss_value,
            }
        )
        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} - "
            f"training loss: {train_loss_value:.6f} - "
            f"validation loss: {validation_loss_value:.6f}"
        )

        if validation_loss_value < best_validation_loss:
            best_validation_loss = validation_loss_value
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(
                    f"Early stopping: validation loss did not improve for "
                    f"{PATIENCE} consecutive epochs."
                )
                break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(HISTORY_PATH, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(history_frame["epoch"], history_frame["training_loss"], label="Training loss")
    plt.plot(history_frame["epoch"], history_frame["validation_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCEWithLogitsLoss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=150)
    plt.close()

    summary = {
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_model_path": str(BEST_MODEL_PATH),
        "history_path": str(HISTORY_PATH),
        "loss_curve_path": str(LOSS_CURVE_PATH),
    }
    print(f"Best validation loss: {best_validation_loss:.6f} at epoch {best_epoch}")
    print(f"Best model weights saved to: {BEST_MODEL_PATH}")
    print(f"Training history saved to: {HISTORY_PATH}")
    print(f"Loss curve saved to: {LOSS_CURVE_PATH}")
    return summary


if __name__ == "__main__":
    train_model()
