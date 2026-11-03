"""Prepare stratified, scaled train/validation/test data for the ANN project.

The scaler is fitted on training features only. The saved CSV files contain
standardized input columns plus the integer ``result`` target (pass=1, fail=0).
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "datasets" / "student_data.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
SCALER_PATH = ARTIFACTS_DIR / "standard_scaler.joblib"
FEATURE_COLUMNS = ["study_hours", "attendance", "previous_marks"]
TARGET_COLUMN = "result"
SEED = 42


def preprocess_dataset(data: pd.DataFrame | None = None) -> dict[str, object]:
    """Stratify, scale, save splits and return a concise report.

    When ``data`` is supplied, that already-loaded frame is used directly.
    Otherwise the project's default CSV is loaded, preserving the original API.
    """
    if data is None:
        if not DATASET_PATH.is_file():
            raise FileNotFoundError(f"Input dataset not found: {DATASET_PATH}")
        frame = pd.read_csv(DATASET_PATH)
    else:
        frame = data.copy()
    expected = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame[expected].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns")

    X = frame[FEATURE_COLUMNS].astype(float)
    label_map = {"fail": 0, "pass": 1}
    normalized_result = frame[TARGET_COLUMN].astype(str).str.strip().str.lower()
    unexpected_labels = sorted(set(normalized_result) - set(label_map))
    if unexpected_labels:
        raise ValueError(f"Unexpected result labels: {unexpected_labels}")
    y = normalized_result.map(label_map).astype("int64")

    # First hold out 30%, then split that equally to produce 15% validation
    # and 15% test. Both stages preserve the target distribution.
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.50,
        random_state=SEED,
        stratify=y_holdout,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    splits = {
        "train": (X_train_scaled, y_train),
        "validation": (X_validation_scaled, y_validation),
        "test": (X_test_scaled, y_test),
    }
    counts: dict[str, dict[int, int]] = {}
    sizes: dict[str, int] = {}
    for split_name, (features, targets) in splits.items():
        split_frame = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        split_frame[TARGET_COLUMN] = targets.to_numpy()
        split_frame.to_csv(PROCESSED_DIR / f"{split_name}.csv", index=False)
        counts[split_name] = {
            int(label): int(count)
            for label, count in targets.value_counts().sort_index().items()
        }
        sizes[split_name] = len(split_frame)

    # The transformed training data should have approximately zero mean and
    # unit population standard deviation per feature.
    scale_check = {
        "train_feature_means": X_train_scaled.mean(axis=0).tolist(),
        "train_feature_stds": X_train_scaled.std(axis=0).tolist(),
        "scaler_fitted_on_training_only": True,
    }
    return {
        "input_path": str(DATASET_PATH),
        "feature_names": FEATURE_COLUMNS,
        "split_sizes": sizes,
        "class_counts": counts,
        "scaler_path": str(SCALER_PATH),
        "processed_dir": str(PROCESSED_DIR),
        "scale_check": scale_check,
    }


if __name__ == "__main__":
    result = preprocess_dataset()
    print(f"Input dataset: {result['input_path']}")
    print(f"Input feature names: {result['feature_names']}")
    for name in ("train", "validation", "test"):
        print(
            f"{name.title()} records: {result['split_sizes'][name]}; "
            f"class counts (0=fail, 1=pass): {result['class_counts'][name]}"
        )
    scale_check = result["scale_check"]
    print(f"Training scaled feature means: {scale_check['train_feature_means']}")
    print(f"Training scaled feature stds: {scale_check['train_feature_stds']}")
    print("Scaling: fitted on training data only; applied to all three splits.")
    print(f"Saved scaler: {result['scaler_path']}")
    print(f"Saved processed CSV splits: {result['processed_dir']}")
