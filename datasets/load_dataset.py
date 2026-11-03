"""Load and inspect the synthetic student performance CSV with Pandas."""

from pathlib import Path

import pandas as pd


DEFAULT_DATASET_PATH = Path(__file__).with_name("student_data.csv")
EXPECTED_COLUMNS = ["study_hours", "attendance", "previous_marks", "result"]


def load_dataset(csv_path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load a CSV and verify its required project columns are present."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    data = pd.read_csv(csv_path)
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return data


def inspect_dataset(data: pd.DataFrame) -> dict[str, object]:
    """Return dimensions, columns, missing values and duplicate count."""
    return {
        "dimensions": data.shape,
        "columns": list(data.columns),
        "missing_values_by_column": data.isna().sum().to_dict(),
        "duplicate_records": int(data.duplicated().sum()),
    }


if __name__ == "__main__":
    dataset = load_dataset()
    report = inspect_dataset(dataset)
    print(f"Loaded synthetic demonstration dataset: {DEFAULT_DATASET_PATH.resolve()}")
    print(f"Dimensions: {report['dimensions']}")
    print(f"Columns: {report['columns']}")
    print(f"Missing values by column: {report['missing_values_by_column']}")
    print(f"Duplicate records: {report['duplicate_records']}")
    print("Preview (first 5 rows):")
    print(dataset.head().to_string(index=False))
