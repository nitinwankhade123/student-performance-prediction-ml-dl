"""Generate a synthetic student performance dataset for pipeline demonstrations.

The generated records do not describe real students and must not be interpreted
as evidence about real-world student performance. Labels are assigned by a
transparent educational demonstration rule, not by observed examination data.

Label rule (pass threshold = 50):
    score = 0.40 * previous_marks
          + 0.30 * attendance
          + 0.30 * min(study_hours * 10, 100)
          + normally distributed noise (mean 0, standard deviation 8)

``result`` is ``pass`` when score >= 50 and ``fail`` otherwise. The noise
creates overlap near the threshold. The values are synthetic, sampled from
reasonable ranges, and are not calibrated to any institution or population.
"""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
DEFAULT_RECORDS = 1000
OUTPUT_PATH = Path(__file__).with_name("student_data.csv")


def generate_synthetic_data(
    n_records: int = DEFAULT_RECORDS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return reproducible synthetic records with four project columns."""
    if n_records <= 0:
        raise ValueError("n_records must be a positive integer")

    rng = np.random.default_rng(seed)
    study_hours = rng.uniform(0.0, 10.0, n_records).round(1)
    attendance = rng.uniform(40.0, 100.0, n_records).round(1)
    previous_marks = rng.uniform(20.0, 100.0, n_records).round(1)

    score = (
        0.40 * previous_marks
        + 0.30 * attendance
        + 0.30 * np.minimum(study_hours * 10.0, 100.0)
        + rng.normal(0.0, 8.0, n_records)
    )
    result = np.where(score >= 50.0, "pass", "fail")

    return pd.DataFrame(
        {
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_marks": previous_marks,
            "result": result,
        }
    )


def save_synthetic_data(
    output_path: Path = OUTPUT_PATH,
    n_records: int = DEFAULT_RECORDS,
    seed: int = SEED,
) -> Path:
    """Generate and save the synthetic CSV, returning its resolved path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_data(n_records=n_records, seed=seed).to_csv(
        output_path, index=False
    )
    return output_path.resolve()


if __name__ == "__main__":
    saved_path = save_synthetic_data()
    print(f"Saved {DEFAULT_RECORDS} synthetic demonstration records to {saved_path}")
