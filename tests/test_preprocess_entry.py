import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from datasets.load_dataset import load_dataset
from preprocessing import preprocess


class PreprocessLoadedDataTests(unittest.TestCase):
    def test_preprocessor_uses_loaded_frame_and_writes_only_to_temp_paths(self):
        source = load_dataset()
        with tempfile.TemporaryDirectory(prefix="student-performance-preprocess-") as temp:
            temp_root = Path(temp)
            processed_dir = temp_root / "processed"
            scaler_path = temp_root / "scaler.joblib"
            with (
                patch.object(preprocess, "ARTIFACTS_DIR", temp_root),
                patch.object(preprocess, "PROCESSED_DIR", processed_dir),
                patch.object(preprocess, "SCALER_PATH", scaler_path),
            ):
                report = preprocess.preprocess_dataset(source)

            self.assertEqual(report["split_sizes"], {"train": 700, "validation": 150, "test": 150})
            self.assertTrue(scaler_path.is_file())
            self.assertTrue(joblib.load(scaler_path).feature_names_in_.tolist() == [
                "study_hours", "attendance", "previous_marks"
            ])
            for split_name, expected_rows in (("train", 700), ("validation", 150), ("test", 150)):
                split = pd.read_csv(processed_dir / f"{split_name}.csv")
                self.assertEqual(len(split), expected_rows)
                self.assertEqual(split.columns.tolist(), [
                    "study_hours", "attendance", "previous_marks", "result"
                ])


if __name__ == "__main__":
    unittest.main()
