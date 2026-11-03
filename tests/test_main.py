import unittest
import contextlib
import io
from unittest.mock import patch

import pandas as pd

import main as project_main


class MainEntryPointTests(unittest.TestCase):
    def test_prepare_loads_dataset_before_preprocessing(self):
        events = []
        sample = pd.DataFrame(
            {
                "study_hours": [2.0],
                "attendance": [70.0],
                "previous_marks": [60.0],
                "result": ["pass"],
            }
        )

        def load():
            events.append("load")
            return sample

        def preprocess(loaded_data):
            self.assertEqual(events, ["load"])
            self.assertIs(loaded_data, sample)
            events.append("preprocess")
            return {
                "feature_names": ["study_hours", "attendance", "previous_marks"],
                "split_sizes": {"train": 1, "validation": 0, "test": 0},
                "class_counts": {"train": {1: 1}, "validation": {}, "test": {}},
                "scaler_path": "scaler.joblib",
                "processed_dir": "processed",
            }

        with patch.object(project_main, "load_dataset", side_effect=load), patch.object(
            project_main, "preprocess_dataset", side_effect=preprocess
        ):
            self.assertEqual(project_main.prepare_command(), 0)
        self.assertEqual(events, ["load", "preprocess"])

    def test_inspect_command_only_loads(self):
        sample = pd.DataFrame(
            {
                "study_hours": [2.0],
                "attendance": [70.0],
                "previous_marks": [60.0],
                "result": ["pass"],
            }
        )
        with patch.object(project_main, "load_dataset", return_value=sample), patch.object(
            project_main, "preprocess_dataset"
        ) as preprocess:
            self.assertEqual(project_main.inspect_command(), 0)
        preprocess.assert_not_called()

    def test_parser_requires_explicit_command(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                project_main.build_parser().parse_args([])


if __name__ == "__main__":
    unittest.main()
