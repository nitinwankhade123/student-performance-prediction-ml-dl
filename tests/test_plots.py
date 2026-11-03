import unittest
from pathlib import Path

import pandas as pd

from visualization.plots import (
    RESULTS_DIR,
    plot_confusion_matrix,
    plot_loss_curves,
    plot_roc_curve,
)


class PlotHelpersTests(unittest.TestCase):
    def setUp(self):
        self.outputs = [
            RESULTS_DIR / "_unit_test_loss_curve.png",
            RESULTS_DIR / "_unit_test_confusion_matrix.png",
            RESULTS_DIR / "_unit_test_roc_curve.png",
            RESULTS_DIR / "_unit_test_existing.png",
        ]
        for path in self.outputs:
            path.unlink(missing_ok=True)

    def tearDown(self):
        for path in self.outputs:
            path.unlink(missing_ok=True)

    def test_loss_curve_saves_under_results(self):
        history = pd.DataFrame(
            {"epoch": [1, 2], "training_loss": [0.8, 0.6], "validation_loss": [0.9, 0.7]}
        )
        saved = plot_loss_curves(history, self.outputs[0].name)
        self.assertEqual(saved.parent, RESULTS_DIR.resolve())
        self.assertTrue(saved.is_file())

    def test_confusion_matrix_plot_saves(self):
        saved = plot_confusion_matrix(
            [0, 0, 1, 1], [0, 1, 0, 1], output_path=self.outputs[1].name
        )
        self.assertTrue(saved.is_file())

    def test_roc_plot_saves_and_reports_auc(self):
        saved, roc_auc = plot_roc_curve(
            [0, 0, 1, 1], [0.2, 0.7, 0.6, 0.9], output_path=self.outputs[2].name
        )
        self.assertTrue(saved.is_file())
        self.assertAlmostEqual(roc_auc, 0.75)

    def test_existing_plot_is_not_overwritten_by_default(self):
        target = self.outputs[3]
        target.write_bytes(b"keep existing file")
        history = pd.DataFrame(
            {"epoch": [1], "training_loss": [0.8], "validation_loss": [0.9]}
        )
        with self.assertRaises(FileExistsError):
            plot_loss_curves(history, target.name)
        self.assertEqual(target.read_bytes(), b"keep existing file")

    def test_overwrite_requires_and_honors_explicit_flag(self):
        target = self.outputs[3]
        target.write_bytes(b"replace only by request")
        history = pd.DataFrame(
            {"epoch": [1], "training_loss": [0.8], "validation_loss": [0.9]}
        )
        plot_loss_curves(history, target.name, overwrite=True)
        self.assertNotEqual(target.read_bytes(), b"replace only by request")

    def test_rejects_output_outside_results(self):
        outside = Path(__file__).resolve().parent / "not_allowed.png"
        with self.assertRaises(ValueError):
            plot_confusion_matrix(matrix=[[1, 0], [0, 1]], output_path=outside)
        self.assertFalse(outside.exists())


if __name__ == "__main__":
    unittest.main()
