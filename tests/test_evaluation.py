import unittest

import numpy as np
import torch

from evaluation.evaluate import evaluate_binary_predictions


class EvaluateBinaryPredictionsTests(unittest.TestCase):
    def setUp(self):
        self.actual = np.array([0, 0, 1, 1])

    def test_logistic_regression_probabilities(self):
        scores = np.array([0.2, 0.7, 0.6, 0.9])
        result = evaluate_binary_predictions(self.actual, y_score=scores)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1_score"], 0.8)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(result["roc_auc"], 0.75)

    def test_ann_raw_logits(self):
        logits = torch.tensor([[-2.0], [2.0], [2.0], [3.0]])
        result = evaluate_binary_predictions(
            self.actual, y_score=logits, score_type="logits", threshold=0.5
        )
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(result["roc_auc"], 0.875)

    def test_logistic_predict_proba_two_column_output(self):
        probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.1, 0.9]])
        result = evaluate_binary_predictions(self.actual, y_score=probabilities)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(result["roc_auc"], 0.75)

    def test_predictions_without_scores_have_no_auc(self):
        result = evaluate_binary_predictions(self.actual, y_pred=[0, 0, 1, 1])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertIsNone(result["roc_auc"])

    def test_rejects_invalid_label_or_score_shapes(self):
        with self.assertRaises(ValueError):
            evaluate_binary_predictions([0, 2], y_pred=[0, 1])
        with self.assertRaises(ValueError):
            evaluate_binary_predictions([0, 1.5], y_pred=[0, 1])
        with self.assertRaises(ValueError):
            evaluate_binary_predictions(self.actual, y_score=[0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
