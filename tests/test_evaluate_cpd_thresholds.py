"""Check the frozen CPD dev threshold and test predictions."""

import contextlib
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_cpd_thresholds as evaluation


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class ThresholdTests(unittest.TestCase):
    def test_equal_threshold_is_positive_and_ba_balances_classes(self):
        metrics = evaluation.compute_metrics(
            np.array([0, 0, 0, 1, 1]), np.array([0., 1., 1., 1., 2.]), 1.,
        )
        self.assertEqual([metrics[key] for key in ("TN", "FP", "FN", "TP")], [1, 2, 0, 2])
        self.assertAlmostEqual(metrics["FPR"], 2 / 3)
        self.assertEqual(metrics["TPR"], 1)
        self.assertAlmostEqual(metrics["Balanced_Accuracy"], 2 / 3)

    def test_tied_scores_at_zero_fpr_require_all_negative_threshold(self):
        threshold = evaluation.choose_threshold(np.array([0, 1]), np.array([2., 2.]), "fpr", 0)
        self.assertTrue(math.isinf(threshold))
        # No finite test score should become positive at the all-negative threshold.
        metrics = evaluation.compute_metrics(np.array([0, 1]), np.array([100., 200.]), threshold)
        self.assertEqual((metrics["FPR"], metrics["TPR"], metrics["Balanced_Accuracy"]), (0, 0, 0.5))

    def test_fpr_and_best_ba_select_different_dev_operating_points(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0., .9, .8, 1.])
        self.assertEqual(evaluation.choose_threshold(labels, scores, "fpr", .5), .8)
        self.assertEqual(evaluation.choose_threshold(labels, scores, "balanced_accuracy", .5), 1.)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dev = self.root / "dev" / "benchmark_2026_GEN_dev_per_sample.csv"
        self.tests = self.root / "test"
        self.output = self.root / "output"
        self.dev.parent.mkdir()
        self.tests.mkdir()
        self.test = self.tests / "benchmark_2026_GEN_test_per_sample.csv"
        evaluation.write_csv(self.dev, [
            {"label": label, "CPD": cpd, "CPD_k1": cpd}
            for label, cpd in [
                ("negative", 1), ("negative", 2),
                ("positive", 3), ("positive", 4),
            ]
        ])
        evaluation.write_csv(self.test, [
            {"label": label, "source_index": index, "CPD": cpd}
            for index, (label, cpd) in enumerate([
                ("negative", 3), ("negative", 1),
                ("positive", 2), ("positive", 4),
            ])
        ])

    def run_evaluation(self, *extra):
        argv = [
            "evaluate_cpd_thresholds.py", "--dev-file", str(self.dev),
            "--test-dir", str(self.tests), "--output-dir", str(self.output), *extra,
        ]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            evaluation.main()

    def test_default_cpd_threshold_predictions_and_no_test_label_leakage(self):
        self.run_evaluation()
        thresholds = load_csv(self.output / "dev_thresholds.csv")
        self.assertEqual(len(thresholds), 1)
        self.assertEqual({row["method"]: float(row["threshold"]) for row in thresholds},
                         {"CPD": 3.})
        metrics = {row["method"]: row for row in load_csv(self.output / "summary.csv")}
        self.assertEqual(set(metrics), {"CPD"})
        self.assertEqual([float(metrics["CPD"][key]) for key in ("FPR", "TPR", "Balanced_Accuracy")],
                         [.5, .5, .5])
        predictions = load_csv(self.output / self.test.name)
        self.assertEqual([row["outcome"] for row in predictions], ["FP", "TN", "FN", "TP"])
        self.assertEqual(set(predictions[0]),
                         {"label", "source_index", "CPD", "predicted_label", "correct", "outcome"})
        manifest = json.loads((self.output / "evaluation_config.json").read_text())
        self.assertEqual(manifest["calibration"]["method"], "CPD")

        # Changing every test label cannot influence the chosen dev threshold.
        rows = load_csv(self.test)
        for row in rows:
            row["label"] = "positive" if row["label"] == "negative" else "negative"
        evaluation.write_csv(self.test, rows)
        self.run_evaluation()
        self.assertEqual(load_csv(self.output / "dev_thresholds.csv"), thresholds)

    def test_legacy_single_method_and_dev_column_mapping(self):
        self.run_evaluation("--method", "CPD", "--dev-method", "CPD_k1")
        threshold = load_csv(self.output / "dev_thresholds.csv")[0]
        self.assertEqual((threshold["method"], threshold["dev_method"]), ("CPD", "CPD_k1"))
        predictions = load_csv(self.output / self.test.name)
        self.assertEqual(predictions[0]["outcome"], "FP")
        self.assertIn("predicted_label", predictions[0])

    def test_best_ba_mode_runs_for_cpd(self):
        self.run_evaluation("--selection", "balanced_accuracy")
        thresholds = load_csv(self.output / "dev_thresholds.csv")
        self.assertEqual(len(thresholds), 1)
        self.assertTrue(all(row["selection"] == "balanced_accuracy" for row in thresholds))
        self.assertTrue(all(row["target_dev_FPR"] == "" for row in thresholds))


if __name__ == "__main__":
    unittest.main()
