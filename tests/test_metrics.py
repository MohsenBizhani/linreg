"""
tests/test_metrics.py
=====================
Unit tests for linreg._metrics:
    mse, rmse, mae, r2_score

Each test verifies a hand-computed expected value so the tests serve as
living documentation of the exact formula in use.
"""

import math
import unittest

from linreg._metrics import mae, mse, r2_score, rmse


class TestMSE(unittest.TestCase):
    def test_perfect_predictions(self):
        y = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(mse(y, y), 0.0)

    def test_known_value(self):
        # errors: [1, -1, 1] → squared: [1, 1, 1] → mean = 1.0
        self.assertAlmostEqual(mse([2, 4, 6], [3, 3, 7]), 1.0)

    def test_symmetry(self):
        # MSE(a, b) == MSE(b, a) because errors are squared
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.5, 2.5, 3.5]
        self.assertAlmostEqual(mse(y_true, y_pred), mse(y_pred, y_true))

    def test_non_negative(self):
        self.assertGreaterEqual(mse([0, 1, 2], [3, 4, 5]), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mse([1, 2, 3], [1, 2])

    def test_single_sample(self):
        self.assertAlmostEqual(mse([3.0], [5.0]), 4.0)


class TestRMSE(unittest.TestCase):
    def test_perfect_predictions(self):
        y = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(rmse(y, y), 0.0)

    def test_known_value(self):
        # MSE = 1.0 → RMSE = 1.0
        self.assertAlmostEqual(rmse([2, 4, 6], [3, 3, 7]), 1.0)

    def test_rmse_equals_sqrt_mse(self):
        y_true = [1.0, 2.0, 5.0, 8.0]
        y_pred = [1.5, 3.0, 4.0, 9.0]
        self.assertAlmostEqual(rmse(y_true, y_pred), math.sqrt(mse(y_true, y_pred)))

    def test_non_negative(self):
        self.assertGreaterEqual(rmse([0, 1], [1, 0]), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            rmse([1, 2], [1])


class TestMAE(unittest.TestCase):
    def test_perfect_predictions(self):
        y = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(mae(y, y), 0.0)

    def test_known_value(self):
        # absolute errors: [1, 1, 1] → mean = 1.0
        self.assertAlmostEqual(mae([2, 4, 6], [3, 3, 7]), 1.0)

    def test_single_sample(self):
        self.assertAlmostEqual(mae([0.0], [5.0]), 5.0)

    def test_non_negative(self):
        self.assertGreaterEqual(mae([1, 2, 3], [4, 5, 6]), 0.0)

    def test_mae_le_rmse(self):
        # MAE ≤ RMSE always holds (Cauchy-Schwarz)
        y_true = [1.0, 2.0, 5.0, 8.0]
        y_pred = [1.5, 3.0, 4.0, 9.0]
        self.assertLessEqual(mae(y_true, y_pred), rmse(y_true, y_pred))

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mae([1, 2, 3], [1, 2])


class TestR2Score(unittest.TestCase):
    def test_perfect_predictions_returns_one(self):
        y = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(r2_score(y, y), 1.0)

    def test_mean_predictions_returns_zero(self):
        # Predicting the mean for every sample → R² = 0
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean_val = sum(y_true) / len(y_true)
        y_pred = [mean_val] * len(y_true)
        self.assertAlmostEqual(r2_score(y_true, y_pred), 0.0)

    def test_worse_than_mean_is_negative(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [10.0, 20.0, 30.0]  # wildly off
        self.assertLess(r2_score(y_true, y_pred), 0.0)

    def test_known_value(self):
        # y_true = [1,2,3], y_pred = [1,2,3] shifted by 0.5 each
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.5, 2.5, 3.5]
        # ss_res = 3 * 0.25 = 0.75;  mean=2, ss_tot = 2;  R² = 1 - 0.75/2 = 0.625
        self.assertAlmostEqual(r2_score(y_true, y_pred), 0.625)

    def test_constant_target_returns_one(self):
        # ss_tot = 0 → sentinel: return 1.0
        y_true = [5.0, 5.0, 5.0]
        y_pred = [5.0, 5.0, 5.0]
        self.assertAlmostEqual(r2_score(y_true, y_pred), 1.0)

    def test_between_negative_inf_and_one(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 2.1, 3.1]
        score = r2_score(y_true, y_pred)
        self.assertLessEqual(score, 1.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            r2_score([1, 2, 3], [1, 2])


if __name__ == "__main__":
    unittest.main()
