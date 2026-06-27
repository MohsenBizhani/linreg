"""
tests/test_scaler.py
====================
Unit tests for linreg._scaler.ZScoreScaler.

Covers:
- fit_transform / transform on 2-D matrices
- fit_transform_1d / transform_1d / inverse_transform_1d on 1-D vectors
- Statistics correctness (mean ≈ 0, std ≈ 1 after scaling)
- No data leakage: test split uses train statistics
- Error paths: transform before fit, inverse before fit
- Round-trip: inverse_transform_1d(fit_transform_1d(y)) ≈ y
- __repr__ string
"""

import unittest

from linreg._scaler import ZScoreScaler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approx_equal(a, b, tol=1e-9):
    return abs(a - b) < tol


def _col_mean(matrix, j):
    return sum(row[j] for row in matrix) / len(matrix)


def _col_std(matrix, j):
    import math

    mu = _col_mean(matrix, j)
    var = sum((row[j] - mu) ** 2 for row in matrix) / len(matrix)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# 2-D interface
# ---------------------------------------------------------------------------


class TestZScoreScaler2D(unittest.TestCase):
    def setUp(self):
        # Deliberately asymmetric so mean ≠ 0 and std ≠ 1 before scaling
        self.X_train = [
            [2.0, 10.0],
            [4.0, 20.0],
            [6.0, 30.0],
            [8.0, 40.0],
        ]
        self.X_test = [
            [1.0, 5.0],
            [9.0, 45.0],
        ]

    def test_fit_transform_returns_same_shape(self):
        scaler = ZScoreScaler()
        result = scaler.fit_transform(self.X_train)
        self.assertEqual(len(result), len(self.X_train))
        self.assertEqual(len(result[0]), len(self.X_train[0]))

    def test_scaled_columns_have_zero_mean(self):
        scaler = ZScoreScaler()
        X_s = scaler.fit_transform(self.X_train)
        for j in range(len(self.X_train[0])):
            col_mean = _col_mean(X_s, j)
            self.assertAlmostEqual(
                col_mean,
                0.0,
                places=10,
                msg=f"Column {j} mean should be 0 after scaling, got {col_mean}",
            )

    def test_scaled_columns_have_unit_std(self):
        scaler = ZScoreScaler()
        X_s = scaler.fit_transform(self.X_train)
        for j in range(len(self.X_train[0])):
            col_std = _col_std(X_s, j)
            self.assertAlmostEqual(
                col_std,
                1.0,
                places=10,
                msg=f"Column {j} std should be 1 after scaling, got {col_std}",
            )

    def test_statistics_stored_correctly(self):
        scaler = ZScoreScaler()
        scaler.fit_transform(self.X_train)
        # Column 0: values [2,4,6,8] → mean=5, std=sqrt(5)
        self.assertAlmostEqual(scaler.means_[0], 5.0)
        # Column 1: values [10,20,30,40] → mean=25
        self.assertAlmostEqual(scaler.means_[1], 25.0)

    def test_transform_uses_train_statistics(self):
        scaler = ZScoreScaler()
        scaler.fit_transform(self.X_train)
        X_test_s = scaler.transform(self.X_test)
        # Manually verify first sample: x0=(1-5)/std0, x1=(5-25)/std1
        import math

        std0 = math.sqrt(sum((v - 5.0) ** 2 for v in [2, 4, 6, 8]) / 4)
        expected_00 = (1.0 - 5.0) / std0
        self.assertAlmostEqual(X_test_s[0][0], expected_00)

    def test_transform_before_fit_raises(self):
        scaler = ZScoreScaler()
        with self.assertRaises(RuntimeError):
            scaler.transform(self.X_test)

    def test_refit_updates_statistics(self):
        scaler = ZScoreScaler()
        scaler.fit_transform(self.X_train)
        old_mean = scaler.means_[0]
        new_X = [[100.0, 200.0], [200.0, 400.0]]
        scaler.fit_transform(new_X)
        self.assertNotAlmostEqual(scaler.means_[0], old_mean)


# ---------------------------------------------------------------------------
# 1-D interface
# ---------------------------------------------------------------------------


class TestZScoreScaler1D(unittest.TestCase):
    def setUp(self):
        self.y_train = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.y_test = [15.0, 35.0, 55.0]

    def test_fit_transform_1d_zero_mean(self):
        scaler = ZScoreScaler()
        y_s = scaler.fit_transform_1d(self.y_train)
        mean_s = sum(y_s) / len(y_s)
        self.assertAlmostEqual(mean_s, 0.0, places=10)

    def test_fit_transform_1d_unit_std(self):
        import math

        scaler = ZScoreScaler()
        y_s = scaler.fit_transform_1d(self.y_train)
        mean_s = sum(y_s) / len(y_s)
        std_s = math.sqrt(sum((v - mean_s) ** 2 for v in y_s) / len(y_s))
        self.assertAlmostEqual(std_s, 1.0, places=10)

    def test_transform_1d_uses_train_statistics(self):
        scaler = ZScoreScaler()
        scaler.fit_transform_1d(self.y_train)
        y_test_s = scaler.transform_1d(self.y_test)
        # mean of y_train = 30, manually check first value
        import math

        sigma = math.sqrt(sum((v - 30.0) ** 2 for v in self.y_train) / 5)
        expected = (15.0 - 30.0) / sigma
        self.assertAlmostEqual(y_test_s[0], expected)

    def test_transform_1d_before_fit_raises(self):
        scaler = ZScoreScaler()
        with self.assertRaises(RuntimeError):
            scaler.transform_1d(self.y_test)

    def test_inverse_transform_1d_before_fit_raises(self):
        scaler = ZScoreScaler()
        with self.assertRaises(RuntimeError):
            scaler.inverse_transform_1d([0.0, 1.0])

    def test_round_trip(self):
        """inverse_transform_1d(fit_transform_1d(y)) must recover y exactly."""
        scaler = ZScoreScaler()
        y_s = scaler.fit_transform_1d(self.y_train)
        y_recovered = scaler.inverse_transform_1d(y_s)
        for orig, rec in zip(self.y_train, y_recovered):
            self.assertAlmostEqual(orig, rec, places=10)

    def test_round_trip_on_test_set(self):
        """Scaling test set and inverting must recover the original test values."""
        scaler = ZScoreScaler()
        scaler.fit_transform_1d(self.y_train)
        y_test_s = scaler.transform_1d(self.y_test)
        y_recovered = scaler.inverse_transform_1d(y_test_s)
        for orig, rec in zip(self.y_test, y_recovered):
            self.assertAlmostEqual(orig, rec, places=10)

    def test_constant_target_no_division_error(self):
        """Constant y should not raise; std sentinel = 1.0."""
        scaler = ZScoreScaler()
        y_s = scaler.fit_transform_1d([5.0, 5.0, 5.0])
        # All values should scale to 0.0 (value - mean) / 1.0
        for v in y_s:
            self.assertAlmostEqual(v, 0.0)


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestZScoreScalerRepr(unittest.TestCase):
    def test_unfitted_repr(self):
        s = ZScoreScaler()
        r = repr(s)
        self.assertIn("fitted=False", r)
        self.assertIn("n_features=0", r)

    def test_fitted_repr(self):
        s = ZScoreScaler()
        s.fit_transform([[1.0, 2.0], [3.0, 4.0]])
        r = repr(s)
        self.assertIn("fitted=True", r)
        self.assertIn("n_features=2", r)


if __name__ == "__main__":
    unittest.main()
