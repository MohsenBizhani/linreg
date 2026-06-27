"""
tests/test_model.py
===================
Unit tests for linreg._model.LinearRegression.

Covers:
- Initialisation / defaults
- fit() returns self (method chaining)
- predict() before fit raises RuntimeError
- Input validation errors
- Loss decreases monotonically on a noiseless problem
- Early stopping fires before max_iter
- Noiseless recovery: weights and bias converge to true values
- score() on perfect and constant predictions
- summary() string content
- verbose=False suppresses output
- Re-fitting resets state cleanly
"""

import io
import sys
import unittest

from linreg._model import LinearRegression
from linreg._scaler import ZScoreScaler

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_noiseless_data(n=100):
    """y = 2*x1 + 3*x2 + 5, no noise. Returns raw (unscaled) X and y."""
    X, y = [], []
    step = 1.0
    for i in range(n):
        x1 = step * i
        x2 = step * i * 0.5
        X.append([x1, x2])
        y.append(2 * x1 + 3 * x2 + 5)
    return X, y


def _scale(X_tr, y_tr, X_te, y_te):
    """Z-score scale both X and y; return scaled arrays + y scaler."""
    xs = ZScoreScaler()
    X_tr_s = xs.fit_transform(X_tr)
    X_te_s = xs.transform(X_te)

    ys = ZScoreScaler()
    y_tr_s = ys.fit_transform_1d(y_tr)
    y_te_s = ys.transform_1d(y_te)

    return X_tr_s, X_te_s, y_tr_s, y_te_s, ys


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestLinearRegressionInit(unittest.TestCase):
    def test_defaults(self):
        m = LinearRegression()
        self.assertAlmostEqual(m.lr, 0.01)
        self.assertEqual(m.max_iter, 10_000)
        self.assertAlmostEqual(m.tol, 1e-6)
        self.assertTrue(m.verbose)

    def test_custom_params(self):
        m = LinearRegression(lr=0.1, max_iter=500, tol=1e-4, verbose=False)
        self.assertAlmostEqual(m.lr, 0.1)
        self.assertEqual(m.max_iter, 500)
        self.assertAlmostEqual(m.tol, 1e-4)
        self.assertFalse(m.verbose)

    def test_initial_weights_empty(self):
        m = LinearRegression()
        self.assertEqual(m.weights, [])
        self.assertAlmostEqual(m.bias, 0.0)
        self.assertEqual(m.loss_history, [])
        self.assertEqual(m.n_iter_, 0)

    def test_repr_contains_params(self):
        m = LinearRegression(lr=0.05, max_iter=200)
        r = repr(m)
        self.assertIn("0.05", r)
        self.assertIn("200", r)


# ---------------------------------------------------------------------------
# fit() — return value and method chaining
# ---------------------------------------------------------------------------


class TestFitReturnValue(unittest.TestCase):
    def test_fit_returns_self(self):
        X = [[1.0], [2.0], [3.0]]
        y = [2.0, 4.0, 6.0]
        m = LinearRegression(verbose=False)
        result = m.fit(X, y)
        self.assertIs(result, m)

    def test_method_chaining(self):
        X = [[1.0], [2.0], [3.0]]
        y = [2.0, 4.0, 6.0]
        preds = LinearRegression(verbose=False).fit(X, y).predict(X)
        self.assertEqual(len(preds), 3)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestFitValidation(unittest.TestCase):
    def test_empty_X_raises(self):
        with self.assertRaises(ValueError):
            LinearRegression(verbose=False).fit([], [])

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            LinearRegression(verbose=False).fit([[1.0], [2.0]], [1.0])

    def test_ragged_X_raises(self):
        with self.assertRaises(ValueError):
            LinearRegression(verbose=False).fit([[1.0, 2.0], [3.0]], [1.0, 2.0])

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            LinearRegression().predict([[1.0, 2.0]])


# ---------------------------------------------------------------------------
# Loss behaviour
# ---------------------------------------------------------------------------


class TestLossBehaviour(unittest.TestCase):
    def test_loss_decreases(self):
        """Loss must be non-increasing on every step for a convex MSE surface."""
        X_raw = [[float(i)] for i in range(50)]
        y_raw = [2.0 * i + 1.0 for i in range(50)]
        xs = ZScoreScaler()
        ys = ZScoreScaler()
        X = xs.fit_transform(X_raw)
        y = ys.fit_transform_1d(y_raw)
        m = LinearRegression(lr=0.01, max_iter=200, tol=0.0, verbose=False)
        m.fit(X, y)
        for i in range(1, len(m.loss_history)):
            self.assertLessEqual(
                m.loss_history[i],
                m.loss_history[i - 1] + 1e-9,
                msg=f"Loss increased at iteration {i}: "
                f"{m.loss_history[i - 1]:.6f} → {m.loss_history[i]:.6f}",
            )

    def test_loss_history_length_matches_n_iter(self):
        X_raw = [[float(i)] for i in range(20)]
        y_raw = [float(i) + 1.0 for i in range(20)]  # +1 avoids all-zero y
        xs = ZScoreScaler()
        ys = ZScoreScaler()
        X = xs.fit_transform(X_raw)
        y = ys.fit_transform_1d(y_raw)
        m = LinearRegression(lr=0.01, max_iter=50, tol=0.0, verbose=False)
        m.fit(X, y)
        self.assertEqual(len(m.loss_history), m.n_iter_)

    def test_initial_loss_equals_variance_of_y(self):
        """With zero-init weights, first MSE = mean((0 - y_i)^2) = mean(y^2)."""
        y = [1.0, 2.0, 3.0, 4.0]
        X = [[0.0]] * 4  # zero features → forward pass = bias = 0
        m = LinearRegression(lr=0.0, max_iter=1, tol=0.0, verbose=False)
        m.fit(X, y)
        expected = sum(v**2 for v in y) / len(y)
        self.assertAlmostEqual(m.loss_history[0], expected)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class TestEarlyStopping(unittest.TestCase):
    def test_early_stop_fires_before_max_iter(self):
        X_raw = [[float(i)] for i in range(80)]
        y_raw = [3.0 * i + 2.0 for i in range(80)]
        xs = ZScoreScaler()
        ys = ZScoreScaler()
        X = xs.fit_transform(X_raw)
        y = ys.fit_transform_1d(y_raw)
        m = LinearRegression(lr=0.01, max_iter=10_000, tol=1e-6, verbose=False)
        m.fit(X, y)
        self.assertLess(m.n_iter_, 10_000)

    def test_tol_zero_runs_all_iterations(self):
        X = [[float(i)] for i in range(10)]
        y = [float(i) for i in range(10)]
        m = LinearRegression(lr=0.001, max_iter=50, tol=0.0, verbose=False)
        m.fit(X, y)
        self.assertEqual(m.n_iter_, 50)

    def test_n_iter_set_after_fit(self):
        X = [[1.0], [2.0], [3.0]]
        y = [1.0, 2.0, 3.0]
        m = LinearRegression(verbose=False)
        m.fit(X, y)
        self.assertGreater(m.n_iter_, 0)


# ---------------------------------------------------------------------------
# Convergence / weight recovery
# ---------------------------------------------------------------------------


class TestConvergence(unittest.TestCase):
    def test_noiseless_high_r2(self):
        """On noiseless data, model should reach R² > 0.999."""
        X_raw, y_raw = _make_noiseless_data(n=120)
        split = 96
        X_tr, y_tr = X_raw[:split], y_raw[:split]
        X_te, y_te = X_raw[split:], y_raw[split:]

        X_tr_s, X_te_s, y_tr_s, y_te_s, _ = _scale(X_tr, y_tr, X_te, y_te)

        m = LinearRegression(lr=0.01, max_iter=10_000, tol=1e-9, verbose=False)
        m.fit(X_tr_s, y_tr_s)
        self.assertGreater(m.score(X_te_s, y_te_s), 0.999)

    def test_bias_near_zero_after_scaling(self):
        """After Z-scoring y, bias should converge to ≈ 0."""
        X_raw, y_raw = _make_noiseless_data(n=80)
        ys = ZScoreScaler()
        xs = ZScoreScaler()
        X_s = xs.fit_transform(X_raw)
        y_s = ys.fit_transform_1d(y_raw)

        m = LinearRegression(lr=0.01, max_iter=10_000, tol=1e-9, verbose=False)
        m.fit(X_s, y_s)
        self.assertAlmostEqual(m.bias, 0.0, places=4)

    def test_single_feature_weight_recovery(self):
        """y = 4*x → after scaling weight should capture that slope."""

        X = [[float(i)] for i in range(1, 51)]
        y = [4.0 * i for i in range(1, 51)]
        xs = ZScoreScaler()
        ys = ZScoreScaler()
        X_s = xs.fit_transform(X)
        y_s = ys.fit_transform_1d(y)
        m = LinearRegression(lr=0.1, max_iter=10_000, tol=1e-9, verbose=False)
        m.fit(X_s, y_s)
        # In scaled space weight[0] should be very close to 1.0
        # (both X and y are perfectly correlated, σ_y / σ_x × slope = 1)
        self.assertAlmostEqual(m.weights[0], 1.0, places=3)


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


class TestScore(unittest.TestCase):
    def test_perfect_score_is_one(self):
        """score() should return exactly 1.0 when predictions are perfect."""
        X = [[1.0], [2.0], [3.0]]
        y = [2.0, 4.0, 6.0]
        # Set weights manually to produce perfect predictions
        m = LinearRegression(verbose=False)
        m.weights = [2.0]
        m.bias = 0.0
        self.assertAlmostEqual(m.score(X, y), 1.0)

    def test_score_below_one_with_noise(self):
        X_raw, y_raw = _make_noiseless_data(n=100)
        split = 80
        X_tr, y_tr = X_raw[:split], y_raw[:split]
        X_te, y_te = X_raw[split:], y_raw[split:]
        X_tr_s, X_te_s, y_tr_s, y_te_s, _ = _scale(X_tr, y_tr, X_te, y_te)
        m = LinearRegression(lr=0.01, max_iter=5_000, verbose=False)
        m.fit(X_tr_s, y_tr_s)
        self.assertLessEqual(m.score(X_te_s, y_te_s), 1.0)


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


class TestSummary(unittest.TestCase):
    def test_summary_before_fit(self):
        s = LinearRegression().summary()
        self.assertIn("not fitted", s)

    def test_summary_after_fit_contains_key_fields(self):
        X = [[1.0], [2.0], [3.0]]
        y = [1.0, 2.0, 3.0]
        m = LinearRegression(verbose=False)
        m.fit(X, y)
        s = m.summary()
        self.assertIn("iterations", s)
        self.assertIn("MSE", s)
        self.assertIn("bias", s)
        self.assertIn("weight", s)


# ---------------------------------------------------------------------------
# verbose flag
# ---------------------------------------------------------------------------


class TestVerbose(unittest.TestCase):
    def test_verbose_false_produces_no_output(self):
        X = [[float(i)] for i in range(20)]
        y = [float(i) for i in range(20)]
        m = LinearRegression(lr=0.01, max_iter=200, verbose=False)
        captured = io.StringIO()
        sys.stdout = captured
        try:
            m.fit(X, y)
        finally:
            sys.stdout = sys.__stdout__
        self.assertEqual(captured.getvalue(), "")


# ---------------------------------------------------------------------------
# Re-fitting resets state
# ---------------------------------------------------------------------------


class TestRefit(unittest.TestCase):
    def test_refit_resets_loss_history(self):
        xs1, ys1 = ZScoreScaler(), ZScoreScaler()
        X_raw = [[float(i)] for i in range(30)]
        y_raw = [float(i) * 2 + 1.0 for i in range(30)]
        X = xs1.fit_transform(X_raw)
        y = ys1.fit_transform_1d(y_raw)
        m = LinearRegression(verbose=False)
        m.fit(X, y)
        first_n_iter = m.n_iter_
        assert first_n_iter > 0

        # Fit again with different (also scaled) data
        xs2, ys2 = ZScoreScaler(), ZScoreScaler()
        X_raw2 = [[float(i) + 100.0] for i in range(30)]
        y_raw2 = [float(i) * 0.5 + 10.0 for i in range(30)]
        X2 = xs2.fit_transform(X_raw2)
        y2 = ys2.fit_transform_1d(y_raw2)
        m.fit(X2, y2)

        # loss_history should match the new run, not accumulate
        self.assertEqual(len(m.loss_history), m.n_iter_)


if __name__ == "__main__":
    unittest.main()
