"""
tests/test_integration.py
=========================
End-to-end integration tests that exercise the full pipeline via the
public API in linreg.__init__:

    generate_dataset → train_test_split → ZScoreScaler → LinearRegression
    → predict → inverse_transform_1d → r2_score / rmse / mae

These tests are intentionally higher-level; they catch regressions that
module-level unit tests might miss (e.g. import surface, API compatibility,
metric-vs-prediction wiring).
"""

import unittest

import linreg
from linreg import (
    LinearRegression,
    ZScoreScaler,
    generate_dataset,
    mae,
    mse,
    plot_all,
    plot_loss,
    plot_predictions,
    plot_residuals,
    r2_score,
    rmse,
    train_test_split,
)

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI(unittest.TestCase):
    def test_version_exists(self):
        self.assertTrue(hasattr(linreg, "__version__"))
        self.assertIsInstance(linreg.__version__, str)

    def test_all_exports_importable(self):
        for name in linreg.__all__:
            self.assertTrue(
                hasattr(linreg, name),
                msg=f"linreg.__all__ lists '{name}' but it is not importable.",
            )


# ---------------------------------------------------------------------------
# Full pipeline — noiseless
# ---------------------------------------------------------------------------


class TestFullPipelineNoiseless(unittest.TestCase):
    """With zero noise the model must recover the relationship almost exactly."""

    @classmethod
    def setUpClass(cls):
        X, y = generate_dataset(n_samples=200, noise=0.0, seed=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, seed=0)

        x_sc = ZScoreScaler()
        X_tr_s = x_sc.fit_transform(X_tr)
        X_te_s = x_sc.transform(X_te)

        y_sc = ZScoreScaler()
        y_tr_s = y_sc.fit_transform_1d(y_tr)

        model = LinearRegression(lr=0.01, max_iter=10_000, tol=1e-9, verbose=False)
        model.fit(X_tr_s, y_tr_s)

        y_pred_s = model.predict(X_te_s)
        y_pred = y_sc.inverse_transform_1d(y_pred_s)

        cls.model = model
        cls.y_te = y_te
        cls.y_pred = y_pred

    def test_r2_near_one(self):
        score = r2_score(self.y_te, self.y_pred)
        self.assertGreater(
            score, 0.999, msg=f"Expected R² > 0.999 on noiseless data, got {score:.6f}"
        )

    def test_rmse_near_zero(self):
        err = rmse(self.y_te, self.y_pred)
        self.assertLess(
            err, 0.5, msg=f"Expected RMSE < 0.5 on noiseless data, got {err:.4f}"
        )

    def test_mae_near_zero(self):
        err = mae(self.y_te, self.y_pred)
        self.assertLess(
            err, 0.5, msg=f"Expected MAE < 0.5 on noiseless data, got {err:.4f}"
        )

    def test_mse_near_zero(self):
        err = mse(self.y_te, self.y_pred)
        self.assertLess(err, 0.5)

    def test_early_stop_fired(self):
        self.assertLess(self.model.n_iter_, 10_000)

    def test_loss_is_non_increasing(self):
        history = self.model.loss_history
        for i in range(1, len(history)):
            self.assertLessEqual(history[i], history[i - 1] + 1e-9)


# ---------------------------------------------------------------------------
# Full pipeline — with noise (realistic)
# ---------------------------------------------------------------------------


class TestFullPipelineWithNoise(unittest.TestCase):
    """With realistic noise the model should still achieve R² > 0.85."""

    @classmethod
    def setUpClass(cls):
        X, y = generate_dataset(n_samples=300, noise=20.0, seed=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, seed=42)

        x_sc = ZScoreScaler()
        X_tr_s = x_sc.fit_transform(X_tr)
        X_te_s = x_sc.transform(X_te)

        y_sc = ZScoreScaler()
        y_tr_s = y_sc.fit_transform_1d(y_tr)

        model = LinearRegression(lr=0.01, max_iter=10_000, tol=1e-6, verbose=False)
        model.fit(X_tr_s, y_tr_s)

        y_pred_s = model.predict(X_te_s)
        y_pred = y_sc.inverse_transform_1d(y_pred_s)

        cls.model = model
        cls.y_te = y_te
        cls.y_pred = y_pred

    def test_r2_above_threshold(self):
        score = r2_score(self.y_te, self.y_pred)
        self.assertGreater(score, 0.85, msg=f"Expected R² > 0.85, got {score:.4f}")

    def test_predictions_same_length_as_targets(self):
        self.assertEqual(len(self.y_pred), len(self.y_te))

    def test_loss_history_non_empty(self):
        self.assertGreater(len(self.model.loss_history), 0)


# ---------------------------------------------------------------------------
# Plotting (smoke tests — just check Figure objects are returned)
# ---------------------------------------------------------------------------


class TestPlottingSmoke(unittest.TestCase):
    """Plotting functions must return matplotlib Figure objects without error."""

    @classmethod
    def setUpClass(cls):
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend; no display required

        X, y = generate_dataset(n_samples=60, noise=5.0, seed=1)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, seed=1)

        x_sc = ZScoreScaler()
        X_tr_s = x_sc.fit_transform(X_tr)
        X_te_s = x_sc.transform(X_te)

        y_sc = ZScoreScaler()
        y_tr_s = y_sc.fit_transform_1d(y_tr)

        model = LinearRegression(lr=0.01, max_iter=2_000, verbose=False)
        model.fit(X_tr_s, y_tr_s)

        y_pred_s = model.predict(X_te_s)
        y_pred = y_sc.inverse_transform_1d(y_pred_s)

        cls.model = model
        cls.y_te = y_te
        cls.y_pred = y_pred

    def _assert_figure(self, fig):
        import matplotlib.pyplot as plt

        self.assertIsInstance(fig, plt.Figure)

    def test_plot_loss_returns_figure(self):
        fig = plot_loss(self.model.loss_history, self.model.n_iter_)
        self._assert_figure(fig)

    def test_plot_predictions_returns_figure(self):
        fig = plot_predictions(self.y_te, self.y_pred)
        self._assert_figure(fig)

    def test_plot_residuals_returns_figure(self):
        fig = plot_residuals(self.y_te, self.y_pred)
        self._assert_figure(fig)

    def test_plot_all_returns_figure(self):
        fig = plot_all(
            self.model.loss_history,
            self.model.n_iter_,
            self.y_te,
            self.y_pred,
        )
        self._assert_figure(fig)


# ---------------------------------------------------------------------------
# No-leakage: scaler fitted on train only
# ---------------------------------------------------------------------------


class TestNoDataLeakage(unittest.TestCase):
    """Scaler statistics must come from the training set only."""

    def test_scaler_mean_matches_train_mean(self):
        X, y = generate_dataset(n_samples=100, noise=0.0, seed=3)
        X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, seed=3)

        x_sc = ZScoreScaler()
        x_sc.fit_transform(X_tr)

        # Column 0 train mean
        train_mean_0 = sum(row[0] for row in X_tr) / len(X_tr)
        self.assertAlmostEqual(x_sc.means_[0], train_mean_0, places=10)

    def test_scaler_does_not_use_test_statistics(self):
        X, y = generate_dataset(n_samples=100, noise=0.0, seed=4)
        X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.5, seed=4)

        x_sc = ZScoreScaler()
        x_sc.fit_transform(X_tr)

        # Full-data mean should differ from train-only mean (different splits)
        full_mean_0 = sum(row[0] for row in X) / len(X)
        train_mean_0 = sum(row[0] for row in X_tr) / len(X_tr)

        # If they happened to be equal this test would be vacuous; verify they differ
        if abs(full_mean_0 - train_mean_0) > 1e-6:
            self.assertAlmostEqual(x_sc.means_[0], train_mean_0, places=10)
            self.assertNotAlmostEqual(x_sc.means_[0], full_mean_0, places=3)


if __name__ == "__main__":
    unittest.main()
