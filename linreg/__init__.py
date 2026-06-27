"""
linreg
======
Linear Regression from scratch — no ML libraries required.

Quick start
-----------
::

    from linreg import LinearRegression, ZScoreScaler, train_test_split
    from linreg import r2_score, rmse, mae
    from linreg import plot_all

    # 1. Load / generate data
    X, y = ...

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 3. Scale (Z-score normalise)
    x_scaler = ZScoreScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s  = x_scaler.transform(X_test)

    y_scaler  = ZScoreScaler()
    y_train_s = y_scaler.fit_transform_1d(y_train)
    y_test_s  = y_scaler.transform_1d(y_test)

    # 4. Train
    model = LinearRegression(lr=0.01, max_iter=10_000, tol=1e-6)
    model.fit(X_train_s, y_train_s)

    # 5. Predict (invert scaling for interpretable output)
    y_pred_s = model.predict(X_test_s)
    y_pred   = y_scaler.inverse_transform_1d(y_pred_s)

    # 6. Evaluate
    print(f"R²   = {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE = {rmse(y_test, y_pred):.4f}")

    # 7. Plot
    fig = plot_all(model.loss_history, model.n_iter_, y_test, y_pred)
    fig.savefig("results.png", dpi=150)

Public API
----------
Classes   : LinearRegression, ZScoreScaler
Functions : train_test_split, generate_dataset,
            mse, rmse, mae, r2_score,
            plot_loss, plot_predictions, plot_residuals, plot_all
"""

from ._data import generate_dataset, train_test_split
from ._metrics import mae, mse, r2_score, rmse
from ._model import LinearRegression
from ._scaler import ZScoreScaler
from .plot import plot_all, plot_loss, plot_predictions, plot_residuals

__all__ = [
    # Core
    "LinearRegression",
    "ZScoreScaler",
    # Data utilities
    "train_test_split",
    "generate_dataset",
    # Metrics
    "mse",
    "rmse",
    "mae",
    "r2_score",
    # Plotting
    "plot_loss",
    "plot_predictions",
    "plot_residuals",
    "plot_all",
]

__version__ = "1.0.0"
__author__ = "Linear Regression from Scratch"
