"""
linreg._metrics
===============
Regression evaluation metrics — all computed from plain Python lists.

Functions
---------
mse(y_true, y_pred)      Mean Squared Error
rmse(y_true, y_pred)     Root Mean Squared Error
mae(y_true, y_pred)      Mean Absolute Error
r2_score(y_true, y_pred) Coefficient of determination R²
"""

import math

from ._math import list_mean


def mse(y_true: list, y_pred: list) -> float:
    """Mean Squared Error.

        MSE = (1/n) · Σ (ŷᵢ − yᵢ)²

    Parameters
    ----------
    y_true : list[float]  ground-truth targets
    y_pred : list[float]  model predictions

    Returns
    -------
    float
    """
    _check_lengths(y_true, y_pred)
    n = len(y_true)
    return sum((y_pred[i] - y_true[i]) ** 2 for i in range(n)) / n


def rmse(y_true: list, y_pred: list) -> float:
    """Root Mean Squared Error.

        RMSE = sqrt( MSE )

    Same units as the target variable — easier to interpret than MSE.

    Parameters
    ----------
    y_true : list[float]
    y_pred : list[float]

    Returns
    -------
    float
    """
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true: list, y_pred: list) -> float:
    """Mean Absolute Error.

        MAE = (1/n) · Σ |ŷᵢ − yᵢ|

    Less sensitive to outliers than RMSE.

    Parameters
    ----------
    y_true : list[float]
    y_pred : list[float]

    Returns
    -------
    float
    """
    _check_lengths(y_true, y_pred)
    n = len(y_true)
    return sum(abs(y_pred[i] - y_true[i]) for i in range(n)) / n


def r2_score(y_true: list, y_pred: list) -> float:
    """Coefficient of determination R².

        R² = 1 − SS_res / SS_tot

        SS_res = Σ (yᵢ − ŷᵢ)²
        SS_tot = Σ (yᵢ − ȳ)²

    R² = 1.0  →  perfect predictions.
    R² = 0.0  →  model is no better than predicting the mean.
    R² < 0.0  →  model is worse than predicting the mean.

    Parameters
    ----------
    y_true : list[float]
    y_pred : list[float]

    Returns
    -------
    float
    """
    _check_lengths(y_true, y_pred)
    mu = list_mean(y_true)
    ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    ss_tot = sum((y_true[i] - mu) ** 2 for i in range(len(y_true)))
    return 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 1.0


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _check_lengths(y_true: list, y_pred: list) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {len(y_true)} and {len(y_pred)}."
        )
