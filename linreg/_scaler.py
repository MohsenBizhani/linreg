"""
linreg._scaler
==============
Z-score (standard-score) normalisation for features and targets.

Formula
-------
    x_scaled = (x − μ) / σ

    where  μ = (1/n) Σ xᵢ
           σ = sqrt( (1/n) Σ (xᵢ − μ)² )

Both μ and σ are computed entirely from the *training* split so that
no information leaks from the test set.
"""

from ._math import list_mean, list_std, transpose


class ZScoreScaler:
    """Z-score normalisation for 2-D feature matrices and 1-D target vectors.

    Workflow
    --------
    1. Call ``fit_transform(X_train)`` — computes μ / σ from training data
       and returns the scaled matrix.
    2. Call ``transform(X_test)`` — applies the *same* μ / σ to new data
       (no re-fitting, no data leakage).
    3. For target vectors use ``fit_transform_1d`` / ``transform_1d`` /
       ``inverse_transform_1d``.

    Attributes
    ----------
    means_ : list[float]
        Per-column means computed during fitting.
    stds_ : list[float]
        Per-column standard deviations computed during fitting.

    Examples
    --------
    >>> scaler = ZScoreScaler()
    >>> X_train_s = scaler.fit_transform(X_train)
    >>> X_test_s  = scaler.transform(X_test)
    """

    def __init__(self):
        self.means_: list = []
        self.stds_: list = []

    # ------------------------------------------------------------------
    # 2-D feature matrix interface
    # ------------------------------------------------------------------

    def fit_transform(self, X: list) -> list:
        """Fit on X and return the scaled matrix.

        Parameters
        ----------
        X : list[list[float]]  shape (n_samples, n_features)

        Returns
        -------
        list[list[float]]  scaled matrix, same shape as X
        """
        cols = transpose(X)
        self.means_ = [list_mean(col) for col in cols]
        self.stds_ = [list_std(col) for col in cols]
        return self._scale(X)

    def transform(self, X: list) -> list:
        """Apply the already-fitted μ / σ to a new matrix.

        Parameters
        ----------
        X : list[list[float]]  shape (n_samples, n_features)

        Returns
        -------
        list[list[float]]
        """
        if not self.means_:
            raise RuntimeError(
                "ZScoreScaler has not been fitted yet. Call fit_transform() first."
            )
        return self._scale(X)

    def _scale(self, X: list) -> list:
        return [
            [(row[j] - self.means_[j]) / self.stds_[j] for j in range(len(row))]
            for row in X
        ]

    # ------------------------------------------------------------------
    # 1-D target vector interface
    # ------------------------------------------------------------------

    def fit_transform_1d(self, y: list) -> list:
        """Fit on a 1-D target vector and return the scaled list.

        Parameters
        ----------
        y : list[float]

        Returns
        -------
        list[float]
        """
        self.means_ = [list_mean(y)]
        self.stds_ = [list_std(y)]
        return [(v - self.means_[0]) / self.stds_[0] for v in y]

    def transform_1d(self, y: list) -> list:
        """Apply the already-fitted μ / σ to a new 1-D list.

        Parameters
        ----------
        y : list[float]

        Returns
        -------
        list[float]
        """
        if not self.means_:
            raise RuntimeError(
                "ZScoreScaler has not been fitted yet. Call fit_transform_1d() first."
            )
        return [(v - self.means_[0]) / self.stds_[0] for v in y]

    def inverse_transform_1d(self, y_scaled: list) -> list:
        """Reverse the scaling: x = x_scaled · σ + μ.

        Use this to convert model predictions back to the original units.

        Parameters
        ----------
        y_scaled : list[float]

        Returns
        -------
        list[float]
        """
        if not self.means_:
            raise RuntimeError(
                "ZScoreScaler has not been fitted yet. Call fit_transform_1d() first."
            )
        return [v * self.stds_[0] + self.means_[0] for v in y_scaled]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = bool(self.means_)
        return f"ZScoreScaler(fitted={fitted}, n_features={len(self.means_)})"
