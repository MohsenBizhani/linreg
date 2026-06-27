"""
linreg._model
=============
Multivariate Linear Regression trained with Batch Gradient Descent.

Algorithm
---------
Model:
    ŷ = b + w₁·x₁ + w₂·x₂ + … + wₙ·xₙ

Loss (MSE):
    L = (1/n) · Σ (ŷᵢ − yᵢ)²

Gradients (mean over full batch):
    ∂L/∂wⱼ = (1/n) · Σ (ŷᵢ − yᵢ) · xᵢⱼ
    ∂L/∂b  = (1/n) · Σ (ŷᵢ − yᵢ)

Parameter update:
    wⱼ ← wⱼ − lr · ∂L/∂wⱼ
    b  ← b  − lr · ∂L/∂b

Early stopping:
    Halt when |MSE(t) − MSE(t−1)| < tol.
"""

from ._math import dot, list_mean


class LinearRegression:
    """Multivariate Linear Regression via Batch Gradient Descent.

    Parameters
    ----------
    lr : float, default 0.01
        Learning rate.  A value of 0.01 is conservative and stable on
        Z-score normalised data; increase cautiously if convergence is slow.
    max_iter : int, default 10_000
        Hard upper bound on gradient-descent iterations.  Early stopping
        will typically fire well before this limit.
    tol : float, default 1e-6
        Early-stopping tolerance.  Training halts as soon as
        |MSE(t) − MSE(t−1)| < tol, meaning the loss surface is
        effectively flat.
    verbose : bool, default True
        Print progress every 500 iterations and at early-stop / convergence.

    Attributes
    ----------
    weights : list[float]
        Fitted weight vector [w₁, w₂, …, wₙ].  Available after ``fit()``.
    bias : float
        Fitted intercept b.  Available after ``fit()``.
    loss_history : list[float]
        MSE value recorded after every iteration.
    n_iter_ : int
        Actual number of iterations executed.

    Examples
    --------
    >>> model = LinearRegression(lr=0.01, max_iter=10_000, tol=1e-6)
    >>> model.fit(X_train_scaled, y_train_scaled)
    >>> predictions = model.predict(X_test_scaled)
    >>> print(model.score(X_test_scaled, y_test_scaled))
    """

    def __init__(
        self,
        lr: float = 0.01,
        max_iter: int = 10_000,
        tol: float = 1e-6,
        verbose: bool = True,
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # Set after fit()
        self.weights: list = []
        self.bias: float = 0.0
        self.loss_history: list = []
        self.n_iter_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: list, y: list) -> "LinearRegression":
        """Train the model using batch gradient descent.

        Parameters
        ----------
        X : list[list[float]]
            Feature matrix, shape (n_samples, n_features).
            Should be Z-score normalised before calling fit.
        y : list[float]
            Target vector, shape (n_samples,).
            Should be Z-score normalised before calling fit.

        Returns
        -------
        self
            Allows method chaining: ``model.fit(X, y).predict(X_test)``.
        """
        self._validate_input(X, y)

        n_samples = len(X)
        n_features = len(X[0])

        # Initialise parameters to zero
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.loss_history = []
        prev_loss = None
        loss = None

        for iteration in range(self.max_iter):
            # ── Forward pass ──────────────────────────────────────────
            y_pred = self._forward(X)

            # ── MSE loss ──────────────────────────────────────────────
            loss = _mse(y, y_pred)
            self.loss_history.append(loss)

            # ── Early stopping check ──────────────────────────────────
            if prev_loss is not None and abs(prev_loss - loss) < self.tol:
                self.n_iter_ = iteration + 1
                if self.verbose:
                    print(
                        f"  Early stop  iter={self.n_iter_:,} "
                        f"| |ΔMSE|={abs(prev_loss - loss):.2e} < {self.tol:.0e}"
                        f" | MSE={loss:.6f}"
                    )
                break

            prev_loss = loss

            # ── Gradients (mean over batch) ───────────────────────────
            errors = [y_pred[i] - y[i] for i in range(n_samples)]

            dw = [
                sum(errors[i] * X[i][j] for i in range(n_samples)) / n_samples
                for j in range(n_features)
            ]
            db = sum(errors) / n_samples

            # ── Parameter update ──────────────────────────────────────
            self.weights = [
                self.weights[j] - self.lr * dw[j] for j in range(n_features)
            ]
            self.bias -= self.lr * db

            if self.verbose and (iteration + 1) % 500 == 0:
                print(f"  Iter {iteration + 1:6,} | MSE: {loss:.6f}")

        else:
            self.n_iter_ = self.max_iter
            if self.verbose:
                print(f"  Reached max_iter={self.max_iter:,} | MSE={loss:.6f}")

        return self

    def predict(self, X: list) -> list:
        """Compute predictions for a feature matrix.

        Parameters
        ----------
        X : list[list[float]]  shape (n_samples, n_features)

        Returns
        -------
        list[float]  predicted values, shape (n_samples,)
        """
        if not self.weights:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict().")
        return self._forward(X)

    def score(self, X: list, y: list) -> float:
        """Compute the coefficient of determination R² on the given data.

        R² = 1 − SS_res / SS_tot

        Parameters
        ----------
        X : list[list[float]]
        y : list[float]

        Returns
        -------
        float
            Best possible score is 1.0; a constant-prediction baseline
            scores 0.0; negative values indicate a worse-than-baseline fit.
        """
        y_pred = self.predict(X)
        mu = list_mean(y)
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(len(y)))
        ss_tot = sum((y[i] - mu) ** 2 for i in range(len(y)))
        return 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 1.0

    def summary(self) -> str:
        """Return a formatted string summarising the fitted model.

        Returns
        -------
        str
        """
        if not self.weights:
            return "LinearRegression (not fitted)"
        lines = [
            "LinearRegression — fitted",
            f"  iterations : {self.n_iter_:,}",
            f"  final MSE  : {self.loss_history[-1]:.6f}",
            f"  bias (b)   : {self.bias:.6f}",
        ]
        for j, w in enumerate(self.weights):
            lines.append(f"  weight[{j}]  : {w:.6f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LinearRegression(lr={self.lr}, max_iter={self.max_iter}, "
            f"tol={self.tol}, verbose={self.verbose})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _forward(self, X: list) -> list:
        """ŷᵢ = b + wᵀ · xᵢ for every sample i."""
        return [dot(self.weights, row) + self.bias for row in X]

    @staticmethod
    def _validate_input(X: list, y: list) -> None:
        if len(X) == 0 or len(y) == 0:
            raise ValueError("X and y must not be empty.")
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got {len(X)} and {len(y)}."
            )
        n_features = len(X[0])
        for i, row in enumerate(X):
            if len(row) != n_features:
                raise ValueError(
                    f"All rows in X must have the same number of features. "
                    f"Row 0 has {n_features}, row {i} has {len(row)}."
                )


# ---------------------------------------------------------------------------
# Module-private helper (not exported)
# ---------------------------------------------------------------------------


def _mse(y_true: list, y_pred: list) -> float:
    n = len(y_true)
    return sum((y_pred[i] - y_true[i]) ** 2 for i in range(n)) / n
