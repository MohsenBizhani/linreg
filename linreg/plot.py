"""
linreg.plot
===========
Plotting helpers for visualising training and evaluation results.

All functions accept plain Python lists and return a ``matplotlib.figure.Figure``
so the caller decides whether to ``show()``, ``savefig()``, or embed in a
notebook — the functions never call ``plt.show()`` themselves.

Functions
---------
plot_loss(loss_history, n_iter)
plot_predictions(y_true, y_pred)
plot_residuals(y_pred, y_true)
plot_all(loss_history, n_iter, y_true, y_pred)
"""

import matplotlib.pyplot as plt


def plot_loss(
    loss_history: list,
    n_iter: int,
    title: str = "Training Loss (MSE)",
) -> plt.Figure:
    """Plot the MSE loss curve with an early-stop marker.

    Parameters
    ----------
    loss_history : list[float]
        MSE recorded at every iteration (``model.loss_history``).
    n_iter : int
        Actual number of iterations run (``model.n_iter_``).
        Used to draw the early-stop vertical line.
    title : str
        Axes title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(loss_history, color="steelblue", linewidth=1.6, label="MSE")
    ax.axvline(
        n_iter - 1,
        color="crimson",
        linestyle="--",
        linewidth=1.3,
        label=f"Early stop (iter {n_iter:,})",
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_predictions(
    y_true: list,
    y_pred: list,
    title: str = "Predicted vs Actual",
    units: str = "",
) -> plt.Figure:
    """Scatter plot of predicted vs actual values with a perfect-fit line.

    Points that lie on the dashed diagonal are predicted exactly.

    Parameters
    ----------
    y_true : list[float]   Ground-truth targets.
    y_pred : list[float]   Model predictions.
    title  : str           Axes title.
    units  : str           Optional label appended to axis titles, e.g. ``"(km)"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.65,
        color="steelblue",
        edgecolors="white",
        linewidths=0.4,
        s=55,
        zorder=3,
    )

    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.8, label="Perfect fit", zorder=4)

    unit_label = f"  {units}" if units else ""
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(f"Actual y{unit_label}")
    ax.set_ylabel(f"Predicted ŷ{unit_label}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_residuals(
    y_true: list,
    y_pred: list,
    title: str = "Residuals vs Predicted",
) -> plt.Figure:
    """Scatter plot of residuals (ŷ − y) against predicted values.

    A well-fitted model shows residuals scattered randomly around zero
    with no visible pattern or trend.

    Parameters
    ----------
    y_true : list[float]  Ground-truth targets.
    y_pred : list[float]  Model predictions.
    title  : str          Axes title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    residuals = [y_pred[i] - y_true[i] for i in range(len(y_true))]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        y_pred,
        residuals,
        alpha=0.65,
        color="coral",
        edgecolors="white",
        linewidths=0.4,
        s=55,
        zorder=3,
    )
    ax.axhline(0, color="black", linewidth=1.4, linestyle="--", zorder=4)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted ŷ")
    ax.set_ylabel("Residual  (ŷ − y)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_all(
    loss_history: list,
    n_iter: int,
    y_true: list,
    y_pred: list,
    suptitle: str = "Linear Regression — Results",
    units: str = "",
) -> plt.Figure:
    """Convenience wrapper: render all three plots side-by-side.

    Parameters
    ----------
    loss_history : list[float]  ``model.loss_history``
    n_iter       : int          ``model.n_iter_``
    y_true       : list[float]  Ground-truth targets (raw units).
    y_pred       : list[float]  Model predictions (raw units).
    suptitle     : str          Figure-level title.
    units        : str          Optional unit label for axis titles.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    # ── Loss curve ────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(loss_history, color="steelblue", linewidth=1.6, label="MSE")
    ax.axvline(
        n_iter - 1,
        color="crimson",
        linestyle="--",
        linewidth=1.3,
        label=f"Early stop (iter {n_iter:,})",
    )
    ax.set_title("Training Loss (MSE)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Predicted vs Actual ───────────────────────────────────────────
    ax = axes[1]
    unit_label = f"  {units}" if units else ""
    ax.scatter(
        y_true,
        y_pred,
        alpha=0.65,
        color="steelblue",
        edgecolors="white",
        linewidths=0.4,
        s=55,
        zorder=3,
    )
    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.8, label="Perfect fit", zorder=4)
    ax.set_title("Predicted vs Actual", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Actual y{unit_label}")
    ax.set_ylabel(f"Predicted ŷ{unit_label}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Residuals ─────────────────────────────────────────────────────
    ax = axes[2]
    residuals = [y_pred[i] - y_true[i] for i in range(len(y_true))]
    ax.scatter(
        y_pred,
        residuals,
        alpha=0.65,
        color="coral",
        edgecolors="white",
        linewidths=0.4,
        s=55,
        zorder=3,
    )
    ax.axhline(0, color="black", linewidth=1.4, linestyle="--", zorder=4)
    ax.set_title(
        "Residuals vs Predicted\n(random scatter = good fit)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Predicted ŷ")
    ax.set_ylabel("Residual  (ŷ − y)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
