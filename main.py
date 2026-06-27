"""
main.py  —  demonstration script for the `linreg` library.

Run:
    python main.py
"""

from linreg import (
    LinearRegression,
    ZScoreScaler,
    generate_dataset,
    mae,
    plot_all,
    r2_score,
    rmse,
    train_test_split,
)


def main():
    print("=" * 60)
    print("  linreg  —  Linear Regression from Scratch")
    print("  Z-score normalisation  |  Early stopping (tol=1e-6)")
    print("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────────────
    X, y = generate_dataset(n_samples=300, noise=20.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(f"\nDataset  : {len(X)} samples, {len(X[0])} features")
    print(f"Train    : {len(X_train)} samples")
    print(f"Test     : {len(X_test)} samples")

    # ── 2. Z-score normalise X ────────────────────────────────────────
    x_scaler = ZScoreScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s = x_scaler.transform(X_test)

    print("\nFeature scaling (Z-score):")
    for j, (mu, sigma) in enumerate(zip(x_scaler.means_, x_scaler.stds_)):
        print(f"  x{j + 1}  μ={mu:.4f}  σ={sigma:.4f}")

    # ── 3. Z-score normalise y ────────────────────────────────────────
    y_scaler = ZScoreScaler()
    y_train_s = y_scaler.fit_transform_1d(y_train)
    y_test_s = y_scaler.transform_1d(y_test)

    print(f"\nTarget scaling (Z-score):")
    print(f"  y  μ={y_scaler.means_[0]:.4f}  σ={y_scaler.stds_[0]:.4f}")

    # ── 4. Train ──────────────────────────────────────────────────────
    model = LinearRegression(lr=0.01, max_iter=10_000, tol=1e-6)
    print("\nTraining…\n")
    model.fit(X_train_s, y_train_s)

    # ── 5. Evaluate ───────────────────────────────────────────────────
    y_pred_s = model.predict(X_test_s)
    y_pred = y_scaler.inverse_transform_1d(y_pred_s)

    print(f"\n{model.summary()}")

    print(f"\n{'─' * 40}")
    print(f"  R²   : {r2_score(y_test, y_pred):.6f}")
    print(f"  RMSE : {rmse(y_test, y_pred):.4f}")
    print(f"  MAE  : {mae(y_test, y_pred):.4f}")
    print(f"{'─' * 40}")

    # ── 6. Plot ───────────────────────────────────────────────────────
    fig = plot_all(
        loss_history=model.loss_history,
        n_iter=model.n_iter_,
        y_true=y_test,
        y_pred=y_pred,
        suptitle="linreg  —  Z-score Normalisation + Early Stopping",
    )
    fig.savefig("linear_regression_results.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved → linear_regression_results.png")
    fig.show()


if __name__ == "__main__":
    main()
