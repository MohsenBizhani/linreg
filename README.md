# linreg

> Linear Regression from scratch — **no ML libraries required**.  
> Pure Python · Batch Gradient Descent · Z-score normalisation · Early stopping

---

## Installation

```bash
# Editable (development) install — changes to the source are reflected immediately
pip install -e .

# Or a regular install
pip install .
```

`linreg` has a single runtime dependency: **matplotlib** (for plotting).  
The model, scaler, metrics, and data utilities use only the Python standard library.

---

## Quick start

```python
from linreg import (
    LinearRegression,
    ZScoreScaler,
    train_test_split,
    r2_score, rmse, mae,
    plot_all,
)

# 1. Prepare data  (list of lists — no NumPy needed)
X = [[x1, x2], ...]   # shape (n_samples, n_features)
y = [y1, y2, ...]     # shape (n_samples,)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Z-score normalise
x_scaler  = ZScoreScaler()
X_train_s = x_scaler.fit_transform(X_train)   # fit + transform on train
X_test_s  = x_scaler.transform(X_test)        # transform only on test

y_scaler  = ZScoreScaler()
y_train_s = y_scaler.fit_transform_1d(y_train)
y_test_s  = y_scaler.transform_1d(y_test)

# 4. Train
model = LinearRegression(lr=0.01, max_iter=10_000, tol=1e-6)
model.fit(X_train_s, y_train_s)

# 5. Predict  (invert target scaling to recover original units)
y_pred_s = model.predict(X_test_s)
y_pred   = y_scaler.inverse_transform_1d(y_pred_s)

# 6. Evaluate
print(f"R²   = {r2_score(y_test, y_pred):.4f}")
print(f"RMSE = {rmse(y_test, y_pred):.4f}")
print(f"MAE  = {mae(y_test, y_pred):.4f}")

# 7. Plot
fig = plot_all(model.loss_history, model.n_iter_, y_test, y_pred)
fig.savefig("results.png", dpi=150)
```

Run the bundled demo:

```bash
python main.py
```

---

## API reference

### `LinearRegression`

```python
LinearRegression(lr=0.01, max_iter=10_000, tol=1e-6, verbose=True)
```

| Method | Description |
|---|---|
| `fit(X, y)` | Train the model; returns `self` for chaining |
| `predict(X)` | Return predicted values |
| `score(X, y)` | Return R² on the given data |
| `summary()` | Print fitted parameters and training stats |

**Key attributes after `fit()`**

| Attribute | Description |
|---|---|
| `weights` | Fitted weight vector `[w₁, …, wₙ]` |
| `bias` | Fitted intercept `b` |
| `loss_history` | MSE at every iteration |
| `n_iter_` | Actual iterations run (early stopping may fire early) |

---

### `ZScoreScaler`

```python
ZScoreScaler()
```

| Method | Description |
|---|---|
| `fit_transform(X)` | Fit on `X`, return scaled matrix |
| `transform(X)` | Apply existing fit to new data |
| `fit_transform_1d(y)` | Fit on target vector, return scaled list |
| `transform_1d(y)` | Apply existing fit to new target list |
| `inverse_transform_1d(y_scaled)` | Recover original units |

---

### Metrics

```python
from linreg import mse, rmse, mae, r2_score
```

All functions accept two plain Python lists: `(y_true, y_pred)`.

---

### Data utilities

```python
from linreg import train_test_split, generate_dataset
```

| Function | Description |
|---|---|
| `train_test_split(X, y, test_size, seed)` | Reproducible random split |
| `generate_dataset(n_samples, noise, seed)` | Synthetic `y = 3x₁ + 5x₂ + 10 + ε` |

---

### Plotting

```python
from linreg import plot_loss, plot_predictions, plot_residuals, plot_all
```

All functions return a `matplotlib.Figure` — they **never** call `plt.show()`, so
you control display, saving, or embedding.

| Function | Description |
|---|---|
| `plot_loss(loss_history, n_iter)` | MSE vs iteration with early-stop marker |
| `plot_predictions(y_true, y_pred)` | Predicted vs actual scatter |
| `plot_residuals(y_true, y_pred)` | Residuals vs predicted scatter |
| `plot_all(loss_history, n_iter, y_true, y_pred)` | All three side-by-side |

---

## Algorithm

```
Model:       ŷ = b + w₁x₁ + w₂x₂ + … + wₙxₙ

Loss (MSE):  L = (1/n) Σ (ŷᵢ − yᵢ)²

Gradients:   ∂L/∂wⱼ = (1/n) Σ (ŷᵢ − yᵢ) · xᵢⱼ
             ∂L/∂b  = (1/n) Σ (ŷᵢ − yᵢ)

Update:      wⱼ ← wⱼ − lr · ∂L/∂wⱼ
             b  ← b  − lr · ∂L/∂b

Stop when:   |MSE(t) − MSE(t−1)| < tol
```

---

## Project layout

```
linreg/
├── __init__.py     public API surface
├── _math.py        dot, list_mean, list_std, transpose
├── _scaler.py      ZScoreScaler
├── _model.py       LinearRegression
├── _metrics.py     mse, rmse, mae, r2_score
├── _data.py        train_test_split, generate_dataset
└── plot.py         plot_loss, plot_predictions, plot_residuals, plot_all
main.py             runnable demo
setup.py            packaging
README.md           this file
```

---

## License

MIT
