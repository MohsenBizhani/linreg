# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2024-01-01

### Added
- `LinearRegression` — Batch Gradient Descent with early stopping (`tol=1e-6`)
- `ZScoreScaler` — Manual Z-score normalisation for features and targets
- `train_test_split` — Reproducible random train/test splitting
- `generate_dataset` — Synthetic multivariate dataset generator
- Metrics: `mse`, `rmse`, `mae`, `r2_score`
- Plotting: `plot_loss`, `plot_predictions`, `plot_residuals`, `plot_all`
- Full unit and integration test suite (`tests/`)
- `pyproject.toml` packaging for PyPI
- MIT License
- GitHub Actions CI workflow
