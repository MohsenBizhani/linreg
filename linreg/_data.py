"""
linreg._data
============
Dataset utilities: train/test splitting and synthetic data generation.
No external dependencies.
"""

import random


def train_test_split(
    X: list,
    y: list,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Randomly split arrays into train and test subsets.

    Parameters
    ----------
    X : list[list[float]]
        Feature matrix, shape (n_samples, n_features).
    y : list[float]
        Target vector, shape (n_samples,).
    test_size : float, default 0.2
        Proportion of the dataset to include in the test split.
        Must be between 0.0 and 1.0.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of four lists

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of samples, got {len(X)} and {len(y)}."
        )

    random.seed(seed)
    indices = list(range(len(X)))
    random.shuffle(indices)

    cut = int(len(indices) * (1 - test_size))
    train_idx, test_idx = indices[:cut], indices[cut:]

    return (
        [X[i] for i in train_idx],
        [X[i] for i in test_idx],
        [y[i] for i in train_idx],
        [y[i] for i in test_idx],
    )


def generate_dataset(
    n_samples: int = 300,
    noise: float = 20.0,
    seed: int = 0,
) -> tuple:
    """Generate a synthetic multivariate linear regression dataset.

    True relationship::

        y = 3·x₁ + 5·x₂ + 10 + ε,   ε ~ N(0, noise)

    with  x₁ ∈ [0, 50]  and  x₂ ∈ [0, 30].

    Useful for testing and demonstration purposes.

    Parameters
    ----------
    n_samples : int, default 300
        Number of data points to generate.
    noise : float, default 20.0
        Standard deviation of the Gaussian noise term.
    seed : int, default 0
        Random seed for reproducibility.

    Returns
    -------
    X : list[list[float]]  shape (n_samples, 2)
    y : list[float]        shape (n_samples,)

    Examples
    --------
    >>> X, y = generate_dataset(n_samples=500, noise=10.0)
    """
    random.seed(seed)
    X, y = [], []
    for _ in range(n_samples):
        x1 = random.uniform(0, 50)
        x2 = random.uniform(0, 30)
        y.append(3 * x1 + 5 * x2 + 10 + random.gauss(0, noise))
        X.append([x1, x2])
    return X, y
