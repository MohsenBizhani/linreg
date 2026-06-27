"""
linreg._math
============
Pure-Python numeric primitives used throughout the library.
No external dependencies.
"""

import math


def dot(a: list, b: list) -> float:
    """Dot product of two equal-length lists.

    Parameters
    ----------
    a, b : list[float]

    Returns
    -------
    float
    """
    return sum(x * y for x, y in zip(a, b))


def list_mean(values: list) -> float:
    """Arithmetic mean of a list.

    Parameters
    ----------
    values : list[float]

    Returns
    -------
    float
    """
    if not values:
        raise ValueError("Cannot compute mean of an empty list.")
    return sum(values) / len(values)


def list_std(values: list) -> float:
    """Population standard deviation using the Z-score formula.

        σ = sqrt( (1/n) · Σ (xᵢ − μ)² )

    Returns 1.0 when σ = 0 to prevent downstream division-by-zero.

    Parameters
    ----------
    values : list[float]

    Returns
    -------
    float
    """
    mu = list_mean(values)
    variance = sum((x - mu) ** 2 for x in values) / len(values)
    return math.sqrt(variance) if variance > 0.0 else 1.0


def transpose(matrix: list) -> list:
    """Transpose a 2-D list.

    Converts shape (n_rows × n_cols) → (n_cols × n_rows).

    Parameters
    ----------
    matrix : list[list[float]]

    Returns
    -------
    list[list[float]]
    """
    return [list(col) for col in zip(*matrix)]
