"""Scratch data utilities.

We intentionally only use scikit-learn for *dataset loading* in this tutorial.
Splitting, scaling, shuffling etc. are implemented from scratch here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split X/y into train and test."""

    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features), got shape={X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (n_samples,), got shape={y.shape}")
    if len(X) != len(y):
        raise ValueError("X and y must have same number of samples")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1")

    n = len(X)
    n_test = int(np.round(n * test_size))
    n_test = max(1, min(n - 1, n_test))

    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


@dataclass
class StandardScaler:
    """Standardize features: z = (x - mean) / std

    Fitted on training data only.
    """

    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None
    eps: float = 1e-12

    def fit(self, X: np.ndarray) -> "StandardScaler":
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler must be fit() before transform()")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def batch_iterator(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    seed: int = 42,
    shuffle: bool = True,
):
    """Yield mini-batches (X_batch, y_batch)."""

    n = len(X)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]
