"""Small preprocessing helpers (scratch-ish)."""

from __future__ import annotations

import numpy as np


def standardize_fit(X: np.ndarray, *, eps: float = 1e-12):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < eps, 1.0, sigma)
    return mu, sigma


def standardize_transform(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (X - mu) / sigma
