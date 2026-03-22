"""Logistic Regression from scratch (binary classification).

This module is shared by multiple tutorial notebooks.

Design goals for the tutorial:
- keep math explicit
- keep code minimal but correct
- allow incremental complexity (naive -> vectorized -> regularized -> early stopping)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid."""
    z = np.asarray(z)
    # stable sigmoid
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out


def binary_cross_entropy(y: np.ndarray, p: np.ndarray, *, eps: float = 1e-12) -> float:
    """Binary cross entropy averaged across samples."""
    y = y.astype(float)
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(X @ w + b)


def predict(X: np.ndarray, w: np.ndarray, b: float, *, threshold: float = 0.5) -> np.ndarray:
    return (predict_proba(X, w, b) >= threshold).astype(int)


def gradients_vectorized(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    *,
    l2_lambda: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Compute gradients of BCE loss (+ optional L2) wrt w and b.

    Loss = mean(BCE) + (l2_lambda/2) * ||w||^2
    """
    n = len(X)
    p = predict_proba(X, w, b)
    dz = (p - y)  # shape: (n,)

    dw = (X.T @ dz) / n
    db = float(dz.mean())

    if l2_lambda != 0.0:
        dw = dw + l2_lambda * w

    return dw, db


def gradients_naive(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    *,
    l2_lambda: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Same as gradients_vectorized, but computed with explicit loops.

    This is intentionally slower but easier to read for beginners.
    """
    n, d = X.shape
    dw = np.zeros(d, dtype=float)
    db = 0.0

    for i in range(n):
        z_i = float(np.dot(X[i], w) + b)
        p_i = float(sigmoid(z_i))
        dz_i = p_i - float(y[i])
        for j in range(d):
            dw[j] += X[i, j] * dz_i
        db += dz_i

    dw /= n
    db /= n

    if l2_lambda != 0.0:
        dw = dw + l2_lambda * w

    return dw, float(db)


@dataclass
class FitConfig:
    lr: float = 0.1
    epochs: int = 200
    batch_size: Optional[int] = None  # None => full-batch
    l2_lambda: float = 0.0
    threshold: float = 0.5
    seed: int = 42
    verbose: bool = True

    # early stopping
    use_early_stopping: bool = False
    patience: int = 20
    min_delta: float = 1e-5


@dataclass
class LogisticRegressionScratch:
    """Binary logistic regression, trained with gradient descent."""

    w: Optional[np.ndarray] = None
    b: float = 0.0
    history_: Optional[Dict[str, List[float]]] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        config: FitConfig = FitConfig(),
        use_naive_gradients: bool = False,
    ) -> "LogisticRegressionScratch":
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        n, d = X_train.shape
        self.w = np.zeros(d, dtype=float)
        self.b = 0.0
        self.history_ = {"train_loss": [], "val_loss": []}

        grad_fn = gradients_naive if use_naive_gradients else gradients_vectorized

        best_val_loss = np.inf
        best_params = None
        bad_epochs = 0

        rng = np.random.default_rng(config.seed)

        for epoch in range(config.epochs):
            # mini-batch or full-batch
            if config.batch_size is None:
                dw, db = grad_fn(
                    X_train,
                    y_train,
                    self.w,
                    self.b,
                    l2_lambda=config.l2_lambda,
                )
                self.w -= config.lr * dw
                self.b -= config.lr * db
            else:
                idx = np.arange(n)
                rng.shuffle(idx)
                for start in range(0, n, config.batch_size):
                    batch_idx = idx[start : start + config.batch_size]
                    Xb = X_train[batch_idx]
                    yb = y_train[batch_idx]
                    dw, db = gradients_vectorized(
                        Xb,
                        yb,
                        self.w,
                        self.b,
                        l2_lambda=config.l2_lambda,
                    )
                    self.w -= config.lr * dw
                    self.b -= config.lr * db

            # track losses
            p_train = predict_proba(X_train, self.w, self.b)
            train_loss = binary_cross_entropy(y_train, p_train)
            if config.l2_lambda != 0.0:
                train_loss += 0.5 * config.l2_lambda * float(np.dot(self.w, self.w))
            self.history_["train_loss"].append(float(train_loss))

            if X_val is not None and y_val is not None:
                p_val = predict_proba(X_val, self.w, self.b)
                val_loss = binary_cross_entropy(np.asarray(y_val, dtype=float), p_val)
                if config.l2_lambda != 0.0:
                    val_loss += 0.5 * config.l2_lambda * float(np.dot(self.w, self.w))
                self.history_["val_loss"].append(float(val_loss))

                if config.use_early_stopping:
                    if val_loss < best_val_loss - config.min_delta:
                        best_val_loss = val_loss
                        best_params = (self.w.copy(), float(self.b))
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                        if bad_epochs >= config.patience:
                            if best_params is not None:
                                self.w, self.b = best_params
                            if config.verbose:
                                print(f"Early stopping at epoch={epoch}")
                            break

            if config.verbose and (epoch % max(1, (config.epochs // 10)) == 0):
                msg = f"epoch={epoch:4d} train_loss={train_loss:.4f}"
                if X_val is not None and y_val is not None:
                    msg += f" val_loss={self.history_['val_loss'][-1]:.4f}"
                print(msg)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted")
        return predict_proba(np.asarray(X, dtype=float), self.w, self.b)

    def predict(self, X: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted")
        return (self.predict_proba(X) >= threshold).astype(int)
