"""Run a quick end-to-end check and print an accuracy progression table.

We avoid executing notebooks here to keep CI/lightweight validation simple.
Instead we reproduce the configurations used in the notebooks.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from sklearn.datasets import load_breast_cancer

# Allow running this script from anywhere by adding tutorial_from_scratch/ to sys.path
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data import StandardScaler, train_test_split
from utils.metrics import accuracy
from scripts.logreg_scratch import FitConfig, LogisticRegressionScratch


def main():
    data = load_breast_cancer()
    X = data.data.astype(float)
    y = data.target.astype(int)

    rows = []

    # 01: minimal naive (unscaled)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    model = LogisticRegressionScratch()
    cfg = FitConfig(lr=0.01, epochs=400, verbose=False)
    model.fit(X_train, y_train, config=cfg, use_naive_gradients=True)
    acc01 = accuracy(y_test, model.predict(X_test))
    rows.append(("01 naive GD (unscaled)", acc01))

    # 02: vectorized (unscaled)
    model = LogisticRegressionScratch()
    cfg = FitConfig(lr=0.01, epochs=800, verbose=False)
    model.fit(X_train, y_train, config=cfg, use_naive_gradients=False)
    acc02 = accuracy(y_test, model.predict(X_test))
    rows.append(("02 vectorized GD (unscaled)", acc02))

    # 03: scaled + mini-batch
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegressionScratch()
    cfg = FitConfig(lr=0.1, epochs=400, batch_size=64, verbose=False)
    model.fit(X_train_s, y_train, config=cfg)
    acc03 = accuracy(y_test, model.predict(X_test_s))
    rows.append(("03 scaled + mini-batch", acc03))

    # 04: scaled + L2 (tune on val)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, seed=123)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    lambdas = [0.0, 1e-4, 1e-3, 1e-2]
    best_model = None
    best_acc = -1.0
    best_lam = None
    for lam in lambdas:
        model = LogisticRegressionScratch()
        cfg = FitConfig(lr=0.1, epochs=600, batch_size=64, l2_lambda=lam, verbose=False)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, config=cfg)
        val_acc = accuracy(y_val, model.predict(X_val))
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            best_lam = lam

    acc04 = accuracy(y_test, best_model.predict(X_test))
    rows.append((f"04 scaled + L2 (best λ={best_lam})", acc04))

    # 06-like: early stopping (strong baseline)
    model = LogisticRegressionScratch()
    cfg = FitConfig(
        lr=0.1,
        epochs=3000,
        batch_size=64,
        l2_lambda=1e-3,
        verbose=False,
        use_early_stopping=True,
        patience=30,
        min_delta=1e-5,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, config=cfg)
    acc06 = accuracy(y_test, model.predict(X_test))
    rows.append(("06 early stopping (λ=1e-3)", acc06))

    print("\nAccuracy progression (higher is better)\n" + "-" * 45)
    for name, acc in rows:
        print(f"{name:35s}  {acc:.4f}")


if __name__ == "__main__":
    main()
