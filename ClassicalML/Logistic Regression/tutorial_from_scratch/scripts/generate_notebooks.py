"""Generate tutorial notebooks programmatically.

Why generate?
- ensures consistent structure across notebooks
- easy to rerun if we tweak explanations/code

The notebooks are written to: ClassicalML/Logistic Regression/tutorial_from_scratch/notebooks/
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def save_notebook(path: Path, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nbf.writes(nb), encoding="utf-8")


def nb00_setup_dataset() -> list:
    return [
        md(
            r"""# 00 — Setup + Breast Cancer Dataset (from scratch split)

**Goal of this notebook**

We will:
1. Load the *Breast Cancer Wisconsin* dataset (binary classification) using `sklearn.datasets.load_breast_cancer()`.
2. Implement a **train/test split** *from scratch* (no `sklearn.model_selection`).
3. Inspect feature scales and class balance.

> In later notebooks, we’ll build logistic regression step-by-step and show how improvements (vectorization, scaling, regularization, early stopping) impact performance.
"""
        ),
        code(
            """import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from utils.data import train_test_split
"""
        ),
        md(
            """## Load dataset (allowed sklearn usage)

We only use sklearn here to **load** the dataset; everything else is scratch."""
        ),
        code(
            """data = load_breast_cancer()
X = data.data.astype(float)
y = data.target.astype(int)

feature_names = data.feature_names

print('X shape:', X.shape)
print('y shape:', y.shape)
print('classes:', np.unique(y, return_counts=True))
"""
        ),
        md(
            """## Scratch train/test split

We’ll implement our own `train_test_split` using random shuffling."""
        ),
        code(
            """X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

print('train:', X_train.shape, y_train.shape)
print('test :', X_test.shape, y_test.shape)
"""
        ),
        md(
            """## Feature scale matters

Logistic regression uses gradient-based optimization. If features have wildly different scales,
optimization can become slow/unstable.

Let’s visualize the distribution of feature standard deviations."""
        ),
        code(
            """stds = X_train.std(axis=0)

plt.figure(figsize=(8, 3))
plt.plot(stds)
plt.title('Feature standard deviations (train set)')
plt.xlabel('feature index')
plt.ylabel('std')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print('min std:', stds.min(), 'max std:', stds.max())
"""
        ),
        md(
            """### What to expect next

In Notebook **01**, we will intentionally build a **minimal** logistic regression.

It will work end-to-end, but because we will train **without scaling** and with a very simple training loop,
accuracy will be good-but-not-amazing.

Then we’ll improve it incrementally."""
        ),
    ]


def nb01_minimal_naive_gd() -> list:
    return [
        md(
            r"""# 01 — Minimal Logistic Regression (naive gradients)

**Goal:** implement logistic regression with the fewest moving parts.

We will:
- write sigmoid
- write binary cross entropy loss
- compute gradients with **explicit loops** (slow but very clear)
- run gradient descent
- compute **accuracy** from scratch

> This notebook is about understanding the mechanics.
> Later notebooks will improve speed and accuracy.
"""
        ),
        code(
            """import numpy as np

from sklearn.datasets import load_breast_cancer

from utils.data import train_test_split
from utils.metrics import accuracy
from scripts.logreg_scratch import (
    LogisticRegressionScratch,
    FitConfig,
)
"""
        ),
        md("""## Data"""),
        code(
            """data = load_breast_cancer()
X = data.data.astype(float)
y = data.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
"""
        ),
        md(
            r"""## Model

For a sample \(x\in\mathbb{R}^d\), logistic regression predicts:

\[
p(y=1\mid x) = \sigma(w^T x + b)
\]

where

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

We train by minimizing **binary cross entropy**."""
        ),
        code(
            """model = LogisticRegressionScratch()

# Intentionally simple config (not tuned)
cfg = FitConfig(lr=0.01, epochs=400, verbose=True)

model.fit(X_train, y_train, config=cfg, use_naive_gradients=True)

y_pred = model.predict(X_test, threshold=0.5)
acc = accuracy(y_test, y_pred)
print('Test accuracy:', acc)
"""
        ),
        md(
            """### Notes

This should run, but training may be:
- slower (loop gradients)
- less stable (unscaled features)

In Notebook **02**, we’ll switch to vectorized gradients and plot the loss curve."""
        ),
    ]


def nb02_vectorized_gd() -> list:
    return [
        md(
            r"""# 02 — Vectorized Logistic Regression + Loss Curve

**Goal:** same math as Notebook 01, but implemented using NumPy vectorization.

Vectorization gives:
- much faster training
- fewer bugs
- easier experimentation (more epochs, hyperparameter search, etc.)

> Accuracy may not improve *yet* (we still haven’t fixed feature scaling).
"""
        ),
        code(
            """import numpy as np

from sklearn.datasets import load_breast_cancer

from utils.data import train_test_split
from utils.metrics import accuracy
from utils.plots import plot_history
from scripts.logreg_scratch import LogisticRegressionScratch, FitConfig
"""
        ),
        code(
            """data = load_breast_cancer()
X = data.data.astype(float)
y = data.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
"""
        ),
        md("""## Train (vectorized gradients)"""),
        code(
            """model = LogisticRegressionScratch()
cfg = FitConfig(lr=0.01, epochs=800, verbose=False)

model.fit(X_train, y_train, config=cfg, use_naive_gradients=False)

y_pred = model.predict(X_test)
print('Test accuracy:', accuracy(y_test, y_pred))

plot_history(model.history_, title='Notebook 02: loss (unscaled features)')
"""
        ),
        md(
            """Next: in Notebook **03** we add **standardization**, which usually gives a big jump in both convergence and accuracy."""
        ),
    ]


def nb03_scaling_and_minibatch() -> list:
    return [
        md(
            r"""# 03 — Feature Scaling (Standardization) + Mini-batch GD

**Goal:** improve optimization by scaling features.

Standardization:
\[
z = \frac{x - \mu}{\sigma}
\]

Where \(\mu\) and \(\sigma\) are computed **only on the training set**.

Why this helps:
- gradients become more balanced across dimensions
- learning rate becomes easier to tune
- convergence is faster and often reaches a better optimum
"""
        ),
        code(
            """import numpy as np

from sklearn.datasets import load_breast_cancer

from utils.data import train_test_split, StandardScaler
from utils.metrics import accuracy
from utils.plots import plot_history
from scripts.logreg_scratch import LogisticRegressionScratch, FitConfig
"""
        ),
        code(
            """data = load_breast_cancer()
X = data.data.astype(float)
y = data.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
"""
        ),
        md("""## Train with mini-batch GD"""),
        code(
            """model = LogisticRegressionScratch()

cfg = FitConfig(lr=0.1, epochs=400, batch_size=64, verbose=False)

model.fit(X_train_s, y_train, config=cfg)

y_pred = model.predict(X_test_s)
print('Test accuracy (scaled):', accuracy(y_test, y_pred))

plot_history(model.history_, title='Notebook 03: loss (scaled features)')
"""
        ),
        md(
            """Next: add **L2 regularization** and a **validation split** so we can tune hyperparameters responsibly."""
        ),
    ]


def nb04_l2_regularization_and_threshold() -> list:
    return [
        md(
            r"""# 04 — L2 Regularization + Threshold Tuning

**Goal:** improve generalization.

L2 regularization adds a penalty on large weights:

\[
\mathcal{L}(w, b) = \text{BCE}(w,b) + \frac{\lambda}{2}\|w\|_2^2
\]

This often:
- reduces overfitting
- makes the solution more stable

We will also show that the default decision threshold 0.5 is arbitrary; for some applications,
you may choose a different threshold to trade off precision vs recall.
"""
        ),
        code(
            """import numpy as np

from sklearn.datasets import load_breast_cancer

from utils.data import train_test_split, StandardScaler
from utils.metrics import accuracy
from scripts.logreg_scratch import LogisticRegressionScratch, FitConfig
"""
        ),
        md("""## Data + scaling"""),
        code(
            """data = load_breast_cancer()
X = data.data.astype(float)
y = data.target.astype(int)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, seed=42)

# validation split from train
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, seed=123)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
"""
        ),
        md("""## Train with L2"""),
        code(
            """lambdas = [0.0, 1e-4, 1e-3, 1e-2]
results = []

for lam in lambdas:
    model = LogisticRegressionScratch()
    cfg = FitConfig(lr=0.1, epochs=600, batch_size=64, l2_lambda=lam, verbose=False)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, config=cfg)

    val_pred = model.predict(X_val)
    val_acc = accuracy(y_val, val_pred)
    results.append((lam, val_acc, model))

best_lam, best_val_acc, best_model = sorted(results, key=lambda t: t[1], reverse=True)[0]
print('Best lambda:', best_lam, 'val_acc:', best_val_acc)

test_pred = best_model.predict(X_test)
print('Test accuracy:', accuracy(y_test, test_pred))
"""
        ),
        md("""## Threshold tuning (simple demo)

We’ll try a few thresholds and pick the one that maximizes validation accuracy.
In practice, you might optimize F1 or recall instead."""),
        code(
            """thresholds = np.linspace(0.1, 0.9, 17)
val_scores = []

val_proba = best_model.predict_proba(X_val)
for t in thresholds:
    val_pred_t = (val_proba >= t).astype(int)
    val_scores.append(accuracy(y_val, val_pred_t))

best_t = float(thresholds[int(np.argmax(val_scores))])
print('Best threshold:', best_t, 'val_acc:', max(val_scores))

test_pred_t = best_model.predict(X_test, threshold=best_t)
print('Test accuracy (threshold tuned):', accuracy(y_test, test_pred_t))
"""
        ),
        md(
            """Next: in Notebook **05**, we’ll compute confusion matrix + precision/recall/F1 from scratch and plot learning curves."""
        ),
    ]


def nb05_metrics_and_learning_curves() -> list:
    return [
        md(
            r"""# 05 — Evaluation Metrics + Learning Curves

Accuracy is useful, but it hides *which kinds* of mistakes we’re making.

In this notebook we implement:
- confusion matrix (TN/FP/FN/TP)
- precision / recall / F1

We also plot learning curves using the loss history.
"""
        ),
        code(
            """import numpy as np

from sklearn.datasets import load_breast_cancer

from utils.data import train_test_split, StandardScaler
from utils.metrics import accuracy, confusion_matrix_binary, precision_recall_f1
from utils.plots import plot_history
from scripts.logreg_scratch import LogisticRegressionScratch, FitConfig
"""
        ),
        code(
            """data = load_breast_cancer()
X = data.data.astype(float)
y = data.target.astype(int)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, seed=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, seed=123)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
"""
        ),
        md("""## Train a reasonably strong model"""),
        code(
            """model = LogisticRegressionScratch()
cfg = FitConfig(lr=0.1, epochs=800, batch_size=64, l2_lambda=1e-3, verbose=False)
model.fit(X_train, y_train, X_val=X_val, y_val=y_val, config=cfg)

y_pred = model.predict(X_test)
acc = accuracy(y_test, y_pred)

cm = confusion_matrix_binary(y_test, y_pred)
prec, rec, f1 = precision_recall_f1(cm)

print('Accuracy:', acc)
print('Confusion matrix:\n', cm.as_array())
print('Precision:', prec)
print('Recall   :', rec)
print('F1       :', f1)

plot_history(model.history_, title='Notebook 05: learning curves')
"""
        ),
        md(
            """Next: in Notebook **06**, we’ll add early stopping + numerical stability tweaks to make training more robust."""
        ),
    ]


def nb06_early_stopping() -> list:
    return [
        md(
            r"""# 06 — Early Stopping (robust training)

When we have a validation set, we can stop training when validation loss stops improving.

This helps:
- prevent overfitting
- avoid wasting compute
- reduce sensitivity to the exact number of epochs

We already implemented early stopping in `FitConfig`.
"""
        ),
        code(
            """import numpy as np

from sklearn.datasets import load_breast_cancer

from utils.data import train_test_split, StandardScaler
from utils.metrics import accuracy
from utils.plots import plot_history
from scripts.logreg_scratch import LogisticRegressionScratch, FitConfig
"""
        ),
        code(
            """data = load_breast_cancer()
X = data.data.astype(float)
y = data.target.astype(int)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, seed=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, seed=123)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
"""
        ),
        md("""## Train with early stopping"""),
        code(
            """model = LogisticRegressionScratch()

cfg = FitConfig(
    lr=0.1,
    epochs=3000,
    batch_size=64,
    l2_lambda=1e-3,
    verbose=True,
    use_early_stopping=True,
    patience=30,
    min_delta=1e-5,
)

model.fit(X_train, y_train, X_val=X_val, y_val=y_val, config=cfg)

y_pred = model.predict(X_test)
print('Test accuracy:', accuracy(y_test, y_pred))

plot_history(model.history_, title='Notebook 06: early stopping learning curves')
"""
        ),
        md(
            """## Summary

You now have:
- a fully scratch logistic regression model
- training loop
- scratch split + standardization
- scratch evaluation metrics
- validation-based early stopping

From here you can extend to:
- polynomial features
- multiclass softmax
- Newton's method
- stochastic gradient descent
"""
        ),
    ]


def main():
    notebooks = {
        "00_setup_and_dataset.ipynb": nb00_setup_dataset(),
        "01_minimal_logreg_naive_gd.ipynb": nb01_minimal_naive_gd(),
        "02_vectorized_logreg_gd.ipynb": nb02_vectorized_gd(),
        "03_feature_scaling_and_minibatch.ipynb": nb03_scaling_and_minibatch(),
        "04_regularization_L2_and_thresholding.ipynb": nb04_l2_regularization_and_threshold(),
        "05_metrics_and_learning_curves.ipynb": nb05_metrics_and_learning_curves(),
        "06_early_stopping.ipynb": nb06_early_stopping(),
    }

    for name, cells in notebooks.items():
        save_notebook(NB_DIR / name, cells)
        print("Wrote", NB_DIR / name)


if __name__ == "__main__":
    main()
