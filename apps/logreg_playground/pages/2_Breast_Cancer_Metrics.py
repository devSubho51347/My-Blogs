from __future__ import annotations

import numpy as np
import streamlit as st

from src.datasets import load_breast_cancer_dataset
from src.metrics import compute_metrics
from src.preprocessing import standardize_fit, standardize_transform
from src.schema import LRParams, validate_params
from src.training import fit_final


st.set_page_config(page_title="Breast Cancer Metrics", layout="wide")
st.title("Breast Cancer Dataset (Metrics Only)")

st.markdown(
    """
The breast cancer dataset is **high-dimensional**, so we do **not** draw a decision boundary.

Instead, this page helps you see how hyperparameters affect:
- accuracy
- confusion matrix
- precision/recall/F1
"""
)


def sidebar_model_controls() -> tuple[LRParams, bool]:
    st.sidebar.header("Model (sklearn LogisticRegression)")

    C = st.sidebar.slider("C (inverse regularization)", 0.001, 100.0, 1.0, step=0.001, format="%.3f")
    penalty = st.sidebar.selectbox("penalty", ["l2", "l1", "elasticnet", "none"], index=0)
    solver = st.sidebar.selectbox("solver", ["lbfgs", "liblinear", "newton-cg", "saga"], index=0)
    max_iter = st.sidebar.slider("max_iter", 10, 10000, 1000, 50)
    tol = st.sidebar.slider("tol", 0.0, 1e-2, 1e-4, step=1e-5, format="%.5f")

    class_weight_ui = st.sidebar.selectbox("class_weight", ["None", "balanced"], index=0)
    class_weight = None if class_weight_ui == "None" else class_weight_ui

    l1_ratio = None
    if penalty == "elasticnet":
        l1_ratio = st.sidebar.slider("l1_ratio", 0.0, 1.0, 0.5, 0.05)

    standardize = st.sidebar.checkbox("Standardize features", value=True)

    params = LRParams(
        C=float(C),
        penalty=str(penalty),
        solver=str(solver),
        max_iter=int(max_iter),
        tol=float(tol),
        class_weight=class_weight,
        l1_ratio=(float(l1_ratio) if l1_ratio is not None else None),
        is_multiclass=False,
        warm_start=False,
        random_state=42,
    )
    return params, bool(standardize)


params, use_standardize = sidebar_model_controls()
ok, msg = validate_params(params)
if not ok:
    st.error(f"Invalid hyperparameter combination: {msg}")
    st.stop()

X, y = load_breast_cancer_dataset()

# scratch split
rng = np.random.default_rng(42)
idx = np.arange(len(X))
rng.shuffle(idx)
split = int(0.8 * len(X))
train_idx, test_idx = idx[:split], idx[split:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

if use_standardize:
    mu, sigma = standardize_fit(X_train)
    X_train = standardize_transform(X_train, mu, sigma)
    X_test = standardize_transform(X_test, mu, sigma)

do_fit = st.button("Train", type="primary")

if do_fit:
    model = fit_final(params, X_train, y_train)
    y_pred = model.predict(X_test)
    m = compute_metrics(y_test, y_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m.accuracy:.4f}")
    c2.metric("Precision (weighted)", f"{m.precision:.4f}")
    c3.metric("Recall (weighted)", f"{m.recall:.4f}")
    c4.metric("F1 (weighted)", f"{m.f1:.4f}")

    st.subheader("Confusion matrix")
    st.dataframe(m.confusion)
