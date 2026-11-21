from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import streamlit as st

from src.datasets import Dataset2DConfig, make_2d_dataset
from src.metrics import compute_metrics
from src.preprocessing import standardize_fit, standardize_transform
from src.reporting import build_pdf_report, make_run_id, save_snapshot
from src.schema import LRParams, validate_params
from src.training import fit_animated, fit_final
from src.viz import make_mesh, plot_decision_boundary


st.set_page_config(page_title="2D Decision Boundary", layout="wide")
st.title("2D Decision Boundary (Animated)")

st.markdown(
    """
This page uses **2D synthetic datasets** so we can **draw** the decision boundary.

Animation is done by repeatedly calling `LogisticRegression.fit()` with:
- `warm_start=True`
- `max_iter=1`

This produces a *best-effort* visualization of how the boundary changes as optimization progresses.
"""
)


def sidebar_dataset_controls() -> Dataset2DConfig:
    st.sidebar.header("Dataset")

    name_ui = st.sidebar.selectbox(
        "Dataset",
        options=[
            ("Linearly separable (Blobs, binary)", "blobs_binary"),
            ("Non-linear (Moons)", "moons"),
            ("Non-linear (Circles)", "circles"),
            ("Multiclass (Blobs, 3 classes)", "blobs_multiclass"),
        ],
        format_func=lambda x: x[0],
    )
    name = name_ui[1]

    # Requested range: 1,000 to 100,000
    n_samples = st.sidebar.slider("n_samples", 1000, 100000, 5000, 1000)
    noise = st.sidebar.slider("noise", 0.0, 1.0, 0.20, 0.05)
    seed = st.sidebar.number_input("seed", min_value=0, max_value=999999, value=42, step=1)

    plot_subsample = st.sidebar.slider(
        "plot subsample (for speed)",
        500,
        20000,
        5000,
        500,
        help="Training can use all samples, but plotting 100k points is slow/cluttered. This limits plotted points.",
    )

    centers = 3
    if name == "blobs_multiclass":
        centers = st.sidebar.slider("n_classes", 3, 5, 3, 1)

    # attach plot_subsample via session_state (keeps Dataset2DConfig simple)
    st.session_state["plot_subsample"] = int(min(plot_subsample, n_samples))

    return Dataset2DConfig(
        name=name, n_samples=int(n_samples), noise=float(noise), seed=int(seed), centers=int(centers)
    )


def sidebar_model_controls(is_multiclass: bool) -> tuple[LRParams, bool, int, int]:
    st.sidebar.header("Model (sklearn LogisticRegression)")

    C = st.sidebar.slider("C (inverse regularization)", 0.001, 100.0, 1.0, step=0.001, format="%.3f")
    penalty = st.sidebar.selectbox("penalty", ["l2", "l1", "elasticnet", "none"], index=0)
    solver = st.sidebar.selectbox("solver", ["lbfgs", "liblinear", "newton-cg", "saga"], index=0)
    max_iter = st.sidebar.slider("max_iter (final fit)", 10, 5000, 200, 10)
    tol = st.sidebar.slider("tol", 0.0, 1e-2, 1e-4, step=1e-5, format="%.5f")

    class_weight_ui = st.sidebar.selectbox("class_weight", ["None", "balanced"], index=0)
    class_weight = None if class_weight_ui == "None" else class_weight_ui

    l1_ratio = None
    if penalty == "elasticnet":
        l1_ratio = st.sidebar.slider("l1_ratio", 0.0, 1.0, 0.5, 0.05)

    standardize = st.sidebar.checkbox("Standardize features", value=True)

    st.sidebar.header("Animation")
    n_frames = st.sidebar.slider("frames", 5, 200, 40, 5)
    frame_delay_ms = st.sidebar.slider("frame delay (ms)", 0, 500, 60, 10)

    params = LRParams(
        C=float(C),
        penalty=str(penalty),
        solver=str(solver),
        max_iter=int(max_iter),
        tol=float(tol),
        class_weight=class_weight,
        l1_ratio=(float(l1_ratio) if l1_ratio is not None else None),
        is_multiclass=bool(is_multiclass),
        warm_start=False,
        random_state=42,
    )
    return params, bool(standardize), int(n_frames), int(frame_delay_ms)


@st.cache_data(show_spinner=False)
def cached_dataset(cfg: Dataset2DConfig):
    return make_2d_dataset(cfg)


cfg = sidebar_dataset_controls()
X, y = cached_dataset(cfg)
is_multiclass = int(np.unique(y).size) > 2

# Subsample for plotting only (keeps UI responsive for huge n_samples)
plot_subsample = int(st.session_state.get("plot_subsample", min(5000, len(X))))
if plot_subsample < len(X):
    rng_plot = np.random.default_rng(0)
    plot_idx = rng_plot.choice(len(X), size=plot_subsample, replace=False)
    X_plot, y_plot = X[plot_idx], y[plot_idx]
else:
    X_plot, y_plot = X, y

params, use_standardize, n_frames, frame_delay_ms = sidebar_model_controls(is_multiclass=is_multiclass)

ok, msg = validate_params(params)
if not ok:
    st.error(f"Invalid hyperparameter combination: {msg}")
    st.stop()

# Train/test split (simple)
rng = np.random.default_rng(42)
idx = np.arange(len(X))
rng.shuffle(idx)
split = int(0.8 * len(X))
train_idx, test_idx = idx[:split], idx[split:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

if use_standardize:
    mu, sigma = standardize_fit(X_train)
    X_train_s = standardize_transform(X_train, mu, sigma)
    X_test_s = standardize_transform(X_test, mu, sigma)
    X_all_s = standardize_transform(X, mu, sigma)
else:
    X_train_s, X_test_s, X_all_s = X_train, X_test, X

mesh = make_mesh(X_all_s, res=250)

col_left, col_right = st.columns([2, 1], gap="large")

with col_right:
    st.subheader("Controls")
    do_fit = st.button("Train (final fit)", type="primary")
    do_animate = st.button("Animate training")
    capture = st.button("Capture snapshot (save + PDF)")
    st.caption("Tip: for non-linear datasets (moons/circles), LR will struggle unless you add features.")


with col_left:
    st.subheader("Decision boundary")
    plot_slot = st.empty()
    metrics_slot = st.empty()


def confusion_matrix_figure(cm: np.ndarray, title: str = "Confusion matrix"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def show_model(model, title: str):
    # Plot uses (optionally) subsampled points; decision surface is always full mesh.
    if use_standardize:
        X_plot_s = standardize_transform(X_plot, mu, sigma)
    else:
        X_plot_s = X_plot

    fig = plot_decision_boundary(X=X_plot_s, y=y_plot, model=model, mesh=mesh, title=title)
    plot_slot.pyplot(fig, clear_figure=True)

    y_pred = model.predict(X_test_s)
    m = compute_metrics(y_test, y_pred)
    metrics_slot.markdown(
        f"""
**Test metrics**
- accuracy: **{m.accuracy:.4f}**
- precision (weighted): **{m.precision:.4f}**
- recall (weighted): **{m.recall:.4f}**
- f1 (weighted): **{m.f1:.4f}**
"""
    )

    # store last shown results in session_state for snapshot capture
    st.session_state["last_model"] = model
    st.session_state["last_boundary_fig"] = fig
    st.session_state["last_metrics"] = m
    st.session_state["last_y_test"] = y_test
    st.session_state["last_y_pred"] = y_pred


if do_fit:
    model = fit_final(params, X_train_s, y_train)
    show_model(model, title="Final fit")

if do_animate:
    # basic animation loop
    for i, model in fit_animated(params, X_train_s, y_train, n_frames=n_frames):
        show_model(model, title=f"Frame {i+1}/{n_frames} (warm_start, max_iter=1)")
        if frame_delay_ms > 0:
            time.sleep(frame_delay_ms / 1000)

if capture:
    model = st.session_state.get("last_model")
    boundary_fig = st.session_state.get("last_boundary_fig")
    m = st.session_state.get("last_metrics")
    y_pred = st.session_state.get("last_y_pred")

    if model is None or boundary_fig is None or m is None or y_pred is None:
        st.warning("Nothing to capture yet. Train (final) or run animation first.")
    else:
        run_id = make_run_id(prefix=f"2d_{cfg.name}")
        cm_fig = confusion_matrix_figure(m.confusion, title="Confusion matrix (test)")

        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "page": "2d",
            "dataset": {
                "name": cfg.name,
                "n_samples": cfg.n_samples,
                "noise": cfg.noise,
                "seed": cfg.seed,
                "centers": cfg.centers,
                "plot_subsample": int(st.session_state.get("plot_subsample", 0)),
            },
            "standardize": bool(use_standardize),
            "hyperparameters": params,
            "metrics": {
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "confusion": m.confusion,
            },
        }

        run_dir = save_snapshot(run_id=run_id, meta=meta, boundary_fig=boundary_fig, confusion_fig=cm_fig)
        pdf_path = build_pdf_report(run_dir)
        st.success(f"Saved snapshot to: {run_dir}")
        st.success(f"PDF report: {pdf_path}")
