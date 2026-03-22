"""Smoke test: create a snapshot + PDF without running Streamlit.

Usage:
  python apps/logreg_playground/scripts/smoke_snapshot.py
"""

from __future__ import annotations

import sys
from datetime import datetime

import numpy as np


def main() -> int:
    # Make `src.*` importable
    sys.path.insert(0, "apps/logreg_playground")

    from src.datasets import Dataset2DConfig, make_2d_dataset
    from src.metrics import compute_metrics
    from src.reporting import build_pdf_report, make_run_id, save_snapshot
    from src.schema import LRParams
    from src.training import fit_final
    from src.viz import make_mesh, plot_decision_boundary

    # 2D dataset
    cfg = Dataset2DConfig(name="blobs_binary", n_samples=2000, noise=0.2, seed=0)
    X, y = make_2d_dataset(cfg)

    # split
    rng = np.random.default_rng(42)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(0.8 * len(X))
    tr, te = idx[:split], idx[split:]
    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]

    p = LRParams(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=200,
        tol=1e-4,
        class_weight=None,
        is_multiclass=False,
    )

    model = fit_final(p, X_train, y_train)
    pred = model.predict(X_test)
    m = compute_metrics(y_test, pred)

    # decision boundary
    mesh = make_mesh(X, res=200)
    fig = plot_decision_boundary(X=X, y=y, model=model, mesh=mesh, title="Smoke boundary")

    # confusion fig
    import matplotlib.pyplot as plt

    cm = m.confusion
    fig_cm, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    fig_cm.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_cm.tight_layout()

    run_id = make_run_id("smoke_2d")
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "page": "2d",
        "dataset": {"name": cfg.name, "n_samples": cfg.n_samples, "noise": cfg.noise, "seed": cfg.seed},
        "standardize": False,
        "hyperparameters": p,
        "metrics": {
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "confusion": m.confusion,
        },
    }

    run_dir = save_snapshot(run_id=run_id, meta=meta, boundary_fig=fig, confusion_fig=fig_cm)
    pdf = build_pdf_report(run_dir)
    print("created", run_dir)
    print("pdf", pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
