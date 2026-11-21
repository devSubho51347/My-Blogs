"""Model training utilities (sklearn LogisticRegression + animation loop)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Generator, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from .schema import LRParams, validate_params


def build_model(p: LRParams, *, max_iter_override: Optional[int] = None) -> LogisticRegression:
    ok, msg = validate_params(p)
    if not ok:
        raise ValueError(msg)

    kwargs = dict(
        C=float(p.C),
        penalty=p.penalty,
        solver=p.solver,
        max_iter=int(p.max_iter if max_iter_override is None else max_iter_override),
        tol=float(p.tol),
        class_weight=p.class_weight,
        warm_start=bool(p.warm_start),
        random_state=int(p.random_state),
    )
    if p.penalty == "elasticnet":
        kwargs["l1_ratio"] = p.l1_ratio

    # Note: sklearn 1.8 emits a FutureWarning about `penalty`.
    # We keep it for tutorial clarity; the app still works.
    # If sklearn changes, we can map penalty->l1_ratio/C here.
    return LogisticRegression(**kwargs)


def fit_final(p: LRParams, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = build_model(p)
    model.fit(X_train, y_train)
    return model


def fit_animated(
    p: LRParams,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_frames: int = 30,
) -> Generator[Tuple[int, LogisticRegression], None, None]:
    """Yield model snapshots for an animation.

    We approximate “epochs/iterations” by repeatedly calling:
    - warm_start=True
    - max_iter=1

    This is best-effort because sklearn solvers differ.
    """

    # Force warm_start for animation
    p2 = LRParams(**{**asdict(p), "warm_start": True})
    model = build_model(p2, max_iter_override=1)

    # Ensure it doesn't early-stop due to tolerance.
    model.set_params(tol=0.0)

    for i in range(int(n_frames)):
        model.fit(X_train, y_train)
        yield i, model
