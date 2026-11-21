"""Schema/validation helpers for LogisticRegression hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LRParams:
    C: float
    penalty: str
    solver: str
    max_iter: int
    tol: float
    class_weight: Optional[str]
    l1_ratio: Optional[float] = None  # elasticnet only
    is_multiclass: bool = False
    warm_start: bool = False
    random_state: int = 42


def validate_params(p: LRParams) -> tuple[bool, str]:
    """Return (ok, message)."""
    if p.C <= 0:
        return False, "C must be > 0"
    if p.max_iter <= 0:
        return False, "max_iter must be >= 1"
    if p.tol < 0:
        return False, "tol must be >= 0"

    penalty = p.penalty
    solver = p.solver

    if penalty == "none":
        if solver == "liblinear":
            return False, "penalty='none' is not supported with solver='liblinear'"

    if penalty == "l1":
        if solver not in {"liblinear", "saga"}:
            return False, "penalty='l1' requires solver='liblinear' or 'saga'"

    if penalty == "elasticnet":
        if solver != "saga":
            return False, "penalty='elasticnet' requires solver='saga'"
        if p.l1_ratio is None:
            return False, "l1_ratio must be set for elasticnet"

    if penalty == "l2":
        if solver not in {"lbfgs", "liblinear", "newton-cg", "saga"}:
            return False, "penalty='l2' requires one of: lbfgs, liblinear, newton-cg, saga"

    if p.is_multiclass and solver == "liblinear":
        return False, "multiclass is not supported with solver='liblinear'"

    return True, "OK"
