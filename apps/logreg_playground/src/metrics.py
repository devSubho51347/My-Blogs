"""Metrics wrappers (sklearn-backed)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    confusion: np.ndarray
    precision: float
    recall: float
    f1: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return Metrics(
        accuracy=acc,
        confusion=cm,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )
