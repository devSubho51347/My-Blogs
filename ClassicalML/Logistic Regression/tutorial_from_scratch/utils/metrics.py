"""Scratch metrics utilities (binary classification)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")
    return float((y_true == y_pred).mean())


@dataclass
class ConfusionMatrix:
    tn: int
    fp: int
    fn: int
    tp: int

    def as_array(self) -> np.ndarray:
        return np.array([[self.tn, self.fp], [self.fn, self.tp]], dtype=int)


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMatrix:
    """Compute TN/FP/FN/TP."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    return ConfusionMatrix(tn=tn, fp=fp, fn=fn, tp=tp)


def precision_recall_f1(cm: ConfusionMatrix):
    precision = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) else 0.0
    recall = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)
