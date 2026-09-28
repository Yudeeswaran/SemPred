from __future__ import annotations

from itertools import pairwise

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true, probabilities, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        mask = (probabilities >= lo) & (probabilities < hi if hi < 1 else probabilities <= hi)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(probabilities[mask].mean() - y_true[mask].mean())
    return float(ece)


def evaluate(y_true, probabilities, threshold=0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = probabilities >= threshold
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "ece": expected_calibration_error(y_true, probabilities),
    }
