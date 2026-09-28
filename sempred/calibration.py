from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def calibrate_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit scalar temperature on held-out logits using NLL."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)

    def loss(log_t: np.ndarray) -> float:
        t = float(np.exp(log_t[0]))
        z = logits / t
        z = np.clip(z, -50, 50)
        p = 1 / (1 + np.exp(-z))
        eps = 1e-12
        return float(-np.mean(labels * np.log(p + eps) + (1 - labels) * np.log(1 - p + eps)))

    result = minimize(loss, np.array([0.0]), method="L-BFGS-B")
    return float(np.exp(result.x[0]))


def calibrate_binary_scores(probabilities, labels) -> tuple[float, float]:
    """Fit a monotone Platt calibrator: sigmoid(scale * logit(p) + bias).

    Use a representative held-out calibration set; never fit this on the final
    evaluation set. A nonnegative slope preserves the ranking of the model.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probabilities.ndim != 1 or labels.ndim != 1 or len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must be one-dimensional arrays of equal length")
    if len(labels) < 2 or set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("calibration labels must contain both binary classes")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite values between 0 and 1")

    eps = np.finfo(float).eps
    logits = np.log(np.clip(probabilities, eps, 1 - eps) / np.clip(1 - probabilities, eps, 1 - eps))

    def loss(params: np.ndarray) -> float:
        scaled = params[0] * logits + params[1]
        return float(np.mean(np.logaddexp(0.0, scaled) - labels * scaled))

    result = minimize(loss, np.array([1.0, 0.0]), method="L-BFGS-B", bounds=((0.0, None), (None, None)))
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"calibration did not converge: {result.message}")
    return float(result.x[0]), float(result.x[1])


def apply_binary_calibration(probabilities, scale: float, bias: float) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    eps = np.finfo(float).eps
    clipped = np.clip(probabilities, eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped))
    return expit(scale * logits + bias)
