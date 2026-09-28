import numpy as np

from sempred.calibration import apply_binary_calibration, calibrate_binary_scores


def test_binary_calibration_improves_offset_scores():
    raw = np.array([0.05, 0.15, 0.2, 0.3, 0.7, 0.8, 0.85, 0.95])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scale, bias = calibrate_binary_scores(raw, labels)
    calibrated = apply_binary_calibration(raw, scale, bias)

    assert scale >= 0
    assert np.all((calibrated >= 0) & (calibrated <= 1))
    assert np.mean((calibrated - labels) ** 2) < np.mean((raw - labels) ** 2)
