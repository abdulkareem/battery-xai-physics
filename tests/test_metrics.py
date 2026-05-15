import numpy as np

from battery_xai.evaluation.metrics import regression_metrics


def test_regression_metrics_exact_prediction():
    y = np.array([1.0, 0.9, 0.8])
    metrics = regression_metrics(y, y, prefix="soh_")
    assert metrics["soh_rmse"] == 0.0
    assert metrics["soh_mae"] == 0.0
    assert metrics["soh_r2"] == 1.0
