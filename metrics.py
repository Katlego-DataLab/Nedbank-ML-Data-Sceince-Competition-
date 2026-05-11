import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

log = logging.getLogger("nedbank.metrics")


def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def evaluate(y_true, y_pred, label=""):
    metrics = {
        "RMSLE": rmsle(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "MAE":   mae(y_true, y_pred),
    }
    prefix = f"[{label}] " if label else ""
    for k, v in metrics.items():
        log.info(f"  {prefix}{k}: {v:.4f}")
    return metrics
