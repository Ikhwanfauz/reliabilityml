import numpy as np


def split_engine_ids(engine_ids, seed=42):
    ids = np.sort(np.unique(np.asarray(list(engine_ids))))
    if len(ids) < 2:
        raise ValueError("Need at least two engines")

    shuffled = np.random.default_rng(seed).permutation(ids)
    validation_count = max(1, int(np.ceil(len(ids) * 0.20)))

    validation_ids = shuffled[:validation_count]
    training_ids = shuffled[validation_count:]
    return training_ids, validation_ids


def predict_median(training_rul, count):
    values = np.asarray(training_rul, dtype=float)
    if values.size == 0 or count < 0:
        raise ValueError("Need training RUL values and a nonnegative count")

    median_rul = float(np.median(values))
    return np.full(count, median_rul)


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Actual and predicted values must have matching nonempty shapes")

    errors = actual - predicted
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }