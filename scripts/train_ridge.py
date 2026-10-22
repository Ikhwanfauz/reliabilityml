from pathlib import Path

import numpy as np
import pandas as pd

from reliabilityml.baseline import (
    predict_median,
    regression_metrics,
    split_engine_ids,
)
from reliabilityml.features import build_features
from reliabilityml.model import make_ridge_model


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train_FD001.txt"


def main():
    history = pd.read_csv(TRAIN_PATH, sep=r"\s+", header=None)
    history.columns = (
        ["engine_id", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    last_cycle = history.groupby("engine_id")["cycle"].max()
    history["rul"] = history["engine_id"].map(last_cycle) - history["cycle"]

    training_ids, validation_ids = split_engine_ids(history["engine_id"])
    training = history[history["engine_id"].isin(training_ids)]
    validation = history[history["engine_id"].isin(validation_ids)]

    x_train = build_features(training)
    x_validation = build_features(validation)
    y_train = training["rul"].to_numpy()
    y_validation = validation["rul"].to_numpy()

    model = make_ridge_model()
    model.fit(x_train, y_train)
    predictions = np.maximum(model.predict(x_validation), 0)

    baseline = predict_median(y_train, len(y_validation))
    baseline_scores = regression_metrics(y_validation, baseline)
    model_scores = regression_metrics(y_validation, predictions)

    print("=== FD001 grouped validation: all rows ===")
    print(f"Training engines: {len(training_ids)}")
    print(f"Validation engines: {len(validation_ids)}")
    print(f"Features per row: {x_train.shape[1]}")
    print(f"Median baseline MAE: {baseline_scores['mae']:.2f} cycles")
    print(f"Ridge MAE: {model_scores['mae']:.2f} cycles")
    print(f"Median baseline RMSE: {baseline_scores['rmse']:.2f} cycles")
    print(f"Ridge RMSE: {model_scores['rmse']:.2f} cycles")


if __name__ == "__main__":
    main()