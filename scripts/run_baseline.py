from pathlib import Path

import pandas as pd

from reliabilityml.baseline import (
    predict_median,
    regression_metrics,
    split_engine_ids,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train_FD001.txt"


def main():
    train = pd.read_csv(TRAIN_PATH, sep=r"\s+", header=None)
    train.columns = (
        ["engine_id", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    last_cycle = train.groupby("engine_id")["cycle"].max()
    train["rul"] = train["engine_id"].map(last_cycle) - train["cycle"]

    training_ids, validation_ids = split_engine_ids(train["engine_id"])
    training = train[train["engine_id"].isin(training_ids)]
    validation = train[train["engine_id"].isin(validation_ids)]

    assert set(training_ids).isdisjoint(set(validation_ids))
    assert len(training_ids) == 80 and len(validation_ids) == 20

    predictions = predict_median(training["rul"], len(validation))
    scores = regression_metrics(validation["rul"], predictions)

    print("=== FD001 median baseline ===")
    print(f"Training engines: {len(training_ids)}, rows: {len(training)}")
    print(f"Validation engines: {len(validation_ids)}, rows: {len(validation)}")
    print(f"Training median RUL: {predictions[0]:.1f} cycles")
    print(f"Validation MAE: {scores['mae']:.2f} cycles")
    print(f"Validation RMSE: {scores['rmse']:.2f} cycles")


if __name__ == "__main__":
    main()