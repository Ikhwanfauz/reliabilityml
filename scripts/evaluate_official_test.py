from pathlib import Path

import numpy as np
import pandas as pd

from reliabilityml.baseline import predict_median, regression_metrics
from reliabilityml.features import build_features
from reliabilityml.model import make_random_forest_model


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def main():
    train = pd.read_csv(RAW / "train_FD001.txt", sep=r"\s+", header=None, names=COLUMNS)
    test = pd.read_csv(RAW / "test_FD001.txt", sep=r"\s+", header=None, names=COLUMNS)

    assert train["engine_id"].nunique() == 100
    assert test["engine_id"].nunique() == 100

    final_cycles = train.groupby("engine_id")["cycle"].max()
    train["rul"] = train["engine_id"].map(final_cycles) - train["cycle"]

    x_train = build_features(train)
    y_train = train["rul"].to_numpy()

    test_features = build_features(test)
    endpoints = test.groupby("engine_id", sort=True).tail(1).sort_values("engine_id")
    assert endpoints["engine_id"].tolist() == list(range(1, 101))
    x_test = test_features.loc[endpoints.index]

    model = make_random_forest_model().set_params(min_samples_leaf=3)
    model.fit(x_train, y_train)
    predictions = np.maximum(model.predict(x_test), 0)

    # Read official answers only after predictions have been produced.
    answers = pd.read_csv(RAW / "RUL_FD001.txt", sep=r"\s+", header=None)
    assert answers.shape == (100, 1)
    actual = answers.iloc[:, 0].to_numpy()

    baseline = predict_median(y_train, len(actual))
    baseline_scores = regression_metrics(actual, baseline)
    model_scores = regression_metrics(actual, predictions)

    print("=== FD001 official test: 100 engine endpoints ===")
    print("CV-selected model: Random Forest, min_samples_leaf=3")
    print(f"Training rows: {len(train)}")
    print(f"Test endpoints: {len(endpoints)}")
    print(
        f"Median baseline: MAE={baseline_scores['mae']:.2f}, "
        f"RMSE={baseline_scores['rmse']:.2f} cycles"
    )
    print(
        f"Random Forest: MAE={model_scores['mae']:.2f}, "
        f"RMSE={model_scores['rmse']:.2f} cycles"
    )


if __name__ == "__main__":
    main()