from pathlib import Path

import numpy as np
import pandas as pd

from reliabilityml.baseline import regression_metrics
from reliabilityml.features import build_features, build_std_features
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

    endpoints = test.groupby("engine_id", sort=True).tail(1).sort_values("engine_id")
    assert endpoints["engine_id"].tolist() == list(range(1, 101))

    predictions_by_features = {}

    for name, feature_builder in (
        ("Original 33", build_features),
        ("33 + std (48)", build_std_features),
    ):
        x_train = feature_builder(train)
        x_test = feature_builder(test).loc[endpoints.index]

        model = make_random_forest_model().set_params(min_samples_leaf=3)
        model.fit(x_train, train["rul"].to_numpy())
        predictions_by_features[name] = (
            x_train.shape[1],
            np.maximum(model.predict(x_test), 0),
        )

    # Read answers after producing predictions for both fixed candidates.
    answers = pd.read_csv(RAW / "RUL_FD001.txt", sep=r"\s+", header=None)
    assert answers.shape == (100, 1)
    actual = answers.iloc[:, 0].to_numpy()

    print("=== FD001 official test: 33 vs 48 features ===")
    print(f"Training engines: {train['engine_id'].nunique()}")
    print(f"Test endpoints: {len(endpoints)}")

    for name, (feature_count, predictions) in predictions_by_features.items():
        scores = regression_metrics(actual, predictions)
        print(
            f"{name}: {feature_count} features, "
            f"MAE={scores['mae']:.2f}, RMSE={scores['rmse']:.2f} cycles"
        )


if __name__ == "__main__":
    main()