from pathlib import Path

import numpy as np
import pandas as pd

from reliabilityml.baseline import (
    predict_median,
    regression_metrics,
    split_engine_ids,
)
from reliabilityml.evaluation import select_observation_rows
from reliabilityml.features import build_features
from reliabilityml.model import (
    make_boosted_model,
    make_random_forest_model,
    make_ridge_model,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train_FD001.txt"


def main():
    history = pd.read_csv(TRAIN_PATH, sep=r"\s+", header=None)
    history.columns = (
        ["engine_id", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    final_cycles = history.groupby("engine_id")["cycle"].max()
    history["rul"] = history["engine_id"].map(final_cycles) - history["cycle"]

    training_ids, validation_ids = split_engine_ids(history["engine_id"])
    training = history[history["engine_id"].isin(training_ids)]
    validation = history[history["engine_id"].isin(validation_ids)]

    x_train = build_features(training)
    x_validation_all = build_features(validation)
    observations = select_observation_rows(validation)
    x_observed = x_validation_all.loc[observations.index]

    y_train = training["rul"].to_numpy()
    y_observed = observations["rul"].to_numpy()

    results = {
        "Median baseline": regression_metrics(
            y_observed,
            predict_median(y_train, len(observations)),
        )
    }

    models = {
        "Ridge": make_ridge_model(),
        "Random Forest": make_random_forest_model(),
        "Boosted Trees": make_boosted_model(),
    }

    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = np.maximum(model.predict(x_observed), 0)
        results[name] = regression_metrics(y_observed, predictions)

    print("=== FD001 simulated endpoint model comparison ===")
    print(f"Training engines: {len(training_ids)}")
    print(f"Validation engines: {len(validation_ids)}")
    print(f"Observation rows evaluated: {len(observations)}")

    for name, scores in results.items():
        print(
            f"{name}: MAE={scores['mae']:.2f}, "
            f"RMSE={scores['rmse']:.2f} cycles"
        )


if __name__ == "__main__":
    main()