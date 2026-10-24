from pathlib import Path

import numpy as np
import pandas as pd

from reliabilityml.baseline import regression_metrics, split_engine_ids
from reliabilityml.evaluation import select_observation_rows
from reliabilityml.features import build_features, build_std_features
from reliabilityml.model import make_random_forest_model


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

    observations = select_observation_rows(validation)
    actual_rul = observations["rul"].to_numpy()

    print("=== FD001 validation: 33 vs 48 features ===")
    print(f"Training engines: {len(training_ids)}")
    print(f"Validation endpoints: {len(observations)}")

    for name, feature_builder in (
        ("Original 33", build_features),
        ("33 + std (48)", build_std_features),
    ):
        x_train = feature_builder(training)
        x_validation = feature_builder(validation).loc[observations.index]

        model = make_random_forest_model().set_params(min_samples_leaf=3)
        model.fit(x_train, training["rul"].to_numpy())
        predictions = np.maximum(model.predict(x_validation), 0)

        scores = regression_metrics(actual_rul, predictions)
        print(
            f"{name}: {x_train.shape[1]} features, "
            f"MAE={scores['mae']:.2f}, RMSE={scores['rmse']:.2f} cycles"
        )


if __name__ == "__main__":
    main()