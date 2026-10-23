from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from reliabilityml.baseline import regression_metrics, split_engine_ids
from reliabilityml.evaluation import select_observation_rows
from reliabilityml.features import (
    BASE_FEATURES,
    SENSORS,
    build_sequence_features,
)
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

    original = BASE_FEATURES + [
        f"{sensor}_mean_5" for sensor in SENSORS
    ]
    additions = {
        "std": [f"{sensor}_std_5" for sensor in SENSORS],
        "lag": [f"{sensor}_lag_1" for sensor in SENSORS],
        "change": [f"{sensor}_change_1" for sensor in SENSORS],
        "trend": [f"{sensor}_trend_5" for sensor in SENSORS],
    }
    feature_sets = {"Original 33": original}
    for name, columns in additions.items():
        feature_sets[f"33 + {name}"] = original + columns
        feature_sets["All 93"] = original + [
            f"{sensor}_{suffix}"
            for sensor in SENSORS
            for suffix in ("std_5", "lag_1", "change_1", "trend_5")
        ]

    scores = {name: [] for name in feature_sets}
    folds = GroupKFold(n_splits=3)

    print("=== FD001 grouped feature ablation ===", flush=True)
    print(
        f"CV engines: {len(training_ids)}; "
        f"untouched validation engines: {len(validation_ids)}",
        flush=True,
    )

    for fold_number, (fit_rows, check_rows) in enumerate(
        folds.split(training, groups=training["engine_id"]),
        start=1,
    ):
        fold_training = training.iloc[fit_rows]
        fold_validation = training.iloc[check_rows]

        assert set(fold_training["engine_id"]).isdisjoint(
            set(fold_validation["engine_id"])
        )

        observations = select_observation_rows(fold_validation)
        x_fit_all = build_sequence_features(fold_training)
        assert feature_sets["All 93"] == x_fit_all.columns.tolist()
        x_check_all = build_sequence_features(fold_validation)
        y_fit = fold_training["rul"].to_numpy()
        y_check = observations["rul"].to_numpy()

        print(f"Fold {fold_number}:", flush=True)
        for name, columns in feature_sets.items():
            model = make_random_forest_model().set_params(
                min_samples_leaf=3
            )
            model.fit(x_fit_all[columns], y_fit)

            x_observed = x_check_all.loc[observations.index, columns]
            predictions = np.maximum(model.predict(x_observed), 0)
            mae = regression_metrics(y_check, predictions)["mae"]
            scores[name].append(mae)

            print(
                f"  {name} ({len(columns)} columns): "
                f"MAE={mae:.2f}",
                flush=True,
            )

    print("\nMean endpoint MAE across 3 folds (lower is better):")
    ranked = sorted(
        scores.items(),
        key=lambda item: np.mean(item[1]),
    )
    for name, fold_scores in ranked:
        print(f"{name}: {np.mean(fold_scores):.2f} cycles")


if __name__ == "__main__":
    main()