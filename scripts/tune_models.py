from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from reliabilityml.baseline import regression_metrics, split_engine_ids
from reliabilityml.evaluation import select_observation_rows
from reliabilityml.features import build_features, build_std_features
from reliabilityml.model import (
    make_boosted_model,
    make_random_forest_model,
    make_ridge_model,
    make_xgboost_model,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train_FD001.txt"


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--features",
        choices=("original", "std"),
        default="original",
    )
    args = parser.parse_args()
    build = build_features if args.features == "original" else build_std_features
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

    candidates = {
        "Ridge alpha=10": make_ridge_model,
        "Forest leaf=3": lambda: make_random_forest_model().set_params(
            min_samples_leaf=3
        ),
        "Forest leaf=5": make_random_forest_model,
        "Boosted leaves=15": make_boosted_model,
        "Boosted leaves=31": lambda: make_boosted_model().set_params(
            max_leaf_nodes=31
        ),
        "XGBoost depth=3": make_xgboost_model,
        "XGBoost depth=5": lambda: make_xgboost_model().set_params(
            max_depth=5
        ),
    }
    mae_by_candidate = {name: [] for name in candidates}

    folds = GroupKFold(n_splits=3)
    print("=== FD001 grouped 3-fold model tuning ===", flush=True)
    print(
        f"Tuning engines: {len(training_ids)}; "
        f"separate validation engines: {len(validation_ids)}",
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

        x_train = build(fold_training)
        x_validation_all = build(fold_validation)
        observations = select_observation_rows(fold_validation)
        x_observed = x_validation_all.loc[observations.index]

        y_train = fold_training["rul"].to_numpy()
        y_observed = observations["rul"].to_numpy()

        print(
            f"Fold {fold_number}: "
            f"{fold_training['engine_id'].nunique()} fit engines, "
            f"{len(observations)} observation engines",
            flush=True,
        )

        for name, factory in candidates.items():
            model = factory()
            model.fit(x_train, y_train)
            predictions = np.maximum(model.predict(x_observed), 0)
            mae = regression_metrics(y_observed, predictions)["mae"]
            mae_by_candidate[name].append(mae)
            print(f"  {name}: MAE={mae:.2f}", flush=True)

    print("\nMean MAE across 3 folds (lower is better):")
    ranked = sorted(
        mae_by_candidate.items(),
        key=lambda item: np.mean(item[1]),
    )
    for name, fold_maes in ranked:
        print(f"{name}: {np.mean(fold_maes):.2f} cycles")

    print(f"CV-selected setting: {ranked[0][0]}")


if __name__ == "__main__":
    main()