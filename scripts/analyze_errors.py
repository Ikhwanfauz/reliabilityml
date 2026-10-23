from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from reliabilityml.baseline import split_engine_ids
from reliabilityml.evaluation import select_observation_rows
from reliabilityml.features import build_std_features
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

    training_ids, _ = split_engine_ids(history["engine_id"])
    training = history[history["engine_id"].isin(training_ids)]

    records = []
    folds = GroupKFold(n_splits=3)

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
        x_fit = build_std_features(fold_training)
        x_check_all = build_std_features(fold_validation)
        x_observed = x_check_all.loc[observations.index]

        model = make_random_forest_model().set_params(
            min_samples_leaf=3
        )
        model.fit(x_fit, fold_training["rul"].to_numpy())
        predictions = np.maximum(model.predict(x_observed), 0)

        records.append(
            pd.DataFrame(
                {
                    "fold": fold_number,
                    "engine_id": observations["engine_id"].astype(int).to_numpy(),
                    "cycle": observations["cycle"].astype(int).to_numpy(),
                    "actual_rul": observations["rul"].to_numpy(),
                    "predicted_rul": predictions,
                }
            )
        )

    results = pd.concat(records, ignore_index=True)
    results["signed_error"] = results["predicted_rul"] - results["actual_rul"]
    results["absolute_error"] = results["signed_error"].abs()

    assert len(results) == 80
    assert results["engine_id"].nunique() == 80

    results["rul_band"] = pd.cut(
        results["actual_rul"],
        bins=[-1, 30, 100, float("inf")],
        labels=["0-30", "31-100", ">100"],
    )

    print("=== FD001 grouped CV endpoint error analysis ===")
    print(f"Engines evaluated once each: {len(results)}")
    print(f"Overall endpoint MAE: {results['absolute_error'].mean():.2f} cycles")
    print(
        "Positive signed error = predicted more remaining cycles "
        "than actually remained"
    )

    print("\nBy actual RUL band:")
    summary = results.groupby("rul_band", observed=True).agg(
        engines=("engine_id", "count"),
        mae=("absolute_error", "mean"),
        mean_signed_error=("signed_error", "mean"),
    )
    print(summary.round(2).to_string())

    print("\nFive largest errors:")
    columns = [
        "fold", "engine_id", "cycle", "actual_rul",
        "predicted_rul", "signed_error", "absolute_error",
    ]
    print(
        results.nlargest(5, "absolute_error")[columns]
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()