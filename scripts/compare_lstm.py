from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from reliabilityml.baseline import regression_metrics, split_engine_ids
from reliabilityml.evaluation import select_observation_rows
from reliabilityml.features import build_std_features
from reliabilityml.model import make_random_forest_model
from reliabilityml.tf_model import make_lstm_model
from reliabilityml.windows import build_windows


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

    scores = {"Forest 48": [], "LSTM 30x18": []}
    folds = GroupKFold(n_splits=3)

    print("=== FD001 grouped endpoint: Forest vs LSTM ===", flush=True)
    print(
        f"CV engines: {len(training_ids)}; "
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

        observations = select_observation_rows(fold_validation)
        y_check = observations["rul"].to_numpy()

        x_fit_seq, fit_endpoints = build_windows(
            fold_training, window_size=30
        )
        y_fit = fold_training.loc[fit_endpoints, "rul"].to_numpy(
            dtype=np.float32
        )

        x_check_all, check_endpoints = build_windows(
            fold_validation, window_size=30
        )
        positions = check_endpoints.get_indexer(observations.index)
        if (positions < 0).any():
            raise ValueError("Observation is missing a complete window")
        x_check_seq = x_check_all[positions]

        print(
            f"Fold {fold_number}: {len(y_fit)} training windows, "
            f"{len(y_check)} endpoint predictions",
            flush=True,
        )

        forest_x_fit = build_std_features(fold_training).loc[fit_endpoints]
        forest_x_check = build_std_features(fold_validation).loc[
            observations.index
        ]
        forest = make_random_forest_model().set_params(
            min_samples_leaf=3
        )
        forest.fit(forest_x_fit, y_fit)
        forest_predictions = np.maximum(
            forest.predict(forest_x_check), 0
        )
        forest_mae = regression_metrics(
            y_check, forest_predictions
        )["mae"]
        scores["Forest 48"].append(forest_mae)
        print(f"  Forest 48: MAE={forest_mae:.2f}", flush=True)

        print("  Training LSTM on CPU...", flush=True)
        lstm = make_lstm_model(x_fit_seq)
        lstm.fit(
            x_fit_seq,
            y_fit,
            epochs=8,
            batch_size=256,
            verbose=0,
        )
        lstm_predictions = np.maximum(
            lstm.predict(x_check_seq, verbose=0).reshape(-1), 0
        )
        lstm_mae = regression_metrics(
            y_check, lstm_predictions
        )["mae"]
        scores["LSTM 30x18"].append(lstm_mae)
        print(f"  LSTM 30x18: MAE={lstm_mae:.2f}", flush=True)

    print("\nMean endpoint MAE across 3 folds (lower is better):")
    for name, fold_scores in scores.items():
        print(f"{name}: {np.mean(fold_scores):.2f} cycles")


if __name__ == "__main__":
    main()