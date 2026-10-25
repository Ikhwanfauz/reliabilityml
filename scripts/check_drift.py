import json
import sqlite3
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from reliabilityml.drift import compare_mean_shift
from reliabilityml.features import SENSORS, build_std_features


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "raw" / "test_FD001.txt"
DB_PATH = ROOT / "monitoring.db"
MODEL_URI = "models:/ReliabilityML-FD001-RUL/2"
INPUT_COLUMNS = ["setting_1", "setting_2", *SENSORS]
MONITORED_COLUMNS = [*INPUT_COLUMNS, "predicted_rul_cycles"]
DATA_COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model = mlflow.sklearn.load_model(MODEL_URI)

    test = pd.read_csv(
        TEST_PATH, sep=r"\s+", header=None, names=DATA_COLUMNS
    )
    endpoints = (
        test.groupby("engine_id", sort=True)
        .tail(1)
        .sort_values("engine_id")
    )
    features = build_std_features(test)
    assert list(model.feature_names_in_) == features.columns.tolist()

    reference = endpoints[INPUT_COLUMNS].copy()
    reference["predicted_rul_cycles"] = np.maximum(
        model.predict(features.loc[endpoints.index]), 0
    )

    if not DB_PATH.exists():
        print("No monitoring.db yet: collect API prediction events first")
        return

    with sqlite3.connect(DB_PATH) as connection:
        events = pd.read_sql_query(
            """
            SELECT engine_id, predicted_rul_cycles, endpoint_inputs_json
            FROM prediction_events
            WHERE model_uri = ?
            ORDER BY id
            """,
            connection,
            params=(MODEL_URI,),
        )

    events = events.drop_duplicates("engine_id", keep="last")
    print("=== FD001 input and prediction drift check ===")
    print(f"Reference endpoint engines: {len(reference)}")
    print(f"Recorded distinct engines for v2: {len(events)}")

    if len(events) < 30:
        print("Insufficient current data: need at least 30 distinct engines")
        return

    current = pd.DataFrame(
        [
            {
                **json.loads(row.endpoint_inputs_json),
                "predicted_rul_cycles": row.predicted_rul_cycles,
            }
            for row in events.itertuples()
        ]
    )

    report = compare_mean_shift(
        reference, current, MONITORED_COLUMNS
    )
    for item in sorted(
        report,
        key=lambda row: row["shift_in_reference_std"],
        reverse=True,
    ):
        status = "ALERT" if item["alert"] else "OK"
        print(
            f"{status:5} {item['column']:22} "
            f"shift={item['shift_in_reference_std']:.2f} "
            f"reference_mean={item['reference_mean']:.2f} "
            f"current_mean={item['current_mean']:.2f}"
        )


if __name__ == "__main__":
    main()