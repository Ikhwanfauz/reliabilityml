import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def record_prediction(
    db_path: str | Path,
    *,
    engine_id: int,
    observed_cycle: int,
    predicted_rul_cycles: float,
    model_uri: str,
    endpoint_inputs: dict[str, float],
) -> None:
    inputs_json = json.dumps(endpoint_inputs, sort_keys=True, allow_nan=False)
    recorded_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_events (
                id INTEGER PRIMARY KEY,
                recorded_at_utc TEXT NOT NULL,
                engine_id INTEGER NOT NULL,
                observed_cycle INTEGER NOT NULL,
                predicted_rul_cycles REAL NOT NULL,
                model_uri TEXT NOT NULL,
                endpoint_inputs_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO prediction_events (
                recorded_at_utc, engine_id, observed_cycle,
                predicted_rul_cycles, model_uri, endpoint_inputs_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                recorded_at,
                engine_id,
                observed_cycle,
                predicted_rul_cycles,
                model_uri,
                inputs_json,
            ),
        )