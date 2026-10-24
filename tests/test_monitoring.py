import json
import sqlite3

from reliabilityml.monitoring import record_prediction


def test_record_prediction_persists_event(tmp_path):
    db_path = tmp_path / "events.db"

    record_prediction(
        db_path,
        engine_id=7,
        observed_cycle=31,
        predicted_rul_cycles=42.5,
        model_uri="models:/ReliabilityML-FD001-RUL/2",
        endpoint_inputs={"setting_1": 0.1, "sensor_2": 642.3},
    )

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT recorded_at_utc, engine_id, observed_cycle,
                   predicted_rul_cycles, model_uri, endpoint_inputs_json
            FROM prediction_events
            """
        ).fetchone()

    assert row[0].endswith("+00:00")
    assert row[1:5] == (
        7,
        31,
        42.5,
        "models:/ReliabilityML-FD001-RUL/2",
    )
    assert json.loads(row[5]) == {"setting_1": 0.1, "sensor_2": 642.3}