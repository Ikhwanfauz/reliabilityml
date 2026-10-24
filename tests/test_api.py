from fastapi.testclient import TestClient

from reliabilityml import api
from reliabilityml.features import SENSORS


client = TestClient(api.app)


class FakeModel:
    def predict(self, features):
        return [17.5]


def make_payload(cycles=(1, 2)):
    return {
        "engine_id": 7,
        "history": [
            {
                "cycle": cycle,
                "setting_1": 0.0,
                "setting_2": 0.0,
                "sensors": {sensor: float(cycle) for sensor in SENSORS},
            }
            for cycle in cycles
        ],
    }


def test_predict_returns_engine_endpoint_and_rul(monkeypatch):
    monkeypatch.setattr(api, "get_model", lambda: FakeModel())

    response = client.post("/predict", json=make_payload())

    assert response.status_code == 200
    assert response.json() == {
        "engine_id": 7,
        "observed_cycle": 2,
        "predicted_rul_cycles": 17.5,
    }


def test_predict_rejects_missing_sensor():
    payload = make_payload()
    del payload["history"][0]["sensors"]["sensor_2"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_missing_cycle(monkeypatch):
    monkeypatch.setattr(api, "get_model", lambda: FakeModel())

    response = client.post("/predict", json=make_payload(cycles=(1, 3)))

    assert response.status_code == 422
    assert "consecutive" in response.json()["detail"]

def test_predict_rejects_nonfinite_sensor_and_setting():
    bad_sensor = make_payload()
    bad_sensor["history"][0]["sensors"]["sensor_2"] = "Infinity"
    response = client.post("/predict", json=bad_sensor)
    assert response.status_code == 422
    assert "finite" in response.json()["detail"]

    bad_setting = make_payload()
    bad_setting["history"][0]["setting_1"] = "NaN"
    response = client.post("/predict", json=bad_setting)
    assert response.status_code == 422
    assert "finite" in response.json()["detail"]

def test_predict_records_latest_endpoint_when_monitoring_enabled(
    monkeypatch, tmp_path
):
    import json
    import sqlite3

    db_path = tmp_path / "events.db"
    monkeypatch.setenv("RELIABILITYML_MONITORING_DB", str(db_path))
    monkeypatch.setenv(
        "RELIABILITYML_MODEL_URI", "models:/ReliabilityML-FD001-RUL/2"
    )
    monkeypatch.setattr(api, "get_model", lambda: FakeModel())

    payload = make_payload()
    payload["history"][0]["sensors"]["sensor_2"] = 999.0
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT model_uri, endpoint_inputs_json FROM prediction_events"
        ).fetchone()

    assert row[0] == "models:/ReliabilityML-FD001-RUL/2"
    assert json.loads(row[1])["sensor_2"] == 2.0


def test_monitoring_write_failure_does_not_block_prediction(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(
        "RELIABILITYML_MONITORING_DB",
        str(tmp_path / "missing_folder" / "events.db"),
    )
    monkeypatch.setattr(api, "get_model", lambda: FakeModel())

    response = client.post("/predict", json=make_payload())

    assert response.status_code == 200
    assert response.json()["predicted_rul_cycles"] == 17.5