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