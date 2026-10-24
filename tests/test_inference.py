import pandas as pd
import pytest

from reliabilityml.features import SENSORS
from reliabilityml.inference import predict_endpoint_rul


class RecordingModel:
    def predict(self, features):
        self.features = features.copy()
        return [42.5]


def make_history(cycles=(1, 2, 3), engine_ids=None):
    if engine_ids is None:
        engine_ids = [7] * len(cycles)

    history = pd.DataFrame(
        {
            "engine_id": engine_ids,
            "cycle": cycles,
            "setting_1": [0.0] * len(cycles),
            "setting_2": [0.0] * len(cycles),
        }
    )
    for sensor in SENSORS:
        history[sensor] = list(cycles)
    return history


def test_predicts_from_latest_cycle_with_causal_features():
    model = RecordingModel()
    result = predict_endpoint_rul(make_history((3, 1, 2)), model)

    assert result == 42.5
    assert model.features.shape == (1, 33)
    assert model.features.iloc[0]["cycle"] == 3
    assert model.features.iloc[0]["sensor_2_mean_5"] == 2.0


@pytest.mark.parametrize(
    "history",
    [
        make_history(()),
        make_history((1, 2), engine_ids=[7, 8]),
        make_history((1, 1)),
        make_history((1, 3)),
    ],
)
def test_rejects_invalid_engine_history(history):
    with pytest.raises(ValueError):
        predict_endpoint_rul(history, RecordingModel())

def test_predicts_with_48_features_when_model_expects_them():
    from reliabilityml.features import build_std_features

    history = make_history()
    model = RecordingModel()
    model.feature_names_in_ = build_std_features(history).columns.to_numpy()

    result = predict_endpoint_rul(history, model)

    assert result == 42.5
    assert model.features.shape == (1, 48)
    assert model.features.iloc[0]["sensor_2_mean_5"] == 2.0
    assert model.features.iloc[0]["sensor_2_std_5"] == 1.0


def test_rejects_unknown_model_feature_schema():
    model = RecordingModel()
    model.feature_names_in_ = ["unexpected_feature"]

    with pytest.raises(ValueError, match="Model feature names"):
        predict_endpoint_rul(make_history(), model)