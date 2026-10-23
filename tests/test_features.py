import pandas as pd

from reliabilityml.features import (
    BASE_FEATURES,
    build_features,
    build_sequence_features,
    build_std_features
)


def make_history():
    rows = []
    for engine_id, values in [(1, [2.0, 4.0, 6.0]), (2, [20.0, 40.0])]:
        for cycle, value in enumerate(values, start=1):
            row = {column: value for column in BASE_FEATURES}
            row.update(
                engine_id=engine_id,
                cycle=cycle,
                setting_1=0.0,
                setting_2=0.0,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_rolling_mean_stays_within_engine():
    features = build_features(make_history())

    assert features["sensor_2_mean_5"].tolist() == [2.0, 3.0, 4.0, 20.0, 30.0]


def test_future_cycle_does_not_change_earlier_features():
    history = make_history()
    original = build_features(history)

    history.loc[2, "sensor_2"] = 600.0
    changed = build_features(history)

    assert original.loc[:1].equals(changed.loc[:1])

def test_sequence_features_reset_at_engine_boundary():
    features = build_sequence_features(make_history())

    assert features.shape[1] == 93
    assert features["sensor_2_lag_1"].tolist() == [
        2.0, 2.0, 4.0, 20.0, 20.0
    ]
    assert features["sensor_2_change_1"].tolist() == [
        0.0, 2.0, 2.0, 0.0, 20.0
    ]
    assert features["sensor_2_trend_5"].tolist() == [
        0.0, 2.0, 2.0, 0.0, 20.0
    ]
    assert features["sensor_2_std_5"].iloc[0] == 0.0
    assert features["sensor_2_std_5"].iloc[3] == 0.0
    assert not features.isna().any().any()


def test_future_cycle_does_not_change_earlier_sequence_features():
    history = make_history()
    original = build_sequence_features(history)

    history.loc[2, "sensor_2"] = 600.0
    changed = build_sequence_features(history)

    assert original.loc[:1].equals(changed.loc[:1])

def test_std_features_have_fixed_columns_and_reset_per_engine():
    features = build_std_features(make_history())

    assert features.shape == (5, 48)
    assert features.columns[-1] == "sensor_21_std_5"
    assert features["sensor_2_std_5"].iloc[0] == 0.0
    assert features["sensor_2_std_5"].iloc[2] == 2.0
    assert features["sensor_2_std_5"].iloc[3] == 0.0
    assert not features.isna().any().any()