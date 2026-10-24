import numpy as np
import pandas as pd
import pytest

from reliabilityml.features import BASE_FEATURES
from reliabilityml.windows import build_windows


def make_history():
    rows = []
    for engine_id, readings in [(1, [2.0, 4.0, 6.0]), (2, [10.0, 20.0, 30.0])]:
        for cycle, reading in enumerate(readings, start=1):
            row = {column: reading for column in BASE_FEATURES}
            row.update(
                engine_id=engine_id,
                cycle=cycle,
                setting_1=0.0,
                setting_2=0.0,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_windows_stay_within_engine_and_end_at_correct_row():
    windows, endpoints = build_windows(make_history(), window_size=2)

    assert windows.shape == (4, 2, 18)
    assert endpoints.tolist() == [1, 2, 4, 5]
    sensor_position = BASE_FEATURES.index("sensor_2")
    assert windows[2, :, sensor_position].tolist() == [10.0, 20.0]


def test_future_reading_does_not_change_earlier_window():
    history = make_history()
    original, _ = build_windows(history, window_size=2)

    history.loc[2, "sensor_2"] = 999.0
    changed, _ = build_windows(history, window_size=2)

    np.testing.assert_array_equal(original[0], changed[0])


def test_windows_reject_missing_cycle():
    history = make_history().drop(index=1)

    with pytest.raises(ValueError, match="contiguous"):
        build_windows(history, window_size=2)