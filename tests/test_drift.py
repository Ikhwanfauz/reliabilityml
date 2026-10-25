import pandas as pd
import pytest

from reliabilityml.drift import compare_mean_shift


def test_mean_shift_flags_changed_prediction_distribution():
    reference = pd.DataFrame(
        {
            "sensor_2": range(50),
            "predicted_rul_cycles": range(50),
        }
    )
    current = pd.DataFrame(
        {
            "sensor_2": range(50),
            "predicted_rul_cycles": range(100, 150),
        }
    )

    report = compare_mean_shift(
        reference, current, ["sensor_2", "predicted_rul_cycles"]
    )
    by_column = {row["column"]: row for row in report}

    assert by_column["sensor_2"]["alert"] is False
    assert by_column["sensor_2"]["shift_in_reference_std"] == 0.0
    assert by_column["predicted_rul_cycles"]["alert"] is True


def test_mean_shift_requires_enough_samples():
    small = pd.DataFrame({"sensor_2": [1.0, 2.0]})

    with pytest.raises(ValueError, match="At least 30 rows"):
        compare_mean_shift(small, small, ["sensor_2"])