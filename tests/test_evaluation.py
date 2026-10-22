import pandas as pd
import pytest

from reliabilityml.evaluation import select_observation_rows


def test_selects_one_repeatable_observation_per_engine():
    history = pd.DataFrame(
        {
            "engine_id": [1] * 5 + [2] * 5,
            "cycle": [1, 2, 3, 4, 5] * 2,
        }
    )

    selected = select_observation_rows(history, minimum_cycle=2)
    selected_again = select_observation_rows(history, minimum_cycle=2)

    assert selected["engine_id"].tolist() == [1, 2]
    assert selected["cycle"].between(2, 4).all()
    pd.testing.assert_frame_equal(selected, selected_again)


def test_rejects_engine_without_observation_cycle():
    history = pd.DataFrame({"engine_id": [1], "cycle": [1]})

    with pytest.raises(ValueError):
        select_observation_rows(history, minimum_cycle=1)