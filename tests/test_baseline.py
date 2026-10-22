import numpy as np
import pytest

from reliabilityml.baseline import (
    predict_median,
    regression_metrics,
    split_engine_ids,
)


def test_engine_split_is_disjoint_and_reproducible():
    engine_ids = np.repeat(np.arange(1, 101), 3)

    train_ids, val_ids = split_engine_ids(engine_ids)
    train_again, val_again = split_engine_ids(engine_ids)

    assert len(train_ids) == 80
    assert len(val_ids) == 20
    assert set(train_ids).isdisjoint(set(val_ids))
    np.testing.assert_array_equal(train_ids, train_again)
    np.testing.assert_array_equal(val_ids, val_again)


def test_median_predictions_and_metrics():
    predictions = predict_median([0, 10, 20], 2)

    np.testing.assert_array_equal(predictions, [10, 10])
    scores = regression_metrics([8, 14], predictions)

    assert scores["mae"] == pytest.approx(3.0)
    assert scores["rmse"] == pytest.approx(np.sqrt(10))


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1])