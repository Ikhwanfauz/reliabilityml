import pandas as pd
import pytest

from reliabilityml.data import add_training_rul, load_history, load_test_rul


def _row(unit_id: int, cycle: int) -> list[float]:
    return [unit_id, cycle, 0.0, 0.0, 100.0] + [1.0] * 21


def _write_history(path, rows):
    pd.DataFrame(rows).to_csv(path, sep=" ", header=False, index=False)


def test_load_history_names_columns_and_reads_engines(tmp_path):
    path = tmp_path / "history.txt"
    _write_history(path, [_row(1, 1), _row(1, 2), _row(2, 1)])

    history = load_history(path)

    assert history.shape == (3, 26)
    assert history.columns[:5].tolist() == [
        "unit_id", "cycle", "setting_1", "setting_2", "setting_3"
    ]
    assert history["unit_id"].tolist() == [1, 1, 2]


def test_load_history_rejects_wrong_column_count(tmp_path):
    path = tmp_path / "history.txt"
    _write_history(path, [_row(1, 1)[:-1]])

    with pytest.raises(ValueError, match="26-column"):
        load_history(path)


def test_load_history_rejects_duplicate_cycle(tmp_path):
    path = tmp_path / "history.txt"
    _write_history(path, [_row(1, 1), _row(1, 1)])

    with pytest.raises(ValueError, match="Duplicate"):
        load_history(path)


def test_load_history_rejects_missing_cycle(tmp_path):
    path = tmp_path / "history.txt"
    _write_history(path, [_row(1, 1), _row(1, 3)])

    with pytest.raises(ValueError, match="missing or unordered"):
        load_history(path)


def test_training_rul_is_calculated_separately_per_engine():
    history = pd.DataFrame({
        "unit_id": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
    })

    labeled = add_training_rul(history)

    assert labeled["rul"].tolist() == [2, 1, 0, 1, 0]
    assert "rul" not in history.columns


def test_load_test_rul_indexes_answers_by_engine(tmp_path):
    path = tmp_path / "answers.txt"
    path.write_text("112\n98\n", encoding="utf-8")

    answers = load_test_rul(path)

    assert answers.to_dict() == {1: 112, 2: 98}


def test_load_test_rul_rejects_negative_answer(tmp_path):
    path = tmp_path / "answers.txt"
    path.write_text("112\n-1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-negative"):
        load_test_rul(path)