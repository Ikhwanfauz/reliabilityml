from pathlib import Path

import numpy as np
import pandas as pd


HISTORY_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_history(path: str | Path) -> pd.DataFrame:
    """Read and validate one FD001 engine-history file."""
    frame = pd.read_csv(path, sep=r"\s+", header=None)

    if frame.empty or frame.shape[1] != len(HISTORY_COLUMNS):
        raise ValueError(f"Expected non-empty, 26-column history: {path}")

    frame.columns = HISTORY_COLUMNS
    frame = frame.apply(pd.to_numeric, errors="raise")

    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"History contains a missing or non-finite value: {path}")

    for column in ("unit_id", "cycle"):
        if (frame[column] < 1).any() or (frame[column] % 1 != 0).any():
            raise ValueError(f"{column} must contain positive integers")
        frame[column] = frame[column].astype("int64")

    if frame.duplicated(["unit_id", "cycle"]).any():
        raise ValueError("Duplicate engine/cycle pair")

    for unit_id, group in frame.groupby("unit_id", sort=False):
        expected = np.arange(1, len(group) + 1)
        if not np.array_equal(group["cycle"].to_numpy(), expected):
            raise ValueError(f"Engine {unit_id} has missing or unordered cycles")

    return frame


def add_training_rul(history: pd.DataFrame) -> pd.DataFrame:
    """Label run-to-failure training rows in operating cycles."""
    labeled = history.copy()
    final_cycle = labeled.groupby("unit_id")["cycle"].transform("max")
    labeled["rul"] = final_cycle - labeled["cycle"]
    return labeled


def load_test_rul(path: str | Path) -> pd.Series:
    """Read endpoint RUL answers, indexed by test engine ID."""
    frame = pd.read_csv(path, sep=r"\s+", header=None)

    if frame.empty or frame.shape[1] != 1:
        raise ValueError(f"Expected a non-empty, one-column RUL file: {path}")

    values = pd.to_numeric(frame.iloc[:, 0], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Test RUL contains a missing or non-finite value")
    if (values < 0).any() or (values % 1 != 0).any():
        raise ValueError("Test RUL must contain non-negative integers")

    return pd.Series(
        values.to_numpy(dtype="int64"),
        index=range(1, len(values) + 1),
        name="rul",
    )