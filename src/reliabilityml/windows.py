import numpy as np
import pandas as pd

from reliabilityml.features import BASE_FEATURES


def build_windows(
    history: pd.DataFrame,
    window_size: int = 30,
) -> tuple[np.ndarray, pd.Index]:
    """Return sensor windows and the row index at each window's last cycle."""
    if window_size < 1:
        raise ValueError("window_size must be positive")

    required = {"engine_id", *BASE_FEATURES}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    windows = []
    endpoint_indices = []

    for engine_id, rows in history.groupby("engine_id", sort=False):
        rows = rows.sort_values("cycle")
        cycles = rows["cycle"].to_numpy()

        if len(cycles) > 1 and not np.all(np.diff(cycles) == 1):
            raise ValueError(f"Cycles must be contiguous for engine {engine_id}")

        values = rows[BASE_FEATURES].to_numpy(dtype=np.float32)

        for end in range(window_size - 1, len(rows)):
            windows.append(values[end - window_size + 1 : end + 1])
            endpoint_indices.append(rows.index[end])

    if not windows:
        raise ValueError("No complete windows available")

    return np.stack(windows), pd.Index(endpoint_indices)