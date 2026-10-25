import numpy as np
import pandas as pd


def compare_mean_shift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    *,
    min_samples: int = 30,
    alert_threshold: float = 1.0,
) -> list[dict]:
    if not columns:
        raise ValueError("At least one column is required")
    if len(reference) < min_samples or len(current) < min_samples:
        raise ValueError(f"At least {min_samples} rows are required in each group")

    missing = set(columns) - set(reference.columns) | set(columns) - set(current.columns)
    if missing:
        raise ValueError(f"Missing monitoring columns: {sorted(missing)}")

    report = []
    for column in columns:
        before = reference[column].to_numpy(dtype=float)
        now = current[column].to_numpy(dtype=float)
        if not np.isfinite(before).all() or not np.isfinite(now).all():
            raise ValueError(f"Non-finite values in {column}")

        before_mean = float(before.mean())
        now_mean = float(now.mean())
        before_std = float(before.std())

        if before_std == 0:
            shift = (
                0.0
                if np.isclose(before_mean, now_mean, rtol=0, atol=1e-9)
                else float("inf")
            )
        else:
            shift = abs(now_mean - before_mean) / before_std

        report.append(
            {
                "column": column,
                "reference_mean": before_mean,
                "current_mean": now_mean,
                "shift_in_reference_std": shift,
                "alert": bool(shift >= alert_threshold),
            }
        )

    return report