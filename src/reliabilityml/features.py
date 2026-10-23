import pandas as pd


SENSORS = [f"sensor_{i}" for i in (2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
BASE_FEATURES = ["cycle", "setting_1", "setting_2"] + SENSORS


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    required = {"engine_id", *BASE_FEATURES}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    features = history[BASE_FEATURES].copy()

    for sensor in SENSORS:
        features[f"{sensor}_mean_5"] = (
            history.groupby("engine_id")[sensor]
            .transform(lambda values: values.rolling(window=5, min_periods=1).mean())
        )

    return features

def build_sequence_features(history: pd.DataFrame) -> pd.DataFrame:
    """Build experimental features using current and earlier engine cycles."""
    features = build_features(history)
    engines = history.groupby("engine_id", sort=False)

    first_cycle = engines["cycle"].transform("first")
    oldest_cycle = engines["cycle"].shift(4).fillna(first_cycle)
    cycle_span = (history["cycle"] - oldest_cycle).replace(0, 1)

    for sensor in SENSORS:
        readings = engines[sensor]
        first_reading = readings.transform("first")
        previous = readings.shift(1).fillna(first_reading)
        oldest = readings.shift(4).fillna(first_reading)

        features[f"{sensor}_std_5"] = readings.transform(
            lambda values: values.rolling(
                window=5, min_periods=2
            ).std().fillna(0.0)
        )
        features[f"{sensor}_lag_1"] = previous
        features[f"{sensor}_change_1"] = history[sensor] - previous
        features[f"{sensor}_trend_5"] = (
            history[sensor] - oldest
        ) / cycle_span

    return features

def build_std_features(history: pd.DataFrame) -> pd.DataFrame:
    """Build the 48-feature candidate selected by grouped CV."""
    columns = (
        BASE_FEATURES
        + [f"{sensor}_mean_5" for sensor in SENSORS]
        + [f"{sensor}_std_5" for sensor in SENSORS]
    )
    return build_sequence_features(history).loc[:, columns]