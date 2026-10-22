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