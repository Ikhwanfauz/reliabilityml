import pandas as pd

from reliabilityml.features import build_features


def predict_endpoint_rul(history: pd.DataFrame, model) -> float:
    if history.empty:
        raise ValueError("Engine history cannot be empty")

    if "engine_id" not in history.columns or "cycle" not in history.columns:
        raise ValueError("Engine history needs engine_id and cycle")

    if history["engine_id"].nunique() != 1:
        raise ValueError("Provide history for exactly one engine")

    ordered = history.sort_values("cycle").reset_index(drop=True)
    cycles = ordered["cycle"].tolist()
    if cycles != list(range(1, len(ordered) + 1)):
        raise ValueError("Cycles must be unique and consecutive from 1")

    features = build_features(ordered)
    endpoint_features = features.tail(1)
    return float(model.predict(endpoint_features)[0])