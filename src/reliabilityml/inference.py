import pandas as pd

from reliabilityml.features import build_features, build_std_features


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

    original_features = build_features(ordered)
    model_columns = getattr(model, "feature_names_in_", None)

    if model_columns is None:
        if getattr(model, "n_features_in_", 33) != 33:
            raise ValueError("Model needs feature names to select its feature set")
        features = original_features
    elif list(model_columns) == list(original_features.columns):
        features = original_features
    else:
        std_features = build_std_features(ordered)
        if list(model_columns) != list(std_features.columns):
            raise ValueError("Model feature names do not match a supported feature set")
        features = std_features

    return float(model.predict(features.tail(1))[0])