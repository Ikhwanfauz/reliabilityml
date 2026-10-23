from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from reliabilityml.features import build_features
from reliabilityml.model import make_random_forest_model


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train_FD001.txt"
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def main():
    train = pd.read_csv(TRAIN_PATH, sep=r"\s+", header=None, names=COLUMNS)
    final_cycles = train.groupby("engine_id")["cycle"].max()
    train["rul"] = train["engine_id"].map(final_cycles) - train["cycle"]

    features = build_features(train)
    targets = train["rul"].to_numpy()

    model = make_random_forest_model().set_params(min_samples_leaf=3)
    model.fit(features, targets)

    example = features.head(5)
    original_predictions = model.predict(example)
    signature = infer_signature(example, original_predictions)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ReliabilityML-FD001")

    with mlflow.start_run(run_name="forest_leaf3_full_training") as run:
        mlflow.log_params(
            {
                "model": "RandomForestRegressor",
                "min_samples_leaf": 3,
                "n_estimators": model.n_estimators,
                "training_engines": train["engine_id"].nunique(),
                "training_rows": len(train),
                "feature_count": features.shape[1],
                "rul_target": "uncapped",
            }
        )
        mlflow.log_metric("v3_grouped_cv_mean_mae", 25.11)
        mlflow.set_tag("metric_source", "V3 grouped 3-fold CV on 80 engines")
        mlflow.set_tag("input_type", "33 engineered features per row")

        model_info = mlflow.sklearn.log_model(
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            sk_model=model,
            name="rul_model",
            signature=signature,
            input_example=example,
        )

        loaded_model = mlflow.sklearn.load_model(model_info.model_uri)
        loaded_predictions = loaded_model.predict(example)
        np.testing.assert_allclose(original_predictions, loaded_predictions)

        print("=== MLflow model run ===")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model URI: {model_info.model_uri}")
        print(f"Features: {features.shape[1]}")
        print("Saved model reload check: passed")


if __name__ == "__main__":
    main()