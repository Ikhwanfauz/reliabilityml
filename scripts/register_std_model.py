from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from reliabilityml.features import build_std_features
from reliabilityml.model import make_random_forest_model


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train_FD001.txt"
REGISTERED_NAME = "ReliabilityML-FD001-RUL"
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def main():
    train = pd.read_csv(TRAIN_PATH, sep=r"\s+", header=None, names=COLUMNS)
    final_cycles = train.groupby("engine_id")["cycle"].max()
    train["rul"] = train["engine_id"].map(final_cycles) - train["cycle"]

    features = build_std_features(train)
    assert features.shape[1] == 48
    targets = train["rul"].to_numpy()

    model = make_random_forest_model().set_params(min_samples_leaf=3)
    model.fit(features, targets)

    example = features.head(5)
    original_predictions = model.predict(example)
    signature = infer_signature(example, original_predictions)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ReliabilityML-FD001")

    with mlflow.start_run(run_name="forest_leaf3_std48_full_training") as run:
        mlflow.log_params(
            {
                "model": "RandomForestRegressor",
                "min_samples_leaf": 3,
                "n_estimators": model.n_estimators,
                "training_engines": train["engine_id"].nunique(),
                "training_rows": len(train),
                "feature_count": features.shape[1],
                "feature_set": "33_original_plus_15_rolling_std",
                "rul_target": "uncapped",
            }
        )
        mlflow.log_metric("grouped_cv_endpoint_mae", 24.13)
        mlflow.log_metric("validation_endpoint_mae", 18.08)
        mlflow.log_metric("official_test_endpoint_mae", 20.58)
        mlflow.set_tag(
            "metric_context",
            "CV: 80 engines/3 folds; validation: 20 engines; "
            "official test: 100 endpoints previously examined",
        )

        model_info = mlflow.sklearn.log_model(
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            sk_model=model,
            name="rul_model_std48",
            signature=signature,
            input_example=example,
        )

        saved_model = mlflow.sklearn.load_model(model_info.model_uri)
        np.testing.assert_allclose(
            original_predictions, saved_model.predict(example)
        )

        version = mlflow.register_model(model_info.model_uri, REGISTERED_NAME)
        client = MlflowClient()
        client.set_model_version_tag(
            REGISTERED_NAME, version.version, "feature_count", "48"
        )
        client.set_model_version_tag(
            REGISTERED_NAME, version.version, "feature_set", "std48"
        )

        registered_uri = f"models:/{REGISTERED_NAME}/{version.version}"
        registered_model = mlflow.sklearn.load_model(registered_uri)
        np.testing.assert_allclose(
            original_predictions, registered_model.predict(example)
        )

        print("=== Registered FD001 model ===")
        print(f"Run ID: {run.info.run_id}")
        print(f"Version: {version.version}")
        print(f"URI: {registered_uri}")
        print(f"Features: {registered_model.n_features_in_}")
        print("Saved and registered reload checks: passed")


if __name__ == "__main__":
    main()