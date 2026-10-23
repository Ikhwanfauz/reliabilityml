import argparse
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from reliabilityml.inference import predict_endpoint_rul


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "raw" / "test_FD001.txt"
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def main():
    parser = argparse.ArgumentParser(description="Predict FD001 engine RUL")
    parser.add_argument("engine_id", type=int, help="FD001 test engine ID")
    args = parser.parse_args()

    model_uri = os.environ.get("RELIABILITYML_MODEL_URI")
    if not model_uri:
        parser.error("Set RELIABILITYML_MODEL_URI to the saved MLflow model URI")

    test = pd.read_csv(TEST_PATH, sep=r"\s+", header=None, names=COLUMNS)
    history = test.loc[test["engine_id"] == args.engine_id]
    if history.empty:
        parser.error(f"Engine {args.engine_id} is not in FD001 test data")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model = mlflow.sklearn.load_model(model_uri)
    prediction = predict_endpoint_rul(history, model)

    print(f"Engine: {args.engine_id}")
    print(f"Latest observed cycle: {int(history['cycle'].max())}")
    print(f"Predicted RUL: {prediction:.2f} cycles")


if __name__ == "__main__":
    main()