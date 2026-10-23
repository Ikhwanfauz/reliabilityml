import os
from functools import lru_cache

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from reliabilityml.features import SENSORS
from reliabilityml.inference import predict_endpoint_rul


app = FastAPI(title="ReliabilityML FD001")


class Observation(BaseModel):
    cycle: int = Field(gt=0)
    setting_1: float
    setting_2: float
    sensors: dict[str, float]


class PredictionRequest(BaseModel):
    engine_id: int = Field(gt=0)
    history: list[Observation] = Field(min_length=1)


@lru_cache
def get_model():
    model_uri = os.environ.get("RELIABILITYML_MODEL_URI")
    if not model_uri:
        raise RuntimeError("RELIABILITYML_MODEL_URI is not set")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    return mlflow.sklearn.load_model(model_uri)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    rows = []
    expected_sensors = set(SENSORS)

    for observation in request.history:
        if set(observation.sensors) != expected_sensors:
            raise HTTPException(
                status_code=422,
                detail=f"Each observation needs exactly these sensors: {SENSORS}",
            )

        rows.append(
            {
                "engine_id": request.engine_id,
                "cycle": observation.cycle,
                "setting_1": observation.setting_1,
                "setting_2": observation.setting_2,
                **observation.sensors,
            }
        )

    history = pd.DataFrame(rows)
    try:
        prediction = predict_endpoint_rul(history, get_model())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "engine_id": request.engine_id,
        "observed_cycle": int(history["cycle"].max()),
        "predicted_rul_cycles": round(prediction, 2),
    }