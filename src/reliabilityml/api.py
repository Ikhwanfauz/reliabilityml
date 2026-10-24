import os
from functools import lru_cache

import mlflow
import mlflow.sklearn
import pandas as pd
import math
import logging
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from reliabilityml.monitoring import record_prediction

from reliabilityml.features import SENSORS
from reliabilityml.inference import predict_endpoint_rul

logger = logging.getLogger("uvicorn.error")
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

        values = (
            observation.setting_1,
            observation.setting_2,
            *observation.sensors.values(),
        )
        if not all(math.isfinite(value) for value in values):
            raise HTTPException(
                status_code=422,
                detail="Settings and sensor readings must be finite",
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
        logger.warning(
            "prediction_rejected engine_id=%s reason=%s",
            request.engine_id,
            error,
        )
        raise HTTPException(status_code=422, detail=str(error)) from error

    result = {
        "engine_id": request.engine_id,
        "observed_cycle": int(history["cycle"].max()),
        "predicted_rul_cycles": round(prediction, 2),
    }

    monitoring_db = os.environ.get("RELIABILITYML_MONITORING_DB")
    if monitoring_db:
        endpoint = history.loc[history["cycle"].idxmax()]
        endpoint_inputs = {
            name: float(endpoint[name])
            for name in ("setting_1", "setting_2", *SENSORS)
        }
        try:
            record_prediction(
                monitoring_db,
                engine_id=result["engine_id"],
                observed_cycle=result["observed_cycle"],
                predicted_rul_cycles=result["predicted_rul_cycles"],
                model_uri=os.environ.get("RELIABILITYML_MODEL_URI", "unset"),
                endpoint_inputs=endpoint_inputs,
            )
        except (sqlite3.Error, OSError):
            logger.exception(
                "monitoring_write_failed engine_id=%s",
                request.engine_id,
            )

    logger.info(
        "prediction_success engine_id=%s observed_cycle=%s "
        "predicted_rul_cycles=%s model_uri=%s",
        result["engine_id"],
        result["observed_cycle"],
        result["predicted_rul_cycles"],
        os.environ.get("RELIABILITYML_MODEL_URI", "unset"),
    )
    return result