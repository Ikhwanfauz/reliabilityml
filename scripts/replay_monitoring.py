from pathlib import Path

import pandas as pd
import argparse
from fastapi.testclient import TestClient

from reliabilityml.api import app
from reliabilityml.features import SENSORS


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "raw" / "test_FD001.txt"
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor2-offset", type=float, default=0.0)
    args = parser.parse_args()

    test = pd.read_csv(TEST_PATH, sep=r"\s+", header=None, names=COLUMNS)

    with TestClient(app) as client:
        for engine_id in range(1, 31):
            engine = test.loc[test["engine_id"] == engine_id]
            payload = {
                "engine_id": engine_id,
                "history": [
                    {
                        "cycle": int(row["cycle"]),
                        "setting_1": float(row["setting_1"]),
                        "setting_2": float(row["setting_2"]),
                        "sensors": {
                            sensor: float(row[sensor])
+ (args.sensor2_offset if sensor == "sensor_2" else 0.0)
for sensor in SENSORS
                        },
                    }
                    for _, row in engine.iterrows()
                ],
            }

            response = client.post("/predict", json=payload)
            response.raise_for_status()
            assert response.json()["engine_id"] == engine_id

    print("Replayed 30 distinct FD001 test engines through FastAPI")


if __name__ == "__main__":
    main()