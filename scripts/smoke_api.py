import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from reliabilityml.features import SENSORS


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "raw" / "test_FD001.txt"
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def main():
    test = pd.read_csv(TEST_PATH, sep=r"\s+", header=None, names=COLUMNS)
    engine = test.loc[test["engine_id"] == 1]

    payload = {
        "engine_id": 1,
        "history": [
            {
                "cycle": int(row["cycle"]),
                "setting_1": float(row["setting_1"]),
                "setting_2": float(row["setting_2"]),
                "sensors": {sensor: float(row[sensor]) for sensor in SENSORS},
            }
            for _, row in engine.iterrows()
        ],
    }

    request = Request(
        os.environ.get("RELIABILITYML_API_URL", "http://127.0.0.1:8000/predict"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        print(json.loads(response.read().decode("utf-8")))


if __name__ == "__main__":
    main()