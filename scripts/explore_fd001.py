from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_file(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        RAW / filename,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )


def main() -> None:
    train = load_file("train_FD001.txt")
    test = load_file("test_FD001.txt")

    assert train.shape == (20631, 26), f"Unexpected train shape: {train.shape}"
    assert test.shape == (13096, 26), f"Unexpected test shape: {test.shape}"

    train_last_cycle = train.groupby("engine_id")["cycle"].max()
    test_last_cycle = test.groupby("engine_id")["cycle"].max()

    train["rul"] = train["engine_id"].map(train_last_cycle) - train["cycle"]

    print("=== FD001 EDA ===")
    print(f"Train: {train.shape[0]} rows, {train['engine_id'].nunique()} engines")
    print(f"Test:  {test.shape[0]} rows, {test['engine_id'].nunique()} engines")
    print(f"Missing values: train={train.isna().sum().sum()}, test={test.isna().sum().sum()}")

    print("\nTraining engine lifetime (cycles):")
    print(train_last_cycle.describe().to_string())

    print("\nObserved test history length (cycles):")
    print(test_last_cycle.describe().to_string())

    print("\nTraining RUL across all rows:")
    print(train["rul"].describe().to_string())

    constant_columns = [
        column
        for column in COLUMNS[2:]
        if train[column].nunique() == 1
    ]
    print(f"\nConstant training columns: {constant_columns}")


if __name__ == "__main__":
    main()