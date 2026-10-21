from pathlib import Path

from reliabilityml.data import add_training_rul, load_history, load_test_rul


def main() -> None:
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"

    train = add_training_rul(load_history(raw_dir / "train_FD001.txt"))
    test = load_history(raw_dir / "test_FD001.txt")
    test_answers = load_test_rul(raw_dir / "RUL_FD001.txt")

    train_engines = train["unit_id"].nunique()
    test_engines = test["unit_id"].nunique()

    if train_engines != 100 or test_engines != 100:
        raise ValueError("FD001 should contain 100 training and 100 test engines")

    if set(test["unit_id"].unique()) != set(test_answers.index):
        raise ValueError("Test engine IDs do not match RUL answer IDs")

    if not (train.groupby("unit_id").tail(1)["rul"] == 0).all():
        raise ValueError("Final training cycle should have RUL zero")

    print(f"Training: {len(train)} rows, {train_engines} engines, {train.shape[1]} columns")
    print(f"Test: {len(test)} rows, {test_engines} engines, {test.shape[1]} columns")
    print(f"Test answers: {len(test_answers)} endpoint RUL values")
    print("FD001 validation passed")


if __name__ == "__main__":
    main()