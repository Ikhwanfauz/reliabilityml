from pathlib import Path

import yaml

from reliabilityml.data import add_training_rul, load_history, load_test_rul


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "fd001.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config["dataset"] != "FD001":
        raise ValueError("This command supports FD001 only")
    if config["target"]["cap"] is not None:
        raise ValueError("V0 uses uncapped training RUL")

    raw_dir = project_root / config["raw_dir"]
    filenames = config["files"]

    train = add_training_rul(load_history(raw_dir / filenames["train"]))
    test = load_history(raw_dir / filenames["test"])
    test_answers = load_test_rul(raw_dir / filenames["test_rul"])

    train_engines = train["unit_id"].nunique()
    test_engines = test["unit_id"].nunique()
    expected = config["expected"]

    if (
        train_engines != expected["train_engines"]
        or test_engines != expected["test_engines"]
    ):
        raise ValueError("Engine counts do not match FD001 configuration")

    if set(test["unit_id"].unique()) != set(test_answers.index):
        raise ValueError("Test engine IDs do not match RUL answer IDs")

    if not (train.groupby("unit_id").tail(1)["rul"] == 0).all():
        raise ValueError("Final training cycle should have RUL zero")

    print("Configuration: FD001, uncapped RUL")
    print(f"Training: {len(train)} rows, {train_engines} engines, {train.shape[1]} columns")
    print(f"Test: {len(test)} rows, {test_engines} engines, {test.shape[1]} columns")
    print(f"Test answers: {len(test_answers)} endpoint RUL values")
    print("FD001 validation passed")


if __name__ == "__main__":
    main()