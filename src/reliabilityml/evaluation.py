import numpy as np
import pandas as pd


def select_observation_rows(
    history: pd.DataFrame,
    seed: int = 42,
    minimum_cycle: int = 31,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected_indices = []

    for engine_id, engine_rows in history.groupby("engine_id", sort=True):
        final_cycle = engine_rows["cycle"].max()
        candidates = engine_rows[
            (engine_rows["cycle"] >= minimum_cycle)
            & (engine_rows["cycle"] < final_cycle)
        ]
        if candidates.empty:
            raise ValueError(f"No observation cycle available for engine {engine_id}")

        selected_indices.append(int(rng.choice(candidates.index.to_numpy())))

    return history.loc[selected_indices].sort_values("engine_id").copy()