# ReliabilityML

ReliabilityML is a learning and portfolio project for estimating the remaining useful life (RUL), in operating cycles, of simulated turbofan engines.

Dataset: NASA C-MAPSS FD001.

## Current scope

V0 establishes the development environment, repository structure, dataset documentation, and a tested data loader. Model training and evaluation begin in later versions.

## Environment

Create and activate the dedicated Conda environment:

```bat
conda env create -f environment.yml
conda activate reliabilityml
```

## V0 verification

Download NASA C-MAPSS and place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in `data/raw/`. The source, schema, and checksums are documented in `data/README.md`.

With the `reliabilityml` Conda environment activated, run:

```bat
python -m pip install --no-deps -e .
python scripts\prepare_data.py
python -m pytest -q
```

## V1 exploration and baseline

FD001 training contains 100 engines and 20,631 cycle rows. The first validation
split keeps all cycles of an engine together: 80 training engines and 20
validation engines. A constant median-RUL baseline predicts 104 cycles for
every validation row. Across all validation rows, its MAE was 50.82 cycles
and RMSE was 59.87 cycles.

## V2 sensor features and Ridge

Each row uses 33 features: cycle, two operating settings, 15 current sensor
readings, and their trailing five-cycle means. The rolling features use only
the current and earlier cycles of the same engine. On the V1 grouped split,
Ridge achieved MAE 26.28 cycles and RMSE 33.16 cycles across all validation
rows. These all-row scores are not directly comparable to V3 endpoint scores.

## V3 model evaluation

FD001 uses uncapped RUL in operating cycles. Model selection used grouped
3-fold cross-validation on 80 training engines, with one simulated observation
per validation engine. Random Forest with `min_samples_leaf=3` had the lowest
mean CV MAE (25.11 cycles); the difference from `leaf=5` (25.23) was small.

The selected model was then trained on all 100 training engines and evaluated
once on the official endpoint of each of the 100 test engines:

| Method | Test MAE (cycles) | Test RMSE (cycles) |
| --- | ---: | ---: |
| Median baseline | 39.08 | 49.82 |
| Random Forest (`min_samples_leaf=3`) | 20.85 | 29.27 |

The simulated validation endpoints and official test endpoints are different
evaluation sets. Official test answers are not used to choose model settings.