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