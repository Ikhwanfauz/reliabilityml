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

## V2 follow-up: sequential sensor features

The original pipeline uses 33 features per cycle. A follow-up experiment
added rolling standard deviation, one-cycle lag, one-cycle change, and
five-cycle trend for each of the 15 selected sensors (93 features total).
Features use only the current and earlier cycles of the same engine.

Using the original 80 training engines, grouped 3-fold cross-validation
compared Random Forest models with identical settings and observation
cycles. Mean endpoint MAE was 25.11 cycles for the original 33 features,
24.38 for all 93 features, and 24.13 for the 48-feature set containing
the original features plus rolling standard deviations.

The 48-feature set is the candidate for further model comparison. These
are cross-validation results, not official test scores. The existing API
and saved model still use the original 33 features.

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

### V3 follow-up: 48-feature model comparison

Using the 48-feature candidate selected in the V2 follow-up, grouped
3-fold cross-validation compared seven model settings on the original
80 training engines. Random Forest with min_samples_leaf=3 remained
the best candidate at 24.13 cycles mean endpoint MAE. XGBoost was also
evaluated: max_depth=3 scored 24.94 and max_depth=5 scored 25.17 cycles.

These are cross-validation results. The existing saved model and API
still use 33 features; they have not been replaced by this experiment.

### V3 follow-up: endpoint error analysis

Out-of-fold analysis evaluated one simulated endpoint for each of the
80 CV engines using the 48-feature Random Forest. Pooled endpoint MAE
was 24.04 cycles. For actual RUL 0-30 cycles (15 engines), MAE was
6.13; for 31-100 (39 engines), MAE was 21.80; and for above 100
(26 engines), MAE was 37.74.

In the 31-100 group, mean prediction minus actual RUL was +13.82
cycles: the model tended to overestimate remaining life in this group.
The largest individual errors show that the mean score does not
describe every engine. These are CV observations, not official test
results or production monitoring.

### V3 follow-up: TensorFlow sequence experiment

A CPU-based TensorFlow LSTM used 30 consecutive cycles with 18 input
features per cycle. Windows stayed within one engine and ended at the
prediction cycle. Normalization was fitted only on each training fold.

On the same grouped 3-fold splits and observed endpoints, a 48-feature
Random Forest scored 24.35 mean endpoint MAE; the LSTM scored 32.70.
For this comparison, both models trained on endpoints with complete
30-cycle windows. The Forest remains the preferred candidate. This
single LSTM configuration does not establish that all LSTMs perform worse.

To reproduce the optional experiment in Anaconda Prompt from the project
folder:

```bat
conda create -n reliabilityml-tf python=3.11 pip -y
conda activate reliabilityml-tf
python -m pip install -e ".[tensorflow]" "pytest>=9,<10" "httpx2>=2,<3"
python scripts\compare_lstm.py
```

The existing saved model and API still use the original 33 features.
These are cross-validation results, not official test scores.

### V3 follow-up: selected feature set

Grouped 3-fold CV on 80 engines selected the 48-feature Random Forest
(min_samples_leaf=3) by endpoint MAE: 24.13 cycles versus 25.11 for
the original 33 features. On the same 20 validation endpoints, MAE was
18.08 versus 18.48 cycles.

On the 100 official FD001 test endpoints, the 48-feature model scored
MAE 20.58 and RMSE 28.96 cycles; the original 33-feature model scored
MAE 20.85 and RMSE 29.27. These are small differences. The official
test had already been examined during earlier development, so this
comparison is descriptive, not an untouched final test. The saved
MLflow model and API still use 33 features.

Reproduce with `python scripts\compare_validation_features.py` and
`python scripts\compare_official_features.py`.

## V4 experiment tracking

Run `python scripts/log_selected_model.py` to train the CV-selected
Random Forest (`min_samples_leaf=3`) on all 100 FD001 training engines
and log it to a local MLflow experiment named `ReliabilityML-FD001`.
The run records model settings, training data counts, 33 input features,
and the previously measured 3-fold grouped CV MAE of 25.11 cycles.
This script does not evaluate the official test set again.

The script saves the model, reloads it, and checks that its predictions
match. To inspect runs locally, start the UI with:

```bat
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000` and select the
`ReliabilityML-FD001` experiment. Local MLflow database and artifacts
are excluded from Git.

### V4 follow-up: local model registry

The local MLflow Registry contains `ReliabilityML-FD001-RUL` version 1
(original 33 features) and version 2 (33 features plus 15 rolling standard
deviations). Version 2 is the `champion` alias. Its training run records
grouped CV endpoint MAE 24.13 and official test endpoint MAE 20.58.
The official test had been examined earlier; see the V3 evaluation notes.

Run `python scripts\register_std_model.py` to train and register the
48-feature model in an MLflow store where version 1 already exists.
Running it again creates another version. The MLflow database and model
artifacts are local and ignored by Git; cloning this repository does not
restore these registered versions.

For the local API, set `RELIABILITYML_MODEL_URI` to
`models:/ReliabilityML-FD001-RUL@champion` before starting Uvicorn.
Restart the API after changing the alias because the loaded model is cached.
The original 33-feature Docker export remains directly in `local_models`.
Version 2 was exported separately to `local_models\v2`. Both directories
are ignored by Git. From Anaconda Prompt in the project folder, test the
version 2 artifact in Docker with:

```bat
docker build -t reliabilityml-api:std48 .
docker run -d --name reliabilityml-std48 -p 8003:8000 -e RELIABILITYML_MODEL_URI=/app/model -v "%cd%\local_models\v2:/app/model:ro" reliabilityml-api:std48
set RELIABILITYML_API_URL=http://127.0.0.1:8003/predict
python scripts\smoke_api.py
```

The local and Docker API smoke tests both returned 180.89 predicted RUL
cycles for FD001 test engine 1 at observed cycle 31.

## V5 local inference API

The saved V4 model can predict RUL from the observed cycle history of one
FD001 engine. `src/reliabilityml/inference.py` builds the same features used
during training and predicts from the latest observed cycle.

On a fresh setup, run `python scripts/log_selected_model.py` first and copy
the Model URI printed by the script. In Anaconda Prompt, from the project
folder, set that URI and start the local API:

```bat
conda activate reliabilityml
set RELIABILITYML_MODEL_URI=models:/YOUR_MODEL_ID
python -m uvicorn reliabilityml.api:app --app-dir src --host 127.0.0.1 --port 8000
```

In a second Anaconda Prompt, from the project folder, try:

```bat
conda activate reliabilityml
python scripts/smoke_api.py
```

The smoke script sends the observed history of FD001 test engine 1 to
`POST /predict`. The API returns the engine ID, latest observed cycle,
and predicted RUL in cycles. `scripts/predict_engine.py` provides a
command-line prediction using the same saved model. The local MLflow
database and saved model artifacts are excluded from Git.

## V6 continuous integration

The GitHub Actions workflow in `.github/workflows/ci.yml` installs the
project with Python 3.11 and runs pytest on pushes and pull requests to
`main`. The tests use small test fixtures, so the downloaded FD001 data
and local MLflow model are not required by CI.

## V7 Docker API

The Docker image contains the API code and its Python dependencies. The
trained MLflow model is kept outside the image and mounted read-only when
the container starts. `configs/model_constraints.txt` records dependency
versions compatible with the saved model. `.dockerignore` excludes the
downloaded dataset and local model from the image build context.

First create a model with `python scripts/log_selected_model.py` if needed.
From the project folder in Anaconda Prompt, replace `YOUR_MODEL_ID` with
the Model URI printed by that script, then export the model:

```bat
set RELIABILITYML_MODEL_URI=models:/YOUR_MODEL_ID
python -c "import os, mlflow, mlflow.artifacts; mlflow.set_tracking_uri('sqlite:///mlflow.db'); print(mlflow.artifacts.download_artifacts(artifact_uri=os.environ['RELIABILITYML_MODEL_URI'], dst_path='local_models'))"
```

Build and run the image from the project folder:

```bat
docker build -t reliabilityml-api:v7 .
docker run --rm --name reliabilityml-v7 -p 8001:8000 --mount "type=bind,source=%CD%\local_models,target=/model,readonly" -e RELIABILITYML_MODEL_URI=/model reliabilityml-api:v7
```

In a second Anaconda Prompt, from the project folder, check the
container API:

```bat
conda activate reliabilityml
set RELIABILITYML_API_URL=http://127.0.0.1:8001/predict
python scripts\smoke_api.py
```

## Monitoring follow-up: local prediction events

Set `RELIABILITYML_MONITORING_DB=monitoring.db` before starting the local
API to save successful prediction events. Each SQLite row records UTC time,
engine ID, observed cycle, predicted RUL, model URI, and the endpoint's
two operating settings plus 15 sensor readings. The API still returns its
prediction if writing a monitoring event fails, and logs that failure.

The database is generated locally and ignored by Git. A smoke request
using Registry `champion` (version 2) stored engine 1, cycle 31, predicted
RUL 180.89, and 17 input values. This verifies event collection; it does
not establish data drift or model performance after deployment. True RUL
labels are not available from live prediction requests.

The current Docker image was built before this monitoring change. A
persistent Docker deployment would also need a mounted database path.

### Drift demonstration

`python scripts\check_drift.py` compares the latest recorded event per
engine for the fixed version 2 model with 100 FD001 test endpoints.
It checks 17 input values and the predicted RUL distribution. The checker
needs at least 30 distinct recorded engines. For each value, it reports
the difference between current and reference means divided by the
reference standard deviation; a value of at least 1.0 raises an alert.

`python scripts\replay_monitoring.py` sends 30 FD001 test engines through
FastAPI's in-process TestClient and records their predictions. This
requires `RELIABILITYML_MODEL_URI=models:/ReliabilityML-FD001-RUL/2`
and `RELIABILITYML_MONITORING_DB=monitoring.db` in Anaconda Prompt.
Replaying with `--sensor2-offset 3.0` deliberately changed sensor 2
and produced a sensor 2 alert (shift 6.59); prediction shift was 0.34.

This is a simulated monitoring demonstration using FD001 data, not
observed production drift. The reference overlaps the replayed dataset,
the threshold is exploratory, and no live failure labels are available
to calculate post-deployment MAE. The local monitoring database contains
synthetic replay events and is ignored by Git.