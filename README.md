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