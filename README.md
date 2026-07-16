# AMEX Default Prediction

An end-to-end machine learning pipeline for predicting customer default risk
from American Express monthly statement histories. It uses DuckDB for
memory-efficient feature preparation and LightGBM for five-fold stratified
cross-validation.

The model produces a default probability for each customer and converts it to a
binary prediction using a threshold selected from out-of-fold predictions.

## Dataset

The project uses data from the
[American Express - Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction)
competition. The data is not included in this repository.

Expected input files:

- `train.parquet`: monthly customer statement records
- `train_labels.csv`: one binary default label per customer

The raw data contains multiple rows per customer. `customer_ID` identifies the
customer, `S_2` is the statement date, and `target` is either `1` for default or
`0` for non-default. The anonymized feature prefixes represent delinquency
(`D_*`), spend (`S_*`), payment (`P_*`), balance (`B_*`), and risk (`R_*`).

The training data used for the reported result contains 5,531,451 statements
from 458,913 customers, with a default rate of approximately 25.9%.

## Pipeline

`audit -> clean -> aggregate customer histories -> train five folds -> evaluate
OOF predictions -> predict with the ensemble`

### Data Preparation

The audit validates customer IDs, dates, labels, duplicate statements, and
customer coverage. Detected integer `-1` sentinels are converted to missing
values before aggregation.

DuckDB then converts each customer's monthly history into one feature row.
Numerical features include summary statistics, first and latest values, recent
means, change features, missing rates, and calendar-aware slopes. Categorical
features include first, latest, mode, unique counts, missing rates, and state
transitions. Statement count, history span, and statement-gap features describe
the available customer history.

The resulting dataset contains 2,726 model features. Numeric calculations are
chunked to control memory usage, and compatible intermediate Parquet files can
be reused.

### Model Training

Training uses five-fold `StratifiedKFold` with LightGBM, seed `42`, a learning
rate of `0.05`, and early stopping based on average precision. Each customer is
predicted once by a model that did not train on that customer, producing a
complete set of out-of-fold (OOF) predictions.

The five trained models are retained as an ensemble. For new data, their
probabilities are averaged before applying the saved classification threshold.

## Results

Results from the complete five-fold training run:

| Metric | Score |
| --- | ---: |
| ROC-AUC | 0.9612 |
| PR-AUC | 0.8977 |
| AMEX metric | 0.7913 |
| Brier score | 0.0681 |
| F1 score | 0.8178 |
| Precision | 0.7767 |
| Recall | 0.8634 |
| Selected threshold | 0.40 |

The threshold maximizes F1 on the OOF predictions. A real financial application
should instead choose a threshold based on the costs of missed defaults and
false alerts.

## Installation

Python 3.12 or newer is required.

```powershell
git clone https://github.com/AlirezaMohamadian1998/Default-Prediction.git
cd Default-Prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Place `train.parquet` and `train_labels.csv` in the project root. Data, generated
features, and trained model artifacts are ignored by Git.

## Usage

### Prepare Features

```powershell
python -m amex_default.prepare `
  --input train.parquet `
  --labels train_labels.csv `
  --output artifacts/train_features.parquet `
  --working-directory artifacts/intermediate `
  --temp-directory artifacts/duckdb_tmp
```

### Train the Model

```powershell
python -m amex_default.train `
  --features-path artifacts/train_features.parquet `
  --labels-path train_labels.csv `
  --output-dir artifacts/model_final `
  --folds 5
```

Use `--sample-size` and fewer folds for a quick smoke test before full training.

### Predict New Customers

```powershell
python -m amex_default.predict `
  --input test.parquet `
  --model-dir artifacts/model_final `
  --output artifacts/test_predictions.csv
```

The output contains `customer_ID`, `default_probability`, and
`predicted_label`. Prepared prediction features are temporary by default; use
`--prepared-output` to retain them.

Training produces five model files, a compatibility manifest, overall and
per-fold metrics, OOF predictions, and feature-importance tables inside the
selected output directory.

## Project Structure

```text
src/amex_default/
|-- audit.py       # Dataset validation
|-- database.py    # DuckDB configuration
|-- features.py    # Temporal feature expressions
|-- metrics.py     # AMEX metric and threshold selection
|-- prepare.py     # Feature preparation CLI
|-- train.py       # Cross-validation training CLI
`-- predict.py     # Ensemble prediction CLI

tests/             # Automated pytest suite
artifacts/         # Generated files, ignored by Git
```

## Tests

```powershell
python -m pytest
```

The tests cover data preparation, caching, metrics, training-input validation,
feature alignment, and ensemble prediction behavior.
