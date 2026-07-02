# AMEX Default Prediction

A implementation of an AMEX default-prediction pipeline using
DuckDB, temporal feature engineering, and LightGBM.

## Dataset

- `train.parquet`: monthly customer statements
- `train_labels.csv`: one default label per customer