import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from lightgbm import Booster
from tempfile import TemporaryDirectory
from amex_default.prepare import prepare_features
from amex_default import constants

def load_model_artifacts(models_dir:str):
    models_path = Path(models_dir)
    manifest_path = models_path / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError("Manifest.json doesn't exist")
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    if len(manifest["feature_names"]) != manifest["feature_count"]:
        raise ValueError("Manifest file is corrupted, feature count is not consistent")
    
    models = []

    for model in manifest["model_files"]:
        path = models_path / model

        if not path.exists():
            raise FileNotFoundError(f"{model} does not exist in this directory")
        
        loaded_model = Booster(model_file=path)

        if loaded_model.feature_name() != manifest["feature_names"]:
            raise ValueError("The order of features does not match")
        
        models.append(loaded_model)
    
    if len(models) != manifest["fold_count"]:
        raise ValueError("Manifest file is corrupted, the number of models is not equal to number of folds")
    
    return {
        "manifest": manifest,
        "models": models
    }

def validate_and_align_features(prepared_features: pd.DataFrame, manifest: dict):
    id_column = manifest["id_column"]
    ordered_feature_names = manifest["feature_names"]

    required_columns = set(ordered_feature_names)
    required_columns.add(id_column)

    available_feature_columns_from_df = set(prepared_features.columns)

    missing_columns = required_columns - available_feature_columns_from_df
    extra_columns = available_feature_columns_from_df - required_columns

    if len(missing_columns) != 0:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing_columns))}")
    if len(extra_columns) != 0:
        warnings.warn(f"Extra columns that will be ignored by the model: {', '.join(sorted(extra_columns))}")

    customer_ids = prepared_features[id_column]
    return {
        "customer_ids": customer_ids,
        "ordered_features": prepared_features.loc[:, ordered_feature_names]
    }

def predict_with_ensemble(models: list[Booster], ordered_features: pd.DataFrame, customer_ids: pd.Series, manifest:dict):
    if len(models) == 0:
        raise ValueError("Models cannot be empty")

    predictions = []
    for model in models:
        predictions.append(model.predict(ordered_features))
    
    probabilities = np.asarray(predictions).mean(axis=0)

    if len(customer_ids) != len(probabilities):
        raise ValueError("Number of customer IDs does not match number of prediction probabilities.")

    binary_predictions = probabilities >= manifest["threshold"]
    binary_predictions = binary_predictions.astype(int)

    return pd.DataFrame(
        {
            manifest["id_column"]: customer_ids,
            "default_probability": probabilities,
            "predicted_label": binary_predictions
        }
    )

def predict_prepared_features(prepared_features_path, models_dir, output_path):
    parquet_path = Path(prepared_features_path)
    output_csv_path = Path(output_path)

    if not parquet_path.exists():
        raise FileNotFoundError("Prepared parquet file doesn't exist")
    
    model_artifacts = load_model_artifacts(models_dir)
    models = model_artifacts["models"]
    manifest = model_artifacts["manifest"]
    
    prepared_features = pd.read_parquet(parquet_path)

    prediction_inputs = validate_and_align_features(prepared_features, manifest)
    customer_ids = prediction_inputs["customer_ids"]
    ordered_features = prediction_inputs["ordered_features"]

    prediction_output = predict_with_ensemble(models, ordered_features, customer_ids, manifest)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_output.to_csv(output_csv_path, index=False)
    return prediction_output

def predict_defaults(raw_input_path, 
                     model_dir, 
                     prediction_output_path, 
                     prepared_output_path=None,
                     threads=constants.DEFAULT_THREADS, 
                     memory_limit=constants.DEFAULT_MEMORY_LIMIT, 
                     chunk_size=constants.DEFAULT_CHUNK_SIZE):
    if not Path(raw_input_path).exists():
        raise FileNotFoundError("Raw parquet file doesn't exist.")
    
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        working = root / "working"
        duckdb_temp = root / "duckdb_temp"
        if prepared_output_path is None:
            prepared_features_path = root / "prepared_features.parquet"
        else:
            prepared_features_path = Path(prepared_output_path)  
            if prepared_features_path.exists():
                raise FileExistsError(f"Prepared output already exists: {prepared_features_path}. Choose another path or remove the existing file")

        prepare_features(
            train_path=str(raw_input_path),
            working_directory_str=str(working),
            temp_directory=str(duckdb_temp),
            final_output_path_str=str(prepared_features_path),
            threads= threads,
            memory_limit=memory_limit,
            chunk_size=chunk_size
            )
        return predict_prepared_features(prepared_features_path, model_dir, prediction_output_path)