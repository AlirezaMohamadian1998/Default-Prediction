import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from amex_default.constants import CUSTOMER_ID, TARGET, DEFAULT_SEED
from amex_default.metrics import amex_metric, select_f1_threshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

OUTPUT_DIR = "artifacts/model"
LABELS_PATH = "train_labels.csv"
FEATURES_PATH = "artifacts/train_features.parquet"
RANDOM_SEED = DEFAULT_SEED

def validate_training_inputs(features:pd.DataFrame, labels:pd.DataFrame) -> None:
    if CUSTOMER_ID not in features.columns:
        raise ValueError(f"Features must include {CUSTOMER_ID}.")
    if CUSTOMER_ID not in labels.columns:
        raise ValueError(f"Labels must include {CUSTOMER_ID}.")
    if TARGET not in labels.columns:
        raise ValueError(f"Labels must include {TARGET}.")

    features_customer_ids = features.loc[:, CUSTOMER_ID]
    labels_customer_ids = labels.loc[:, CUSTOMER_ID]

    if features_customer_ids.isna().any():
        raise ValueError(f"(features): {CUSTOMER_ID} cannot be NULL.")
    if labels_customer_ids.isna().any():
        raise ValueError(f"(labels): {CUSTOMER_ID} cannot be NULL.")
    if labels.loc[:, TARGET].isna().any():
        raise ValueError(f"{TARGET} cannot be NULL.")
    if features_customer_ids.duplicated().any():
        raise ValueError(f"The features contain duplicate {CUSTOMER_ID}s")
    if labels_customer_ids.duplicated().any():
        raise ValueError(f"The labels contain duplicate {CUSTOMER_ID}s")

    actual_targets = set(labels.loc[:, TARGET].unique())
    if actual_targets != {0, 1}:
        raise ValueError(f"Expected binary targets {{0, 1}}, found {actual_targets}")

    features_customer_ids_set = set(features_customer_ids)
    labels_customer_ids_set = set(labels_customer_ids)
    missing_customer_ids_from_labels = features_customer_ids_set - labels_customer_ids_set
    missing_customer_ids_from_features = labels_customer_ids_set - features_customer_ids_set

    if len(missing_customer_ids_from_features) != 0:
        raise ValueError(f"There are {len(missing_customer_ids_from_features)} {CUSTOMER_ID}s missing from features.")
    if len(missing_customer_ids_from_labels) != 0:
        raise ValueError(f"There are {len(missing_customer_ids_from_labels)} {CUSTOMER_ID}s missing from labels.")

def train_model(
        output_dir= OUTPUT_DIR,
        features_path=FEATURES_PATH,
        labels_path=LABELS_PATH,
        sample_size=None,
        random_seed=RANDOM_SEED,
        folds= 5
    ):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if folds < 2:
        raise ValueError("Number of folds cannot be less than 2")
    if sample_size is not None and sample_size <= 0:
        raise ValueError("Sample size should be greater than 0")

    features = pd.read_parquet(features_path)
    labels = pd.read_csv(labels_path)
    validate_training_inputs(features, labels)

    if sample_size is not None:
        features = features.sample(n=sample_size, random_state=random_seed)

    labels.set_index(CUSTOMER_ID, inplace=True)
    customer_ID = features.pop(CUSTOMER_ID)

    X = features
    y = labels.loc[customer_ID, TARGET]

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    fold_metrics = []
    fold_predictions = []
    fold_importances = []
    model_names = []

    for fold, (train_idx, validation_idx) in enumerate(skf.split(X, y)):
        train_X = X.iloc[train_idx]
        train_y = y.iloc[train_idx]
        validation_X = X.iloc[validation_idx]
        validation_y = y.iloc[validation_idx]
        validation_ids = customer_ID.iloc[validation_idx]

        result = train_one_fold(train_X, train_y, validation_X, validation_y, validation_ids, random_seed, fold + 1)
        model = result["model"]
        model.booster_.save_model(output_path / f"model_fold_{fold + 1}.txt")
        model_names.append(f"model_fold_{fold + 1}.txt")

        fold_predictions.append(result["predictions_df"])
        fold_metrics.append(result["metrics"])
        fold_importances.append(
            pd.DataFrame({
                "feature": model.booster_.feature_name(),
                "importance_gain": model.booster_.feature_importance(importance_type="gain"),
                "fold": fold + 1
            })
        )
    
    oof_predictions = pd.concat(fold_predictions, ignore_index=True)
    combined_fold_importances = pd.concat(fold_importances, ignore_index=True)

    average_feature_importance = (
        combined_fold_importances.groupby("feature")["importance_gain"]
        .mean()
        .reset_index()
        .sort_values(by="importance_gain", ascending=False)
    )

    if len(oof_predictions) != len(X): 
        raise ValueError("Some customers default probability hasn't been predicted")
    if oof_predictions["customer_ID"].duplicated().any():
        raise ValueError("There are duplicate predictions for some customers")
    
    combined_fold_importances.to_csv(output_path / "feature_importance_folds.csv", index=False)
    oof_predictions.to_csv(output_path / f"oof_predictions.csv", index=False)
    average_feature_importance.to_csv(output_path / "feature_importance.csv", index=False)

    oof_auc_score = roc_auc_score(oof_predictions["target"], oof_predictions["prediction_probability"])
    oof_pr_auc_score = average_precision_score(oof_predictions["target"], oof_predictions["prediction_probability"])
    oof_amex_score = amex_metric(oof_predictions["target"], oof_predictions["prediction_probability"])
    oof_brier_score = brier_score_loss(oof_predictions["target"], oof_predictions["prediction_probability"])

    thresholds = np.arange(0.05, 0.96, 0.01)
    select_threshold = select_f1_threshold(oof_predictions["target"], oof_predictions["prediction_probability"], thresholds)

    manifest = {
        "feature_names": X.columns.tolist(),
        "id_column": "customer_ID",
        "feature_count": len(X.columns),
        "fold_count": folds,
        "random_seed": random_seed,
        "threshold": select_threshold["threshold"],
        "model_files": model_names,
        "lightgbm_params": model.get_params()
    }

    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    summary_metrics = {
        "roc_auc": oof_auc_score,
        "pr_auc": oof_pr_auc_score,
        "amex_metric": oof_amex_score,
        "brier_score": oof_brier_score,
        "total_customer_count": len(oof_predictions),
        "random_seed": random_seed,
        "fold_count": folds,
        "threshold_selection": select_threshold,
        "individual_fold_metrics": fold_metrics
    }

    with open(output_path / "metrics.json", "w") as f:
        json.dump(summary_metrics, f, indent=4)


def train_one_fold(train_X, train_y, validation_X, validation_y, validation_ids, random_seed, fold_num):
    model = LGBMClassifier(
        objective="binary",
        n_estimators=1500,
        random_state=random_seed,
        learning_rate=0.05,
        n_jobs=-1,
        verbose=-1
    )

    callbacks = [
        early_stopping(stopping_rounds=100, verbose=True, first_metric_only=True),
        log_evaluation(period=50)
    ]

    model.fit(
        X=train_X, y=train_y,
        eval_set=[(validation_X, validation_y)],
        eval_metric=['average_precision', 'auc'],
        callbacks=callbacks
    )

    probabilities = np.asarray(model.predict_proba(validation_X))
    validation_probabilities = probabilities[:, 1]

    auc_score = roc_auc_score(validation_y, validation_probabilities)
    pr_auc_score= average_precision_score(validation_y, validation_probabilities)
    amex_score = amex_metric(validation_y, validation_probabilities)
    brier_score = brier_score_loss(validation_y, validation_probabilities)

    metrics = {
        "roc_auc": auc_score,
        "pr_auc": pr_auc_score,
        "amex_metric": amex_score,
        "brier_score": brier_score,
        "training_size": len(train_X),
        "random_seed": random_seed,
        "validation_size": len(validation_X),
        "best_iteration": model.best_iteration_,
        "fold": fold_num
    }

    predictions_df = pd.DataFrame({
        "customer_ID": validation_ids,
        "target": validation_y.values,
        "prediction_probability": validation_probabilities,
        "fold": fold_num
    })

    return {
        "model": model,
        "predictions_df": predictions_df,
        "metrics": metrics
    }


def _argument_parser():
    parser = argparse.ArgumentParser(
        description="Train a stratified cross-validation LightGBM default-risk model from prepared AMEX customer features."
    )
    parser.add_argument(
        "--output-dir",
        type=str, 
        required=False, 
        default=OUTPUT_DIR, 
        help="Directory where the trained model, metrics, and OOF predictions will be saved."
    )
    parser.add_argument(
        "--features-path",
        type=str, 
        required=False, 
        default=FEATURES_PATH, 
        help="Path to prepared one-row-per-customer feature parquet"
    )
    parser.add_argument(
        "--labels-path", 
        type=str, 
        required=False, 
        default=LABELS_PATH, 
        help="Path to training labels CSV containing customer_ID and target"
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        required=False, 
        default=None, 
        help="Number of customer rows to randomly sample before training. Omit to use all rows"
    )
    parser.add_argument(
        "--random-seed", 
        type=int, 
        required=False, 
        default=RANDOM_SEED, 
        help="Random seed used for sampling and stratified fold splitting"
    )
    parser.add_argument(
        "--folds", 
        type=int, 
        required=False, 
        default=5, 
        help="Number of stratified cross-validation folds"
    )
    return parser


def main():
    parser = _argument_parser()
    args = parser.parse_args()

    train_model(
        output_dir=args.output_dir,
        features_path=args.features_path,
        labels_path=args.labels_path,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
        folds=args.folds
    )

if __name__ == "__main__":
    main()