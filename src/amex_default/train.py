import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from amex_default import constants
from amex_default.metrics import amex_metric
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

OUTPUT_DIR = "artifacts/model"
LABELS_PATH = "train_labels.csv"
FEATURES_PATH = "artifacts/train_features.parquet"
RANDOM_SEED = constants.DEFAULT_SEED


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

    if sample_size is None:
        features = pd.read_parquet(features_path)
    elif sample_size <= 0:
        raise ValueError("Sample size should be greater than 0")
    else:
        features = pd.read_parquet(features_path).sample(n=sample_size, random_state=random_seed)

    labels = pd.read_csv(labels_path)
    labels.set_index('customer_ID', inplace=True)
    customer_ID = features.pop("customer_ID")

    X = features
    y = labels.loc[customer_ID, 'target']

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    fold_metrics = []
    fold_predictions = []
    fold_importances = []

    for fold, (train_idx, validation_idx) in enumerate(skf.split(X, y)):
        train_X = X.iloc[train_idx]
        train_y = y.iloc[train_idx]
        validation_X = X.iloc[validation_idx]
        validation_y = y.iloc[validation_idx]
        validation_ids = customer_ID.iloc[validation_idx]

        result = train_one_fold(train_X, train_y, validation_X, validation_y, validation_ids, random_seed, fold + 1)
        model = result["model"]
        model.booster_.save_model(output_path / f"model_fold_{fold + 1}.txt")

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
    oof_pr_auc_score= average_precision_score(oof_predictions["target"], oof_predictions["prediction_probability"])
    oof_amex_score = amex_metric(oof_predictions["target"], oof_predictions["prediction_probability"])

    summary_metrics = {
        "roc_auc": oof_auc_score,
        "pr_auc": oof_pr_auc_score,
        "amex_metric": oof_amex_score,
        "total_customer_count": len(oof_predictions),
        "random_seed": random_seed,
        "fold_count": folds,
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
        early_stopping(stopping_rounds=100, verbose=True),
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

    metrics = {
        "roc_auc": auc_score,
        "pr_auc": pr_auc_score,
        "amex_metric": amex_score,
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