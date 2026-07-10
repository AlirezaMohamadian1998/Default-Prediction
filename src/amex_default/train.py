import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from amex_default import constants
from amex_default.metrics import amex_metric
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

OUTPUT_DIR = "artifacts/model"
LABELS_PATH = "train_labels.csv"
FEATURES_PATH = "artifacts/train_features.parquet"
RANDOM_SEED = constants.DEFAULT_SEED
TEST_SIZE = 0.2

def train_model(
        output_dir= OUTPUT_DIR,
        features_path=FEATURES_PATH,
        labels_path=LABELS_PATH,
        sample_size=None,
        test_size=TEST_SIZE,
        random_seed=RANDOM_SEED
    ):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if sample_size is None:
        features = pd.read_parquet(features_path)
    elif sample_size <= 0:
        raise ValueError("Sample size should be greater than 0")
    else:
        features = pd.read_parquet(features_path).sample(n=sample_size, random_state=random_seed)

    labels = pd.read_csv(labels_path)
    labels.set_index('customer_ID', inplace=True)
    customer_IDs = features["customer_ID"].copy()
    customer_ID = features.pop("customer_ID")

    X = features
    y = labels.loc[customer_ID, 'target']

    train_X, test_X, train_y, test_y, _, test_ids = train_test_split(X, y, customer_IDs, random_state=random_seed, test_size=test_size, stratify=y)

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
        eval_set=[(test_X, test_y)],
        eval_metric=['average_precision', 'auc'],
        callbacks=callbacks
    )

    probabilities = np.asarray(model.predict_proba(test_X))
    validation_probabilities = probabilities[:, 1]

    auc_score = roc_auc_score(test_y, validation_probabilities)
    pr_auc_score= average_precision_score(test_y, validation_probabilities)
    amex_score = amex_metric(test_y, validation_probabilities)

    model.booster_.save_model(output_path / "model.txt")
    metrics = {
        "roc_auc": auc_score,
        "pr_auc": pr_auc_score,
        "amex_metric": amex_score,
        "sample_size": sample_size if sample_size is not None else len(features),
        "random_seed": random_seed,
        "validation_size": test_size,
        "best_iteration": model.best_iteration_
    }
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {output_path}/metrics.json")

    predictions_df = pd.DataFrame({
        "customer_ID": test_ids,
        "target": test_y.values,
        "prediction_probability": validation_probabilities
    })
    predictions_df.to_csv(output_path / "validation_predictions.csv", index=False)

    print(f"ROC-AUC Score: {auc_score:.4f}")
    print(f"PR-AUC Score: {pr_auc_score:.4f}")
    print(f"AMEX score: {amex_score:.4f}")

def _argument_parser():
    parser = argparse.ArgumentParser(
        description="Train a baseline LightGBM default-risk model from prepared AMEX customer features."
    )
    parser.add_argument(
        "--output-dir",
        type=str, 
        required=False, 
        default=OUTPUT_DIR, 
        help="Directory where the trained model, metrics, and validation predictions will be saved."
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
        "--validation-size", 
        type=float, 
        required=False, 
        default=TEST_SIZE, 
        help="Fraction of sampled customers held out for validation and early stopping"
    )
    parser.add_argument(
        "--random-seed", 
        type=int, 
        required=False, 
        default=RANDOM_SEED, 
        help="Random seed used for sampling and train/validation splitting."
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
        test_size=args.validation_size,
        random_seed=args.random_seed

    )

if __name__ == "__main__":
    main()