import pandas as pd
import numpy as np
from pathlib import Path
from amex_default import constants
from amex_default.metrics import amex_metric
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

LABELS_PATH = Path("train_labels.csv")
FEATURES_PATH = Path("artifacts/train_features.parquet")
SAMPLE_SIZE = 200000
RANDOM_SEED = constants.DEFAULT_SEED
TEST_SIZE = 0.2

def train_model():
    labels = pd.read_csv(LABELS_PATH)
    features = pd.read_parquet(FEATURES_PATH).sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

    labels.set_index('customer_ID', inplace=True)
    customer_ID = features.pop("customer_ID")

    X = features
    y = labels.loc[customer_ID, 'target']

    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=RANDOM_SEED, test_size=TEST_SIZE, stratify=y)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=1500,
        random_state=RANDOM_SEED,
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

    print(f"ROC-AUC Score: {auc_score:.4f}")
    print(f"PR-AUC Score: {pr_auc_score:.4f}")
    print(f"AMEX score: {amex_score:.4f}")

if __name__ == "__main__":
    train_model()