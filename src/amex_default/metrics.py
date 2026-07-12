import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score

def _build_metric_table(y_true, y_pred):
    table = pd.DataFrame(
        {
            "target": np.asarray(y_true),
            "prediction": np.asarray(y_pred),
        }
    )
    if len(table) == 0:
        raise ValueError("AMEX metric cannot be calculated on empty data.")

    if table["target"].isna().any() or table["prediction"].isna().any():
        raise ValueError("AMEX metric inputs cannot contain missing values.")

    if set(table["target"].unique()) - {0, 1}:
        raise ValueError("AMEX metric target values must be only 0 and 1.")

    return table

def top_four_percent_captured(y_true, y_pred):
    table = _build_metric_table(y_true, y_pred)
    table = table.sort_values("prediction", ascending=False, kind="mergesort")

    table["weight"] = np.where(table["target"] == 0, 20, 1)
    cutoff = int(0.04 * table["weight"].sum())

    #Only takes the rows of top 4%, by cumsum until it hits the cutoff
    top_rows = table[table["weight"].cumsum() <= cutoff]
    total_defaults = table["target"].sum()

    if total_defaults == 0:
        raise ValueError("AMEX metric needs at least one default customer.")
    
    return top_rows["target"].sum() / total_defaults

def weighted_gini(y_true, y_pred):
    table = _build_metric_table(y_true, y_pred)
    table = table.sort_values("prediction", ascending=False, kind="mergesort")

    table["weight"] = np.where(table["target"] == 0, 20, 1)
    table["weighted_target"] = table["target"] * table["weight"]

    total_weight = table["weight"].sum()
    total_weighted_defaults = table["weighted_target"].sum()

    if total_weighted_defaults == 0:
        raise ValueError("AMEX metric needs at least one default customer.")

    random_cumulative = (table["weight"] / total_weight).cumsum()
    default_cumulative = table["weighted_target"].cumsum() / total_weighted_defaults

    return ((default_cumulative - random_cumulative) * table["weight"]).sum()

def select_f1_threshold(true_y, probabilities, thresholds):
    probabilities = np.asarray(probabilities)
    if len(thresholds) == 0:
        raise ValueError("Cannot have an empty list for thresholds")
    result = []
    for threshold in thresholds:
        predicted_labels = probabilities >= threshold
        score_f1 = f1_score(true_y, predicted_labels)
        result.append({"threshold": threshold, "f1_score": score_f1})

    result_df = pd.DataFrame(result)
    best = result_df.loc[result_df["f1_score"].idxmax()]
    best_threshold = best["threshold"]
    best_f1 = best["f1_score"]

    predicted_labels = probabilities >= best_threshold
    score_precision = precision_score(true_y, predicted_labels)
    score_recall = recall_score(true_y, predicted_labels)
    predicted_default_count = predicted_labels.sum()
    return {
        "threshold": float(best_threshold), #type: ignore
        "f1_score": float(best_f1), #type: ignore
        "precision_score": score_precision,
        "recall_score": score_recall,
        "predicted_default_count": int(predicted_default_count)
    }


def amex_metric(y_true, y_pred) -> float:
    capture_score = top_four_percent_captured(y_true, y_pred)

    model_gini = weighted_gini(y_true, y_pred)
    perfect_gini = weighted_gini(y_true, y_true)
    normalized_gini = model_gini / perfect_gini

    return 0.5 * capture_score + 0.5 * normalized_gini