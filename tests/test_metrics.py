import numpy as np
import math
import pytest
import json
from amex_default.metrics import amex_metric, select_f1_threshold

def test_amex_metric_rewards_better_predictions():
    y_true = [1, 0, 1, 0, 0, 1, 0, 1, 0, 0]

    perfect_predictions = np.array([0.95, 0.10, 0.90, 0.20, 0.15, 0.85, 0.25, 0.80, 0.30, 0.05])
    inverse_predictions = 1 - perfect_predictions
    mixed_predictions = np.array([0.80, 0.75, 0.70, 0.30, 0.20, 0.65, 0.55, 0.60, 0.10, 0.25])

    perfect = amex_metric(y_true, perfect_predictions)
    bad = amex_metric(y_true, inverse_predictions)
    mix = amex_metric(y_true, mixed_predictions)

    assert perfect > mix > bad
    assert math.isclose(perfect, 1.0)

def test_amex_metric_rejects_invalid_labels():
    invalid_y_true = [0, 1, 2]
    predictions = [0.1, 0.8, 0.4]
    with pytest.raises(ValueError): 
        amex_metric(invalid_y_true, predictions)

def test_select_threshold():
    y_true = [0, 0, 1, 1]
    prob = [0.1, 0.4, 0.6, 0.9]
    thresholds = [0.3, 0.5, 0.7]

    result = select_f1_threshold(y_true, prob, thresholds)

    assert result["threshold"] == 0.5
    assert result["f1_score"] == result["precision_score"] == result["recall_score"] == 1.0
    assert result["predicted_default_count"] == 2

    with pytest.raises(ValueError):
        select_f1_threshold(y_true, prob, [])
    
    json.dumps(result)