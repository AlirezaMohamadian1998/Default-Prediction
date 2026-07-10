import numpy as np
import math
import pytest
from amex_default.metrics import amex_metric

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