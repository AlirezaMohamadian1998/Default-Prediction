import pytest
import pandas as pd
from amex_default.train import validate_training_inputs

def test_validate_training_inputs_rejects_customer_missing_from_labels():
    features = pd.DataFrame(
        {
            "customer_ID": ["A", "B", "C"],
            "dummy_feature": [0.1, 0.2, 0.3]
        }
    )
    labels = pd.DataFrame(
        {
            "customer_ID": ['A', 'B'],
            "target": [0, 1]
        }
    )

    with pytest.raises(ValueError, match="missing from labels"):
        validate_training_inputs(features, labels)
    