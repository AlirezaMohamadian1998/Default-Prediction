import pandas as pd
from amex_default.constants import CUSTOMER_ID
from amex_default.predict import validate_and_align_features


def test_validate_and_align_features_uses_manifest_order():
    manifest = {
        "id_column": CUSTOMER_ID,
        "feature_names": ["feature_a", "feature_b"],
    }

    prepared_features = pd.DataFrame(
        {
            "feature_b": [2.0, 4.0],
            CUSTOMER_ID: ["A", "B"],
            "feature_a": [1.0, 3.0],
        }
    )

    result = validate_and_align_features(prepared_features, manifest)

    customer_ids = result["customer_ids"]
    ordered_features = result["ordered_features"]

    assert customer_ids.tolist() == ["A", "B"]
    assert ordered_features.columns.tolist() == ["feature_a", "feature_b"]
    assert ordered_features.to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]
