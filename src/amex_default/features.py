from amex_default import constants

def classify_feature_columns(schema):
    column_names = [
        col
        for col in schema["column_name"].tolist() 
        if col.startswith(constants.FEATURE_PREFIXES) and col != constants.DATE_COLUMN
    ]

    numeric_features = [col for col in column_names if col not in constants.CATEGORICAL_COLUMNS]
    categorical_features = [col for col in column_names if col in constants.CATEGORICAL_COLUMNS]

    return numeric_features, categorical_features