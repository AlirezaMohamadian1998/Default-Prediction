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

def build_numeric_aggregation_expressions(column_name):
    return [
        f"AVG({column_name}) AS {column_name}_mean",
        f"MIN({column_name}) AS {column_name}_min",
        f"MAX({column_name}) AS {column_name}_max",
        f"COUNT({column_name}) AS {column_name}_non_null_count",
        f"FIRST({column_name} ORDER BY {constants.DATE_COLUMN}) AS {column_name}_first",
        f"LAST({column_name} ORDER BY {constants.DATE_COLUMN}) AS {column_name}_latest",
        f"STDDEV_SAMP({column_name}) AS {column_name}_std",
        f"{column_name}_latest - {column_name}_mean AS {column_name}_latest_minus_mean",
        f"{column_name}_latest - {column_name}_first AS {column_name}_latest_minus_first",
        f"1.0 - {column_name}_non_null_count / statement_count AS {column_name}_missing_rate"
    ]

def build_all_numeric_expressions(numeric_columns):
    expressions = []
    for column_name in numeric_columns:
        column_expressions = build_numeric_aggregation_expressions(column_name)
        expressions.extend(column_expressions)
    return expressions

def build_clean_source_expressions(schema, sentinel_counts: dict):
    expressions =[]
    sentinel_columns = set(sentinel_counts)
    for column_name, column_type in zip(schema['column_name'], schema['column_type']):
        if(column_name == constants.CUSTOMER_ID):
            expressions.append(column_name)
        elif(column_name == constants.DATE_COLUMN):
            expressions.append(f"TRY_CAST({column_name} AS DATE) AS {column_name}")
        elif (column_name in sentinel_columns):
            expressions.append(f"CAST(NULLIF({column_name}, -1) AS {column_type}) AS {column_name}")
        else:
            expressions.append(column_name)
    return expressions