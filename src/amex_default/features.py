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
        f"CAST(AVG({column_name}) AS FLOAT) AS {column_name}_mean",
        f"MIN({column_name}) AS {column_name}_min",
        f"MAX({column_name}) AS {column_name}_max",
        f"CAST(COUNT({column_name}) AS SMALLINT) AS {column_name}_non_null_count",
        f"FIRST({column_name} ORDER BY {constants.DATE_COLUMN}) AS {column_name}_first",
        f"LAST({column_name} ORDER BY {constants.DATE_COLUMN}) AS {column_name}_latest",
        f"CAST(STDDEV_SAMP({column_name}) AS FLOAT) AS {column_name}_std",
        f"CAST({column_name}_latest - {column_name}_mean AS FLOAT) AS {column_name}_latest_minus_mean",
        f"{column_name}_latest - {column_name}_first AS {column_name}_latest_minus_first",
        f"CAST(1.0 - {column_name}_non_null_count / statement_count AS FLOAT) AS {column_name}_missing_rate"
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

def build_categorical_aggregation_expressions(column_name):
    return [
        f"FIRST({column_name} ORDER BY {constants.DATE_COLUMN}) AS {column_name}_first",
        f"LAST({column_name} ORDER BY {constants.DATE_COLUMN}) AS {column_name}_latest",
        f"CAST(COUNT(DISTINCT {column_name}) AS SMALLINT) AS {column_name}_nunique",
        f"MODE({column_name}) AS {column_name}_mode",
        f"CAST(1.0 - COUNT({column_name}) / statement_count AS FLOAT) AS {column_name}_missing_rate",
        f"CAST(SUM(CASE WHEN {column_name} IS NOT NULL AND previous_{column_name} IS NOT NULL AND {column_name} <> previous_{column_name} THEN 1 ELSE 0 END) AS SMALLINT) AS {column_name}_transition_count"
    ]

def build_all_categorical_expressions(categorical_columns):
    expressions = []
    for column in categorical_columns:
        column_expressions = build_categorical_aggregation_expressions(column)
        expressions.extend(column_expressions)
    return expressions

def build_lag_categorical_expressions(categorical_columns):
    expressions = []
    for col in categorical_columns:
        expressions.append(f"LAG({col}) OVER (PARTITION BY {constants.CUSTOMER_ID} ORDER BY {constants.DATE_COLUMN}) AS previous_{col}")
    return expressions