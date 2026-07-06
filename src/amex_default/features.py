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
        f"{column_name}_max - {column_name}_min AS {column_name}_range",
        f"CAST(COUNT({column_name}) AS SMALLINT) AS {column_name}_non_null_count",
        f"FIRST({column_name} ORDER BY {constants.DATE_COLUMN}) AS {column_name}_first",
        f"LAST({column_name} ORDER BY {constants.DATE_COLUMN}) AS {column_name}_latest",
        f"CAST(STDDEV_SAMP({column_name}) AS FLOAT) AS {column_name}_std",
        f"CAST({column_name}_latest - {column_name}_mean AS FLOAT) AS {column_name}_latest_minus_mean",
        f"{column_name}_latest - {column_name}_first AS {column_name}_latest_minus_first",
        f"CAST(1.0 - {column_name}_non_null_count / statement_count AS FLOAT) AS {column_name}_missing_rate",
        f"CAST(LIST_AVG(LIST_SLICE(LIST({column_name} ORDER BY {constants.DATE_COLUMN} DESC), 1, 3)) AS FLOAT) AS {column_name}_recent_3_mean",
        f"CAST({column_name}_recent_3_mean - {column_name}_mean AS FLOAT) AS {column_name}_recent_3_minus_mean",
        f"CAST(LIST({column_name} ORDER BY {constants.DATE_COLUMN} DESC)[1] - LIST({column_name} ORDER BY {constants.DATE_COLUMN} DESC)[2] AS FLOAT) AS {column_name}_latest_minus_previous",
        f"""
        CASE 
            WHEN COUNT({column_name}) >= 2 
            THEN CAST(REGR_SLOPE({column_name}, DATEDIFF('month', DATE '2017-01-01', {constants.DATE_COLUMN})) AS FLOAT) 
            ELSE NULL 
        END AS {column_name}_monthly_slope
        """
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
        f"MODE({column_name} ORDER BY {constants.DATE_COLUMN} DESC) AS {column_name}_mode",
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
        expressions.append(f"LAG({col} IGNORE NULLS) OVER (PARTITION BY {constants.CUSTOMER_ID} ORDER BY {constants.DATE_COLUMN}) AS previous_{col}")
    return expressions

def build_history_sequence_expression() -> str:
    return f"""
    CAST(
        DATEDIFF(
            'day',
            LAG({constants.DATE_COLUMN}) OVER (PARTITION BY {constants.CUSTOMER_ID} ORDER BY {constants.DATE_COLUMN}),
            {constants.DATE_COLUMN}
        ) AS SMALLINT
    ) AS gap_days
    """

def build_customer_history_aggregate_expressions() -> list[str]:
    return [
        f"CAST(DATEDIFF('day', MIN({constants.DATE_COLUMN}), MAX({constants.DATE_COLUMN})) AS SMALLINT) AS history_span_days",
        f"CAST(STDDEV_SAMP(gap_days) AS FLOAT) AS std_gap_days",
        f"CAST(MAX(gap_days) AS SMALLINT) AS max_gap_days",
        f"CAST(MIN(gap_days) AS SMALLINT) AS min_gap_days"
    ]
