from amex_default import features
from amex_default.database import connect
from amex_default.audit import audit_dataset
from pathlib import Path

def prepare_numeric_features(train_path_str: str, labels_path_str: str, output_path_str: str, temp_directory, threads):
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audit_result = audit_dataset(train_path_str, labels_path_str)
    schema = audit_result['schema']
    sentinel_columns = audit_result['sentinel_counts']
    numerical_columns, _ = features.classify_feature_columns(schema)

    clean_expressions = features.build_clean_source_expressions(schema, sentinel_columns)
    all_numeric_aggregate_expressions = features.build_all_numeric_expressions(numerical_columns)

    joined_clean_expression = ",\n".join(clean_expressions)
    joined_numerical_aggregate_expression = ",\n".join(all_numeric_aggregate_expressions)

    connection = connect(threads=threads, temp_directory=temp_directory)
    try:
        query = f"""
            COPY(
                SELECT 
                    customer_ID,
                    CAST(COUNT(*) AS SMALLINT) AS statement_count,
                    {joined_numerical_aggregate_expression}
                FROM (
                    SELECT {joined_clean_expression}
                    FROM read_parquet($train_path)
                )
                GROUP BY customer_ID
            ) TO $output_path (FORMAT PARQUET, COMPRESSION ZSTD)
        """

        connection.execute(query, {"train_path": train_path_str, "output_path": str(output_path),})

    finally:
        connection.close()

    return output_path

def prepare_categorical_history_features(train_path_str: str, labels_path_str: str, output_path_str: str, temp_directory, threads):
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audit_result = audit_dataset(train_path_str, labels_path_str)
    schema = audit_result['schema']
    sentinel_columns = audit_result['sentinel_counts']
    _, categorical_columns = features.classify_feature_columns(schema)

    clean_expressions = features.build_clean_source_expressions(schema, sentinel_columns)
    all_categorical_aggregate_expressions = features.build_all_categorical_expressions(categorical_columns)
    lagged_expressions = features.build_lag_categorical_expressions(categorical_columns)
    customer_history_aggregate_expressions = features.build_customer_history_aggregate_expressions()

    customer_history_sequence_expression = features.build_history_sequence_expression()
    joined_clean_expression = ",\n".join(clean_expressions)
    joined_categorical_aggregate_expression = ",\n".join(all_categorical_aggregate_expressions)
    joined_lagged_expression = ",\n".join(lagged_expressions)
    joined_customer_history_aggregate_expression = ",\n".join(customer_history_aggregate_expressions)
    joined_categorical_column_names = ",\n".join(categorical_columns)

    connection = connect(threads=threads, temp_directory=temp_directory)
    try:
        query = f"""
            COPY(
                WITH cleaned AS (
                    SELECT {joined_clean_expression}
                    FROM read_parquet($train_path)
                ),
                sequenced AS (
                    SELECT 
                        customer_ID,
                        S_2,
                        {joined_categorical_column_names},
                        {joined_lagged_expression},
                        {customer_history_sequence_expression}
                    FROM cleaned
                )
                SELECT
                    customer_ID,
                    CAST (COUNT(*) AS SMALLINT) AS statement_count,
                    {joined_categorical_aggregate_expression},
                    {joined_customer_history_aggregate_expression}
                FROM sequenced
                GROUP BY customer_ID 
            ) TO $output_path (FORMAT PARQUET, COMPRESSION ZSTD)
        """

        connection.execute(query, {"train_path": train_path_str, "output_path": str(output_path),})
    finally:
        connection.close()
    
    return output_path