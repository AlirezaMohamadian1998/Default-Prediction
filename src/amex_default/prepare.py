from amex_default import features
from amex_default.database import connect
from amex_default.audit import audit_dataset
from pathlib import Path

def _build_numerical_query(numerical_columns, clean_expression):
    all_numeric_aggregate_expressions = features.build_all_numeric_expressions(numerical_columns)

    joined_numerical_aggregate_expression = ",\n".join(all_numeric_aggregate_expressions)

    return f"""
        SELECT 
            customer_ID,
            CAST(COUNT(*) AS SMALLINT) AS statement_count,
            {joined_numerical_aggregate_expression}
        FROM (
            SELECT {clean_expression}
            FROM read_parquet($train_path)
        )
        GROUP BY customer_ID
    """   

def _build_categorical_history_query(categorical_columns, clean_expression):
    all_categorical_aggregate_expressions = features.build_all_categorical_expressions(categorical_columns)
    lagged_expressions = features.build_lag_categorical_expressions(categorical_columns)
    customer_history_aggregate_expressions = features.build_customer_history_aggregate_expressions()

    customer_history_sequence_expression = features.build_history_sequence_expression()
    joined_categorical_aggregate_expression = ",\n".join(all_categorical_aggregate_expressions)
    joined_lagged_expression = ",\n".join(lagged_expressions)
    joined_customer_history_aggregate_expression = ",\n".join(customer_history_aggregate_expressions)
    joined_categorical_column_names = ",\n".join(categorical_columns)

    return f"""
        WITH cleaned AS (
            SELECT {clean_expression}
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
    """

def _copy_query_to_parquet(connection, query, parameters):
    output_path = Path(parameters['output_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(f"COPY({query}) TO $output_path (FORMAT PARQUET, COMPRESSION ZSTD)", parameters)

def prepare_features(
        train_path: str, 
        labels_path: str, 
        final_output_path_str: str,
        working_directory_str: str, 
        temp_directory, 
        threads
    ):
    working_directory = Path(working_directory_str)
    working_directory.mkdir(parents=True, exist_ok=True)

    audit_result = audit_dataset(train_path, labels_path)
    schema = audit_result['schema']
    sentinel_counts = audit_result['sentinel_counts']

    clean_expression = ",\n".join(features.build_clean_source_expressions(schema, sentinel_counts))
    numerical_columns, categorical_columns = features.classify_feature_columns(schema)

    numerical_query = _build_numerical_query(numerical_columns, clean_expression)
    categorical_history_query = _build_categorical_history_query(categorical_columns, clean_expression)

    connection = connect(temp_directory= temp_directory, threads= threads)
    try:
        parameters = {"train_path": str(train_path)}
        numerical_output_path = f"{working_directory_str}/numeric_features.parquet"
        categorical_output_path = f"{working_directory_str}/categorical_history_features.parquet"
        
        _copy_query_to_parquet(
            connection,
            numerical_query,
            {**parameters, "output_path": numerical_output_path}
        )
        _copy_query_to_parquet(
            connection, 
            categorical_history_query, 
            {**parameters, "output_path": categorical_output_path}
        )

        _validate_intermediate_features(connection, numerical_output_path, categorical_output_path, audit_result["customer_count"])

        parameters = {
            "numeric_path": numerical_output_path,
            "categorical_path": categorical_output_path,
            "output_path": final_output_path_str,
        }
        final_join_query = f"""
            SELECT
                numeric.*,
                categorical.* EXCLUDE (customer_ID, statement_count)
            FROM read_parquet($numeric_path) AS numeric
            INNER JOIN read_parquet($categorical_path) AS categorical
            USING (customer_ID)
        """
        _copy_query_to_parquet(connection, final_join_query, parameters)

    finally:
        connection.close()

    return final_output_path_str

def _validate_intermediate_features(connection, numerical_output_path, categorical_output_path, expected_customer_count):
    count_numeric_customer = connection.execute(f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT customer_ID)
        FROM read_parquet(?)
    """, [numerical_output_path]  
    ).fetchone()
    
    count_categorical_customer = connection.execute(f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT customer_ID)
        FROM read_parquet(?)
    """ , [categorical_output_path]      
    ).fetchone()

    if(count_numeric_customer[0] == count_numeric_customer[1] == count_categorical_customer[0] == count_categorical_customer[1] == expected_customer_count):
        return
    else:
        raise ValueError("There are duplicates or less rows")