from amex_default import features
from amex_default.database import connect
from amex_default.audit import audit_dataset
from amex_default import constants
from pathlib import Path
from math import ceil
import argparse

def _build_numerical_query(numerical_columns, clean_expression):
    numerical_aggregate_expression = ",\n".join(features.build_all_numeric_expressions(numerical_columns))

    return f"""
        SELECT 
            customer_ID,
            CAST(COUNT(*) AS SMALLINT) AS statement_count,
            {numerical_aggregate_expression}
        FROM (
            SELECT {clean_expression}
            FROM read_parquet($train_path)
        )
        GROUP BY customer_ID
    """   

def _build_categorical_history_query(categorical_columns, clean_expression):
    categorical_aggregate_expression = ",\n".join(features.build_all_categorical_expressions(categorical_columns))
    lagged_expression = ",\n".join(features.build_lag_categorical_expressions(categorical_columns))
    customer_history_aggregate_expression = ",\n".join(features.build_customer_history_aggregate_expressions())
    categorical_column_names = ",\n".join(categorical_columns)

    return f"""
        WITH cleaned AS (
            SELECT {clean_expression}
            FROM read_parquet($train_path)
        ),
        sequenced AS (
            SELECT 
                customer_ID,
                S_2,
                {categorical_column_names},
                {lagged_expression},
                {features.build_history_sequence_expression()}
            FROM cleaned
        )
        SELECT
            customer_ID,
            CAST (COUNT(*) AS SMALLINT) AS statement_count,
            {categorical_aggregate_expression},
            {customer_history_aggregate_expression}
        FROM sequenced
        GROUP BY customer_ID 
    """

def build_final_join_query(chunk_nums):
    select_queries = [f"n{i}.* EXCLUDE (customer_ID, statement_count)" for i in range(1, chunk_nums)]
    select_queries.insert(1, "categorical.* EXCLUDE (customer_ID, statement_count)")

    join_queries = [f"INNER JOIN read_parquet($numeric_path_{i}) AS n{i} USING (customer_ID)" for i in range(1, chunk_nums)]  

    return f"""
    SELECT 
        n0.*,
        {",\n".join(select_queries)}
        FROM read_parquet($numeric_path_0) AS n0
        {'\n'.join(join_queries)}
        INNER JOIN read_parquet($categorical_path) AS categorical USING (customer_ID)
    """

def _copy_query_to_parquet(connection, query, parameters):
    output_path = Path(parameters['output_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(f"COPY({query}) TO $output_path (FORMAT PARQUET, COMPRESSION ZSTD)", parameters)

def chunk_helper(columns:list, chunk_size:int) -> list[list]:
    copy_column = columns.copy()
    chunck_columns = []

    while copy_column:
        chunck_columns.append(copy_column[:chunk_size])
        del copy_column[:chunk_size]
    return chunck_columns

def _validate_intermediate_features(connection, numerical_output_paths, categorical_output_path, expected_customer_count):

    numerical_summary = set()
    for path in numerical_output_paths:
        numerical_summary.add(
            connection.execute(
                """
                    SELECT
                        COUNT(*),
                        COUNT(DISTINCT customer_ID)
                    FROM read_parquet(?)
                """, [path]  
            ).fetchone()
        )

    if len(numerical_summary) != 1:
        raise ValueError("Intermediate numeric feature files contain duplicate or missing customers")
    
    ((numeric_row_count, numeric_customer_count), ) = numerical_summary

    count_categorical_customer = connection.execute(f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT customer_ID)
        FROM read_parquet(?)
    """ , [categorical_output_path]      
    ).fetchone()

    if(numeric_row_count == numeric_customer_count == count_categorical_customer[0] == count_categorical_customer[1] == expected_customer_count):
        return
    else:
        raise ValueError("Intermediate feature files contain duplicate or missing customers")    

def argument_parser():
    parser = argparse.ArgumentParser(description="Prepare one-row-per-customer AMEX features from monthly statements.")
    parser.add_argument("--input", type=str, required=True, help="Path to the raw monthly statement parquet file")
    parser.add_argument("--labels", type=str, required=True, help="Path to the training labels CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path where the final prepared feature parquet will be written")
    parser.add_argument("--working-directory", type=str, required=False, default="artifacts/intermediate", help="Directory for intermediate numeric and categorical parquet files")
    parser.add_argument("--temp-directory", type=str, required=False, default="artifacts/duckdb_tmp", help="Directory DuckDB can use for temporary spill files")
    parser.add_argument("--threads", type=int, required=False, default=None, help="Number of DuckDB worker threads to use")
    return parser

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
    numerical_columns, categorical_columns = features.classify_feature_columns(schema)

    chunks_num = ceil(len(numerical_columns) / constants.DEFAULT_CHUNK_SIZE)
    numerical_chunk_columns = chunk_helper(numerical_columns,constants.DEFAULT_CHUNK_SIZE)

    clean_expression = ",\n".join(features.build_clean_source_expressions(schema, sentinel_counts))
    categorical_history_query = _build_categorical_history_query(categorical_columns, clean_expression)

    connection = connect(temp_directory= temp_directory, threads= threads, memory_limit="12GB")
    try:
        parameters = {"train_path": str(train_path)}

        numerical_output_paths = [
            f"{working_directory_str}/numeric_features_{i}.parquet" for i in range(chunks_num)
        ]
        categorical_output_path = f"{working_directory_str}/categorical_history_features.parquet"
        
        for i in range(chunks_num):
            _copy_query_to_parquet(
                connection,
                _build_numerical_query(numerical_chunk_columns[i], clean_expression),
                {**parameters, "output_path": numerical_output_paths[i]}
            )
        _copy_query_to_parquet(
            connection, 
            categorical_history_query, 
            {**parameters, "output_path": categorical_output_path}
        )

        _validate_intermediate_features(connection, numerical_output_paths, categorical_output_path, audit_result["customer_count"])

        parameters = {
            **{f"numeric_path_{i}": numerical_output_paths[i] for i in range(chunks_num)},
            "categorical_path": categorical_output_path,
            "output_path": final_output_path_str,
        }

        _copy_query_to_parquet(connection, build_final_join_query(chunks_num), parameters)

    finally:
        connection.close()

    return final_output_path_str


def main():
    parser = argument_parser()
    args = parser.parse_args()

    output_path = prepare_features(
        train_path=args.input,
        labels_path=args.labels,
        final_output_path_str=args.output,
        working_directory_str=args.working_directory,
        temp_directory=args.temp_directory,
        threads=args.threads,
    )

    print(f"Prepared features written to: {output_path}")

if __name__ == "__main__":
    main()