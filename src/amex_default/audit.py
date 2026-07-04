from amex_default.database import connect

def audit_dataset(train_path: str, labels_path: str):
    connection = connect()
    try:
        schema = connection.execute(
            """
            DESCRIBE
            SELECT * 
            FROM read_parquet(?)
            """,
            [train_path]
        ).fetchdf()

        sentinel_counts = audit_sentinel_cols(train_path, connection, schema)

        statement_count, customer_count = connection.execute(
            """
            SELECT COUNT(*) AS statement_count, COUNT(DISTINCT customer_ID) AS customer_count
            FROM read_parquet(?)
            """,
            [train_path]
        ).fetchone()  # type: ignore

        min_statements, max_statements, avg_statements = connection.execute(
            """
            WITH customer_stats AS (
                SELECT customer_ID,
                COUNT(*) AS history_stats
                FROM read_parquet(?)
                GROUP BY customer_ID
            )
            SELECT MIN(history_stats) AS min_statements,
            MAX(history_stats) AS max_statements,
            AVG(history_stats) AS avg_statements
            FROM customer_stats
            """,
            [train_path]
        ).fetchone()  # type: ignore

        label_counts = connection.execute(
            """
            SELECT target, 
            COUNT(*) AS label_count
            FROM read_csv_auto(?, header=true)
            GROUP BY target
            ORDER BY target
            """,
            [labels_path]
        ).fetchall()  # type: ignore

        positive_rate = label_counts[1][1] / (label_counts[0][1] + label_counts[1][1])
    finally:
        connection.close()
    
    return {
        "schema": schema,
        "column_count": len(schema),
        "statement_count": statement_count,
        "customer_count": customer_count,
        "min_statements": min_statements,
        "max_statements": max_statements,
        "avg_statements": avg_statements,
        "label_counts": label_counts,
        "positive_rate": positive_rate,
        "sentinel_counts": sentinel_counts
    }

def audit_sentinel_cols(train_path: str, connection, schema):
    integer_columns = sorted(schema.loc[schema['column_type'].isin(['SMALLINT', 'TINYINT']), 'column_name'])
    expressions = []
    for col in integer_columns:
        expressions.append(f"SUM(CASE WHEN {col} = -1 THEN 1 ELSE 0 END) AS {col}_sentinel_count")

    joined_expression = ",\n".join(expressions)
    query = f"""
    SELECT
        {joined_expression}
    FROM read_parquet(?)
    """


    result = connection.execute(query, [train_path]).fetchone()
    sentinel_dict = {
    col: int(count) for col, count in zip(integer_columns, result) if count > 0
    }
    return sentinel_dict