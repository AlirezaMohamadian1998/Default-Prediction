from amex_default.database import connect
from amex_default.constants import CUSTOMER_ID, TARGET, DATE_COLUMN

def audit_dataset(train_path: str, labels_path: str|None = None):
    label_counts = None
    positive_rate = None
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

        if not {CUSTOMER_ID, DATE_COLUMN}.issubset(set(schema["column_name"])):
            raise ValueError(f"{CUSTOMER_ID} or {DATE_COLUMN} is missing from the dataset.")

        statement_count, customer_count, null_customer_count, null_date_count, duplicate_statement_count = connection.execute(
            f"""
            WITH statement_rows AS (
                SELECT
                    {CUSTOMER_ID},
                    {DATE_COLUMN}
                FROM read_parquet(?)
            ),
            duplicate_pairs AS (
                SELECT *
                FROM statement_rows
                GROUP BY *
                HAVING COUNT(*) > 1
            )
            SELECT
                COUNT(*) AS statement_count,
                COUNT(DISTINCT {CUSTOMER_ID}) AS customer_count,
                COUNT(*) FILTER (WHERE {CUSTOMER_ID} IS NULL) AS null_customer_count,
                COUNT(*) FILTER (WHERE {DATE_COLUMN} IS NULL) AS null_date_count,
                (SELECT COUNT(*) FROM duplicate_pairs) AS duplicate_statement_count
            FROM statement_rows
            """,
            [train_path]
        ).fetchone()  # type: ignore

        if null_customer_count > 0:
            raise ValueError(f"Some {CUSTOMER_ID}s are Null which is unacceptable.")
        if null_date_count > 0:
            raise ValueError(f"Some {DATE_COLUMN}s are Null which is unacceptable.")
        if duplicate_statement_count > 0:
            raise ValueError(f"{duplicate_statement_count} duplicate customer-date combinations")
        
        sentinel_counts = audit_sentinel_cols(train_path, connection, schema)


        min_statements, max_statements, avg_statements = connection.execute(
            f"""
            WITH customer_stats AS (
                SELECT {CUSTOMER_ID},
                COUNT(*) AS history_stats
                FROM read_parquet(?)
                GROUP BY {CUSTOMER_ID}
            )
            SELECT MIN(history_stats) AS min_statements,
            MAX(history_stats) AS max_statements,
            AVG(history_stats) AS avg_statements
            FROM customer_stats
            """,
            [train_path]
        ).fetchone()  # type: ignore

        if labels_path is not None:
            distinct_label_customers, null_customer_ids = connection.execute(
                f"""
                SELECT
                    COUNT(DISTINCT {CUSTOMER_ID}) AS distinct_label_customers,
                    COUNT(*) FILTER (WHERE {CUSTOMER_ID} IS NULL) AS null_customer_ids
                FROM read_csv_auto(?, header=true)
                """,
                [labels_path]
            ).fetchone() #type: ignore

            label_counts = connection.execute(
                f"""
                SELECT {TARGET},
                COUNT(*) AS label_count
                FROM read_csv_auto(?, header=true)
                GROUP BY {TARGET}
                ORDER BY {TARGET}
                """,
                [labels_path]
            ).fetchall()  # type: ignore

            label_count_by_target = dict(label_counts)
            actual_targets = set(label_count_by_target)

            if actual_targets != {0, 1}:
                raise ValueError(f"Expected binary targets {{0, 1}}, found {actual_targets}")

            total_labels = sum(label_count_by_target.values())

            if null_customer_ids != 0:
                raise ValueError("labels contain missing customer IDs")
            if total_labels != distinct_label_customers:
                raise ValueError("Some customers have duplicate labels.")
            if distinct_label_customers != customer_count:
                raise ValueError("Number of labelled customers does not match the statement customers")

            missing_label_count, extra_label_count = connection.execute(
                F"""
                WITH statement_customers AS (
                    SELECT DISTINCT {CUSTOMER_ID} FROM read_parquet(?)
                ),
                label_customers AS (
                    SELECT DISTINCT {CUSTOMER_ID} FROM read_csv_auto(?, header=true)
                ),
                missing_labels AS (
                    SELECT {CUSTOMER_ID} FROM statement_customers
                    EXCEPT
                    SELECT {CUSTOMER_ID} FROM label_customers
                ),
                extra_labels AS (
                    SELECT {CUSTOMER_ID} FROM label_customers
                    EXCEPT
                    SELECT {CUSTOMER_ID} FROM statement_customers
                )
                SELECT
                    (SELECT COUNT(*) FROM missing_labels) AS missing_label_count,
                    (SELECT COUNT(*) FROM extra_labels) AS extra_label_count
                """,
                [train_path, labels_path]
            ).fetchone() #type:ignore

            if missing_label_count != 0:
                raise ValueError(f"Statement customers are missing labels, {missing_label_count} labels are missing.")
            if extra_label_count != 0:
                raise ValueError(f"Labels contain {extra_label_count} unknown customers.")

            positive_rate = label_count_by_target[1] / total_labels
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