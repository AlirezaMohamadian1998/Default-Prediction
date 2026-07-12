import duckdb
from amex_default.prepare import prepare_features

def test_prepare_features_end_to_end(tmp_path):
    train_path = tmp_path / "train.parquet"
    labels_path = tmp_path / "labels.csv"
    final_features_path = tmp_path / "final_features.parquet"
    
    work_dir = tmp_path / "work"
    duckdb_tmp_dir = tmp_path / "duckdb_tmp"

    connection = duckdb.connect()

    connection.execute("""
        CREATE TABLE mock_train (
            customer_ID VARCHAR,
            S_2 DATE,
            P_2 DOUBLE, -- Standard numeric/float type for AMEX data
            B_30 TINYINT
        );
    """)

    connection.execute("""
        CREATE TABLE mock_labels (
            customer_ID VARCHAR,
            target TINYINT
        );
    """)
    
    connection.execute("""
        INSERT INTO mock_train VALUES
        ('customer a', '2017-01-01', 1, 1),
        ('customer a', '2017-02-01', 2, NULL),
        ('customer a', '2017-03-01', 3, 2),
        ('customer b', '2017-01-01', 4, 0);
    """)

    connection.execute("""
        INSERT INTO mock_labels VALUES
        ('customer a', 0),
        ('customer b', 1);
    """)
    
    connection.execute("""
        COPY mock_train TO ? (FORMAT 'PARQUET');
    """, [str(train_path)])
    
    connection.execute("""
        COPY mock_labels TO ? (FORMAT 'CSV', HEADER);
    """, [str(labels_path)])

    connection.close()

    returned_path = prepare_features(
        train_path=str(train_path),
        labels_path=str(labels_path),
        final_output_path_str=str(final_features_path),
        working_directory_str=str(work_dir),
        temp_directory=str(duckdb_tmp_dir),
        threads=1
    )

    assert returned_path == str(final_features_path)
    assert final_features_path.is_file()

    connection = duckdb.connect()
    result_count = connection.execute(
        """
        SELECT
            COUNT(*),
            COUNT(DISTINCT customer_ID)
        FROM read_parquet(?)
        """, [str(final_features_path)]
    ).fetchone()

    assert result_count[0] == result_count[1] == 2 # type: ignore

    result = connection.execute(
        """
        SELECT
            customer_ID,
            statement_count,
            P_2_monthly_slope,
            B_30_transition_count,
            history_span_days
        FROM read_parquet(?)
        ORDER BY customer_ID
        """, [str(final_features_path)]
    ).fetchall()

    connection.close()

    assert result[0] == ("customer a", 3, 1.0, 1, 59)
    assert result[1] == ("customer b", 1, None, 0, 0)

    prediction_features_path = tmp_path / "prediction_features.parquet"
    prediction_work_dir = tmp_path / "prediction_work"
    prediction_duckdb_tmp_dir = tmp_path / "prediction_duckdb_tmp"

    prepare_features(
        train_path=str(train_path),
        labels_path=None,
        final_output_path_str=str(prediction_features_path),
        working_directory_str=str(prediction_work_dir),
        temp_directory=str(prediction_duckdb_tmp_dir),
        threads=1,
    )

    assert prediction_features_path.is_file()

    connection = duckdb.connect()
    prediction_result_count = connection.execute(
        """
        SELECT
            COUNT(*),
            COUNT(DISTINCT customer_ID)
        FROM read_parquet(?)
        """,
        [str(prediction_features_path)],
    ).fetchone()
    connection.close()

    assert prediction_result_count[0] == prediction_result_count[1] == 2  # type: ignore
