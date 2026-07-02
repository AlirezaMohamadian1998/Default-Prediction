from amex_default import database
def test_database_connection():
    connection = database.connect(threads=2)

    test = connection.execute(
        """
        SELECT 1
        """
    ).fetchone()[0] # type: ignore

    assert test == 1, f"Expected 1, but got {test}"
    result = connection.execute("SELECT current_setting('threads')").fetchone()[0] # type: ignore
    assert result == 2, f"Expected 2, but got {result}"
    connection.close()