import duckdb as db
from amex_default import constants

def connect(memory_limit: str = constants.DEFAULT_MEMORY_LIMIT, threads: int | None = None):
    if threads is not None and threads < 1:
        raise ValueError("threads must be at least 1")
    
    connection = db.connect(
        database=":memory:",
        config={
            "memory_limit": memory_limit,
            "threads": threads if threads is not None and threads > 0 else constants.DEFAULT_THREADS,
            "preserve_insertion_order": False
            }
        )
    return connection