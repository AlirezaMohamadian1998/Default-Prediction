from pathlib import Path

import duckdb as db

from amex_default import constants


def connect(
    memory_limit: str = constants.DEFAULT_MEMORY_LIMIT,
    threads: int | None = None,
    temp_directory: str | Path | None = None,
):
    if threads is not None and threads < 1:
        raise ValueError("threads must be at least 1")

    config = {
        "memory_limit": memory_limit,
        "threads": threads if threads is not None else constants.DEFAULT_THREADS,
        "preserve_insertion_order": False,
    }

    if temp_directory is not None:
        temp_path = Path(temp_directory)
        temp_path.mkdir(parents=True, exist_ok=True)
        config["temp_directory"] = str(temp_path.resolve())

    connection = db.connect(
        database=":memory:",
        config=config,
    )
    return connection
