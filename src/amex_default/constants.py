import os
from typing import Final

CUSTOMER_ID: Final[str] = "customer_ID"
DATE_COLUMN: Final[str] = "S_2"
TARGET: Final[str] = "target"
FEATURE_PREFIXES: Final[tuple[str, ...]] = (
    "P_",
    "B_",
    "D_",
    "R_",
    "S_"
)
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68"
)
DEFAULT_SEED: Final[int] = 42
DEFAULT_MEMORY_LIMIT: Final[str] = "8GB"
DEFAULT_THREADS: Final[int] = max(1, (os.cpu_count() or 1) - 2)

