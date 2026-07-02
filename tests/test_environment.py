def test_project_package_is_importable():
    import amex_default

    assert amex_default.__version__ == "0.1.0"


def test_core_dependencies_are_importable():
    import duckdb
    import lightgbm
    import numpy
    import optuna
    import pandas
    import pyarrow
    import sklearn