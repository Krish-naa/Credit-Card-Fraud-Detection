"""Shared test fixtures: a small synthetic dataset matching the real schema."""
import numpy as np
import pandas as pd
import pytest

from src.fraud_detection.utils.main_utils import read_yaml
from src.fraud_detection import constants as C


@pytest.fixture(scope="session")
def feature_columns():
    return read_yaml(C.SCHEMA_FILE)["feature_columns"]


@pytest.fixture
def synthetic_df(feature_columns):
    """Small labeled dataset with a rare fraud class (imbalanced).

    Fraud rows are shifted in feature space so a model can separate them.
    """
    rng = np.random.default_rng(42)
    n = 400
    y = (rng.random(n) < 0.08).astype(int)

    data = {}
    for col in feature_columns:
        base = rng.normal(0, 1, n)
        # push fraud rows away from legit rows for learnability
        base = base + y * 3.0
        data[col] = base
    data["Time"] = rng.uniform(0, 100000, n)
    data["Amount"] = rng.uniform(0, 500, n) + y * 200
    data["Class"] = y

    return pd.DataFrame(data)
