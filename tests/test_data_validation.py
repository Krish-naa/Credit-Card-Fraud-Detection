"""Tests for the DataValidation component."""
import pandas as pd

from src.fraud_detection.components.data_validation import DataValidation
from src.fraud_detection.entity.config_entity import DataValidationConfig


def _make_validator():
    # ingestion_artifact is not used by the column/missing checks directly
    return DataValidation(DataValidationConfig(), ingestion_artifact=None)


def test_valid_columns_pass(synthetic_df):
    validator = _make_validator()
    assert validator.validate_columns(synthetic_df) is True


def test_missing_column_fails(synthetic_df):
    validator = _make_validator()
    bad = synthetic_df.drop(columns=["V1"])
    assert validator.validate_columns(bad) is False


def test_missing_values_detected(synthetic_df):
    validator = _make_validator()
    assert validator.check_missing(synthetic_df) is True
    bad = synthetic_df.copy()
    bad.loc[0, "Amount"] = None
    assert validator.check_missing(bad) is False


def test_drift_report_structure(synthetic_df):
    validator = _make_validator()
    report = validator.detect_drift(synthetic_df, synthetic_df)
    assert "_summary" in report
    # identical distributions -> no drift
    assert report["_summary"]["any_drift"] is False
