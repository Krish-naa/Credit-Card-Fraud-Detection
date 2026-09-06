"""Tests for PredictionPipeline using a small model trained on synthetic data.

These tests write temporary artifacts into final_models/ so the pipeline can
load them, then verify single and batch prediction output structure.
"""
import os

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.fraud_detection.components.data_transformation import DataTransformation
from src.fraud_detection.entity.config_entity import DataTransformationConfig
from src.fraud_detection.pipeline.prediction_pipeline import PredictionPipeline
from src.fraud_detection.utils.main_utils import save_object
from src.fraud_detection import constants as C


@pytest.fixture
def trained_artifacts(synthetic_df, feature_columns, tmp_path, monkeypatch):
    """Fit a preprocessor + tiny model on synthetic data and save to final_models/."""
    t = DataTransformation(DataTransformationConfig(imbalance_method="none"), None, None)
    pre = t.build_preprocessor()
    x = pre.fit_transform(synthetic_df[feature_columns])
    y = synthetic_df["Class"].values

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x, y)

    # Redirect artifact paths to a temp location for the test
    model_file = str(tmp_path / "model.pkl")
    pre_file = str(tmp_path / "preprocessor.pkl")
    save_object(model_file, {"model": model, "threshold": 0.5, "name": "logistic_regression"})
    save_object(pre_file, pre)

    monkeypatch.setattr(C, "FINAL_MODEL_FILE", model_file)
    monkeypatch.setattr(C, "FINAL_PREPROCESSOR_FILE", pre_file)
    return feature_columns


def test_predict_single_structure(trained_artifacts, synthetic_df):
    pipeline = PredictionPipeline()
    sample = synthetic_df.iloc[0][trained_artifacts].to_dict()
    result = pipeline.predict_single(sample)
    assert set(result.keys()) == {"prediction", "label", "probability"}
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("Fraud", "Legitimate")
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_batch_adds_columns(trained_artifacts, synthetic_df):
    pipeline = PredictionPipeline()
    df = synthetic_df[trained_artifacts].head(10)
    out = pipeline.predict_batch(df)
    assert "predicted_class" in out.columns
    assert "fraud_probability" in out.columns
    assert "predicted_label" in out.columns
    assert len(out) == 10


def test_missing_column_raises(trained_artifacts, synthetic_df):
    pipeline = PredictionPipeline()
    bad = synthetic_df[trained_artifacts].drop(columns=["V1"])
    with pytest.raises(Exception):
        pipeline.predict_batch(bad)
