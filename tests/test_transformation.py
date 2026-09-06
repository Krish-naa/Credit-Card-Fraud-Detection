"""Tests for DataTransformation: preprocessor behavior and imbalance handling."""
import numpy as np

from src.fraud_detection.components.data_transformation import DataTransformation
from src.fraud_detection.entity.config_entity import DataTransformationConfig


def _make_transformer():
    return DataTransformation(
        DataTransformationConfig(imbalance_method="smote"),
        validation_artifact=None,
        ingestion_artifact=None,
    )


def test_preprocessor_output_shape(synthetic_df, feature_columns):
    t = _make_transformer()
    pre = t.build_preprocessor()
    x = synthetic_df[feature_columns]
    out = pre.fit_transform(x)
    # 30 feature columns (Time, V1-V28, Amount)
    assert out.shape == (len(synthetic_df), 30)


def test_scaled_columns_are_standardized(synthetic_df, feature_columns):
    t = _make_transformer()
    pre = t.build_preprocessor()
    out = pre.fit_transform(synthetic_df[feature_columns])
    # First two output columns are scaled Time and Amount -> mean ~0, std ~1
    assert abs(out[:, 0].mean()) < 1e-6
    assert abs(out[:, 0].std() - 1.0) < 1e-6


def test_smote_balances_training_only(synthetic_df, feature_columns):
    t = _make_transformer()
    pre = t.build_preprocessor()
    x = pre.fit_transform(synthetic_df[feature_columns])
    y = synthetic_df["Class"].values

    x_res, y_res = t.apply_imbalance(x, y)
    # SMOTE should balance the classes
    counts = np.bincount(y_res)
    assert counts[0] == counts[1]
    # and produce more rows than the original imbalanced set
    assert len(y_res) >= len(y)
