"""Artifact dataclasses returned by each pipeline stage."""
from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_path: str
    test_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    drift_report_path: str
    message: str


@dataclass
class DataTransformationArtifact:
    preprocessor_path: str
    transformed_train_path: str
    transformed_test_path: str


@dataclass
class ModelTrainerArtifact:
    model_path: str
    best_model_name: str
    threshold: float
    train_pr_auc: float


@dataclass
class ModelEvaluationArtifact:
    metrics_path: str
    primary_metric_value: float
    is_accepted: bool
