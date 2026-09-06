"""Configuration dataclasses for each pipeline stage."""
from dataclasses import dataclass, field
from typing import List

from src.fraud_detection import constants as C


@dataclass
class DataIngestionConfig:
    raw_path: str = C.RAW_DATA_FILE
    train_path: str = C.TRAIN_FILE
    test_path: str = C.TEST_FILE
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass
class DataValidationConfig:
    schema_path: str = C.SCHEMA_FILE
    status_path: str = C.VALIDATION_STATUS_FILE
    drift_report_path: str = C.DRIFT_REPORT_FILE


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = C.FINAL_PREPROCESSOR_FILE
    transformed_train_path: str = C.TRANSFORMED_TRAIN_FILE
    transformed_test_path: str = C.TRANSFORMED_TEST_FILE
    imbalance_method: str = "smote"
    smote_random_state: int = 42


@dataclass
class ModelTrainerConfig:
    model_path: str = C.FINAL_MODEL_FILE
    models: List[str] = field(default_factory=lambda: ["logistic_regression", "random_forest", "xgboost"])
    cv_folds: int = 3
    scoring: str = "average_precision"
    threshold_search: bool = True
    random_state: int = 42


@dataclass
class ModelEvaluationConfig:
    metrics_path: str = C.METRICS_FILE
    plots_dir: str = C.PLOTS_DIR
    primary_metric: str = "pr_auc"
    min_pr_auc: float = 0.70
