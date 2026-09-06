"""Stage 2: Data Validation - check schema, dtypes, missing values, and data drift."""
import pandas as pd
from scipy.stats import ks_2samp

from src.fraud_detection.entity.config_entity import DataValidationConfig
from src.fraud_detection.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.fraud_detection.exception import FraudException
from src.fraud_detection.logger import get_logger
from src.fraud_detection.utils.main_utils import read_yaml, write_json

logger = get_logger(__name__)


class DataValidation:
    def __init__(self, config: DataValidationConfig, ingestion_artifact: DataIngestionArtifact):
        self.config = config
        self.ingestion_artifact = ingestion_artifact
        self.schema = read_yaml(config.schema_path)

    def validate_columns(self, df: pd.DataFrame) -> bool:
        expected = set(self.schema["columns"].keys())
        actual = set(df.columns)
        missing = expected - actual
        if missing:
            logger.error("Missing columns: %s", missing)
            return False
        return True

    def check_missing(self, df: pd.DataFrame) -> bool:
        nulls = df.isnull().sum().sum()
        if nulls > 0:
            logger.error("Found %d missing values", nulls)
            return False
        return True

    def detect_drift(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """KS test per feature column. Drift if p-value < 0.05."""
        report = {}
        drift_found = False
        for col in self.schema["feature_columns"]:
            stat, p_value = ks_2samp(train_df[col], test_df[col])
            drifted = bool(p_value < 0.05)
            if drifted:
                drift_found = True
            report[col] = {"p_value": float(p_value), "drift": drifted}
        report["_summary"] = {"any_drift": drift_found}
        return report

    def initiate(self) -> DataValidationArtifact:
        try:
            train_df = pd.read_csv(self.ingestion_artifact.train_path)
            test_df = pd.read_csv(self.ingestion_artifact.test_path)

            checks = {
                "columns_train": self.validate_columns(train_df),
                "columns_test": self.validate_columns(test_df),
                "missing_train": self.check_missing(train_df),
                "missing_test": self.check_missing(test_df),
            }
            status = all(checks.values())

            drift_report = self.detect_drift(train_df, test_df)
            write_json(self.config.drift_report_path, drift_report)
            write_json(self.config.status_path, {"validation_status": status, "checks": checks})

            message = "Validation passed" if status else f"Validation failed: {checks}"
            logger.info(message)
            # Drift is reported but does not block the pipeline (train/test come from
            # the same source; drift report is informational for monitoring).
            return DataValidationArtifact(
                validation_status=status,
                drift_report_path=self.config.drift_report_path,
                message=message,
            )
        except Exception as e:
            raise FraudException(e)
