"""Orchestrates all five stages in order and promotes artifacts to final_models/."""
from src.fraud_detection.components.data_ingestion import DataIngestion
from src.fraud_detection.components.data_validation import DataValidation
from src.fraud_detection.components.data_transformation import DataTransformation
from src.fraud_detection.components.model_trainer import ModelTrainer
from src.fraud_detection.components.model_evaluation import ModelEvaluation
from src.fraud_detection.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)
from src.fraud_detection.exception import FraudException
from src.fraud_detection.logger import get_logger
from src.fraud_detection.utils.main_utils import read_yaml
from src.fraud_detection import constants as C

logger = get_logger(__name__)


class TrainingPipeline:
    def __init__(self):
        self.params = read_yaml(C.PARAMS_FILE)

    def _ingestion_config(self) -> DataIngestionConfig:
        p = self.params["data_ingestion"]
        return DataIngestionConfig(
            test_size=p["test_size"],
            random_state=p["random_state"],
            stratify=p["stratify"],
        )

    def _transformation_config(self) -> DataTransformationConfig:
        p = self.params["data_transformation"]
        return DataTransformationConfig(
            imbalance_method=p["imbalance_method"],
            smote_random_state=p["smote_random_state"],
        )

    def _trainer_config(self) -> ModelTrainerConfig:
        p = self.params["model_trainer"]
        return ModelTrainerConfig(
            models=p["models"],
            cv_folds=p["cv_folds"],
            scoring=p["scoring"],
            threshold_search=p["threshold_search"],
            random_state=p["random_state"],
        )

    def _evaluation_config(self) -> ModelEvaluationConfig:
        p = self.params["evaluation"]
        return ModelEvaluationConfig(
            primary_metric=p["primary_metric"],
            min_pr_auc=p["min_pr_auc"],
        )

    def run(self) -> dict:
        try:
            logger.info("=== Starting training pipeline ===")

            ingestion_artifact = DataIngestion(self._ingestion_config()).initiate()

            validation_artifact = DataValidation(
                DataValidationConfig(), ingestion_artifact
            ).initiate()
            if not validation_artifact.validation_status:
                raise FraudException(Exception(f"Pipeline stopped: {validation_artifact.message}"))

            transformation_artifact = DataTransformation(
                self._transformation_config(), validation_artifact, ingestion_artifact
            ).initiate()

            trainer_artifact = ModelTrainer(
                self._trainer_config(), transformation_artifact
            ).initiate()

            evaluation_artifact = ModelEvaluation(
                self._evaluation_config(), trainer_artifact, transformation_artifact
            ).initiate()

            summary = {
                "best_model": trainer_artifact.best_model_name,
                "threshold": trainer_artifact.threshold,
                "test_pr_auc": evaluation_artifact.primary_metric_value,
                "model_accepted": evaluation_artifact.is_accepted,
                "metrics_file": evaluation_artifact.metrics_path,
                "model_file": trainer_artifact.model_path,
                "preprocessor_file": transformation_artifact.preprocessor_path,
            }
            logger.info("=== Training pipeline complete === %s", summary)
            return summary
        except Exception as e:
            raise FraudException(e)
