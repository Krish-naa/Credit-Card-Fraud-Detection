"""Stage 3: Data Transformation - scale Time/Amount and handle class imbalance.

Leakage guards (explicit):
  - The preprocessor (StandardScaler) is fitted on TRAIN data only.
  - SMOTE / undersampling is applied to the TRAIN set only, never to TEST.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.fraud_detection.entity.config_entity import DataTransformationConfig
from src.fraud_detection.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
    DataIngestionArtifact,
)
from src.fraud_detection.exception import FraudException
from src.fraud_detection.logger import get_logger
from src.fraud_detection.utils.main_utils import read_yaml, save_object, save_numpy
from src.fraud_detection import constants as C

logger = get_logger(__name__)


class DataTransformation:
    def __init__(
        self,
        config: DataTransformationConfig,
        validation_artifact: DataValidationArtifact,
        ingestion_artifact: DataIngestionArtifact,
    ):
        self.config = config
        self.validation_artifact = validation_artifact
        self.ingestion_artifact = ingestion_artifact
        self.schema = read_yaml(C.SCHEMA_FILE)

    def build_preprocessor(self) -> ColumnTransformer:
        """StandardScaler on Time/Amount; passthrough for V1-V28.

        The ColumnTransformer is arranged so its output column order equals
        feature_columns from the schema (Time, V1..V28, Amount).
        """
        numerical = self.schema["numerical_columns"]  # [Time, Amount]
        passthrough = [c for c in self.schema["feature_columns"] if c not in numerical]
        # remainder order: sklearn appends 'passthrough' columns after transformed ones,
        # so we track the resulting order explicitly for downstream use.
        self._output_order = ["Time", "Amount"] + passthrough
        return ColumnTransformer(
            transformers=[("scale", StandardScaler(), numerical)],
            remainder="passthrough",
        )

    def apply_imbalance(self, x_train: np.ndarray, y_train: np.ndarray):
        method = self.config.imbalance_method
        before = int(y_train.sum())
        if method == "smote":
            sampler = SMOTE(random_state=self.config.smote_random_state)
        elif method == "undersample":
            sampler = RandomUnderSampler(random_state=self.config.smote_random_state)
        else:
            logger.info("No imbalance handling applied")
            return x_train, y_train
        x_res, y_res = sampler.fit_resample(x_train, y_train)
        logger.info(
            "Imbalance (%s): fraud rows %d -> %d, total %d -> %d",
            method, before, int(y_res.sum()), len(y_train), len(y_res),
        )
        return x_res, y_res

    def initiate(self) -> DataTransformationArtifact:
        try:
            if not self.validation_artifact.validation_status:
                raise FraudException(Exception("Cannot transform: data validation failed"))

            train_df = pd.read_csv(self.ingestion_artifact.train_path)
            test_df = pd.read_csv(self.ingestion_artifact.test_path)
            target = self.schema["target_column"]
            features = self.schema["feature_columns"]

            x_train, y_train = train_df[features], train_df[target].values
            x_test, y_test = test_df[features], test_df[target].values

            preprocessor = self.build_preprocessor()
            # Fit on TRAIN only, then transform both.
            x_train_t = preprocessor.fit_transform(x_train)
            x_test_t = preprocessor.transform(x_test)

            # SMOTE / undersample on TRAIN only.
            x_train_res, y_train_res = self.apply_imbalance(x_train_t, y_train)

            train_arr = np.c_[x_train_res, y_train_res]
            test_arr = np.c_[x_test_t, y_test]

            save_object(self.config.preprocessor_path, preprocessor)
            save_numpy(self.config.transformed_train_path, train_arr)
            save_numpy(self.config.transformed_test_path, test_arr)
            logger.info("Saved preprocessor.pkl and transformed arrays")

            return DataTransformationArtifact(
                preprocessor_path=self.config.preprocessor_path,
                transformed_train_path=self.config.transformed_train_path,
                transformed_test_path=self.config.transformed_test_path,
            )
        except Exception as e:
            raise FraudException(e)
