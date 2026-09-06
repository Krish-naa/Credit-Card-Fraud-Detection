"""Stage 1: Data Ingestion - read raw CSV and produce stratified train/test split."""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.fraud_detection.entity.config_entity import DataIngestionConfig
from src.fraud_detection.entity.artifact_entity import DataIngestionArtifact
from src.fraud_detection.exception import FraudException
from src.fraud_detection.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def read_data(self) -> pd.DataFrame:
        if not os.path.exists(self.config.raw_path):
            raise FraudException(
                FileNotFoundError(
                    f"Dataset not found at {self.config.raw_path}. "
                    "Please place 'creditcard.csv' in the data/ folder."
                )
            )
        logger.info("Reading raw dataset from %s", self.config.raw_path)
        df = pd.read_csv(self.config.raw_path)
        # Class column can be read as string when quoted; coerce to int
        df["Class"] = df["Class"].astype(int)
        logger.info("Loaded dataset with shape %s", df.shape)
        return df

    def split_data(self, df: pd.DataFrame):
        stratify = df["Class"] if self.config.stratify else None
        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify,
        )
        logger.info("Train shape: %s, Test shape: %s", train_df.shape, test_df.shape)
        return train_df, test_df

    def initiate(self) -> DataIngestionArtifact:
        try:
            df = self.read_data()
            train_df, test_df = self.split_data(df)
            os.makedirs(os.path.dirname(self.config.train_path), exist_ok=True)
            train_df.to_csv(self.config.train_path, index=False)
            test_df.to_csv(self.config.test_path, index=False)
            logger.info("Saved train/test to artifacts/ingestion/")
            return DataIngestionArtifact(
                train_path=self.config.train_path,
                test_path=self.config.test_path,
            )
        except Exception as e:
            raise FraudException(e)
