"""Loads saved artifacts and serves single and batch predictions.

The preprocessor is a ColumnTransformer that maps columns by name, so inputs
must be DataFrames whose columns match the schema feature_columns.
"""
import os

import pandas as pd

from src.fraud_detection.exception import FraudException
from src.fraud_detection.logger import get_logger
from src.fraud_detection.utils.main_utils import load_object, read_yaml
from src.fraud_detection import constants as C

logger = get_logger(__name__)


class PredictionPipeline:
    def __init__(self):
        self._loaded = False
        self.model = None
        self.threshold = 0.5
        self.model_name = None
        self.preprocessor = None
        self.feature_columns = read_yaml(C.SCHEMA_FILE)["feature_columns"]

    def _ensure_loaded(self):
        if self._loaded:
            return
        if not (os.path.exists(C.FINAL_MODEL_FILE) and os.path.exists(C.FINAL_PREPROCESSOR_FILE)):
            raise FraudException(
                FileNotFoundError(
                    "Model artifacts not found. Run training (python train.py) first."
                )
            )
        payload = load_object(C.FINAL_MODEL_FILE)
        self.model = payload["model"]
        self.threshold = payload["threshold"]
        self.model_name = payload.get("name", "model")
        self.preprocessor = load_object(C.FINAL_PREPROCESSOR_FILE)
        self._loaded = True
        logger.info("Loaded model (%s) and preprocessor", self.model_name)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise FraudException(Exception(f"schema mismatch: missing columns {missing}"))
        return df[self.feature_columns].astype(float)

    def predict_single(self, features: dict) -> dict:
        try:
            self._ensure_loaded()
            df = pd.DataFrame([features])
            df = self._prepare(df)
            x = self.preprocessor.transform(df)
            proba = float(self.model.predict_proba(x)[:, 1][0])
            prediction = int(proba >= self.threshold)
            return {
                "prediction": prediction,
                "label": "Fraud" if prediction == 1 else "Legitimate",
                "probability": round(proba, 6),
            }
        except FraudException:
            raise
        except Exception as e:
            raise FraudException(e)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            self._ensure_loaded()
            prepared = self._prepare(df)
            x = self.preprocessor.transform(prepared)
            proba = self.model.predict_proba(x)[:, 1]
            preds = (proba >= self.threshold).astype(int)
            result = df.copy()
            result["fraud_probability"] = proba.round(6)
            result["predicted_class"] = preds
            result["predicted_label"] = ["Fraud" if p == 1 else "Legitimate" for p in preds]
            return result
        except FraudException:
            raise
        except Exception as e:
            raise FraudException(e)
