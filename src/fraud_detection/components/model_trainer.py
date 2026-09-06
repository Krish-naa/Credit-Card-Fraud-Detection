"""Stage 4: Model Trainer - train/compare models, tune threshold, select best.

Selection metric: PR-AUC (average_precision) via stratified cross-validation,
which is the correct choice for a highly imbalanced fraud problem.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.fraud_detection.entity.config_entity import ModelTrainerConfig
from src.fraud_detection.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.fraud_detection.exception import FraudException
from src.fraud_detection.logger import get_logger
from src.fraud_detection.utils.main_utils import save_object, load_numpy

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, transformation_artifact: DataTransformationArtifact):
        self.config = config
        self.transformation_artifact = transformation_artifact

    def get_models(self) -> dict:
        rs = self.config.random_state
        catalog = {
            "logistic_regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=rs
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=rs
            ),
            "xgboost": XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                eval_metric="aucpr", n_jobs=-1, random_state=rs,
            ),
        }
        return {name: catalog[name] for name in self.config.models if name in catalog}

    def cross_validate(self, model, x, y) -> float:
        skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        scores = cross_val_score(model, x, y, cv=skf, scoring=self.config.scoring, n_jobs=-1)
        return float(np.mean(scores))

    def tune_threshold(self, model, x_val, y_val) -> float:
        """Pick threshold maximizing F1 on the fraud (positive) class."""
        proba = model.predict_proba(x_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, proba)
        # thresholds has len-1 vs precision/recall; compute F1 per threshold
        f1s = []
        for t in thresholds:
            preds = (proba >= t).astype(int)
            f1s.append(f1_score(y_val, preds, zero_division=0))
        if not f1s:
            return 0.5
        best_idx = int(np.argmax(f1s))
        return float(thresholds[best_idx])

    def initiate(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy(self.transformation_artifact.transformed_train_path)
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1].astype(int)

            models = self.get_models()
            results = {}
            for name, model in models.items():
                score = self.cross_validate(model, x_train, y_train)
                results[name] = score
                logger.info("Model %s CV PR-AUC: %.4f", name, score)

            best_name = max(results, key=results.get)
            best_model = models[best_name]
            logger.info("Best model: %s (PR-AUC=%.4f)", best_name, results[best_name])

            # Fit best model on full (resampled) training set
            best_model.fit(x_train, y_train)

            # Tune threshold on the training set predictions
            threshold = 0.5
            if self.config.threshold_search:
                threshold = self.tune_threshold(best_model, x_train, y_train)
                logger.info("Tuned decision threshold: %.4f", threshold)

            # Package model + threshold together for inference
            payload = {"model": best_model, "threshold": threshold, "name": best_name}
            save_object(self.config.model_path, payload)
            logger.info("Saved model.pkl (%s) to %s", best_name, self.config.model_path)

            return ModelTrainerArtifact(
                model_path=self.config.model_path,
                best_model_name=best_name,
                threshold=threshold,
                train_pr_auc=results[best_name],
            )
        except Exception as e:
            raise FraudException(e)
