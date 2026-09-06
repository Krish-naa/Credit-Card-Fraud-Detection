"""Stage 5: Model Evaluation - compute imbalance-aware metrics and plots."""
import os

import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.fraud_detection.entity.config_entity import ModelEvaluationConfig
from src.fraud_detection.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact,
)
from src.fraud_detection.exception import FraudException
from src.fraud_detection.logger import get_logger
from src.fraud_detection.utils.main_utils import load_numpy, load_object, write_json

logger = get_logger(__name__)


class ModelEvaluation:
    def __init__(
        self,
        config: ModelEvaluationConfig,
        trainer_artifact: ModelTrainerArtifact,
        transformation_artifact: DataTransformationArtifact,
    ):
        self.config = config
        self.trainer_artifact = trainer_artifact
        self.transformation_artifact = transformation_artifact

    def _save_plots(self, y_true, proba, cm):
        os.makedirs(self.config.plots_dir, exist_ok=True)

        # PR curve
        precision, recall, _ = precision_recall_curve(y_true, proba)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
        plt.savefig(os.path.join(self.config.plots_dir, "pr_curve.png"), bbox_inches="tight")
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, proba)
        plt.figure()
        plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
        plt.savefig(os.path.join(self.config.plots_dir, "roc_curve.png"), bbox_inches="tight")
        plt.close()

        # Confusion matrix
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix"); plt.colorbar()
        plt.xticks([0, 1], ["Legit", "Fraud"]); plt.yticks([0, 1], ["Legit", "Fraud"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.savefig(os.path.join(self.config.plots_dir, "confusion_matrix.png"), bbox_inches="tight")
        plt.close()

    def initiate(self) -> ModelEvaluationArtifact:
        try:
            test_arr = load_numpy(self.transformation_artifact.transformed_test_path)
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1].astype(int)

            payload = load_object(self.trainer_artifact.model_path)
            model, threshold = payload["model"], payload["threshold"]

            proba = model.predict_proba(x_test)[:, 1]
            preds = (proba >= threshold).astype(int)

            pr_auc = float(average_precision_score(y_test, proba))
            roc_auc = float(roc_auc_score(y_test, proba))
            precision = float(precision_score(y_test, preds, zero_division=0))
            recall = float(recall_score(y_test, preds, zero_division=0))
            f1 = float(f1_score(y_test, preds, zero_division=0))
            cm = confusion_matrix(y_test, preds)

            metrics = {
                "model": payload["name"],
                "threshold": threshold,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm.tolist(),
                "test_fraud_count": int(y_test.sum()),
                "test_total": int(len(y_test)),
            }
            write_json(self.config.metrics_path, metrics)
            self._save_plots(y_test, proba, cm)

            logger.info(
                "Test metrics -> PR-AUC: %.4f | ROC-AUC: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
                pr_auc, roc_auc, precision, recall, f1,
            )

            is_accepted = pr_auc >= self.config.min_pr_auc
            logger.info("Model accepted: %s (min PR-AUC=%.2f)", is_accepted, self.config.min_pr_auc)

            return ModelEvaluationArtifact(
                metrics_path=self.config.metrics_path,
                primary_metric_value=pr_auc,
                is_accepted=is_accepted,
            )
        except Exception as e:
            raise FraudException(e)
