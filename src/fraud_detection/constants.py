"""Paths and constant values used across the project."""
import os

# Root of the repository (three levels up from this file:
# src/fraud_detection/constants.py -> project root)
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Config
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
SCHEMA_FILE = os.path.join(CONFIG_DIR, "schema.yaml")
PARAMS_FILE = os.path.join(CONFIG_DIR, "params.yaml")

# Data
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_FILE = os.path.join(DATA_DIR, "creditcard.csv")

# Artifacts (generated per run)
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
INGESTION_DIR = os.path.join(ARTIFACTS_DIR, "ingestion")
VALIDATION_DIR = os.path.join(ARTIFACTS_DIR, "validation")
TRANSFORMATION_DIR = os.path.join(ARTIFACTS_DIR, "transformation")
EVALUATION_DIR = os.path.join(ARTIFACTS_DIR, "evaluation")

TRAIN_FILE = os.path.join(INGESTION_DIR, "train.csv")
TEST_FILE = os.path.join(INGESTION_DIR, "test.csv")

VALIDATION_STATUS_FILE = os.path.join(VALIDATION_DIR, "status.json")
DRIFT_REPORT_FILE = os.path.join(VALIDATION_DIR, "drift_report.json")

TRANSFORMED_TRAIN_FILE = os.path.join(TRANSFORMATION_DIR, "train.npy")
TRANSFORMED_TEST_FILE = os.path.join(TRANSFORMATION_DIR, "test.npy")

METRICS_FILE = os.path.join(EVALUATION_DIR, "metrics.json")
PLOTS_DIR = os.path.join(EVALUATION_DIR, "plots")

# Final models (committed for deployment)
FINAL_MODELS_DIR = os.path.join(ROOT_DIR, "final_models")
FINAL_MODEL_FILE = os.path.join(FINAL_MODELS_DIR, "model.pkl")
FINAL_PREPROCESSOR_FILE = os.path.join(FINAL_MODELS_DIR, "preprocessor.pkl")
