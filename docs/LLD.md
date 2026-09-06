# Low-Level Design (LLD)
## Credit Card Fraud Detection — End-to-End Machine Learning System

**Document version:** 1.0
**Date:** September 6, 2026
**Status:** Draft — awaiting approval

This document specifies the implementation-level design: folder structure, each module's classes/functions with their inputs and outputs, data schema, configuration, API contracts, error handling, testing, and deployment files. It complements the HLD (which describes the overall architecture).

---

## 1. Project Folder Structure

```
MLProjectNew/
│
├── data/
│   └── creditcard.csv                # placed by user (NOT committed)
│
├── docs/
│   ├── HLD.md
│   └── LLD.md
│
├── config/
│   ├── schema.yaml                   # expected columns, dtypes, target
│   └── params.yaml                   # test split, seed, model params, threshold
│
├── artifacts/                        # generated per run (NOT committed)
│   ├── ingestion/{train.csv,test.csv}
│   ├── validation/{status.json,drift_report.json}
│   └── evaluation/{metrics.json,plots/*.png}
│
├── final_models/
│   ├── model.pkl                     # committed for deployment
│   └── preprocessor.pkl              # committed for deployment
│
├── src/
│   └── fraud_detection/
│       ├── __init__.py
│       ├── constants.py              # paths, filenames, constants
│       ├── config.py                 # loads yaml -> config objects
│       ├── logger.py                 # logging setup
│       ├── exception.py              # custom exception class
│       ├── entity/
│       │   ├── config_entity.py      # dataclasses for stage configs
│       │   └── artifact_entity.py    # dataclasses for stage outputs
│       ├── components/
│       │   ├── data_ingestion.py
│       │   ├── data_validation.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_evaluation.py
│       ├── pipeline/
│       │   ├── training_pipeline.py  # runs all stages in order
│       │   └── prediction_pipeline.py# load artifacts, predict
│       └── utils/
│           └── main_utils.py         # read/write yaml, save/load objects
│
├── app/
│   ├── main.py                       # FastAPI app, routes
│   ├── schemas.py                    # Pydantic request/response models
│   ├── static/{style.css,script.js}
│   └── templates/{index.html,result.html}
│
├── tests/
│   ├── test_data_validation.py
│   ├── test_transformation.py
│   ├── test_prediction_pipeline.py
│   └── test_api.py
│
├── requirements.txt
├── Procfile                          # Render/gunicorn start command
├── render.yaml                       # Render service config
├── .gitignore
├── README.md
└── train.py                          # entry point to run training pipeline
```

---

## 2. Configuration Files

### 2.1 `config/schema.yaml`
```yaml
columns:
  Time: float64
  V1: float64
  # ... V2 through V28 ...
  V28: float64
  Amount: float64
  Class: int64
target_column: Class
numerical_columns: [Time, Amount]   # columns needing scaling (V1-V28 already PCA-scaled)
```

### 2.2 `config/params.yaml`
```yaml
data_ingestion:
  test_size: 0.2
  random_state: 42
  stratify: true

data_transformation:
  scaler: standard          # StandardScaler for Time and Amount
  imbalance_method: smote   # smote | undersample | none
  smote_random_state: 42

model_trainer:
  models: [logistic_regression, random_forest, xgboost]
  cv_folds: 5
  scoring: average_precision   # PR-AUC based selection
  threshold_search: true       # tune decision threshold for best F1 on fraud

evaluation:
  primary_metric: pr_auc
  report_metrics: [pr_auc, roc_auc, precision, recall, f1, confusion_matrix]
```

---

## 3. Data Schema (Input Contract)

| Column | Type | Notes |
|--------|------|-------|
| Time | float | seconds since first transaction |
| V1–V28 | float | PCA-anonymized features |
| Amount | float | transaction amount |
| Class | int | target: 0 legit, 1 fraud (absent at prediction time) |

Validation rejects data with missing columns, wrong dtypes, or unexpected nulls.

---

## 4. Module-Level Design

### 4.1 `logger.py`
- Configures a rotating file + console logger.
- Function: `get_logger(name) -> logging.Logger`.

### 4.2 `exception.py`
- Class `FraudException(Exception)` — captures error message + file + line number for clear debugging.

### 4.3 `utils/main_utils.py`
- `read_yaml(path) -> dict`
- `write_json(path, obj)`
- `save_object(path, obj)` / `load_object(path)` — joblib wrappers.
- `save_numpy(path, arr)` / `load_numpy(path)`.

### 4.4 `entity/config_entity.py` (dataclasses)
- `DataIngestionConfig(raw_path, train_path, test_path, test_size, random_state, stratify)`
- `DataValidationConfig(schema_path, status_path, drift_report_path)`
- `DataTransformationConfig(preprocessor_path, transformed_train_path, transformed_test_path, imbalance_method)`
- `ModelTrainerConfig(model_path, models, cv_folds, scoring, threshold_search)`
- `ModelEvaluationConfig(metrics_path, plots_dir, primary_metric)`

### 4.5 `entity/artifact_entity.py` (dataclasses)
- `DataIngestionArtifact(train_path, test_path)`
- `DataValidationArtifact(validation_status: bool, drift_report_path)`
- `DataTransformationArtifact(preprocessor_path, transformed_train_path, transformed_test_path)`
- `ModelTrainerArtifact(model_path, train_metric, best_model_name, threshold)`
- `ModelEvaluationArtifact(metrics_path, primary_metric_value, is_accepted: bool)`

---

## 5. Component Design (Detailed)

### 5.1 Data Ingestion — `components/data_ingestion.py`
**Class:** `DataIngestion(config: DataIngestionConfig)`
- `read_data() -> pd.DataFrame` — reads `data/creditcard.csv`; raises `FraudException` with a clear message if the file is missing.
- `split_data(df) -> (train_df, test_df)` — `train_test_split` with `stratify=Class`, `test_size`, `random_state`.
- `initiate() -> DataIngestionArtifact` — orchestrates read + split + save to `artifacts/ingestion/`.

### 5.2 Data Validation — `components/data_validation.py`
**Class:** `DataValidation(config, ingestion_artifact)`
- `validate_columns(df) -> bool` — checks all expected columns present.
- `validate_dtypes(df) -> bool` — checks dtypes against schema.
- `check_missing(df) -> bool` — flags unexpected nulls.
- `detect_drift(train_df, test_df) -> dict` — per-column distribution check (Kolmogorov–Smirnov test); writes `drift_report.json`.
- `initiate() -> DataValidationArtifact` — combines checks, writes `status.json`; `validation_status=False` stops the pipeline.

### 5.3 Data Transformation — `components/data_transformation.py`
**Class:** `DataTransformation(config, validation_artifact)`
- `build_preprocessor() -> ColumnTransformer` — `StandardScaler` on `Time` and `Amount`; passthrough for `V1–V28`.
- `apply_imbalance(X_train, y_train) -> (X_res, y_res)` — SMOTE (from imbalanced-learn) **on training data only**.
- `initiate() -> DataTransformationArtifact`:
  1. Split into X/y.
  2. Fit preprocessor on **train X only**, transform train and test.
  3. Apply SMOTE to transformed **train** set only.
  4. Save `preprocessor.pkl` and transformed arrays.

**Leakage guards (explicit):** scaler fitted on train only; SMOTE never applied to test.

### 5.4 Model Trainer — `components/model_trainer.py`
**Class:** `ModelTrainer(config, transformation_artifact)`
- `get_models() -> dict` — LogisticRegression (`class_weight='balanced'`), RandomForestClassifier, XGBClassifier (`scale_pos_weight`).
- `cross_validate(model, X, y) -> float` — Stratified K-Fold, scoring = `average_precision` (PR-AUC).
- `tune_threshold(model, X_val, y_val) -> float` — sweep thresholds, pick best F1 on fraud class.
- `select_best(results) -> (name, model)` — highest CV PR-AUC.
- `initiate() -> ModelTrainerArtifact` — trains all, selects best, tunes threshold, saves `model.pkl` (model + threshold packaged together).

### 5.5 Model Evaluation — `components/model_evaluation.py`
**Class:** `ModelEvaluation(config, trainer_artifact, transformation_artifact)`
- `evaluate(model, X_test, y_test, threshold) -> dict` — computes PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix.
- `save_plots()` — PR curve, ROC curve, confusion matrix heatmap to `artifacts/evaluation/plots/`.
- `initiate() -> ModelEvaluationArtifact` — writes `metrics.json`; sets `is_accepted` if primary metric meets minimum threshold in params.

---

## 6. Pipelines

### 6.1 Training Pipeline — `pipeline/training_pipeline.py`
```
run():
  a = DataIngestion(cfg).initiate()
  b = DataValidation(cfg, a).initiate()
  if not b.validation_status: stop with clear error
  c = DataTransformation(cfg, b).initiate()
  d = ModelTrainer(cfg, c).initiate()
  e = ModelEvaluation(cfg, d, c).initiate()
  copy best model.pkl + preprocessor.pkl -> final_models/
  return summary
```
Triggered by `train.py`.

### 6.2 Prediction Pipeline — `pipeline/prediction_pipeline.py`
**Class:** `PredictionPipeline`
- `__init__` — loads `final_models/model.pkl` and `final_models/preprocessor.pkl` once.
- `predict_single(features: dict) -> dict` — validate → DataFrame → preprocess → predict proba → apply threshold → `{prediction, probability}`.
- `predict_batch(df: pd.DataFrame) -> pd.DataFrame` — same, vectorized; appends `predicted_class` and `fraud_probability`.

---

## 7. API Design — `app/main.py`

### 7.1 Endpoints
| Method | Path | Purpose | Request | Response |
|--------|------|---------|---------|----------|
| GET | `/` | Web UI home (form) | – | HTML |
| GET | `/health` | Health check | – | `{"status":"ok"}` |
| POST | `/predict` | Single prediction | JSON (Pydantic) | `{prediction, probability, label}` |
| POST | `/predict/csv` | Batch prediction | CSV file upload | HTML table + downloadable CSV |
| GET | `/docs` | Swagger (auto) | – | interactive docs |

### 7.2 Pydantic Models — `app/schemas.py`
```python
class TransactionInput(BaseModel):
    Time: float
    V1: float
    ...            # V1 through V28
    V28: float
    Amount: float
    # validators: Amount >= 0

class PredictionResponse(BaseModel):
    prediction: int          # 0 or 1
    label: str               # "Fraud" | "Legitimate"
    probability: float       # fraud probability
```

### 7.3 Request/Response Example
Request `POST /predict`:
```json
{ "Time": 0.0, "V1": -1.35, "...": "...", "V28": -0.02, "Amount": 149.62 }
```
Response:
```json
{ "prediction": 0, "label": "Legitimate", "probability": 0.012 }
```

### 7.4 Error Handling
| Case | HTTP | Body |
|------|------|------|
| Missing/invalid field | 422 | validation error detail |
| Model artifact missing | 503 | `{"error":"model not available"}` |
| Bad CSV (wrong columns) | 400 | `{"error":"schema mismatch: <detail>"}` |
| Unexpected error | 500 | `{"error":"internal error"}` (details logged) |

---

## 8. Frontend Design — `app/templates/` + `app/static/`
- `index.html` — form with the input fields (grouped; V1–V28 collapsible), a "Predict" button, and a CSV upload control.
- `result.html` — shows prediction, label, probability; for batch, renders an HTML table.
- `style.css` — clean, minimal styling.
- `script.js` — sends `fetch` POST to `/predict`, renders JSON result without full page reload.

---

## 9. Testing Design — `tests/`
| Test file | Covers |
|-----------|--------|
| `test_data_validation.py` | column/dtype checks pass on valid data, fail on malformed data |
| `test_transformation.py` | preprocessor fitted on train only; output shapes; SMOTE not applied to test |
| `test_prediction_pipeline.py` | loads artifacts, returns valid prediction structure on sample input |
| `test_api.py` | `/health` returns ok; `/predict` returns 200 on valid, 422 on invalid |

Run with `pytest`. Tests use a small synthetic sample so they run without the full dataset.

---

## 10. Deployment Files

### 10.1 `requirements.txt` (pinned)
```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
imbalanced-learn==0.12.3
xgboost==2.1.1
joblib==1.4.2
fastapi==0.114.0
uvicorn==0.30.6
pydantic==2.9.0
python-multipart==0.0.9
jinja2==3.1.4
pyyaml==6.0.2
matplotlib==3.9.2
scipy==1.14.1
pytest==8.3.2
gunicorn==22.0.0
```
(Versions finalized at build time; kept pinned for reproducibility.)

### 10.2 `Procfile`
```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 10.3 `render.yaml`
```yaml
services:
  - type: web
    name: credit-card-fraud-detection
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
    plan: free
```

### 10.4 `.gitignore`
```
data/creditcard.csv
artifacts/
venv/
__pycache__/
*.pyc
.env
```
Note: `final_models/*.pkl` are intentionally **committed** so Render can serve without retraining.

---

## 11. Sequence: Single Prediction (Runtime)
```
Browser ──POST /predict (JSON)──> FastAPI
FastAPI ──validate──> Pydantic (TransactionInput)
FastAPI ──> PredictionPipeline.predict_single()
   load preprocessor.pkl (cached) ─> transform
   load model.pkl (cached) ─> predict_proba
   apply tuned threshold ─> prediction
FastAPI ──JSON {prediction,label,probability}──> Browser
```

---

## 12. Reproducibility and Correctness Controls
- Fixed `random_state=42` across split, SMOTE, and models.
- Preprocessor fitted only on training data; identical object reused at inference.
- SMOTE applied only to training data.
- Metrics chosen for imbalance (PR-AUC primary); accuracy reported but not used for selection.
- Pinned dependency versions.
- Deterministic artifact paths documented.

---

## 13. Traceability to HLD
| HLD Component | LLD Module(s) |
|---------------|---------------|
| Data Ingestion | `components/data_ingestion.py` |
| Data Validation | `components/data_validation.py` |
| Data Transformation | `components/data_transformation.py` |
| Model Trainer | `components/model_trainer.py` |
| Model Evaluation | `components/model_evaluation.py` |
| Prediction Service | `app/main.py`, `pipeline/prediction_pipeline.py` |
| Web UI | `app/templates/`, `app/static/` |
| Tests | `tests/` |
| Deployment | `Procfile`, `render.yaml`, `requirements.txt` |
