# Credit Card Fraud Detection — End-to-End Machine Learning System

An end-to-end machine learning system for detecting fraudulent credit card transactions. The project implements the complete lifecycle: data ingestion, schema validation, data transformation, class-imbalance handling, model training and selection, evaluation with imbalance-aware metrics, a REST API with a web interface, an automated test suite, and containerised deployment.

**Live application:** https://credit-card-fraud-detection-1-done.onrender.com/

Design documentation is available in [`docs/HLD.md`](docs/HLD.md) (High-Level Design) and [`docs/LLD.md`](docs/LLD.md) (Low-Level Design).

---

## Overview

Fraudulent transactions represent approximately 0.17% of the dataset, resulting in a severe class imbalance. A model that always predicts "not fraud" would achieve roughly 99.8% accuracy while detecting no fraud at all. The system is therefore designed around this constraint:

- **Accuracy is intentionally not the optimisation target.** The primary metric is **PR-AUC** (average precision), reported alongside precision, recall, and F1 on the fraud class.
- Class imbalance is addressed using **SMOTE**, applied exclusively to the training set.
- Data leakage is prevented by fitting the scaler on training data only and never applying SMOTE to the test set.

---

## Architecture

```
creditcard.csv
      │
      ▼
Data Ingestion  ->  Data Validation  ->  Data Transformation  ->  Model Trainer  ->  Model Evaluation
 (stratified       (schema, dtypes,     (scale Time/Amount,      (compare models,   (PR-AUC, ROC-AUC,
  train/test)       nulls, drift)        SMOTE on train only)     PR-AUC CV,         precision/recall,
                                         saves preprocessor.pkl   threshold tuning,  confusion matrix)
                                                                  saves model.pkl)
                                                       │
                                                       ▼
                                          FastAPI service (+ HTML/CSS/JS UI)
                                          /  /predict  /predict/csv  /health  /docs
                                                       │
                                                       ▼
                                        Docker container hosted on Render
```

---

## Project structure

```
MLProjectNew/
├── data/creditcard.csv          # dataset (not committed - you add it)
├── docs/                        # HLD.md, LLD.md
├── config/                      # schema.yaml, params.yaml
├── artifacts/                   # generated per run (not committed)
├── final_models/                # model.pkl, preprocessor.pkl (committed for deploy)
├── src/fraud_detection/
│   ├── components/              # ingestion, validation, transformation, trainer, evaluation
│   ├── entity/                  # config + artifact dataclasses
│   ├── pipeline/                # training_pipeline, prediction_pipeline
│   ├── utils/                   # helpers (yaml, save/load)
│   ├── constants.py, config...  # paths, logger, exceptions
├── app/                         # FastAPI app, schemas, templates, static
├── tests/                       # pytest suite
├── train.py                     # run the full training pipeline
├── requirements.txt
├── Dockerfile, .dockerignore    # container definition
├── render.yaml                  # Render deployment configuration
└── README.md
```

---

## Local setup

The project requires Python 3.12 (3.11 is also supported). The pinned dependencies provide prebuilt wheels for these versions.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Download 'creditcard.csv' from Kaggle:
#   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place the file in the data/ directory: data/creditcard.csv
```

---

## Train the model

```bash
python train.py
```

This runs all five stages and writes:
- `final_models/model.pkl` (best model + tuned threshold)
- `final_models/preprocessor.pkl`
- `artifacts/evaluation/metrics.json` and plots (PR curve, ROC curve, confusion matrix)

The training summary (best model, threshold, test PR-AUC) is printed at the end.

---

## Running the API locally

```bash
uvicorn app.main:app --reload
```

The following are then available:
- **http://127.0.0.1:8000/** — web interface (single-transaction form and CSV upload)
- **http://127.0.0.1:8000/docs** — interactive Swagger documentation

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Web UI |
| GET | `/health` | Health check |
| POST | `/predict` | Single prediction (JSON body) |
| POST | `/predict/csv` | Batch prediction (CSV upload) |
| GET | `/docs` | Swagger UI |

### Example single prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time":0,"V1":-1.36,"V2":-0.07, ... ,"V28":-0.02,"Amount":149.62}'
```

Response:
```json
{ "prediction": 0, "label": "Legitimate", "probability": 0.0123 }
```

The web interface includes a **Load Sample** button that populates the form with a real transaction for immediate testing.

---

## Testing

```bash
pytest
```

The test suite covers data validation, data transformation (including leakage and imbalance safeguards), the prediction pipeline, and the API. The tests operate on a small synthetic dataset and therefore run without requiring the full `creditcard.csv` file.

---

## Deployment

The application is deployed on Render as a Docker container and is publicly accessible:

**https://credit-card-fraud-detection-1-done.onrender.com/**

Additional endpoints on the live service:

- Interactive API documentation: [`/docs`](https://credit-card-fraud-detection-1-done.onrender.com/docs)
- Health check: [`/health`](https://credit-card-fraud-detection-1-done.onrender.com/health)

> **Note:** The service runs on Render's free tier, which suspends the instance after a period of inactivity. The first request following a period of inactivity may take 30–50 seconds while the instance restarts (a cold start), after which responses are immediate.

Deployment is defined by the [`Dockerfile`](Dockerfile) and [`render.yaml`](render.yaml). The container image pins the Python version internally, so the runtime environment is fully deterministic and independent of the host's default Python version. The trained artefacts (`final_models/*.pkl`) are committed to the repository, allowing the service to serve predictions without retraining.

### Running the container locally

```bash
docker build -t fraud-app .
docker run -p 8000:8000 fraud-app
# The application is then available at http://localhost:8000
```

### Environment determinism

Pinned dependencies such as `pandas==2.2.2` provide prebuilt wheels only for specific Python versions. If the host uses a newer Python version, pip attempts to compile the package from source, which typically fails. To eliminate this class of failure, the `Dockerfile` fixes the interpreter to `python:3.12.7-slim`, ensuring the correct wheels are always used regardless of where the image runs.

---

## Configuration

- `config/schema.yaml` — expected columns, dtypes, target, feature order.
- `config/params.yaml` — test split, imbalance method (`smote` / `undersample` / `none`), models to compare, CV folds, metrics, and the minimum PR-AUC to accept a model.

The imbalance method and the list of candidate models can be changed here without modifying any code.

---

## Key design decisions

1. **Imbalance-aware metrics.** PR-AUC drives model selection; accuracy is reported but is never used as the selection criterion.
2. **No data leakage.** The scaler is fitted on the training set only, SMOTE is applied to the training set only, and the test set remains untouched until evaluation.
3. **Threshold tuning.** The decision threshold is tuned to maximise F1 on the fraud class rather than defaulting to 0.5.
4. **Modular pipeline.** Each stage is an isolated component with typed configuration and artefact objects, reflecting production machine learning system design.
5. **Reproducibility.** Random seeds are fixed and dependencies are pinned; the container fixes the Python version.
6. **Inference-only deployment.** Training is performed offline and the deployed service loads pre-computed artefacts, which keeps the hosted application lightweight and reliable.
```https://credit-card-fraud-detection-1-done.onrender.com/
