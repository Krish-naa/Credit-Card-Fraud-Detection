# Credit Card Fraud Detection — End-to-End ML System

An end-to-end machine learning system that detects fraudulent credit card transactions. It covers the full lifecycle: data ingestion, validation, transformation, class-imbalance handling, model training and selection, evaluation with imbalance-aware metrics, a REST API with a web UI, tests, and free deployment on Render.

Design documents: [`docs/HLD.md`](docs/HLD.md) (High-Level Design) and [`docs/LLD.md`](docs/LLD.md) (Low-Level Design).

---

## Why this project

Fraud is rare (~0.17% of transactions), so the data is severely imbalanced. A model that always predicts "not fraud" would score ~99.8% accuracy while catching zero fraud. This project is built around that reality:

- **Accuracy is deliberately not the target.** The primary metric is **PR-AUC** (average precision), with precision, recall, and F1 on the fraud class.
- Class imbalance is handled with **SMOTE**, applied to the training set only.
- Data leakage is prevented: the scaler is fit on training data only, and SMOTE never touches the test set.

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
                                              Render (free hosting)
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
├── Procfile, render.yaml, runtime.txt   # deployment
└── README.md
```

---

## Setup

Requires Python 3.12 (3.11 also works). The pinned dependencies have prebuilt wheels for these versions.

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
# Place it in the data/ folder: data/creditcard.csv
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

## Run the API locally

```bash
uvicorn app.main:app --reload
```

Then open:
- **http://127.0.0.1:8000/** — web UI (single prediction form + CSV upload)
- **http://127.0.0.1:8000/docs** — interactive Swagger docs

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

The web UI has a **Load Sample** button that fills a real transaction so you can test instantly.

---

## Run tests

```bash
pytest
```

Tests cover data validation, transformation (including leakage/imbalance guards), the prediction pipeline, and the API. They use a small synthetic dataset, so they run without the full `creditcard.csv`.

---

## Deploy to Render (free)

1. Push this repo to GitHub (the trained `final_models/*.pkl` are committed so the app serves without retraining).
2. On [Render](https://render.com), create a new **Web Service** from your repo.
3. Render reads `render.yaml` automatically. Otherwise set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Deploy. The free tier sleeps when idle, so the first request after inactivity may be slow (cold start).

---

## Configuration

- `config/schema.yaml` — expected columns, dtypes, target, feature order.
- `config/params.yaml` — test split, imbalance method (`smote` / `undersample` / `none`), models to compare, CV folds, metrics, and the minimum PR-AUC to accept a model.

Change the imbalance method or model list here without touching code.

---

## Key design decisions (interview talking points)

1. **Imbalance-aware metrics.** PR-AUC drives model selection; accuracy is reported but never used to choose a model.
2. **No data leakage.** Scaler fit on train only; SMOTE on train only; test set stays untouched until evaluation.
3. **Threshold tuning.** The decision threshold is tuned for best F1 on the fraud class rather than left at 0.5.
4. **Modular pipeline.** Each stage is an isolated component with typed config and artifact objects, mirroring production ML systems.
5. **Reproducibility.** Fixed random seeds and pinned dependencies.
6. **Inference-only deployment.** Training runs offline; the deployed service just loads artifacts, which keeps it reliable on free hosting.
```
