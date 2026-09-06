# High-Level Design (HLD)
## Credit Card Fraud Detection — End-to-End Machine Learning System

**Document version:** 1.0
**Date:** September 6, 2026
**Author:** Project Team
**Status:** Draft — awaiting approval

---

## 1. Introduction

### 1.1 Purpose
This document describes the high-level design of an end-to-end Machine Learning system that detects fraudulent credit card transactions. It covers the problem, objectives, overall architecture, major components, technology choices, deployment approach, and non-functional requirements. It is written so that a reviewer (or interviewer) can understand *what* the system does and *how* the pieces fit together, without needing implementation-level detail (that is covered in the LLD).

### 1.2 Scope
The system:
- Ingests a credit card transactions dataset.
- Validates the data against a defined schema.
- Cleans and transforms the data (scaling, class-imbalance handling).
- Trains and compares multiple ML models and selects the best one.
- Evaluates the model with metrics appropriate for imbalanced data.
- Serves predictions through a REST API with a simple web UI.
- Is deployable for free on Render.

Out of scope (documented as future enhancements): real-time streaming ingestion, live bank integration, automated retraining pipelines, and a production feature store.

### 1.3 Definitions and Acronyms
| Term | Meaning |
|------|---------|
| HLD | High-Level Design |
| LLD | Low-Level Design |
| PCA | Principal Component Analysis (dataset features V1–V28 are PCA components) |
| SMOTE | Synthetic Minority Over-sampling Technique |
| PR-AUC | Area Under the Precision-Recall Curve |
| ROC-AUC | Area Under the Receiver Operating Characteristic Curve |
| API | Application Programming Interface |
| EDA | Exploratory Data Analysis |

---

## 2. Problem Statement

Credit card fraud causes major financial losses. Fraudulent transactions are rare compared to legitimate ones (typically well under 1% of all transactions). This creates a **severe class imbalance** problem: a naive model that predicts "not fraud" for everything can score ~99.8% accuracy while catching zero fraud.

The goal is to build a model that **catches fraud (high recall on the fraud class)** while **keeping false alarms manageable (reasonable precision)**, and to expose it as a working, deployed service.

---

## 3. Objectives

1. Build a reproducible, modular ML pipeline (ingestion → validation → transformation → training → evaluation).
2. Correctly handle class imbalance and choose metrics that reflect real performance (PR-AUC, precision, recall, F1 — not raw accuracy).
3. Compare multiple models and select the best based on defined criteria.
4. Persist the trained model and the preprocessing object so training and inference use identical transformations.
5. Serve predictions via a FastAPI REST API with a simple HTML/CSS/JS frontend.
6. Provide automated tests, a clear README, and an architecture diagram.
7. Deploy for free on Render.

---

## 4. Dataset Overview

- **Source:** Kaggle "Credit Card Fraud Detection" dataset (`creditcard.csv`), free with a Kaggle login.
- **Size:** ~284,807 transactions.
- **Features:**
  - `Time` — seconds elapsed between each transaction and the first transaction.
  - `V1`–`V28` — anonymized numerical features (PCA-transformed for confidentiality).
  - `Amount` — transaction amount.
  - `Class` — target label: `0` = legitimate, `1` = fraud.
- **Imbalance:** fraud is ~0.17% of records.
- **Handling:** The dataset is not committed to the repo (it is large and license-bound). The user drops `creditcard.csv` into the `data/` folder; the pipeline reads it from there.

---

## 5. System Architecture

### 5.1 High-Level Flow

```
        ┌──────────────────────┐
        │   creditcard.csv     │   (placed in data/ by user)
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   Data Ingestion     │  read CSV, split train/test (stratified)
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   Data Validation    │  check schema, columns, dtypes, nulls, drift
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Data Transformation │  scale Amount/Time, handle imbalance (SMOTE)
        └──────────┬───────────┘   save preprocessor.pkl
                   │
                   ▼
        ┌──────────────────────┐
        │    Model Training     │  train + compare models, tune threshold
        └──────────┬───────────┘   save model.pkl
                   │
                   ▼
        ┌──────────────────────┐
        │   Model Evaluation    │  PR-AUC, ROC-AUC, precision/recall/F1, CM
        └──────────┬───────────┘   save metrics report
                   │
                   ▼
        ┌──────────────────────┐
        │   FastAPI Service     │  loads model.pkl + preprocessor.pkl
        └──────────┬───────────┘
                   │
          ┌────────┴─────────┐
          ▼                  ▼
   ┌─────────────┐    ┌───────────────┐
   │  Web UI     │    │ REST Endpoints │  /predict (single + CSV batch)
   │ (HTML/JS)   │    │  /health /docs │
   └─────────────┘    └───────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Render Deployment    │  free hosting
        └──────────────────────┘
```

### 5.2 Architectural Style
- **Modular pipeline architecture:** each stage is a separate component with a well-defined input and output artifact. This mirrors production ML systems and makes the project easy to explain, test, and extend.
- **Separation of concerns:** training pipeline is independent from the serving API. The API only loads saved artifacts and serves predictions.
- **Config-driven:** schema and parameters live in config files, not hard-coded in logic.

---

## 6. Major Components (High Level)

| # | Component | Responsibility | Key Output |
|---|-----------|----------------|------------|
| 1 | Data Ingestion | Read raw CSV, stratified train/test split | `train.csv`, `test.csv` |
| 2 | Data Validation | Verify schema, columns, dtypes, missing values; basic drift check | validation status + drift report |
| 3 | Data Transformation | Scale `Amount`/`Time`, apply SMOTE on training data only, build preprocessor | `preprocessor.pkl`, transformed arrays |
| 4 | Model Trainer | Train and compare models, tune decision threshold, select best | `model.pkl` |
| 5 | Model Evaluation | Compute imbalance-aware metrics, confusion matrix, curves | `metrics.json`, plots |
| 6 | Prediction Service | FastAPI app loading artifacts and serving predictions | JSON responses / HTML table |
| 7 | Web UI | Simple form for single prediction + CSV upload for batch | rendered result page |
| 8 | Tests | Unit tests for pipeline components and API | test report |

---

## 7. Technology Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| Language | Python 3.10+ | Standard for ML |
| Data | Pandas, NumPy | Data manipulation |
| ML | Scikit-learn, imbalanced-learn (SMOTE), XGBoost | Modeling + imbalance handling |
| Serialization | joblib | Save/load model and preprocessor |
| API | FastAPI, Uvicorn | Fast, typed, auto-docs (Swagger) |
| Validation | Pydantic | Request validation at the API boundary |
| Frontend | HTML, CSS, JavaScript, Jinja2 | Simple UI, no heavy framework |
| Config | YAML | Schema and parameters |
| Testing | Pytest | Unit/integration tests |
| Deployment | Render (free tier), Gunicorn/Uvicorn | Free hosting |
| Version control | Git / GitHub | Source management |

Everything listed is free and open source.

---

## 8. Deployment Architecture

```
Developer ──push──> GitHub ──auto-deploy──> Render (free web service)
                                              │
                                              ▼
                                      Uvicorn + FastAPI
                                              │
                                     loads model.pkl + preprocessor.pkl
                                              │
                                     serves /predict, /health, UI
User Browser ──HTTPS──────────────────────────┘
```

- **Model artifacts** (`model.pkl`, `preprocessor.pkl`) are committed so the deployed service can load them without retraining on the free tier.
- **Training** runs locally (or on demand); the deployed app is inference-only for reliability on free hosting.
- **Cold starts** on Render free tier are expected and documented.

---

## 9. Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| Correctness | Same preprocessing at train and inference time (shared `preprocessor.pkl`). |
| Reproducibility | Fixed random seeds; documented steps; pinned dependency versions. |
| Performance | Single prediction returns in < 1 second (excluding cold start). |
| Reliability | API validates input and returns clear errors for malformed requests. |
| Maintainability | Modular components, config-driven, documented, tested. |
| Security | No secrets in repo; input validation; dataset not committed. |
| Usability | Simple UI; Swagger docs at `/docs`; clear README. |
| Cost | Entirely free (open-source tools + Render free tier). |

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Class imbalance skews model | Model misses fraud | SMOTE on train only, use PR-AUC/recall, threshold tuning |
| Data leakage from scaling before split | Inflated metrics | Fit scaler on train only, apply to test |
| Applying SMOTE to test data | Misleading evaluation | SMOTE applied strictly to training set |
| Large dataset not in repo | Build fails without data | Documented data-drop step; validation fails clearly if missing |
| Render free tier cold starts | Slow first response | Documented; inference-only deployment |
| Overfitting to majority class | Poor real-world use | Cross-validation, holdout test, imbalance-aware metrics |

---

## 11. Success Criteria

- Pipeline runs end-to-end and produces `model.pkl`, `preprocessor.pkl`, and a metrics report.
- Model achieves strong PR-AUC and high fraud-class recall with acceptable precision (targets finalized during training; accuracy alone is explicitly not the target).
- API serves single and batch predictions correctly and validates input.
- Tests pass.
- App is deployed and reachable on Render.
- README + architecture diagram present.

---

## 12. Deliverables

1. Modular training pipeline (all 5 stages).
2. Trained `model.pkl` and `preprocessor.pkl`.
3. Metrics report and evaluation plots.
4. FastAPI service + HTML/CSS/JS UI.
5. Pytest test suite.
6. README with setup, run, and deployment instructions.
7. Architecture diagram.
8. Render deployment configuration.
