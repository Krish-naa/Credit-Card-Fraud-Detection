"""FastAPI application: web UI + REST endpoints for fraud prediction."""
import io
import os

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import TransactionInput, PredictionResponse
from src.fraud_detection.exception import FraudException
from src.fraud_detection.logger import get_logger
from src.fraud_detection.pipeline.prediction_pipeline import PredictionPipeline

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="Credit Card Fraud Detection",
    description="End-to-end ML service that predicts whether a transaction is fraudulent.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Cache-busting token appended to static asset URLs so browsers always fetch the
# latest CSS/JS after a deploy instead of serving a stale cached copy.
ASSET_VERSION = str(int(os.path.getmtime(os.path.join(BASE_DIR, "static", "style.css"))))

# Single shared pipeline instance (artifacts loaded lazily on first prediction)
pipeline = PredictionPipeline()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "v": ASSET_VERSION})


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TransactionInput):
    try:
        result = pipeline.predict_single(payload.model_dump())
        return result
    except FraudException as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=503, detail="Model not available. Train the model first.")
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        logger.exception("Unexpected error in /predict")
        raise HTTPException(status_code=500, detail="Internal error")


@app.post("/predict/csv", response_class=HTMLResponse)
async def predict_csv(request: Request, file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        result_df = pipeline.predict_batch(df)

        fraud_count = int(result_df["predicted_class"].sum())
        total = len(result_df)
        # Show first 100 rows in the UI to keep the page light
        table_html = result_df.head(100).to_html(
            classes="result-table", index=False, border=0
        )
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "table": table_html,
                "fraud_count": fraud_count,
                "total": total,
                "shown": min(100, total),
                "v": ASSET_VERSION,
            },
        )
    except FraudException as e:
        msg = str(e)
        if "schema mismatch" in msg:
            raise HTTPException(status_code=400, detail=msg)
        if "not found" in msg.lower():
            raise HTTPException(status_code=503, detail="Model not available. Train the model first.")
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        logger.exception("Unexpected error in /predict/csv")
        raise HTTPException(status_code=500, detail="Internal error processing CSV")
