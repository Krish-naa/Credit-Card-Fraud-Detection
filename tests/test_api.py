"""API tests for the FastAPI service."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_home_page_loads():
    res = client.get("/")
    assert res.status_code == 200
    assert "Credit Card Fraud Detection" in res.text


def test_predict_rejects_invalid_payload():
    # Missing all required fields -> 422 validation error
    res = client.post("/predict", json={"Amount": 100})
    assert res.status_code == 422


def test_predict_rejects_negative_amount():
    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload["Time"] = 0.0
    payload["Amount"] = -5.0  # violates ge=0
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
