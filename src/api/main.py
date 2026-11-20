"""
FraudShield-RT scoring API (FastAPI).

POST /score  -> {"score", "risk_level", "decision", "top_reasons", "latency_ms"}
GET  /health -> {"status": "ok", "model_loaded": bool, "threshold": float}

Design for a real-time path:
  * the model is loaded once at startup, not per request;
  * velocity features are accepted pre-computed by the caller (the
    transaction stream / feature store owns the sliding windows in
    production — see README "Production notes"). If omitted they
    default to 0, and the API still returns a score;
  * explanations use the fast importance-based path (see src/explain.py),
    not SHAP per request, so p95 latency stays in single-digit ms.
Run:  uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.explain import top_reasons
from src.features import MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "model.pkl"
STATS_PATH = ROOT / "models" / "feature_stats.json"

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load()
    yield


app = FastAPI(title="FraudShield-RT", version="0.1.0",
              description="Real-time fraud scoring: score + top 3 reasons in ms.",
              lifespan=lifespan)

_model = None
_stats: dict = {"feature_medians": {}, "feature_importance": {}, "threshold": 0.5}


def _load():
    global _model, _stats
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    if STATS_PATH.exists():
        _stats = json.loads(STATS_PATH.read_text())
    return _model


class TransactionRequest(BaseModel):
    transaction_id: str = "TX-UNKNOWN"
    amount_mxn: float = Field(..., gt=0)
    account_id: str = "ACC-00000"
    device_id: str = "DEV-00000"
    merchant_id: str = "MER-0000"
    channel: str = "online"
    hour: int = Field(12, ge=0, le=23)
    account_age_days: int = 365
    device_age_days: int = 180
    # Velocity features (pre-computed by the caller / feature store).
    acct_count_10m: float = 0
    acct_count_1h: float = 0
    acct_count_24h: float = 0
    acct_amt_1h: float = 0
    acct_amt_24h: float = 0
    acct_avg_amt_24h: float = 0
    device_count_1h: float = 0
    device_count_24h: float = 0
    merch_count_1h: float = 0
    merch_count_24h: float = 0
    time_since_last_acct_min: float = 1440
    amount_vs_acct_avg: float = 1.0
    is_new_device: int = 0


class ScoreResponse(BaseModel):
    transaction_id: str
    score: float
    risk_level: str
    decision: str
    threshold: float
    top_reasons: list[str]
    latency_ms: float
    model_version: str = "fraudshield-rt-0.1.0"


@app.get("/health")
def health():
    model = _load()
    return {"status": "ok", "model_loaded": model is not None,
            "threshold": float(_stats.get("threshold", 0.5)),
            "features": len(MODEL_FEATURES)}


@app.post("/score", response_model=ScoreResponse)
def score(tx: TransactionRequest):
    t0 = time.perf_counter()
    model = _load()
    if model is None:
        raise HTTPException(503, "Model not trained yet — run: python -m src.train")
    hour = tx.hour
    feats = {
        "amount_mxn": tx.amount_mxn,
        "amount_log": float(np.log1p(tx.amount_mxn)),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "is_night": 1 if (hour < 5 or hour >= 23) else 0,
        "account_age_days": tx.account_age_days,
        "device_age_days": tx.device_age_days,
        "channel_code": {"online": 0, "pos": 1, "ecommerce": 2, "mobile_app": 3}.get(tx.channel, 0),
        "acct_count_10m": tx.acct_count_10m, "acct_count_1h": tx.acct_count_1h,
        "acct_count_24h": tx.acct_count_24h, "acct_amt_1h": tx.acct_amt_1h,
        "acct_amt_24h": tx.acct_amt_24h, "acct_avg_amt_24h": tx.acct_avg_amt_24h,
        "device_count_1h": tx.device_count_1h, "device_count_24h": tx.device_count_24h,
        "merch_count_1h": tx.merch_count_1h, "merch_count_24h": tx.merch_count_24h,
        "time_since_last_acct_min": tx.time_since_last_acct_min,
        "amount_vs_acct_avg": tx.amount_vs_acct_avg,
        "is_new_device": tx.is_new_device,
    }
    x = np.array([[feats[f] for f in MODEL_FEATURES]], dtype=float)
    score_val = float(model.predict_proba(x)[0, 1])
    thr = float(_stats.get("threshold", 0.5))
    if score_val >= thr:
        decision, risk = "block", "high"
    elif score_val >= thr * 0.5:
        decision, risk = "review", "medium"
    else:
        decision, risk = "approve", "low"
    reasons = top_reasons(feats, _stats, k=3)
    latency_ms = (time.perf_counter() - t0) * 1000
    return ScoreResponse(
        transaction_id=tx.transaction_id, score=round(score_val, 4),
        risk_level=risk, decision=decision, threshold=thr,
        top_reasons=reasons, latency_ms=round(latency_ms, 2))
