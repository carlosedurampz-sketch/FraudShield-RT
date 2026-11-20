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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.explain import top_reasons
from src.features import CHANNEL_CODES, MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "model.pkl"
STATS_PATH = ROOT / "models" / "feature_stats.json"

# Maximum single-transaction amount accepted by the API (MXN). A single
# card/payment transaction above this is not a plausible scoring input;
# reject it with 422 instead of scoring an out-of-distribution value.
MAX_AMOUNT_MXN = 10_000_000.0
MAX_AGE_DAYS = 36_500  # 100 years, upper bound for account/device age
MAX_ID_LENGTH = 64


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load()
    yield


app = FastAPI(
    title="FraudShield-RT",
    version="0.1.0",
    description="Real-time fraud scoring: score + top 3 reasons in ms.",
    lifespan=lifespan,
)

_model: Any = None
_stats: dict[str, Any] = {
    "feature_medians": {},
    "feature_importance": {},
    "threshold": 0.5,
}


def _load() -> Any:
    """Load the model and feature stats once; return the model (or None)."""
    global _model, _stats
    if _model is None and MODEL_PATH.exists():
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception:  # noqa: BLE001 — any load failure means 'model unavailable'
            # A corrupt/unreadable model file must not crash the service
            # or leak a traceback: the API reports the model as unavailable.
            _model = None
    if STATS_PATH.exists():
        try:
            _stats = json.loads(STATS_PATH.read_text())
        except (OSError, ValueError):
            _stats = {"feature_medians": {}, "feature_importance": {}, "threshold": 0.5}
    return _model


def _threshold() -> float:
    """Cost-optimal threshold from the trained stats, clamped to [0, 1]."""
    try:
        return float(min(1.0, max(0.0, float(_stats.get("threshold", 0.5)))))
    except (TypeError, ValueError):
        return 0.5


class TransactionRequest(BaseModel):
    """Strict input schema for POST /score.

    Amounts must be positive and within a documented maximum; hours are
    0-23; velocity counts are non-negative integers; ages and window
    amounts are non-negative. Anything outside these ranges is rejected
    with a 422 validation error by Pydantic before scoring runs.
    """

    transaction_id: str = Field(default="TX-UNKNOWN", max_length=MAX_ID_LENGTH)
    amount_mxn: float = Field(..., gt=0, le=MAX_AMOUNT_MXN)
    account_id: str = Field(default="ACC-00000", max_length=MAX_ID_LENGTH)
    device_id: str = Field(default="DEV-00000", max_length=MAX_ID_LENGTH)
    merchant_id: str = Field(default="MER-0000", max_length=MAX_ID_LENGTH)
    channel: Literal["online", "pos", "ecommerce", "mobile_app"] = "online"
    hour: int = Field(default=12, ge=0, le=23)
    account_age_days: int = Field(default=365, ge=0, le=MAX_AGE_DAYS)
    device_age_days: int = Field(default=180, ge=0, le=MAX_AGE_DAYS)
    # Velocity features (pre-computed by the caller / feature store).
    acct_count_10m: int = Field(default=0, ge=0)
    acct_count_1h: int = Field(default=0, ge=0)
    acct_count_24h: int = Field(default=0, ge=0)
    acct_amt_1h: float = Field(default=0, ge=0)
    acct_amt_24h: float = Field(default=0, ge=0)
    acct_avg_amt_24h: float = Field(default=0, ge=0)
    device_count_1h: int = Field(default=0, ge=0)
    device_count_24h: int = Field(default=0, ge=0)
    merch_count_1h: int = Field(default=0, ge=0)
    merch_count_24h: int = Field(default=0, ge=0)
    time_since_last_acct_min: float = Field(default=1440, ge=0)
    amount_vs_acct_avg: float = Field(default=1.0, ge=0)
    is_new_device: int = Field(default=0, ge=0, le=1)


class ScoreResponse(BaseModel):
    transaction_id: str
    score: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]
    decision: Literal["approve", "review", "block"]
    threshold: float = Field(ge=0.0, le=1.0)
    top_reasons: list[str]
    latency_ms: float = Field(ge=0.0)
    model_version: str = "fraudshield-rt-0.1.0"


@app.get("/health")
def health() -> dict[str, Any]:
    model = _load()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "threshold": _threshold(),
        "features": len(MODEL_FEATURES),
    }


@app.post("/score", response_model=ScoreResponse)
def score(tx: TransactionRequest) -> ScoreResponse:
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
        "channel_code": CHANNEL_CODES[tx.channel],
        "acct_count_10m": tx.acct_count_10m,
        "acct_count_1h": tx.acct_count_1h,
        "acct_count_24h": tx.acct_count_24h,
        "acct_amt_1h": tx.acct_amt_1h,
        "acct_amt_24h": tx.acct_amt_24h,
        "acct_avg_amt_24h": tx.acct_avg_amt_24h,
        "device_count_1h": tx.device_count_1h,
        "device_count_24h": tx.device_count_24h,
        "merch_count_1h": tx.merch_count_1h,
        "merch_count_24h": tx.merch_count_24h,
        "time_since_last_acct_min": tx.time_since_last_acct_min,
        "amount_vs_acct_avg": tx.amount_vs_acct_avg,
        "is_new_device": tx.is_new_device,
    }
    x = np.array([[feats[f] for f in MODEL_FEATURES]], dtype=float)
    try:
        score_val = float(model.predict_proba(x)[0, 1])
    except Exception:  # noqa: BLE001 — any scoring failure maps to a generic 500
        # Never expose a model traceback to the caller: report a generic
        # 500 and let the server log carry the details.
        raise HTTPException(500, "Scoring failed for this transaction") from None
    thr = _threshold()
    if score_val >= thr:
        decision, risk = "block", "high"
    elif score_val >= thr * 0.5:
        decision, risk = "review", "medium"
    else:
        decision, risk = "approve", "low"
    reasons = top_reasons(feats, _stats, k=3)
    latency_ms = (time.perf_counter() - t0) * 1000
    return ScoreResponse(
        transaction_id=tx.transaction_id,
        score=round(score_val, 4),
        risk_level=risk,
        decision=decision,
        threshold=thr,
        top_reasons=reasons,
        latency_ms=round(latency_ms, 2),
    )
