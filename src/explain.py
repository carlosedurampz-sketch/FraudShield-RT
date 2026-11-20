"""
Per-case explanations: why did this transaction score high?

Two paths, in order of preference:
  1. SHAP (if installed): TreeExplainer on the raw boosting model —
     exact per-feature contributions for tree models.
  2. Fallback (always available, and what the API uses for latency):
     feature importance x how far the value sits above its training
     median. Fast (<1 ms), deterministic, and honest about being an
     approximation — the reason strings are the same either way.

Reason strings are written for a human reviewer, not a data scientist:
"5 transactions from this account in the last hour" beats
"acct_count_1h = 5".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.features import MODEL_FEATURES

REASONS = {
    "acct_count_10m": lambda v: f"{int(v)} transactions from this account in the last 10 minutes",
    "acct_count_1h": lambda v: f"{int(v)} transactions from this account in the last hour",
    "acct_count_24h": lambda v: f"{int(v)} transactions from this account in the last 24 hours",
    "acct_amt_1h": lambda v: f"${v:,.0f} MXN from this account in the last hour",
    "acct_amt_24h": lambda v: f"${v:,.0f} MXN from this account in the last 24 hours",
    "device_count_1h": lambda v: f"{int(v)} transactions from this device in the last hour",
    "device_count_24h": lambda v: f"{int(v)} transactions from this device in the last 24 hours",
    "merch_count_1h": lambda v: f"{int(v)} transactions at this merchant in the last hour",
    "merch_count_24h": lambda v: f"{int(v)} transactions at this merchant in the last 24 hours",
    "amount_mxn": lambda v: f"High amount: ${v:,.0f} MXN",
    "amount_log": lambda v: "High amount for this channel",
    "amount_vs_acct_avg": lambda v: f"Amount is {v:.1f}x the account's usual amount",
    "is_new_device": lambda v: "First transaction ever from this device",
    "is_night": lambda v: "Transaction at night (23:00-05:00)",
    "time_since_last_acct_min": lambda v: f"Only {v:.0f} min since the account's previous transaction",
    "account_age_days": lambda v: "Very new account",
    "device_age_days": lambda v: "Very new device",
    "channel_code": lambda v: "High-risk channel for this pattern",
    "hour_sin": lambda v: "Unusual hour for this account",
    "hour_cos": lambda v: "Unusual hour for this account",
    "acct_avg_amt_24h": lambda v: "Account's recent average amount is elevated",
}


def top_reasons(feature_values: dict, feature_stats: dict, k: int = 3) -> list[str]:
    """Importance-weighted deviation from the training median."""
    importances = feature_stats.get("feature_importance", {})
    medians = feature_stats.get("feature_medians", {})
    scored = []
    for name in MODEL_FEATURES:
        v = float(feature_values.get(name, 0.0))
        med = float(medians.get(name, 0.0))
        imp = float(importances.get(name, 0.0))
        if imp <= 0:
            continue
        # Deviation, scaled so count/amount features dominate when extreme.
        denom = abs(med) if abs(med) > 1 else 1.0
        dev = max(0.0, (v - med) / denom) if name not in ("time_since_last_acct_min",) else \
            max(0.0, (med - v) / denom if med else 0.0)
        # Binary flags (new device, night) fire at full weight when set.
        if name in ("is_new_device", "is_night") and v >= 1:
            dev = 1.0
        if name in ("is_new_device", "is_night") and v < 1:
            dev = 0.0
        score = imp * (1.0 + dev)
        if dev > 0 or (name in ("is_new_device", "is_night") and v >= 1):
            scored.append((score, name, v))
    scored.sort(reverse=True)
    reasons = []
    for _, name, v in scored[:k]:
        fn = REASONS.get(name)
        reasons.append(fn(v) if fn else f"{name} = {v:.2f}")
    if not reasons:
        reasons = ["No single dominant signal; score driven by the combination of features"]
    return reasons[:k]


def shap_reasons(model, feature_values: dict, k: int = 3) -> list[str] | None:
    """Best-effort SHAP explanation; returns None if SHAP is unavailable."""
    try:
        import shap
        import pandas as pd
        raw = model
        # Unwrap IsotonicCalibratedModel -> underlying booster.
        if hasattr(model, "base_model"):
            raw = model.base_model
        explainer = shap.TreeExplainer(raw)
        row = pd.DataFrame([[feature_values.get(f, 0.0) for f in MODEL_FEATURES]],
                           columns=MODEL_FEATURES)
        vals = explainer.shap_values(row)
        if isinstance(vals, list):
            vals = vals[1]
        vals = np.asarray(vals).ravel()
        order = np.argsort(-np.abs(vals))[:k]
        out = []
        for j in order:
            name = MODEL_FEATURES[j]
            fn = REASONS.get(name)
            out.append(fn(float(feature_values.get(name, 0.0))) if fn else name)
        return out
    except Exception:
        return None


def explain_case(feature_values: dict, model=None,
                 stats_path: str | Path = "models/feature_stats.json") -> list[str]:
    if model is not None:
        shap_out = shap_reasons(model, feature_values)
        if shap_out:
            return shap_out
    try:
        stats = json.loads(Path(stats_path).read_text())
    except Exception:
        stats = {"feature_medians": {}, "feature_importance": {f: 1.0 for f in MODEL_FEATURES}}
    return top_reasons(feature_values, stats)
