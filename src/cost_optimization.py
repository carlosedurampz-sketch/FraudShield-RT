"""
Cost-based threshold selection — the metric that matters in production.

A fraud model is not judged by AUC alone: every decision has a peso cost.

Cost matrix (MXN, documented assumptions — tune them with the business):
  * False Negative (fraud missed):  the full transaction amount
    (the money is gone; chargeback / loss).
  * False Positive (legit blocked): 120 MXN — manual review time +
    customer friction / lost sale / support contact.
  * True Positive (fraud caught):    60 MXN — manual review of the alert.
  * True Negative:                    0 MXN.

Total cost = sum over all scored transactions. The optimal threshold is
the one that minimises total cost on the *calibration* split (never on
the test split — that would be threshold leakage). A threshold of 0.50
is almost never optimal under <1% fraud: missing one $8,000 fraud costs
as much as ~65 false alarms.
"""
from __future__ import annotations

import numpy as np

# MXN — see module docstring for the rationale. Override per business.
COST_FP = 120.0   # legit transaction blocked: review + friction
COST_TP = 60.0    # fraud caught: manual review of the alert
COST_TN = 0.0


def total_cost(y_true, y_score, threshold: float, amounts,
               cost_fp: float = COST_FP, cost_tp: float = COST_TP) -> float:
    y_true = np.asarray(y_true); amounts = np.asarray(amounts, dtype=float)
    pred = np.asarray(y_score) >= threshold
    fn = (~pred) & (y_true == 1)
    fp = pred & (y_true == 0)
    tp = pred & (y_true == 1)
    return float(amounts[fn].sum() + cost_fp * fp.sum() + cost_tp * tp.sum())


def cost_breakdown(y_true, y_score, threshold: float, amounts,
                   cost_fp: float = COST_FP, cost_tp: float = COST_TP) -> dict:
    y_true = np.asarray(y_true); amounts = np.asarray(amounts, dtype=float)
    pred = np.asarray(y_score) >= threshold
    tp = int(((pred) & (y_true == 1)).sum()); fp = int(((pred) & (y_true == 0)).sum())
    fn = int(((~pred) & (y_true == 1)).sum()); tn = int(((~pred) & (y_true == 0)).sum())
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fraud_caught": tp, "fraud_missed": fn,
        "fraud_missed_amount_mxn": float(amounts[(~pred) & (y_true == 1)].sum()),
        "false_positive_cost_mxn": float(cost_fp * fp),
        "review_cost_mxn": float(cost_tp * tp),
        "total_cost_mxn": float(total_cost(y_true, y_score, threshold, amounts, cost_fp, cost_tp)),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def find_optimal_threshold(y_true, y_score, amounts,
                           cost_fp: float = COST_FP, cost_tp: float = COST_TP,
                           grid: np.ndarray | None = None) -> dict:
    """Scan thresholds; return the cost-minimising one + the full curve."""
    if grid is None:
        grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    rows = [cost_breakdown(y_true, y_score, float(t), amounts, cost_fp, cost_tp)
            for t in grid]
    best = min(rows, key=lambda r: r["total_cost_mxn"])
    return {"optimal_threshold": best["threshold"], "best": best, "curve": rows}


def static_rule_cost(y_true, amounts, rule_pred) -> dict:
    """Cost of a static rule (e.g. block amount > $5,000) for comparison."""
    y_true = np.asarray(y_true); amounts = np.asarray(amounts, dtype=float)
    pred = np.asarray(rule_pred, dtype=bool)
    tp = int(((pred) & (y_true == 1)).sum()); fp = int(((pred) & (y_true == 0)).sum())
    fn = int(((~pred) & (y_true == 1)).sum()); tn = int(((~pred) & (y_true == 0)).sum())
    total = float(amounts[(~pred) & (y_true == 1)].sum() + COST_FP * fp + COST_TP * tp)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "total_cost_mxn": total,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0}
