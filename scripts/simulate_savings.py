"""
Net-savings simulator: model (cost-optimal threshold) vs. static rules.

Static baselines a team might actually run in production:
  * Rule A — block if amount > $5,000 MXN
  * Rule B — block if amount > $5,000 OR >= 5 txns from the account in 1 h
  * Rule C — block if amount > $2,000 (aggressive)

Reads outputs/test_predictions.csv (produced by src/train.py) and the
velocity features from data/synthetic_transactions.csv, recomputes the
cost of each policy with the same cost matrix (src/cost_optimization.py),
and writes outputs/savings_simulation.json + a readable summary.

Usage: python scripts/simulate_savings.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.cost_optimization import cost_breakdown, static_rule_cost

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    preds = pd.read_csv(ROOT / "outputs" / "test_predictions.csv")
    metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    thr = metrics["optimal_threshold_calibration"]

    # Static rules need the raw features for the test rows.
    raw = pd.read_csv(ROOT / "data" / "synthetic_transactions.csv",
                      parse_dates=["timestamp"])
    from src.features import build_features
    feat = build_features(raw)
    feat = feat.sort_values("timestamp").reset_index(drop=True)
    test_feat = feat.iloc[len(feat) - len(preds):].reset_index(drop=True)
    assert len(test_feat) == len(preds)

    y = preds["is_fraud"].to_numpy()
    amounts = preds["amount_mxn"].to_numpy()
    scores = preds["score"].to_numpy()

    model = cost_breakdown(y, scores, thr, amounts)
    rules = {
        "static_A_amount_gt_5000": static_rule_cost(
            y, amounts, test_feat["amount_mxn"].to_numpy() > 5_000),
        "static_B_amount_gt_5000_or_velocity_ge5_1h": static_rule_cost(
            y, amounts, (test_feat["amount_mxn"].to_numpy() > 5_000)
            | (test_feat["acct_count_1h"].to_numpy() >= 5)),
        "static_C_amount_gt_2000": static_rule_cost(
            y, amounts, test_feat["amount_mxn"].to_numpy() > 2_000),
        "no_controls_approve_all": static_rule_cost(
            y, amounts, __import__("numpy").zeros(len(y), dtype=bool)),
    }
    result = {
        "test_n": int(len(y)), "fraud_n": int(y.sum()),
        "optimal_threshold": float(thr),
        "model_at_optimal_threshold": model,
        "static_rules": rules,
        "net_savings_vs_best_static_rule_mxn": float(
            min(r["total_cost_mxn"] for r in rules.values()) - model["total_cost_mxn"]),
        "net_savings_vs_approve_all_mxn": float(
            rules["no_controls_approve_all"]["total_cost_mxn"] - model["total_cost_mxn"]),
        "note": "ILLUSTRATIVE — synthetic data. Costs use the matrix in "
                "src/cost_optimization.py (FN = amount, FP = $120, TP = $60).",
    }
    for name, r in rules.items():
        result[f"savings_vs_{name}_mxn"] = float(r["total_cost_mxn"] - model["total_cost_mxn"])
    out = ROOT / "outputs" / "savings_simulation.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Test: {len(y):,} txns, {int(y.sum())} fraud, "
          f"fraud volume ${amounts[y == 1].sum():,.0f} MXN")
    print(f"Model @ thr {thr}: cost ${model['total_cost_mxn']:,.0f} | "
          f"P {model['precision']:.2f} R {model['recall']:.2f} | "
          f"fraud missed ${model['fraud_missed_amount_mxn']:,.0f}")
    for name, r in rules.items():
        print(f"{name}: cost ${r['total_cost_mxn']:,.0f} | "
              f"P {r['precision']:.2f} R {r['recall']:.2f} | "
              f"savings of model vs rule: ${r['total_cost_mxn'] - model['total_cost_mxn']:,.0f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
