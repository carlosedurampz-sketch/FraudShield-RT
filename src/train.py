"""
Train the fraud model with an out-of-time split and isotonic calibration.

Split (chronological — never random, never shuffled):
  * train        first 60% of time  — fit the gradient-boosting model
  * calibration  next  10% of time  — isotonic calibration + threshold search
  * test         last  30% of time  — untouched until final evaluation

Why PR-AUC, not ROC-AUC: with <1% fraud, ROC-AUC is flattering
(most negatives are trivially easy). Precision-Recall AUC measures
exactly what operations cares about: of the alerts we raise, how many
are fraud, and how much fraud do we catch.

Model: LightGBM if installed, else XGBoost, else sklearn
HistGradientBoosting (fallback, no extra dependency). All are
optimised for average precision via the class-weight / scale_pos_weight.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.calibrated_model import IsotonicCalibratedModel
from src.cost_optimization import find_optimal_threshold, cost_breakdown
from src.features import MODEL_FEATURES, build_features
from src.generate_synthetic import generate


def get_model(n_pos: int, n_neg: int):
    """LightGBM > XGBoost > HistGradientBoosting, by availability."""
    try:
        from lightgbm import LGBMClassifier
        print("Model: LightGBM")
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=40, subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=n_neg / max(n_pos, 1),
            random_state=42, verbose=-1, n_jobs=-1), "LightGBM"
    except Exception as e:
        print(f"LightGBM unavailable ({e}), trying XGBoost")
    try:
        from xgboost import XGBClassifier
        print("Model: XGBoost")
        return XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=n_neg / max(n_pos, 1),
            eval_metric="logloss", random_state=42, n_jobs=-1), "XGBoost"
    except Exception as e:
        print(f"XGBoost unavailable ({e}), using HistGradientBoosting fallback")
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        class_weight="balanced", random_state=42), "HistGradientBoosting"


def ks_statistic(y_true, y_score) -> float:
    order = np.argsort(y_score)
    y = np.asarray(y_true)[order]
    pos = np.cumsum(y == 1) / max((y == 1).sum(), 1)
    neg = np.cumsum(y == 0) / max((y == 0).sum(), 1)
    return float(np.max(np.abs(pos - neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="CSV with transaction columns; "
                    "if omitted, synthetic data are generated")
    ap.add_argument("--n", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    root = Path(args.out_dir)

    if args.data:
        df = pd.read_csv(args.data, parse_dates=["timestamp"])
        print(f"Loaded {len(df):,} rows from {args.data}")
    else:
        df = generate(n=args.n, seed=args.seed)
        (root / "data").mkdir(exist_ok=True, parents=True)
        df.to_csv(root / "data" / "synthetic_transactions.csv", index=False)
        print(f"Generated {len(df):,} synthetic rows "
              f"(fraud {df['is_fraud'].mean():.3%})")

    df = build_features(df)
    df = df.sort_values("timestamp").reset_index(drop=True)
    X = df[MODEL_FEATURES].to_numpy(dtype=float)
    y = df["is_fraud"].to_numpy(dtype=int)
    amounts = df["amount_mxn"].to_numpy(dtype=float)

    n = len(df)
    i_train = int(n * 0.60)
    i_cal = int(n * 0.70)
    splits = {
        "train": (0, i_train), "calibration": (i_train, i_cal), "test": (i_cal, n),
    }
    print({k: f"{v[1]-v[0]:,} rows, fraud "
              f"{y[v[0]:v[1]].mean():.3%}" for k, v in splits.items()})
    print(f"Cut dates: train ≤ {df['timestamp'].iloc[i_train-1]}, "
          f"calibration ≤ {df['timestamp'].iloc[i_cal-1]}, "
          f"test → {df['timestamp'].iloc[-1]}")

    model, model_name = get_model(int(y[:i_train].sum()), int((y[:i_train] == 0).sum()))
    model.fit(X[:i_train], y[:i_train])

    # Isotonic calibration on the calibration split (see src/calibrated_model.py;
    # sklearn >= 1.6 removed CalibratedClassifierCV(cv="prefit")).
    calibrated = IsotonicCalibratedModel(base_model=model)
    calibrated.fit_calibration(X[i_train:i_cal], y[i_train:i_cal])

    def proba(m, X_):
        return m.predict_proba(X_)[:, 1]

    p_cal = proba(calibrated, X[i_train:i_cal])
    p_test = proba(calibrated, X[i_cal:])
    y_test = y[i_cal:]

    # Threshold: chosen on calibration only (no test leakage).
    opt = find_optimal_threshold(y[i_train:i_cal], p_cal, amounts[i_train:i_cal])
    thr = opt["optimal_threshold"]
    test_cost = cost_breakdown(y_test, p_test, thr, amounts[i_cal:])
    cost_050 = cost_breakdown(y_test, p_test, 0.50, amounts[i_cal:])

    metrics = {
        "model": model_name,
        "calibration": "isotonic (fitted on calibration split)",
        "split": "out-of-time 60/10/30",
        "n_train": int(i_train), "n_calibration": int(i_cal - i_train),
        "n_test": int(n - i_cal),
        "fraud_rate_overall": float(y.mean()),
        "fraud_rate_test": float(y_test.mean()),
        "pr_auc_test": float(average_precision_score(y_test, p_test)),
        "pr_auc_baseline_random": float(y_test.mean()),
        "roc_auc_test": float(roc_auc_score(y_test, p_test)),
        "ks_test": ks_statistic(y_test, p_test),
        "optimal_threshold_calibration": float(thr),
        "test_at_optimal_threshold": test_cost,
        "test_at_threshold_0.50": cost_050,
        "feature_importance": dict(sorted(
            zip(MODEL_FEATURES, [float(v) for v in
                 getattr(model, "feature_importances_", np.zeros(len(MODEL_FEATURES)))]),
            key=lambda kv: -kv[1])),
        "note": "ILLUSTRATIVE metrics on synthetic data — see README. "
                "Do not quote as real-world performance.",
    }
    (root / "models").mkdir(exist_ok=True, parents=True)
    (root / "outputs").mkdir(exist_ok=True, parents=True)
    joblib.dump(calibrated, root / "models" / "model.pkl")
    joblib.dump(model, root / "models" / "model_raw.pkl")
    # Stats the API / explainer need: medians + importances.
    stats = {
        "feature_medians": {f: float(np.median(X[:, j])) for j, f in enumerate(MODEL_FEATURES)},
        "feature_importance": metrics["feature_importance"],
        "threshold": float(thr),
    }
    (root / "models" / "feature_stats.json").write_text(json.dumps(stats, indent=2))
    (root / "outputs" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    test_df = df.iloc[i_cal:][["transaction_id", "timestamp", "account_id",
                               "amount_mxn", "is_fraud"]].copy()
    test_df["score"] = p_test
    test_df["pred_block"] = (p_test >= thr).astype(int)
    test_df.to_csv(root / "outputs" / "test_predictions.csv", index=False)

    print(json.dumps({k: metrics[k] for k in
                      ["model", "pr_auc_test", "roc_auc_test", "ks_test",
                       "optimal_threshold_calibration"]}, indent=2))
    print(f"Test @ optimal thr {thr}: cost ${test_cost['total_cost_mxn']:,.0f} MXN | "
          f"precision {test_cost['precision']:.2f} | recall {test_cost['recall']:.2f} | "
          f"fraud missed ${test_cost['fraud_missed_amount_mxn']:,.0f}")
    print(f"Test @ thr 0.50:        cost ${cost_050['total_cost_mxn']:,.0f} MXN | "
          f"precision {cost_050['precision']:.2f} | recall {cost_050['recall']:.2f}")
    print("Saved models/model.pkl, models/feature_stats.json, outputs/metrics.json, "
          "outputs/test_predictions.csv")


if __name__ == "__main__":
    main()
