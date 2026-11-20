# Using real data instead of synthetic data

The project runs end-to-end on synthetic data (no download). To reproduce
it on a real dataset, adapt the columns below and pass `--data`.

## IEEE-CIS Fraud Detection (Kaggle)

1. Create a Kaggle account and accept the competition rules:
   https://www.kaggle.com/competitions/ieee-fraud-detection
2. Download `train_transaction.csv` and `train_identity.csv`.
3. Map columns to the project schema:

| Project column | IEEE-CIS |
|---|---|
| `timestamp` | `TransactionDT` (seconds since a reference date — convert to a datetime) |
| `account_id` | `card1` (card identifier, proxy for account) |
| `device_id` | `DeviceInfo` / `id_30` |
| `merchant_id` | `addr1` / `card4` (proxy) |
| `amount_mxn` | `TransactionAmt` (USD in the original; rename the unit) |
| `channel` | `ProductCD` |
| `is_fraud` | `isFraud` |

Then: `python -m src.train --data data/ieee_train.csv`

Notes: IEEE-CIS has ~3.5% fraud in train (higher than production <1%),
so re-weight / re-calibrate and report PR-AUC, not ROC-AUC. The
`V1..V339` anonymised features are *not* used here on purpose: the
velocity features in `src/features.py` are the transferable part.

## PaySim (Kaggle / synthetic mobile-money simulator)

https://www.kaggle.com/datasets/ealaxi/paysim1 — `PS_20174392719_1491204439457_log.csv`.
Map `nameOrig` → `account_id`, `nameDest` → `merchant_id`,
`amount` → `amount_mxn`, `isFraud` → `is_fraud`, `step` → timestamp
(1 step = 1 hour). PaySim fraud rate is ~0.13% — even more imbalanced,
so the cost-based threshold matters more, not less.

## What changes, what doesn't

* Changes: column mapping, fraud rate, cost matrix (amounts are in a
  different currency/scale — re-estimate FP/TP costs with the business).
* Doesn't change: out-of-time split, past-only velocity features,
  isotonic calibration, cost-optimal threshold, FastAPI contract.
