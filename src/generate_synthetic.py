"""
Generate realistic synthetic card/payment transactions with <1% fraud.

Why synthetic by default
-------------------------
The recommended real datasets (IEEE-CIS Fraud Detection, PaySim) require a
Kaggle account / download and cannot be redistributed. This generator
reproduces the *structure* that matters for the pipeline — extreme class
imbalance, chronological order, velocity bursts, new-device and night-time
fraud patterns — so the whole project runs with zero downloads:

    python -m src.generate_synthetic --n 60000 --out data/synthetic_transactions.csv

To use a real dataset instead, see docs/real_data.md. The training code
accepts any CSV with the columns documented in README.md.

Fraud signals baked into the generator (so the model has something real
to learn, and the signals are the ones fraud teams actually use):
  * velocity bursts: a compromised account fires 5-8 transactions in <10 min
  * new device + night hours (00-05) + unusually high amount
  * amount far above the account's own historical average

All randomness is seeded (default 42) for reproducibility.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ["online", "pos", "ecommerce", "mobile_app"]


def generate(n: int = 60_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01")
    n_accounts, n_devices, n_merchants = 3_000, 4_000, 200

    account_age = rng.integers(30, 1500, size=n_accounts)
    device_age = rng.integers(1, 900, size=n_devices)

    # --- base (mostly legitimate) transactions ---------------------------
    timestamps = start + pd.to_timedelta(
        rng.uniform(0, 60 * 24 * 60, size=n), unit="m"
    )
    account_id = rng.integers(0, n_accounts, size=n)
    # Most accounts reuse 1-2 devices; a minority use a brand-new device.
    device_id = np.where(
        rng.random(n) < 0.85,
        (account_id * 7 + rng.integers(0, 2, size=n)) % n_devices,
        rng.integers(0, n_devices, size=n),
    )
    merchant_id = rng.integers(0, n_merchants, size=n)
    channel = rng.choice(CHANNELS, size=n, p=[0.35, 0.30, 0.20, 0.15])
    amount = rng.lognormal(mean=np.log(750), sigma=0.9, size=n).round(2)
    hour = timestamps.hour

    is_new_device = rng.random(n) < 0.08
    is_night = (hour < 5) | (hour >= 23)
    # Fraud propensity: night + new device + high amount raise it sharply.
    p_fraud = (
        0.0010
        + 0.008 * is_night
        + 0.010 * is_new_device
        + 0.006 * (amount > 5_000)
        + 0.005 * (amount > 10_000)
    )
    is_fraud = rng.random(n) < p_fraud
    # Fraudulent amounts are drawn larger, as in real card fraud.
    amount = np.where(
        is_fraud,
        rng.lognormal(mean=np.log(3_200), sigma=1.0, size=n).round(2),
        amount,
    )

    df = pd.DataFrame(
        {
            "transaction_id": [f"TX-{i:06d}" for i in range(n)],
            "timestamp": timestamps,
            "account_id": [f"ACC-{a:05d}" for a in account_id],
            "device_id": [f"DEV-{d:05d}" for d in device_id],
            "merchant_id": [f"MER-{m:04d}" for m in merchant_id],
            "amount_mxn": amount,
            "channel": channel,
            "account_age_days": account_age[account_id],
            "device_age_days": device_age[device_id],
            "is_fraud": is_fraud.astype(int),
        }
    )

    # --- velocity-burst fraud rings: 45 accounts, 6 txns in <10 minutes ---
    bursts = []
    for b in range(45):
        acc = f"ACC-{rng.integers(0, n_accounts):05d}"
        dev = f"DEV-{rng.integers(0, n_devices):05d}"
        t0 = start + pd.to_timedelta(rng.uniform(0, 58 * 24 * 60), unit="m")
        for k in range(rng.integers(5, 9)):
            bursts.append(
                {
                    "transaction_id": f"TX-BURST-{b:03d}-{k}",
                    "timestamp": t0 + pd.to_timedelta(int(rng.integers(0, 10)), unit="m"),
                    "account_id": acc,
                    "device_id": dev,
                    "merchant_id": f"MER-{rng.integers(0, n_merchants):04d}",
                    "amount_mxn": round(float(rng.lognormal(np.log(2_800), 0.7)), 2),
                    "channel": "online",
                    "account_age_days": int(rng.integers(30, 1500)),
                    "device_age_days": int(rng.integers(1, 30)),  # new device
                    "is_fraud": 1,
                }
            )
    df = pd.concat([df, pd.DataFrame(bursts)], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["transaction_id"] = [f"TX-{i:06d}" for i in range(len(df))]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic_transactions.csv")
    args = ap.parse_args()

    df = generate(n=args.n, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    rate = df["is_fraud"].mean()
    print(f"Wrote {len(df):,} transactions to {out}")
    print(f"Fraud rate: {rate:.3%} ({int(df['is_fraud'].sum())} fraud)")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Total volume: ${df['amount_mxn'].sum():,.0f} MXN | "
          f"Fraud volume: ${df.loc[df['is_fraud'] == 1, 'amount_mxn'].sum():,.0f} MXN")


if __name__ == "__main__":
    main()
