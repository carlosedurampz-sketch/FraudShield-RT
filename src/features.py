"""
Velocity features — strictly causal (past only, no temporal leakage).

For every transaction we compute, using ONLY transactions with an
earlier timestamp:

  * counts / amounts in sliding windows 10 min, 1 h, 24 h, per
    account, per device and per merchant
  * time since the account's previous transaction
  * amount vs. the account's own prior average
  * whether the device has ever been seen before

Implementation: per entity group, timestamps are sorted and
``np.searchsorted`` finds the window boundaries, so a window count is
``(# prior txns with t - window <= t_prev < t)`` — the current row is
never included (``side="left"`` on the current timestamp).

This is the single most important correctness property of the project:
in production a scorer only ever sees the past. A feature that peeks at
the future inflates offline metrics and fails live. tests/test_features.py
asserts exactly this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS_MIN = {"10m": 10, "1h": 60, "24h": 1440}

MODEL_FEATURES = [
    "amount_mxn",
    "amount_log",
    "hour_sin",
    "hour_cos",
    "is_night",
    "account_age_days",
    "device_age_days",
    "channel_code",
    "acct_count_10m",
    "acct_count_1h",
    "acct_count_24h",
    "acct_amt_1h",
    "acct_amt_24h",
    "acct_avg_amt_24h",
    "device_count_1h",
    "device_count_24h",
    "merch_count_1h",
    "merch_count_24h",
    "time_since_last_acct_min",
    "amount_vs_acct_avg",
    "is_new_device",
]

CHANNEL_CODES = {"online": 0, "pos": 1, "ecommerce": 2, "mobile_app": 3}


def _window_counts_sums(
    times_ns: np.ndarray, amounts: np.ndarray, window_min: int
) -> tuple[np.ndarray, np.ndarray]:
    """For each row i, count/sum of rows j with t-window <= t_j < t_i."""
    window_ns = window_min * 60 * 1_000_000_000
    counts = np.zeros(len(times_ns), dtype=float)
    sums = np.zeros(len(times_ns), dtype=float)
    csum = np.concatenate([[0.0], np.cumsum(amounts)])
    for i, t in enumerate(times_ns):
        lo = np.searchsorted(times_ns, t - window_ns, side="left")
        hi = np.searchsorted(times_ns, t, side="left")  # strictly before t
        counts[i] = hi - lo
        sums[i] = csum[hi] - csum[lo]
    return counts, sums


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["amount_log"] = np.log1p(df["amount_mxn"])
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["is_night"] = (
        (df["timestamp"].dt.hour < 5) | (df["timestamp"].dt.hour >= 23)
    ).astype(int)
    df["channel_code"] = df["channel"].map(CHANNEL_CODES).fillna(0).astype(int)
    times_ns = df["timestamp"].values.astype("datetime64[ns]").astype(np.int64)
    amounts = df["amount_mxn"].to_numpy(dtype=float)

    for prefix, col in [
        ("acct", "account_id"),
        ("device", "device_id"),
        ("merch", "merchant_id"),
    ]:
        # Per-group window features, computed on the group's own sorted times.
        for win_name, win_min in WINDOWS_MIN.items():
            if prefix == "merch" and win_name == "10m":
                continue  # merchant 10-min window adds noise, not signal
            counts = np.zeros(len(df))
            sums = np.zeros(len(df))
            for idx in df.groupby(col, sort=False).groups.values():
                idx = np.asarray(sorted(idx))
                c, s = _window_counts_sums(times_ns[idx], amounts[idx], win_min)
                counts[idx] = c
                sums[idx] = s
            df[f"{prefix}_count_{win_name}"] = counts
            if prefix in ("acct",):
                df[f"{prefix}_amt_{win_name}"] = sums
        # device / merchant amounts are not model features; keep counts only.
        if prefix != "acct":
            for win_name in WINDOWS_MIN:
                if f"{prefix}_amt_{win_name}" in df.columns:
                    df = df.drop(columns=[f"{prefix}_amt_{win_name}"])

    # Account history: prior average amount, time since last txn, new device.
    acct_avg = np.ones(len(df))
    since_last = np.full(len(df), 1440.0)
    for idx in df.groupby("account_id", sort=False).groups.values():
        idx = np.asarray(sorted(idx))
        csum = np.concatenate([[0.0], np.cumsum(amounts[idx])])
        for k, i in enumerate(idx):
            acct_avg[i] = csum[k] / k if k else amounts[i]
            if k:
                since_last[i] = (times_ns[i] - times_ns[idx[k - 1]]) / 60e9
    df["acct_avg_amt_24h"] = np.where(
        df["acct_count_24h"] > 0,
        df["acct_amt_24h"] / np.maximum(df["acct_count_24h"], 1),
        acct_avg,
    )
    df["amount_vs_acct_avg"] = df["amount_mxn"] / np.maximum(acct_avg, 1.0)
    df["time_since_last_acct_min"] = since_last
    # Device ever seen before (strictly prior transactions on that device).
    seen: dict[str, int] = {}
    is_new = np.ones(len(df), dtype=int)
    for i, dev in enumerate(df["device_id"]):
        is_new[i] = 0 if seen.get(dev, 0) > 0 else 1
        seen[dev] = seen.get(dev, 0) + 1
    df["is_new_device"] = is_new
    # device_count_24h==0 is equivalent for a first-ever device; keep both.
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add velocity + derived features and return only model columns + keys."""
    out = add_velocity_features(df)
    for c in MODEL_FEATURES:
        if c not in out.columns:
            out[c] = 0.0
    return out
