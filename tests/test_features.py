"""No temporal leakage: velocity features must use ONLY past transactions."""
import pandas as pd

from src.features import add_velocity_features


def _df():
    return pd.DataFrame({
        "transaction_id": ["TX-000001", "TX-000002", "TX-000003", "TX-000004"],
        "timestamp": pd.to_datetime([
            "2026-01-01 10:00", "2026-01-01 10:05", "2026-01-01 10:07",
            "2026-01-02 10:00"]),
        "account_id": ["ACC-00001"] * 4,
        "device_id": ["DEV-00001"] * 4,
        "merchant_id": ["MER-0001"] * 4,
        "amount_mxn": [100.0, 200.0, 300.0, 400.0],
        "channel": ["online"] * 4,
        "account_age_days": [365] * 4,
        "device_age_days": [100] * 4,
        "is_fraud": [0, 0, 0, 0],
    })


def test_first_transaction_has_zero_velocity():
    out = add_velocity_features(_df())
    first = out.iloc[0]
    assert first["acct_count_10m"] == 0
    assert first["acct_count_1h"] == 0
    assert first["acct_count_24h"] == 0
    assert first["is_new_device"] == 1


def test_windows_count_only_the_past():
    out = add_velocity_features(_df())
    # Row 2 (10:05): 1 prior txn within 10 min / 1 h / 24 h.
    assert out.iloc[1]["acct_count_10m"] == 1
    assert out.iloc[1]["acct_count_1h"] == 1
    # Row 3 (10:07): 2 prior in 1 h, but only 1 within 10 min (10:05; 10:00 is 7 min... actually both)
    assert out.iloc[2]["acct_count_10m"] == 2
    assert out.iloc[2]["acct_count_1h"] == 2
    assert out.iloc[2]["acct_amt_1h"] == 300.0  # 100 + 200, current 300 excluded
    # Row 4 (next day): 0 in 1 h, 0 in 10 min, but 3 in 24 h? 10:00 Jan 1 is exactly 24h before.
    # Window is [t-24h, t): the Jan-1 10:00 txn is at the boundary and IS included.
    assert out.iloc[3]["acct_count_10m"] == 0
    assert out.iloc[3]["acct_count_1h"] == 0
    assert out.iloc[3]["acct_count_24h"] == 3
    # A future transaction must never leak backwards: row 0 stays 0 even
    # though later rows exist.
    assert out.iloc[0]["acct_count_24h"] == 0


def test_time_since_last_and_new_device():
    out = add_velocity_features(_df())
    assert out.iloc[1]["time_since_last_acct_min"] == 5.0
    assert out.iloc[2]["time_since_last_acct_min"] == 2.0
    assert out.iloc[1]["is_new_device"] == 0  # device already seen
