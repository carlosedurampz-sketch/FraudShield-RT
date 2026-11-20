"""API smoke tests: /health and /score contract."""
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True, "train first: python -m src.train"


def test_score_returns_score_and_three_reasons():
    r = client.post("/score", json={
        "transaction_id": "TX-TEST-001", "amount_mxn": 8500.0,
        "account_id": "ACC-00001", "device_id": "DEV-99999",
        "merchant_id": "MER-0001", "channel": "online", "hour": 2,
        "account_age_days": 400, "device_age_days": 3,
        "acct_count_10m": 6, "acct_count_1h": 8, "acct_count_24h": 12,
        "acct_amt_1h": 20000, "acct_amt_24h": 30000, "acct_avg_amt_24h": 900,
        "device_count_1h": 5, "device_count_24h": 7,
        "merch_count_1h": 1, "merch_count_24h": 3,
        "time_since_last_acct_min": 2, "amount_vs_acct_avg": 4.0,
        "is_new_device": 1,
    })
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["decision"] in ("approve", "review", "block")
    assert body["risk_level"] in ("low", "medium", "high")
    assert len(body["top_reasons"]) == 3
    assert body["latency_ms"] >= 0
    # A burst + new device + night + high amount should not be "approve".
    assert body["decision"] in ("review", "block")


def test_score_legit_transaction_is_low_risk():
    r = client.post("/score", json={
        "transaction_id": "TX-TEST-002", "amount_mxn": 450.0,
        "account_id": "ACC-00001", "device_id": "DEV-00001",
        "merchant_id": "MER-0001", "channel": "pos", "hour": 14,
        "account_age_days": 900, "device_age_days": 800,
        "acct_count_10m": 0, "acct_count_1h": 0, "acct_count_24h": 1,
        "acct_amt_1h": 0, "acct_amt_24h": 450, "acct_avg_amt_24h": 500,
        "device_count_1h": 0, "device_count_24h": 1,
        "merch_count_1h": 0, "merch_count_24h": 2,
        "time_since_last_acct_min": 600, "amount_vs_acct_avg": 0.9,
        "is_new_device": 0,
    })
    assert r.status_code == 200
    assert r.json()["score"] < 0.5
