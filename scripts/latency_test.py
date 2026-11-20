"""
Latency test for the scoring API (in-process, via FastAPI TestClient).

Sends N scoring requests (a mix of normal and suspicious transactions)
and reports mean / p50 / p95 / p99 latency and throughput. No server
needs to be running.

Usage:
    python scripts/latency_test.py [--n 500]
    python scripts/latency_test.py --url http://localhost:8000   # live server
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def payloads(n: int) -> list[dict]:
    rng = np.random.default_rng(7)
    out = []
    for i in range(n):
        suspicious = i % 5 == 0
        out.append(
            {
                "transaction_id": f"TX-LAT-{i:05d}",
                "amount_mxn": float(
                    rng.lognormal(np.log(3500 if suspicious else 700), 0.8)
                ),
                "account_id": "ACC-00001",
                "device_id": "DEV-00001",
                "merchant_id": "MER-0001",
                "channel": "online",
                "hour": 2 if suspicious else 14,
                "account_age_days": 400,
                "device_age_days": 5 if suspicious else 300,
                "acct_count_10m": 6 if suspicious else 0,
                "acct_count_1h": 8 if suspicious else 1,
                "acct_count_24h": 12 if suspicious else 2,
                "acct_amt_1h": 20000 if suspicious else 800,
                "acct_amt_24h": 30000 if suspicious else 1500,
                "acct_avg_amt_24h": 900,
                "device_count_1h": 5 if suspicious else 0,
                "device_count_24h": 7 if suspicious else 1,
                "merch_count_1h": 1,
                "merch_count_24h": 3,
                "time_since_last_acct_min": 2 if suspicious else 600,
                "amount_vs_acct_avg": 4.0 if suspicious else 1.0,
                "is_new_device": 1 if suspicious else 0,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument(
        "--url", default=None, help="live API base URL; if omitted, uses TestClient"
    )
    args = ap.parse_args()

    reqs = payloads(args.n)
    lat: list[float] = []
    if args.url:
        import httpx

        # trust_env=False: httpx chokes parsing NO_PROXY entries such as
        # "[::1]" in some environments; localhost needs no proxy anyway.
        with httpx.Client(base_url=args.url, timeout=10, trust_env=False) as c:
            assert c.get("/health").json()["model_loaded"], "model not loaded on server"
            for p in reqs[:20]:  # warm-up: exclude cold start from the numbers
                c.post("/score", json=p)
            t0 = time.perf_counter()
            for p in reqs:
                s = time.perf_counter()
                r = c.post("/score", json=p)
                r.raise_for_status()
                lat.append((time.perf_counter() - s) * 1000)
            total = time.perf_counter() - t0
    else:
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as c:
            assert c.get("/health").json()["model_loaded"], (
                "train first: python -m src.train"
            )
            for p in reqs[:20]:  # warm-up: exclude cold start from the numbers
                c.post("/score", json=p)
            t0 = time.perf_counter()
            for p in reqs:
                s = time.perf_counter()
                r = c.post("/score", json=p)
                r.raise_for_status()
                lat.append((time.perf_counter() - s) * 1000)
            total = time.perf_counter() - t0
    lat_sorted = sorted(lat)

    def pct(q):
        return lat_sorted[min(len(lat_sorted) - 1, int(q * len(lat_sorted)))]

    print(
        f"Requests: {len(lat)} | total {total:.2f}s | throughput {len(lat) / total:.0f} req/s"
    )
    print(
        f"Latency ms — mean {statistics.mean(lat):.2f} | p50 {pct(0.50):.2f} | "
        f"p95 {pct(0.95):.2f} | p99 {pct(0.99):.2f} | max {max(lat):.2f}"
    )
    print(
        "Note: in-process TestClient latency excludes network; "
        "re-run with --url against a live uvicorn server for end-to-end numbers."
    )


if __name__ == "__main__":
    main()
