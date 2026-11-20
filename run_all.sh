#!/usr/bin/env bash
# FraudShield-RT — end-to-end run: data -> train -> savings -> latency -> tests
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. Synthetic data =="
python -m src.generate_synthetic --n 60000 --out data/synthetic_transactions.csv
echo "== 2. Train (out-of-time split, isotonic calibration) =="
python -m src.train
echo "== 3. Net-savings simulation vs static rules =="
python scripts/simulate_savings.py
echo "== 4. Latency test =="
python scripts/latency_test.py --n 500
echo "== 5. Tests =="
pytest -q
echo "== Done. Start the API with: uvicorn src.api.main:app --port 8000 =="
