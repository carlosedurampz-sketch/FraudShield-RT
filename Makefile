.PHONY: install data train simulate latency test api run-all clean

install:
	pip install -r requirements.txt

data:
	python -m src.generate_synthetic --n 60000 --out data/synthetic_transactions.csv

train: data
	python -m src.train

simulate: train
	python scripts/simulate_savings.py

latency: train
	python scripts/latency_test.py --n 500

test: train
	pytest -q

api: train
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

run-all: test simulate latency

clean:
	rm -rf __pycache__ src/__pycache__ src/api/__pycache__ tests/__pycache__ \
		.pytest_cache models/*.pkl outputs/*.json outputs/*.csv data/*.csv
