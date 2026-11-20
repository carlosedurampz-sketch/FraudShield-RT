"""Cost optimisation: the optimal threshold must beat threshold 0.50
and beat approving everything, on a simple hand-made case."""

import numpy as np

from src.cost_optimization import find_optimal_threshold, total_cost


def test_optimal_threshold_beats_defaults():
    rng = np.random.default_rng(0)
    y = np.array([0] * 900 + [1] * 100)
    # Scores: fraud mostly high, legit mostly low, with overlap.
    scores = np.concatenate([rng.beta(2, 8, 900), rng.beta(6, 2, 100)])
    amounts = np.where(y == 1, 5000.0, 500.0)
    opt = find_optimal_threshold(y, scores, amounts)
    best_cost = opt["best"]["total_cost_mxn"]
    assert best_cost <= total_cost(y, scores, 0.50, amounts)
    assert best_cost < total_cost(y, scores, 1.00, amounts)  # approve-all
    assert 0.02 <= opt["optimal_threshold"] <= 0.98


def test_missing_fraud_costs_the_amount():
    # One fraud of $8,000 missed at a very high threshold costs $8,000.
    assert total_cost([1], [0.1], 0.9, [8000.0]) == 8000.0
    # ...and catching it costs only the $60 review.
    assert total_cost([1], [0.95], 0.9, [8000.0]) == 60.0
