"""
tests/test_pipeline.py — Tests unitaires (bonus CI/CD /2).

Ces tests ne dépendent PAS de TensorFlow : ils valident la logique métier
(régularisation, diagnostic biais/variance, distribution des classes) sur des
données synthétiques. Ils tournent donc rapidement en CI.

Lancement : pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import CLASS_NAMES, NUM_CLASSES, class_distribution
from src.evaluation import diagnose_bias_variance
from src.ml_baseline import compare_regularizations, train_baseline


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n_feat = 16 * 16 * 3
    Xtr, ytr = rng.random((90, n_feat)), rng.integers(0, 3, 90)
    Xva, yva = rng.random((30, n_feat)), rng.integers(0, 3, 30)
    return Xtr, ytr, Xva, yva


def test_class_names_consistent():
    assert len(CLASS_NAMES) == NUM_CLASSES == 3


def test_class_distribution_sums():
    y = np.array([0, 0, 1, 2, 2, 2])
    dist = class_distribution(y)
    assert sum(dist.values()) == len(y)
    assert dist["healthy"] == 3


def test_lasso_creates_sparsity(synthetic_data):
    """Cœur du cours 1.5 : L1 doit annuler plus de coefficients que L2."""
    Xtr, ytr, Xva, yva = synthetic_data
    results = {r.penalty: r for r in compare_regularizations(Xtr, ytr, Xva, yva, C=0.1)}
    assert results["l1"].n_zero_coefs > results["l2"].n_zero_coefs


def test_baseline_returns_valid_accuracy(synthetic_data):
    Xtr, ytr, Xva, yva = synthetic_data
    r = train_baseline(Xtr, ytr, Xva, yva, penalty="l2", C=1.0)
    assert 0.0 <= r.train_acc <= 1.0
    assert 0.0 <= r.val_acc <= 1.0


def test_bias_variance_diagnosis():
    assert "VARIANCE" in diagnose_bias_variance(0.99, 0.60)
    assert "BIAIS" in diagnose_bias_variance(0.55, 0.52)
    assert "CORRECT" in diagnose_bias_variance(0.92, 0.89)