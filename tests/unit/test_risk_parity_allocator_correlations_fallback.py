import numpy as np


def test_risk_parity_allocator_falls_back_when_correlations_invalid_shape():
    from src.algorithms.risk_parity_allocator import RiskParityAllocator

    allocator = RiskParityAllocator()
    vols = np.array([0.2, 0.3, 0.1], dtype=float)
    # Invalid correlation matrix shape triggers ValueError inside dot products
    correlations = np.eye(2)

    weights = allocator._risk_parity(vols, correlations, constraints={})

    assert weights.shape == (3,)
    assert np.isfinite(weights).all()
    assert np.isclose(weights.sum(), 1.0)

