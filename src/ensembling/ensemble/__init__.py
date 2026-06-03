"""
Ensemble module - exports from stacked_ensemble for backward compatibility.
"""

from ..stacked_ensemble import (
    StackedEnsemble,
    ensemble_forecast,
    EnsembleResult
)

__all__ = [
    'StackedEnsemble',
    'ensemble_forecast',
    'EnsembleResult',
]
