"""
Ensemble module - exports from stacked_ensemble for backward compatibility.
"""

from ..stacked_ensemble import EnsembleResult, StackedEnsemble, ensemble_forecast

__all__ = [
    'StackedEnsemble',
    'ensemble_forecast',
    'EnsembleResult',
]
