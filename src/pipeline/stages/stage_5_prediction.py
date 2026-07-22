"""
Stage 5: Prediction - Facade for Modular Prediction Stage.
Maintains backward compatibility with the original PredictionStage.
"""

from .prediction.orchestrator import PredictionStage as ModularPredictionStage


class PredictionStage(ModularPredictionStage):
    """
    Facade for PredictionStage.
    Delegates to the modular components in the 'prediction' subdirectory.
    """
    pass
