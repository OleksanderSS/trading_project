"""
Stage 7: Evaluation - Facade for Modular Evaluation Stage.
Maintains backward compatibility with the original EvaluationStage.
"""

from .evaluation.orchestrator import EvaluationStage as ModularEvaluationStage


class EvaluationStage(ModularEvaluationStage):
    """
    Facade for EvaluationStage.
    Delegates to the modular components in the 'evaluation' subdirectory.
    """
    pass
