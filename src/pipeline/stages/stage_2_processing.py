"""
Stage 2: Data Processing - Facade for Modular Processing Stage.
Maintains backward compatibility with the original ProcessingStage.
"""

from .processing.orchestrator import ProcessingStage as ModularProcessingStage


class ProcessingStage(ModularProcessingStage):
    """
    Facade for ProcessingStage.
    Delegates to the modular components in the 'processing' subdirectory.
    """
    pass
