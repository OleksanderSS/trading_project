"""
Stage 4: Modeling - Facade for Modular Modeling Stage.
Maintains backward compatibility with the original ModelingStage.
"""

from .modeling.orchestrator import ModelingStage as ModularModelingStage
from .modeling.orchestrator import TargetProcessingConfig


class ModelingStage(ModularModelingStage):
    """
    Facade for ModelingStage.
    Delegates to the modular components in the 'modeling' subdirectory.
    """
    pass
