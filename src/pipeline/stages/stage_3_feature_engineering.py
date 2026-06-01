"""
Stage 3: Feature Engineering - Facade for Modular Feature Engineering Stage.
Maintains backward compatibility with the original FeatureEngineeringStage.
"""

from .feature_engineering.orchestrator import FeatureEngineeringStage as ModularFeatureEngineeringStage

class FeatureEngineeringStage(ModularFeatureEngineeringStage):
    """
    Facade for FeatureEngineeringStage.
    Delegates to the modular components in the 'feature_engineering' subdirectory.
    """
    pass
