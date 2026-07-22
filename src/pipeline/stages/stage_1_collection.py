"""
Stage 1: Collection - Facade for Modular Collection Stage.
Maintains backward compatibility with the original CollectionStage.
"""

from .collection.orchestrator import CollectionStage as ModularCollectionStage


class CollectionStage(ModularCollectionStage):
    """
    Facade for CollectionStage.
    Delegates to the modular components in the 'collection' subdirectory.
    """
    pass
