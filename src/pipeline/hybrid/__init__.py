# src/pipeline/hybrid/__init__.py
"""
Hybrid Pipeline Components.

This module contains refactored components for the hybrid pipeline orchestrator,
split into logical components for better maintainability and testability.
Hybrid Orchestrator Components.
Modular components for the hybrid pipeline orchestrator.
"""

from .colab_manager import ColabManager
from .contracts import HybridFinalStagesRequest, HybridPipelineRequest
from .data_manager import HybridDataManager
from .data_utils import DataUtils
from .feature_processor import FeatureProcessor
from .metadata_manager import MetadataManager, MetadataParams, SaveMetadataParams
from .orchestrator_config import OrchestratorConfigManager, PipelineConfig
from .pipeline_config import ColabBatchParams, FinalStagesParams, PipelineParams
from .pipeline_manager import PipelineManager
from .storage_manager import StorageManager

# CacheManager, FeatureSelectionManager, OrchestratorInterface and
# PipelineExecutor were re-exported here but are now archived under
# src/archive/pipeline_hybrid_dormant/ -- they were constructed by
# component_factory and attached to the orchestrator, yet never called.
__all__ = [
    'HybridDataManager',
    'FeatureProcessor',
    'ColabManager',
    'PipelineManager',
    'OrchestratorConfigManager',
    'PipelineConfig',
    'StorageManager',
    'DataUtils',
    'MetadataManager',
    'PipelineParams',
    'FinalStagesParams',
    'ColabBatchParams',
    'HybridFinalStagesRequest',
    'HybridPipelineRequest',
    'MetadataParams',
    'SaveMetadataParams'
]
