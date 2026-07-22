"""Compatibility exports for older hybrid dataclass imports."""

from src.pipeline.hybrid.contracts import (
    ColabBatchRequest,
    HybridFinalStagesRequest,
    HybridMockFeaturesRequest,
    HybridPipelineConfig,
    ModelTrainingContext,
)

__all__ = [
    "ColabBatchRequest",
    "HybridFinalStagesRequest",
    "HybridMockFeaturesRequest",
    "HybridPipelineConfig",
    "ModelTrainingContext",
]
