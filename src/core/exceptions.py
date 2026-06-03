"""
Centralized exception hierarchy for the trading pipeline.
"""


class PipelineError(Exception):
    """Base exception for all pipeline-related errors."""

    pass


class DataLoadError(PipelineError):
    """Raised when data loading fails."""

    pass


class DataProcessingError(PipelineError):
    """Raised when data processing/cleaning fails."""

    pass


class ModelTrainingError(PipelineError):
    """Raised when model training fails."""

    pass


class FeatureSelectionError(PipelineError):
    """Raised when feature selection fails."""

    pass


class ConfigurationError(PipelineError):
    """Raised when configuration is invalid or missing."""

    pass
