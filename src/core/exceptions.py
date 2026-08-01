"""
Centralized exception hierarchy for the trading pipeline.

There used to be two of these. This module defined PipelineError(Exception)
and ConfigurationError(PipelineError), while
src/core/error_handling/error_handler.py separately defined
TradingSystemError(Exception), PipelineError(TradingSystemError) and
ConfigurationError(TradingSystemError). Two unrelated classes shared each of
those names, so

    from src.core.exceptions import ConfigurationError
    ...
    except ConfigurationError:

would not have caught a ConfigurationError raised by a module that imported
the other one. Nothing was hitting that in practice -- the single raise and
the single except live in the same file with the same import, checked -- but
it is a trap that springs the first time someone catches one of these across
a module boundary.

One tree now, rooted at TradingSystemError. error_handler imports these names
instead of redefining them, so imports from either module keep working.

    TradingSystemError
    └── PipelineError
        ├── DataLoadError
        ├── DataProcessingError
        ├── ModelTrainingError
        ├── FeatureSelectionError
        ├── ConfigurationError
        └── StageError
            ├── StageExecutionError
            └── ModelLoadingError

Several of these are raised nowhere yet (PipelineError, FeatureSelectionError,
StageError, StageExecutionError). They are kept rather than deleted because
with a single root they become useful as catch-alls -- `except PipelineError`
finally means "any pipeline failure", which is what they were always meant to
mean.
"""


class TradingSystemError(Exception):
    """Base exception class for trading system errors."""

    pass


class PipelineError(TradingSystemError):
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


class StageError(PipelineError):
    """Base exception for individual pipeline stage failures."""

    pass


class StageExecutionError(StageError):
    """Exception raised when a stage fails during execution."""

    pass


class ModelLoadingError(StageError):
    """Exception raised when loading a model fails."""

    pass
