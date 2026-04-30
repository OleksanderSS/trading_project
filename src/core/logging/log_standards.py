"""
Logging Standards: Consistent logging patterns across the trading pipeline

Defines standardized logging macros and formats for better readability,
monitoring, and debugging. All components should use these patterns.

Standards:
- Use emojis for visual distinction
- Consistent message formats
- Appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Structured information for monitoring

Usage:
    from src.core.logging.log_standards import log_stage_start, log_success, log_error

    log_stage_start(logger, "FeatureEngineering", {"tickers": 5})
    # Output: "⏹️ Starting FeatureEngineering (tickers=5)"

    log_success(logger, "Model training completed", {"accuracy": 0.95})
    # Output: "✅ Model training completed: accuracy=0.95"
"""

import logging
from typing import Any, Dict, Optional


# Standard logging macros
def log_stage_start(logger: logging.Logger, component: str, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log the start of a pipeline stage or major component.

    Args:
        logger: Logger instance
        component: Component name (e.g., "Stage3", "FeatureEngineering")
        context: Optional context info (e.g., {"tickers": 5, "timeframe": "1d"})
    """
    context_str = _format_context(context)
    logger.info(f"⏹️ Starting {component}{context_str}")


def log_stage_end(logger: logging.Logger, component: str, duration: float,
                  context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log the completion of a pipeline stage or major component.

    Args:
        logger: Logger instance
        component: Component name
        duration: Execution time in seconds
        context: Optional context info
    """
    context_str = _format_context(context)
    logger.info(f"✅ {component} completed in {duration:.2f}s{context_str}")


def log_progress(logger: logging.Logger, current: int, total: int, component: str = "",
                 context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log progress updates for long-running operations.

    Args:
        logger: Logger instance
        current: Current progress count
        total: Total expected count
        component: Optional component name
        context: Optional additional context
    """
    percentage = (current / total * 100) if total > 0 else 0
    component_str = f" {component}" if component else ""
    context_str = _format_context(context)
    logger.info(f"📊 Progress{component_str}: {current}/{total} ({percentage:.1f}%){context_str}")


def log_success(logger: logging.Logger, message: str, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log successful operations or milestones.

    Args:
        logger: Logger instance
        message: Success message
        context: Optional context info
    """
    context_str = _format_context(context)
    logger.info(f"✅ {message}{context_str}")


def log_warning(logger: logging.Logger, message: str, context: Optional[Dict[str, Any]] = None,
                fallback: Optional[str] = None) -> None:
    """
    Log warnings with optional fallback information.

    Args:
        logger: Logger instance
        message: Warning message
        context: Optional context info
        fallback: Description of fallback action taken
    """
    context_str = _format_context(context)
    fallback_str = f" → {fallback}" if fallback else ""
    logger.warning(f"⚠️ {message}{context_str}{fallback_str}")


def log_error(logger: logging.Logger, message: str, error: Optional[Exception] = None,
              context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log errors with optional exception details.

    Args:
        logger: Logger instance
        message: Error message
        error: Optional exception object
        context: Optional context info
    """
    context_str = _format_context(context)
    logger.error(f"❌ {message}{context_str}", exc_info=error)


def log_cache_hit(logger: logging.Logger, cache_type: str, key_info: str,
                  context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log cache hits for performance monitoring.

    Args:
        logger: Logger instance
        cache_type: Type of cache (e.g., "prediction", "feature")
        key_info: Cache key information
        context: Optional context
    """
    context_str = _format_context(context)
    logger.debug(f"🚀 {cache_type.title()} cache hit: {key_info}{context_str}")


def log_cache_miss(logger: logging.Logger, cache_type: str, key_info: str,
                   context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log cache misses for performance monitoring.

    Args:
        logger: Logger instance
        cache_type: Type of cache
        key_info: Cache key information
        context: Optional context
    """
    context_str = _format_context(context)
    logger.debug(f"🔄 {cache_type.title()} cache miss: {key_info}{context_str}")


def log_memory_usage(logger: logging.Logger, operation: str, memory_mb: float,
                     delta_mb: Optional[float] = None) -> None:
    """
    Log memory usage for operations.

    Args:
        logger: Logger instance
        operation: Operation name
        memory_mb: Current memory usage in MB
        delta_mb: Optional memory change from start
    """
    if delta_mb is not None:
        sign = '+' if delta_mb and delta_mb >= 0 else ''
        delta_str = f" ({sign}{delta_mb:.1f}MB)"
    else:
        delta_str = ""
    logger.debug(f"🧠 {operation}: {memory_mb:.1f}MB{delta_str}")


def log_model_loaded(logger: logging.Logger, model_id: str, source: str,
                     context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log successful model loading.

    Args:
        logger: Logger instance
        model_id: Model identifier
        source: Loading source (e.g., "pool", "disk", "cache")
        context: Optional context
    """
    context_str = _format_context(context)
    logger.info(f"📦 Model loaded: {model_id} (from {source}){context_str}")


def log_prediction_stats(logger: logging.Logger, model_count: int, ensemble_count: int,
                        cache_stats: Optional[Dict[str, Any]] = None) -> None:
    """
    Log prediction generation statistics.

    Args:
        logger: Logger instance
        model_count: Number of individual models used
        ensemble_count: Number of ensemble predictions
        cache_stats: Optional cache performance stats
    """
    cache_str = ""
    if cache_stats:
        hit_rate = cache_stats.get('hit_rate', 0)
        cache_str = f", cache: {hit_rate:.1f}% hit rate"

    logger.info(f"🎯 Predictions generated: {model_count} models, {ensemble_count} ensembles{cache_str}")


def _format_context(context: Optional[Dict[str, Any]]) -> str:
    """
    Format context dictionary into readable string.

    Args:
        context: Context dictionary

    Returns:
        Formatted context string (e.g., " (key1=value1, key2=value2)")
    """
    if not context:
        return ""

    items = []
    for key, value in context.items():
        if isinstance(value, float):
            items.append(f"{key}={value:.2f}")
        elif isinstance(value, int):
            items.append(f"{key}={value}")
        else:
            items.append(f"{key}={value}")

    return f" ({', '.join(items)})"


# Logging level helpers
def set_component_log_level(logger: logging.Logger, level: str) -> None:
    """
    Set log level for a component.

    Args:
        logger: Logger instance
        level: Log level string ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    if level.upper() in level_map:
        logger.setLevel(level_map[level.upper()])
    else:
        logger.warning(f"Invalid log level: {level}. Using INFO.")


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)

    # Test all macros
    log_stage_start(logger, "TestComponent", {"param": "value"})
    log_progress(logger, 5, 10, "processing")
    log_success(logger, "Operation completed", {"result": 42})
    log_warning(logger, "Minor issue", {"code": 123}, "using fallback")
    log_error(logger, "Critical error", ValueError("test error"))
    log_cache_hit(logger, "feature", "AAPL_2024")
    log_memory_usage(logger, "computation", 512.5, 25.3)
    log_stage_end(logger, "TestComponent", 1.23, {"status": "ok"})