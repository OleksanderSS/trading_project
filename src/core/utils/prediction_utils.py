"""
Utilities for handling predictions and type conversions.
Centralizes common prediction processing logic.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def normalize_prediction(pred: float | int | list | tuple | np.ndarray) -> float:
    """
    Convert any prediction type to a normalized float value.

    Handles various prediction formats that different models might return:
    - float, int: direct conversion
    - list, tuple: takes last value
    - numpy arrays/scalars: converts to Python float

    Args:
        pred: Prediction value in any supported format

    Returns:
        float: Normalized prediction value

    Raises:
        TypeError: If prediction type cannot be normalized
    """
    # Direct numeric types
    if isinstance(pred, float):
        return pred

    if isinstance(pred, int):
        return float(pred)

    # Sequence types - take last element
    if isinstance(pred, (list, tuple)):
        if len(pred) == 0:
            logger.warning("Empty sequence prediction, returning 0.0")
            return 0.0
        return float(pred[-1])

    # NumPy scalars and arrays
    if hasattr(pred, 'item'):  # numpy scalar
        return float(pred.item())

    if isinstance(pred, np.ndarray):
        if pred.size == 0:
            logger.warning("Empty array prediction, returning 0.0")
            return 0.0
        return float(pred.flat[-1])  # Last element

    # Unknown type
    raise TypeError(
        f"Cannot normalize prediction of type {type(pred).__name__}. "
        f"Supported types: float, int, list, tuple, numpy.ndarray"
    )


def normalize_predictions_batch(predictions: dict) -> dict:
    """
    Normalize a batch of predictions from multiple models.

    Args:
        predictions: Dict mapping model_id -> prediction

    Returns:
        dict: Dict mapping model_id -> normalized float

    Example:
        >>> preds = {
        ...     'model_1': [0.7, 0.3],
        ...     'model_2': np.array([0.6]),
        ...     'model_3': 0.65
        ... }
        >>> normalize_predictions_batch(preds)
        {'model_1': 0.3, 'model_2': 0.6, 'model_3': 0.65}
    """
    normalized = {}
    failed_models = []

    for model_id, pred in predictions.items():
        try:
            normalized[model_id] = normalize_prediction(pred)
        except TypeError as e:
            logger.warning(f"Failed to normalize prediction for {model_id}: {e}")
            failed_models.append(model_id)

    if failed_models:
        logger.warning(f"Could not normalize predictions for: {failed_models}")

    return normalized


def validate_prediction_value(value: float, min_val: float = -1.0, max_val: float = 1.0) -> bool:
    """
    Validate if a prediction value is within expected range.

    Args:
        value: Prediction value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        bool: True if value is within range
    """
    return min_val <= value <= max_val


def clamp_prediction(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """
    Clamp a prediction value to the specified range.

    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        float: Clamped value
    """
    return max(min_val, min(max_val, value))
