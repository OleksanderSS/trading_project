"""
Type Conversion Utilities: Centralized type conversion and normalization

Provides consistent handling of predictions, arrays, and data structures
across the trading pipeline. Eliminates scattered conversion logic and
ensures robust type handling.

Benefits:
- Eliminates type-related bugs
- Consistent prediction handling
- Robust array/dataframe conversions
- Centralized validation logic

Usage:
    from src.utils.type_conversion import normalize_prediction, ensure_array
    
    pred = normalize_prediction(model_output)  # Always returns float
    features = ensure_array(input_data)        # Always returns numpy array
"""

import numpy as np
import pandas as pd
from typing import Any, Union, Optional
import logging


logger = logging.getLogger(__name__)


def _handle_none_nan(pred: Any) -> Optional[float]:
    """Handle None and NaN values."""
    if pred is None or (isinstance(pred, float) and np.isnan(pred)):
        return 0.0
    return None


def _convert_numeric_types(pred: Any) -> Optional[float]:
    """Convert direct numeric types."""
    if isinstance(pred, (int, float)):
        return float(pred)
    return None


def _convert_numpy_array(pred: Any) -> Optional[float]:
    """Convert numpy arrays to float."""
    if isinstance(pred, np.ndarray):
        if pred.size == 0:
            return 0.0
        return float(pred.flatten()[-1])
    return None


def _convert_list_tuple(pred: Any) -> Optional[float]:
    """Convert lists and tuples to float."""
    if isinstance(pred, (list, tuple)):
        if not pred:
            return 0.0
        return float(pred[-1])
    return None


def _convert_pandas_series(pred: Any) -> Optional[float]:
    """Convert pandas Series to float."""
    if isinstance(pred, pd.Series):
        if pred.empty:
            return 0.0
        return float(pred.iloc[-1])
    return None


def _convert_numpy_scalar(pred: Any) -> Optional[float]:
    """Convert numpy scalars to float."""
    if hasattr(pred, 'item'):  # numpy scalar
        return float(pred.item())
    return None


def _handle_unknown_type(pred: Any, strict: bool) -> float:
    """Handle unknown prediction types."""
    if strict:
        raise TypeError(f"Cannot convert prediction of type {type(pred)} to float")
    else:
        logger.warning(f"Unknown prediction type {type(pred)}, returning 0.0")
        return 0.0


def _handle_conversion_error(pred: Any, error: Exception, strict: bool) -> float:
    """Handle conversion errors."""
    if strict:
        raise TypeError(f"Failed to convert prediction {pred} to float: {error}") from error
    else:
        logger.warning(f"Failed to convert prediction {pred} to float: {error}, returning 0.0")
        return 0.0


def normalize_prediction(pred: Any, strict: bool = False) -> float:
    """
    Convert any prediction format to standardized float.
    
    Handles common prediction formats from different models:
    - Single float/int values
    - Numpy arrays (takes last element)
    - Lists/tuples (takes last element)
    - Pandas Series (takes last element)
    - Numpy scalars
    
    Args:
        pred: Prediction value in any supported format
        strict: If True, raise error for unknown types instead of returning 0.0
    
    Returns:
        Prediction as float
        
    Raises:
        TypeError: If strict=True and type cannot be converted
        
    Examples:
        >>> normalize_prediction(0.75)
        0.75
        >>> normalize_prediction(np.array([0.1, 0.2, 0.8]))
        0.8
        >>> normalize_prediction([0.5, 0.7])
        0.7
    """
    try:
        # Handle None/NaN
        result = _handle_none_nan(pred)
        if result is not None:
            return result
        
        # Direct numeric types
        result = _convert_numeric_types(pred)
        if result is not None:
            return result
        
        # Numpy arrays
        result = _convert_numpy_array(pred)
        if result is not None:
            return result
        
        # Lists and tuples
        result = _convert_list_tuple(pred)
        if result is not None:
            return result
        
        # Pandas Series
        result = _convert_pandas_series(pred)
        if result is not None:
            return result
        
        # Numpy scalars
        result = _convert_numpy_scalar(pred)
        if result is not None:
            return result
        
        # Unknown type
        return _handle_unknown_type(pred, strict)
                
    except (ValueError, IndexError, TypeError) as e:
        return _handle_conversion_error(pred, e, strict)


def _convert_dataframe(data: Any) -> Optional[pd.DataFrame]:
    """Handle already DataFrame input."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return None


def _convert_dict_to_dataframe(data: Any) -> Optional[pd.DataFrame]:
    """Convert dict of arrays/lists to DataFrame."""
    if isinstance(data, dict):
        return pd.DataFrame(data)
    return None


def _convert_numpy_to_dataframe(data: Any, columns: Optional[list]) -> Optional[pd.DataFrame]:
    """Convert numpy array to DataFrame."""
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
        if columns and len(columns) == df.shape[1]:
            df.columns = columns
        return df
    return None


def _convert_list_to_dataframe(data: Any, columns: Optional[list]) -> Optional[pd.DataFrame]:
    """Convert list data to DataFrame."""
    if not isinstance(data, list):
        return None
        
    if not data:
        return pd.DataFrame()
    
    # List of dicts
    if isinstance(data[0], dict):
        return pd.DataFrame(data)
    
    # List of lists
    elif isinstance(data[0], (list, tuple)):
        return pd.DataFrame(data, columns=columns)
    
    # Simple list
    else:
        return pd.DataFrame(data, columns=columns or ['value'])


def _convert_series_to_dataframe(data: Any) -> Optional[pd.DataFrame]:
    """Convert pandas Series to DataFrame."""
    if isinstance(data, pd.Series):
        return data.to_frame()
    return None


def ensure_dataframe(data: Any, columns: Optional[list] = None) -> pd.DataFrame:
    """
    Convert various data formats to pandas DataFrame.
    
    Args:
        data: Input data in supported format
        columns: Optional column names for resulting DataFrame
        
    Returns:
        DataFrame representation of input
        
    Raises:
        ValueError: If data cannot be converted
        
    Examples:
        >>> ensure_dataframe([1, 2, 3])
           0
        0  1
        1  2
        2  3
        
        >>> ensure_dataframe({'a': [1, 2], 'b': [3, 4]})
           a  b
        0  1  3
        1  2  4
    """
    try:
        # Already a DataFrame
        result = _convert_dataframe(data)
        if result is not None:
            return result
        
        # Dict of arrays/lists
        result = _convert_dict_to_dataframe(data)
        if result is not None:
            return result
        
        # Numpy array
        result = _convert_numpy_to_dataframe(data, columns)
        if result is not None:
            return result
        
        # List data
        result = _convert_list_to_dataframe(data, columns)
        if result is not None:
            return result
        
        # Pandas Series
        result = _convert_series_to_dataframe(data)
        if result is not None:
            return result
        
        # Unknown type
        raise ValueError(f"Cannot convert {type(data)} to DataFrame")
            
    except Exception as e:
        raise ValueError(f"Failed to convert data to DataFrame: {e}") from e


def ensure_array(data: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Convert various data formats to numpy array.
    
    Args:
        data: Input data in supported format
        dtype: Desired numpy dtype for output
        
    Returns:
        Numpy array representation
        
    Raises:
        ValueError: If data cannot be converted
        
    Examples:
        >>> ensure_array([1, 2, 3])
        array([1., 2., 3.], dtype=float32)
        
        >>> ensure_array(pd.DataFrame({'a': [1, 2], 'b': [3, 4]}))
        array([[1., 3.],
               [2., 4.]], dtype=float32)
    """
    try:
        # Already a numpy array
        if isinstance(data, np.ndarray):
            return data.astype(dtype, copy=False)
        
        # Pandas DataFrame/Series
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values.astype(dtype)
        
        # List/tuple
        elif isinstance(data, (list, tuple)):
            return np.array(data, dtype=dtype)
        
        # Scalar
        elif isinstance(data, (int, float)):
            return np.array([data], dtype=dtype)
        
        else:
            raise ValueError(f"Cannot convert {type(data)} to numpy array")
            
    except Exception as e:
        raise ValueError(f"Failed to convert data to numpy array: {e}") from e


def safe_divide(a: Union[float, np.ndarray], b: Union[float, np.ndarray], 
                default: float = 0.0) -> Union[float, np.ndarray]:
    """
    Safe division that handles division by zero.
    
    Args:
        a: Numerator
        b: Denominator
        default: Value to return when b == 0
        
    Returns:
        a/b or default if b == 0
        
    Examples:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
    """
    try:
        # Handle numpy arrays
        if isinstance(b, np.ndarray):
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.divide(a, b)
                result[np.isinf(result) | np.isnan(result)] = default
                return result
        else:
            return a / b if b != 0 else default
    except Exception:
        return default


def clamp_value(value: float, min_val: float = -np.inf, max_val: float = np.inf) -> float:
    """
    Clamp value to specified range.
    
    Args:
        value: Input value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Clamped value
        
    Examples:
        >>> clamp_value(10, 0, 5)
        5.0
        >>> clamp_value(-1, 0, 5)
        0.0
    """
    return max(min_val, min(max_val, value))


def normalize_array(arr: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize array using specified method.
    
    Args:
        arr: Input array
        method: Normalization method ('zscore', 'minmax', 'robust')
        
    Returns:
        Normalized array
        
    Raises:
        ValueError: For unknown normalization method
    """
    if method == 'zscore':
        mean = np.mean(arr)
        std = np.std(arr)
        return (arr - mean) / std if std > 0 else arr - mean
    
    elif method == 'minmax':
        min_val = np.min(arr)
        max_val = np.max(arr)
        return (arr - min_val) / (max_val - min_val) if max_val > min_val else arr - min_val
    
    elif method == 'robust':
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        return (arr - median) / mad if mad > 0 else arr - median
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def validate_numeric_range(value: float, min_val: Optional[float] = None, 
                          max_val: Optional[float] = None, name: str = "value") -> None:
    """
    Validate that numeric value is within acceptable range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (None = no minimum)
        max_val: Maximum allowed value (None = no maximum)
        name: Name of value for error messages
        
    Raises:
        ValueError: If value is outside allowed range
    """
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} {value} is below minimum {min_val}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} {value} is above maximum {max_val}")


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format float as percentage string.
    
    Args:
        value: Value to format (0.5 = 50%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
        
    Examples:
        >>> format_percentage(0.856)
        '85.6%'
        >>> format_percentage(0.1, decimals=0)
        '10%'
    """
    return f"{value * 100:.{decimals}f}%"