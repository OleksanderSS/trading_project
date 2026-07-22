"""
Public API for market data operations in DEAN OS.

This module exposes public functions for loading and preparing market data,
avoiding the need to import private functions across modules.
"""
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def read_market_frame(pd: Any, path: Path) -> Any:
    """
    Read market data from CSV or Parquet file.
    
    Args:
        pd: pandas module
        path: Path to market data file (.csv or .parquet)
    
    Returns:
        DataFrame with market data
    
    Raises:
        ValueError: If file type is not supported
    """
    from dean_os.draft.dean_os_agent_system_v7.dean_os.dean_paths import DeanPaths

    try:
        return DeanPaths.load_data_file(path)
    except Exception as exc:
        raise ValueError(f"Failed to load market data from {path}: {exc}")


def prepare_market_frame(pd: Any, frame: Any, close_col: str = "close", datetime_col: str = "datetime") -> Any:
    """
    Prepare market data frame with standardized columns for analysis.
    
    Args:
        pd: pandas module
        frame: Raw market data DataFrame
        close_col: Name of close price column
        datetime_col: Name of datetime column
    
    Returns:
        Prepared DataFrame with _dean_close, _dean_datetime, _dean_ticker columns
    
    Raises:
        ValueError: If required columns are missing
    """
    def _resolve_column(frame: Any, col: str) -> str:
        """Resolve column name with case-insensitive matching."""
        if col in frame.columns:
            return col
        for column in frame.columns:
            if str(column).lower() == col.lower():
                return column
        return col

    def _first_existing_column(frame: Any, candidates: list[str]) -> str | None:
        """Find first existing column from candidates."""
        for candidate in candidates:
            if candidate in frame.columns:
                return candidate
        return None

    close_col = _resolve_column(frame, close_col)
    datetime_col = _resolve_column(frame, datetime_col)
    if close_col not in frame.columns:
        raise ValueError(f"Missing close column: {close_col}")
    if datetime_col not in frame.columns:
        raise ValueError(f"Missing datetime column: {datetime_col}")
    prepared = frame.copy()
    prepared["_dean_close"] = pd.to_numeric(prepared[close_col], errors="coerce")
    prepared["_dean_datetime"] = pd.to_datetime(prepared[datetime_col], utc=True, errors="coerce")
    ticker_col = _first_existing_column(prepared, ["ticker", "symbol", "Ticker", "Symbol"])
    prepared["_dean_ticker"] = prepared[ticker_col].astype(str).str.upper() if ticker_col else ""
    prepared = prepared.dropna(subset=["_dean_close", "_dean_datetime"])
    return prepared.sort_values("_dean_datetime")


def parse_datetime(value: str) -> datetime:
    """
    Parse datetime string with UTC timezone.
    
    Args:
        value: Datetime string (ISO format)
    
    Returns:
        datetime object with UTC timezone
    """
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
