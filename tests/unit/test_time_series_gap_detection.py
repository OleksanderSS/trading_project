import pandas as pd

from src.validation.time_series_validator import TimeSeriesValidator
from src.validation.validators import UnifiedValidator


def test_validate_time_gaps_detects_a_real_missing_trading_day():
    """validate_time_gaps previously crashed with AttributeError
    (TradingCalendar has no get_trading_days method) - the real attribute
    is calendar.trading_days, a pre-generated DatetimeIndex to be sliced
    by date range."""
    dates = pd.date_range("2026-01-05", "2026-01-09", freq="D")
    df = pd.DataFrame({"close": range(len(dates))}, index=dates)
    df = df.drop(df.index[2])  # drop 2026-01-07 (a Wednesday)

    validator = TimeSeriesValidator()
    report = validator.validate_time_gaps(df)

    assert report["is_valid"] is False
    assert report["missing_points_count"] == 1


def test_validate_time_gaps_reports_valid_for_complete_business_days():
    full_dates = pd.bdate_range("2026-01-05", "2026-01-09")
    df = pd.DataFrame({"close": range(len(full_dates))}, index=full_dates)

    validator = TimeSeriesValidator()
    report = validator.validate_time_gaps(df)

    assert report["is_valid"] is True
    assert report["missing_points_count"] == 0


def test_unified_validator_check_time_continuity_flags_real_gaps():
    """UnifiedValidator._check_time_continuity previously read
    gaps.get('has_gaps')/gaps.get('gap_count'), keys validate_time_gaps
    never produces (real keys: is_valid/missing_points_count) - the check
    was a permanent silent no-op regardless of real gaps."""
    dates = pd.date_range("2026-01-05", "2026-01-09", freq="D")
    df = pd.DataFrame({"close": range(len(dates))}, index=dates)
    df = df.drop(df.index[2])

    validator = UnifiedValidator()
    issues = validator._check_time_continuity("test_data", df)

    assert len(issues) == 1
    assert "gaps" in issues[0]


def test_unified_validator_check_time_continuity_no_issues_for_complete_data():
    full_dates = pd.bdate_range("2026-01-05", "2026-01-09")
    df = pd.DataFrame({"close": range(len(full_dates))}, index=full_dates)

    validator = UnifiedValidator()
    issues = validator._check_time_continuity("test_data", df)

    assert issues == []
