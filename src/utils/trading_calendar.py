# src/utils/trading_calendar.py

from datetime import date, datetime, timedelta
from typing import List, Set, Optional, Union

import holidays
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay

from src.core.logging.logger import ProjectLogger

# Initialize logger for the module
logger = ProjectLogger.get_logger("TradingCalendar")

class TradingCalendar:
    """
    A comprehensive utility for handling market trading days, holidays, and earnings dates.
    Uses pre-generated trading days index for efficient lookups.
    """

    def __init__(self, start_year: int = 2020, end_year: int = datetime.now().year + 1, country: str = 'US'):
        self.country = country
        self.start_year = start_year
        self.end_year = end_year
        self.holidays: Set[date] = self._get_holidays()
        self.trading_days: pd.DatetimeIndex = self._generate_trading_days()
        self.earnings_dates: Set[date] = set()
        logger.info(f"TradingCalendar initialized for {country} from {start_year} to {end_year}. Found {len(self.holidays)} holidays.")

    def _get_holidays(self) -> Set[date]:
        """Fetches holidays for the specified country and year range."""
        try:
            return set(holidays.CountryHoliday(
                self.country, years=range(self.start_year, self.end_year + 1), observed=True
            ))
        except Exception as e:
            logger.error(f"Could not fetch holidays for country '{self.country}'. Defaulting to empty set. Error: {e}", exc_info=True)
            return set()

    def _generate_trading_days(self) -> pd.DatetimeIndex:
        """Generates a DatetimeIndex of business days, excluding holidays."""
        business_days = pd.bdate_range(start=f'{self.start_year}-01-01', end=f'{self.end_year}-12-31')
        trading_days = business_days.drop(self.holidays, errors='ignore')
        logger.info(f"Generated {len(trading_days)} trading days.")
        return trading_days

    def is_trading_day(self, day: Union[date, datetime, str]) -> bool:
        """Checks if a given date is a trading day using the pre-generated index."""
        try:
            dt = pd.to_datetime(day).normalize()
            return dt in self.trading_days
        except Exception as e:
            logger.error(f"Could not parse date {day}: {e}")
            return False

    def get_next_trading_day(self, from_date: Union[date, datetime, str]) -> date:
        """Returns the next trading day after the given date from the index."""
        dt = pd.to_datetime(from_date).normalize()
        
        # Find the first trading day strictly after the given date
        future_days = self.trading_days[self.trading_days > dt]
        if not future_days.empty:
            return future_days[0].date()
        
        # Fallback if outside pre-generated range (not efficient but safe)
        next_day = (dt + BDay(1)).date()
        while next_day in self.holidays:
            next_day = (pd.to_datetime(next_day) + BDay(1)).date()
        return next_day

    def get_previous_trading_days(self, from_date: Union[date, datetime, str], count: int) -> List[date]:
        """Returns a list of the previous `count` trading days from a given date using the index."""
        dt = pd.to_datetime(from_date).normalize()
        
        try:
            loc = self.trading_days.get_loc(dt, method='pad')
        except KeyError:
            if dt < self.trading_days[0]:
                logger.warning(f"Date {dt} is before the start of the calendar.")
                return []
            loc = len(self.trading_days) - 1

        # Adjust indices to exclude current day if it matches dt
        if self.trading_days[loc] >= dt:
            end_index = loc
        else:
            end_index = loc + 1
            
        start_index = max(0, end_index - count)
        return [d.date() for d in self.trading_days[start_index:end_index]]

    def fetch_and_set_earnings_dates(self, tickers: List[str]):
        """Fetches earnings dates for tickers and updates the calendar."""
        all_earnings = set()
        for ticker_str in tickers:
            try:
                ticker = yf.Ticker(ticker_str)
                earnings = ticker.get_earnings_dates(limit=20)
                if earnings is not None and not earnings.empty:
                    dates = pd.to_datetime(earnings.index).normalize().date
                    all_earnings.update(dates)
            except Exception as e:
                logger.error(f"Failed to fetch earnings for ticker '{ticker_str}': {e}")
        
        self.earnings_dates.update(all_earnings)
        logger.info(f"Updated earnings dates. Total unique dates: {len(self.earnings_dates)}")

    def is_earnings_day(self, day: Union[date, datetime, str], ticker: Optional[str] = None) -> bool:
        """Checks if a given date is an earnings announcement day."""
        try:
            dt = pd.to_datetime(day).normalize().date()
            return dt in self.earnings_dates
        except Exception:
            return False