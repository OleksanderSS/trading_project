# utils/trading_calendar.py

import holidays
import pandas as pd
from datetime import timedelta
from utils.logger import ProjectLogger
from config.calendar_config import CALENDAR_DEFAULTS
import yfinance as yf

logger = ProjectLogger.get_logger("TradingCalendar")

class TradingCalendar:
    def __init__(self,
                 trading_dates,
                 country=CALENDAR_DEFAULTS["country"],
                 years=None,
                 max_shift_days=CALENDAR_DEFAULTS["max_shift_days"],
                 earnings_days=None):
        #  Нормалandforцandя дат
        self.trading_dates = set(
            pd.to_datetime(trading_dates, errors="coerce").dropna().map(lambda d: d.normalize())
        )
        try:
            self.holidays = holidays.CountryHoliday(
                country,
                years=years or [CALENDAR_DEFAULTS["default_year"]],
                observed=True
            )
        except Exception as e:
            logger.error(f"[TradingCalendar] Неможливо create holidays for країни {country}: {e}")
            self.holidays = set()

        self.max_shift_days = max_shift_days

        # Межand календаря
        self.min_date = min(self.trading_dates) if self.trading_dates else pd.Timestamp("1900-01-01", tz="UTC")
        self.max_date = max(self.trading_dates) if self.trading_dates else pd.Timestamp("2100-01-01", tz="UTC")

        # Днand withвandтностand
        if earnings_days is None:
            self.earnings_days = set()
        else:
            self.earnings_days = set(pd.to_datetime(earnings_days).dropna().normalize())

        logger.info(f"[TradingCalendar]  Initialized with {len(self.trading_dates)} торговими даandми")
        logger.info(f"[TradingCalendar]  Днandв withвandтностand: {len(self.earnings_days)}")

    @classmethod
    def from_year(cls, year=CALENDAR_DEFAULTS["default_year"], country=CALENDAR_DEFAULTS["country"], tickers=None):
        trading_dates = cls.generate_trading_dates(
            start=f"{year}-01-01", end=f"{year}-12-31", country=country
        )
        earnings_days = []
        if tickers:
            for t in tickers:
                earnings_days += cls.fetch_earnings_dates(t)
        return cls(trading_dates, country=country, years=[year], earnings_days=earnings_days)

    def is_trading_day(self, date) -> bool:
        if pd.isna(date):
            return False
        return pd.to_datetime(date).normalize() in self.trading_dates

    def shift_to_next_trading_day(self, date) -> pd.Timestamp:
        if pd.isna(date):
            raise ValueError(" Некоректна даand: NaT or None")
        date = pd.to_datetime(date).normalize()
        original_date = date

        if self.is_trading_day(date):
            return date

        if date > self.max_date:
            logger.warning(f"[TradingCalendar] {date.date()} > max trading date  поверandю {self.max_date.date()}")
            logger.warning(f"[TradingCalendar] [WARN] {date.date()} > max trading date  поверandю {self.max_date.date()}")
            return self.max_date

        logger.debug(f"[TradingCalendar]  Зсув {original_date.date()} до наступного торгового дня")

        for _ in range(self.max_shift_days):
            date += timedelta(days=1)
            if self.is_trading_day(date):
                return date

        fallback = max([d for d in self.trading_dates if d <= original_date], default=self.min_date)
        logger.warning(f"[TradingCalendar] [WARN] Зсув {original_date.date()} not withнайwhereно  поверandю {fallback.date()}")
        return fallback

    def get_next_trading_date(self, reference_date, max_offset=10) -> pd.Timestamp:
        if pd.isna(reference_date):
            return None
        reference_date = pd.to_datetime(reference_date).normalize()
        for offset in range(1, max_offset + 1):
            candidate = reference_date + timedelta(days=offset)
            if self.is_trading_day(candidate):
                logger.debug(f"[TradingCalendar] [OK] Наступна торгова даand пandсля {reference_date.date()}  {candidate.date()}")
                return candidate
        logger.warning(f"[TradingCalendar] [WARN] Не withнайwhereно наступну торгову дату пandсля {reference_date.date()}")
        return None

    @staticmethod
    def generate_trading_dates(start="2000-01-01",
        end="2030-12-31",
        country=CALENDAR_DEFAULTS["country"]) -> pd.DatetimeIndex:
        all_days = pd.date_range(start=start, end=end, freq="B")
        try:
            holiday_set = holidays.CountryHoliday(
                country,
                years=range(pd.to_datetime(start).year, pd.to_datetime(end).year + 1),
                observed=True
            )
        except Exception as e:
            logger.error(f"[TradingCalendar] [ERROR] Неможливо create holidays: {e}")
            holiday_set = set()

        trading_days = [d.normalize() for d in all_days if d.normalize() not in holiday_set]
        logger.info(f"[TradingCalendar]  Згеnotровано {len(trading_days)} торгових дат with {start} до {end}")
        return pd.DatetimeIndex(trading_days)

    @staticmethod
    def fetch_earnings_dates(ticker: str):
        """
        Пandдтягує earningsдати for компанandї череwith Yahoo Finance.
        Для ETF (SPY, QQQ) поверandє порожнandй список.
        """
        if ticker.upper() in ["SPY", "QQQ"]:
            logger.info(f"[TradingCalendar]  {ticker}  ETF, notмає власних earnings")
            return []

        try:
            stock = yf.Ticker(ticker)
            cal = stock.get_earnings_dates(limit=10)
            return pd.to_datetime(cal.index).normalize().tolist()
        except Exception as e:
            logger.error(f"[TradingCalendar] [ERROR] Не вдалося отримати earnings for {ticker}: {e}")
            return []

# Кешований календар
_default_calendar = None

def align_to_trading_day(date, trading_calendar=None):
    global _default_calendar
    if pd.isna(date):
        return None
    date = pd.to_datetime(date).normalize()
    if trading_calendar is None:
        if _default_calendar is None:
            trading_dates = TradingCalendar.generate_trading_dates(
                start="2000-01-01", end="2030-12-31"
            )
            _default_calendar = TradingCalendar(trading_dates)
            logger.info("[TradingCalendar] [OK] Створено кешований календар")
        trading_calendar = _default_calendar
    return trading_calendar.shift_to_next_trading_day(date)