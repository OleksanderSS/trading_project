# utils/trading_calendar.py

import holidays
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from config.calendar_config import CALENDAR_DEFAULTS
import yfinance as yf
import os
import sys

# Додаємо шлях до project root for andмпортandв
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger("TradingCalendar")

class TradingCalendar:
    def __init__(self,
                 trading_dates,
                 country=CALENDAR_DEFAULTS["country"],
                 years=None,
                 max_shift_days=CALENDAR_DEFAULTS["max_shift_days"],
                 earnings_days=None):
        # Робимо all дати timezone-naive for сумandсностand
        self.trading_dates = set(
            pd.to_datetime(trading_dates,
                errors="coerce").dropna().map(lambda d: d.normalize().tz_convert(None) if hasattr(d.normalize(),
                'tz') and d.normalize().tz is not None else d.normalize())
        )
        try:
            self.holidays = holidays.CountryHoliday(
                country,
                years=years or [CALENDAR_DEFAULTS["default_year"]],
                observed=True
            )
        except Exception as e:
            logger.error(f"[TradingCalendar] [ERROR] Неможливо create holidays for країни {country}: {e}")
            self.holidays = set()

        self.max_shift_days = max_shift_days

        #  Межand календаря
        self.min_date = min(self.trading_dates) if self.trading_dates else pd.Timestamp("1900-01-01")
        self.max_date = max(self.trading_dates) if self.trading_dates else pd.Timestamp("2100-01-01")

        #  Днand withвandтностand
        if earnings_days is None:
            self.earnings_days = set()
        else:
            self.earnings_days = set(pd.to_datetime(earnings_days,
                errors="coerce").dropna().map(lambda d: d.normalize().tz_convert(None) if hasattr(d.normalize(),
                'tz') and d.normalize().tz is not None else d.normalize()))

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
        # Робимо дату timezone-naive for сумandсностand
        date_normalized = pd.to_datetime(date, errors="coerce")
        if hasattr(date_normalized, 'tz') and date_normalized.tz is not None:
            date_normalized = date_normalized.tz_convert(None)
        date_normalized = date_normalized.normalize()
        return date_normalized in self.trading_dates

    def shift_to_next_trading_day(self, date) -> pd.Timestamp:
        if pd.isna(date):
            raise ValueError("[ERROR] Некоректна даand: NaT or None")
        # Робимо дату timezone-naive for сумandсностand
        date = pd.to_datetime(date, errors="coerce")
        if hasattr(date, 'tz') and date.tz is not None:
            date = date.tz_convert(None)
        date = date.normalize()
        original_date = date

        if self.is_trading_day(date):
            return date

        if date > self.max_date:
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
        # Робимо дату timezone-naive for сумandсностand
        reference_date = pd.to_datetime(reference_date, errors="coerce")
        if hasattr(reference_date, 'tz') and reference_date.tz is not None:
            reference_date = reference_date.tz_convert(None)
        reference_date = reference_date.normalize()
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
        """
        try:
            stock = yf.Ticker(ticker)
            # [OK] ВИПРАВЛЕННЯ: Збandльшення лandмandту до 50.
            # This доwithволить отримати бandльше andсторичних and майбутнandх дат withвandтностand
            cal = stock.get_earnings_dates(limit=50)

            # Робимо дати timezone-naive for сумandсностand
            earnings_dates = pd.to_datetime(cal.index, errors="coerce")
            earnings_dates = earnings_dates.map(lambda d: d.normalize().tz_convert(None) if hasattr(d.normalize(),
                'tz') and d.normalize().tz is not None else d.normalize()).tolist()

            if not earnings_dates:
                # [IDEA] Попередження, якщо API not повернув дат навandть andwith великим лandмandтом
                logger.warning(f"[TradingCalendar] [WARN] Не withнайwhereно жодної дати withвandтностand for {ticker} (limit=50).")

            return earnings_dates

        except Exception as e:
            # [IDEA] ВИПРАВЛЕННЯ: Покращеnot logging errors for дandагностики
            logger.error(f"[TradingCalendar] [ERROR] КРИТИЧНА ПОМИЛКА API. Не вдалося отримати earnings for {ticker}: {e}")
            return []

# Кешований календар
_default_calendar = None

def align_to_trading_day(date, trading_calendar=None):
    global _default_calendar
    if pd.isna(date):
        return None
    # Перевandряємо чи даand вже має timezone and обробляємо вandдповandдно
    date = pd.to_datetime(date, errors="coerce")
    if hasattr(date, 'tz') and date.tz is not None:
        date = date.tz_convert(None)
    date = date.normalize()
    if trading_calendar is None:
        if _default_calendar is None:
            trading_dates = TradingCalendar.generate_trading_dates(
                start="2000-01-01", end="2030-12-31"
            )
            _default_calendar = TradingCalendar(trading_dates)
            logger.info("[TradingCalendar] [OK] Створено кешований календар")
        trading_calendar = _default_calendar
    return trading_calendar.shift_to_next_trading_day(date)

def get_last_candles_info(timeframe: str = "1d", tickers: list = None) -> dict:
    """
    Отримує andнформацandю про осandннand свandчки for allх тandкерandв or конкретних
    
    Args:
        timeframe: andймфрейм ("15m", "60m", "1d")
        tickers: список тandкерandв (якщо None, то all with конфandгурацandї)
    
    Returns:
        dict with andнформацandєю про осandннand свandчки
    """
    try:
        from config.config import TICKERS
        if tickers is None:
            tickers = list(TICKERS.keys())
        
        candle_info = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                
                # Отримуємо данand forлежно вandд andймфрейму
                if timeframe == "15m":
                    data = stock.history(period="60d", interval="15m")
                elif timeframe == "60m":
                    data = stock.history(period="60d", interval="60m")
                else:  # 1d
                    data = stock.history(period="2y", interval="1d")
                
                if data.empty:
                    logger.warning(f"[TradingCalendar] [WARN] Немає data for {ticker} {timeframe}")
                    continue
                
                # Осandння свandчка
                last_candle = data.iloc[-1]
                
                # RSI (14 periodandв)
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = 100 - (100 / (1 + gain / loss))
                rsi_value = rs.iloc[-1] if not pd.isna(rs.iloc[-1]) else "N/A"
                
                # Баwithовand покаwithники
                candle_info[ticker.lower()] = {
                    'datetime': last_candle.name,
                    'close': round(last_candle['Close'], 2),
                    'volume': int(last_candle['Volume']) if not pd.isna(last_candle['Volume']) else 0,
                    'high': round(last_candle['High'], 2),
                    'low': round(last_candle['Low'], 2),
                    'open': round(last_candle['Open'], 2),
                    'rsi': round(rsi_value, 1) if isinstance(rsi_value, (int, float)) else "N/A",
                    'change_pct': round(((last_candle['Close'] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100, 2) if len(data) > 1 else 0,
                    'timeframe': timeframe
                }
                
                logger.debug(f"[TradingCalendar] {ticker} {timeframe}: "
                           f"Close={candle_info[ticker.lower()]['close']}, "
                           f"RSI={candle_info[ticker.lower()]['rsi']}, "
                           f"Volume={candle_info[ticker.lower()]['volume']}")
                
            except Exception as e:
                logger.error(f"[TradingCalendar] [ERROR] Error отримання data for {ticker}: {e}")
                candle_info[ticker.lower()] = {
                    'datetime': 'N/A',
                    'close': 'N/A',
                    'volume': 0,
                    'high': 'N/A',
                    'low': 'N/A',
                    'open': 'N/A',
                    'rsi': 'N/A',
                    'change_pct': 0,
                    'timeframe': timeframe
                }
        
        logger.info(f"[TradingCalendar] [OK] Отримано andнформацandю про свandчки {timeframe}: {list(candle_info.keys())}")
        return candle_info
        
    except Exception as e:
        logger.error(f"[TradingCalendar] [ERROR] Критична error get_last_candles_info: {e}")
        return {}

def calculate_gaps_and_impact(candle_info: dict, timeframe: str = "1d", news_time: datetime = None) -> dict:
    """
    Обчислює гепи (GAP) and andмпакт (IMPACT) на основand candles
    
    Args:
        candle_info: словник with andнформацandєю про свandчки
        timeframe: andймфрейм for аналandwithу
        news_time: час публandкацandї новини (for обчислення andмпакту)
        
    Returns:
        dict with обчисленими гепами and andмпакandми
    """
    try:
        if not candle_info:
            return {}
        
        gaps_impact = {}
        
        for ticker, data in candle_info.items():
            if data.get('datetime') == 'N/A':
                continue
                
            # Обчислення гепandв (рandwithниця мandж свandчками)
            prev_close = data.get('prev_close', data.get('close', 0))  # Попередня цandна closing
            current_close = data.get('close', 0)
            current_open = data.get('open', 0)
            
            if prev_close > 0 and current_close > 0:
                # Геп вandд попередньої цandни closing до поточної вandдкриття
                gap_open = ((current_open - prev_close) / prev_close) * 100
                # Геп вandд попередньої цandни closing до поточної closing  
                gap_close = ((current_close - prev_close) / prev_close) * 100
                
                # Загальний геп
                gap_total = ((current_close - prev_close) / prev_close) * 100
                
                # Імпакт гепу
                if abs(gap_total) >= 2.0:  # Великий геп > 2%
                    gap_impact = "HIGH"
                elif abs(gap_total) >= 1.0:  # Середнandй геп > 1%
                    gap_impact = "MEDIUM"
                elif abs(gap_total) >= 0.5:  # Малий геп > 0.5%
                    gap_impact = "LOW"
                else:
                    gap_impact = "MINIMAL"
            else:
                gap_open = 0
                gap_close = 0
                gap_total = 0
                gap_impact = "NONE"
            
            # Обчислення волатильностand
            high_low = data.get('high', 0) - data.get('low', 0)
            volatility = (high_low / data.get('close', 1)) * 100 if data.get('close', 1) > 0 else 0
            
            # RSI сигнали
            rsi = data.get('rsi', 50)
            if rsi == 'N/A':
                rsi_signal = "NEUTRAL"
            elif rsi >= 70:
                rsi_signal = "OVERBOUGHT"
            elif rsi <= 30:
                rsi_signal = "OVERSOLD"
            else:
                rsi_signal = "NEUTRAL"
            
            # Обсяг сигнали
            volume = data.get('volume', 0)
            volume_signal = "NORMAL"
            if volume > 0:
                # Порandвнюємо with середнandм обсягом (спрощено)
                avg_volume = 5000000  # Баwithовий середнandй обсяг
                if volume >= avg_volume * 1.5:
                    volume_signal = "HIGH"
                elif volume >= avg_volume * 1.2:
                    volume_signal = "MEDIUM"
                elif volume <= avg_volume * 0.8:
                    volume_signal = "LOW"
            
            # Обчислення andмпакту (рandwithниця покаwithникandв до/пandсля новини)
            impact_before = "NONE"
            impact_after = "NONE"
            
            if news_time and data.get('datetime') != 'N/A':
                # Покаwithники до публandкацandї новини (осandння доступна свandчка)
                impact_before = {
                    'rsi': rsi,
                    'rsi_signal': rsi_signal,
                    'volume': volume,
                    'volume_signal': volume_signal,
                    'price': current_close,
                    'volatility_pct': volatility,
                    'gap_total_pct': gap_total,
                    'gap_impact': gap_impact
                }
                
                # Для andмпакту пandсля новини потрandбнand першand свandчки пandсля публandкацandї
                # This can отримати тandльки при наступному виклику функцandї
                # or add параметр for отримання candles пandсля певної дати
                logger.info(f"[TradingCalendar]  Імпакт до новини {ticker.upper()}: "
                           f"RSI={rsi} ({rsi_signal}), Gap={gap_total:.2f}% ({gap_impact})")
            
            # Загальний andмпакт
            if gap_impact in ["HIGH", "MEDIUM"] or rsi_signal in ["OVERBOUGHT", "OVERSOLD"] or volume_signal in ["HIGH", "MEDIUM"]:
                overall_impact = "HIGH"
            elif gap_impact in ["LOW"] or rsi_signal != "NEUTRAL" or volume_signal in ["LOW"]:
                overall_impact = "MEDIUM"
            else:
                overall_impact = "LOW"
            
            gaps_impact[ticker] = {
                'gap_open_pct': round(gap_open, 2),
                'gap_close_pct': round(gap_close, 2),
                'gap_total_pct': round(gap_total, 2),
                'gap_impact': gap_impact,
                'volatility_pct': round(volatility, 2),
                'rsi': rsi,
                'rsi_signal': rsi_signal,
                'volume': volume,
                'volume_signal': volume_signal,
                'overall_impact': overall_impact,
                'timeframe': timeframe,
                'datetime': data.get('datetime'),
                'price': data.get('close'),
                'change_pct': data.get('change_pct', 0),
                'impact_before_news': impact_before,
                'impact_after_news': impact_after,
                'news_time': news_time
            }
            
            logger.debug(f"[TradingCalendar] {ticker.upper()}: "
                       f"Gap={gap_total:.2f}%, Impact={gap_impact}, "
                       f"RSI={rsi} ({rsi_signal}), Vol={volatility:.1f}%")
        
        logger.info(f"[TradingCalendar] [OK] Обчислено гепи and andмпакт for {timeframe}: {list(gaps_impact.keys())}")
        return gaps_impact
        
    except Exception as e:
        logger.error(f"[TradingCalendar] [ERROR] Error обчислення гепandв and andмпакту: {e}")
        return {}