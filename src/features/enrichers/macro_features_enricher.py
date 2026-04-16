import pandas as pd
import pandas_datareader.data as web
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

from src.config.unified_config_manager import get_current_config
from src.features.enrichers.base import BaseEnricher
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("MacroFeaturesEnricher")

class MacroFeaturesEnricher(BaseEnricher):
    """
    Enriches the DataFrame with macroeconomic indicators from FRED.
    Implements caching to avoid repeated downloads.
    """
    
    @property
    def name(self) -> str:
        return "macro_features"
    
    @property
    def priority(self) -> int:
        """Execution order - run after technical analysis (20) but before NLP (30)"""
        return 27

    def __init__(self, config: dict = None):
        """Initialize with optional config dict from FeatureOrchestrator"""
        # ✅ FIX: Правильний шлях до конфігурації
        # Спочатку пробуємо enrichment.macro_features.macro_fred_series
        config_manager = get_current_config()
        self.config = config_manager.get('enrichment.macro_features.macro_fred_series', {})
        
        # Fallback: якщо не знайдено, пробуємо macro_features.macro_fred_series
        if not self.config:
            self.config = config_manager.get('macro_features.macro_fred_series', {})
        
        self.cache_path = Path('./cache') / 'macro_data.parquet'
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.config:
            logger.warning("Configuration for macro features ('macro_fred_series') not found.")
        else:
            logger.info(f"✅ MacroFeaturesEnricher initialized with {len(self.config)} series")

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Adds macro features to the DataFrame.
        First tries to use macro_data from kwargs (collected in Stage 1),
        then falls back to FRED API if needed.

        Args:
            df: DataFrame with a DatetimeIndex.
            **kwargs: May contain 'macro_data' from Stage 1

        Returns:
            DataFrame with added macro features.
        """
        if df.empty:
            return df

        # ✅ Перевіряємо індекс - має бути DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df = df.set_index('datetime')
                logger.info("Converted 'datetime' column to DatetimeIndex")
            else:
                logger.error("Cannot enrich macro features: no DatetimeIndex or 'datetime' column")
                return df
        
        start_date = df.index.min()
        end_date = df.index.max()

        # ✅ СПЕЦІАЛЬНА ФІЛЬТРАЦІЯ (Розумна логіка)
        import json
        config_manager = get_current_config()
        params_path = config_manager.get_runtime_params_path()
        if params_path.exists():
            try:
                with open(params_path, 'r') as f:
                    runtime_params = json.load(f)
                test_ticker = runtime_params.get('test_mode', {}).get('test_ticker')
                if test_ticker and 'ticker' in df.columns:
                    logger.info(f"🧪 MacroFeaturesEnricher: фільтрація для тікера {test_ticker}")
                    df = df[df['ticker'] == test_ticker]
            except Exception:
                pass

        unique_dates = len(df.index.unique())
        logger.info(f"MacroFeaturesEnricher processing {len(df)} records ({unique_dates} unique dates) from {start_date} to {end_date}")

        # ✅ ПРІОРИТЕТ: Використовуємо дані з Stage 1, якщо є
        macro_data = kwargs.get('macro_data')
        if macro_data is not None and isinstance(macro_data, pd.DataFrame) and not macro_data.empty:
            logger.info(f"✅ Using macro_data from Stage 1 ({len(macro_data)} rows)")
            
            # ✅ PIVOT: Розгортаємо series_id в окремі колонки
            if 'series_id' in macro_data.columns and 'value' in macro_data.columns:
                # Знаходимо колонку з датою
                date_col = None
                for col in ['date', 'datetime', 'realtime_start']:
                    if col in macro_data.columns:
                        date_col = col
                        break
                
                if date_col:
                    macro_data[date_col] = pd.to_datetime(macro_data[date_col])
                    # Pivot: series_id стають колонками
                    macro_pivoted = macro_data.pivot_table(
                        index=date_col,
                        columns='series_id',
                        values='value',
                        aggfunc='last'  # Беремо останнє значення якщо є дублікати
                    )
                    # Додаємо префікс FRED_
                    macro_pivoted.columns = [f'FRED_{col}' for col in macro_pivoted.columns]
                    macro_data = macro_pivoted
                    logger.info(f"✅ Pivoted macro data into {len(macro_data.columns)} FRED columns")
                else:
                    logger.warning("No date column found in macro_data for pivoting")
        else:
            logger.info("No macro_data in kwargs, loading from FRED API...")
            macro_data = self._load_macro_data(start_date, end_date)
        
        if macro_data.empty:
            logger.warning("Could not load macro data. Skipping enrichment.")
            return df

        logger.info("Joining macro data with the main DataFrame...")
        # Видаляємо дублікати індексу для коректного reindex
        if not isinstance(macro_data.index, pd.DatetimeIndex):
            if 'datetime' in macro_data.columns:
                macro_data = macro_data.set_index('datetime')
            elif 'date' in macro_data.columns:
                macro_data = macro_data.set_index('date')
                
        # Ensure macro_data is sorted and deduplicated
        macro_data = macro_data.sort_index()
        macro_data = macro_data[~macro_data.index.duplicated(keep='last')]
        
        # ✅ КРИТИЧНИЙ FIX: Нормалізуємо timezone/precision в macro_data.index ДО reset_index()
        if isinstance(macro_data.index, pd.DatetimeIndex):
            # Remove timezone if present
            if macro_data.index.tz is not None:
                macro_data.index = macro_data.index.tz_localize(None)
            # Convert to ns precision
            if macro_data.index.dtype != 'datetime64[ns]':
                macro_data.index = macro_data.index.astype('datetime64[ns]')
        
        # ✅ FIX: Якщо df має дублікати індексу (кілька тікерів), використовуємо merge замість reindex
        if df.index.duplicated().any():
            logger.info("Detected duplicate index labels (multiple tickers). Using merge instead of reindex.")
            # Зберігаємо оригінальний індекс
            df_reset = df.reset_index()
            macro_reset = macro_data.reset_index()
            
            # Визначаємо назву колонки datetime
            datetime_col = None
            for col in ['datetime', 'index', 'date']:
                if col in df_reset.columns:
                    datetime_col = col
                    break
            
            if datetime_col is None:
                logger.error("❌ Cannot find datetime column after reset_index")
                return df
            
            # Перейменовуємо в 'datetime' якщо потрібно
            if datetime_col != 'datetime':
                df_reset = df_reset.rename(columns={datetime_col: 'datetime'})
            
            # ✅ Нормалізуємо timezone в df_reset
            if 'datetime' in df_reset.columns:
                if pd.api.types.is_datetime64_any_dtype(df_reset['datetime']):
                    # Remove timezone if present
                    if hasattr(df_reset['datetime'].dtype, 'tz') and df_reset['datetime'].dt.tz is not None:
                        df_reset['datetime'] = df_reset['datetime'].dt.tz_localize(None)
                    # Convert to ns precision
                    if df_reset['datetime'].dtype != 'datetime64[ns]':
                        df_reset['datetime'] = df_reset['datetime'].astype('datetime64[ns]')
            
            # Визначаємо назву колонки datetime в macro_reset
            macro_datetime_col = None
            for col in ['datetime', 'index', 'date']:
                if col in macro_reset.columns:
                    macro_datetime_col = col
                    break
            
            if macro_datetime_col and macro_datetime_col != 'datetime':
                macro_reset = macro_reset.rename(columns={macro_datetime_col: 'datetime'})
            
            # ✅ Нормалізуємо timezone в macro_reset
            if 'datetime' in macro_reset.columns:
                if pd.api.types.is_datetime64_any_dtype(macro_reset['datetime']):
                    # Remove timezone if present
                    if hasattr(macro_reset['datetime'].dtype, 'tz') and macro_reset['datetime'].dt.tz is not None:
                        macro_reset['datetime'] = macro_reset['datetime'].dt.tz_localize(None)
                    # Convert to ns precision
                    if macro_reset['datetime'].dtype != 'datetime64[ns]':
                        macro_reset['datetime'] = macro_reset['datetime'].astype('datetime64[ns]')
            
            # Merge з forward fill
            df_merged = pd.merge_asof(
                df_reset.sort_values('datetime'),
                macro_reset.sort_values('datetime'),
                on='datetime',
                direction='backward'
            )
            
            # Відновлюємо індекс
            if 'datetime' in df_merged.columns:
                df = df_merged.set_index('datetime')
            else:
                logger.error("❌ 'datetime' column missing after merge_asof")
                df = df_merged
        else:
            # Старий метод для випадку без дублікатів
            macro_dates = macro_data.index.unique()
            aligned_macro_data = macro_data.loc[macro_dates].reindex(df.index, method='ffill')
            
            # Об'єднуємо без Join, щоб не плодити рядки при дублікатах міток часу
            for col in aligned_macro_data.columns:
                if col not in df.columns:  # Не перезаписуємо існуючі колонки
                    df[col] = aligned_macro_data[col].values
        
        df = df.bfill()
        
        # ✅ Додатковий forward fill для місячних показників (CPIAUCSL, UNRATE)
        fred_cols = [col for col in df.columns if col.startswith('FRED_')]
        if fred_cols:
            # ✅ КРИТИЧНИЙ FIX: Конвертуємо в numeric перед операціями
            for col in fred_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df[fred_cols] = df[fred_cols].ffill(limit=60)  # Forward fill до 60 днів (2 місяці)
            remaining_nans = df[fred_cols].isna().sum()
            if remaining_nans.any():
                logger.warning(f"Some FRED columns still have NaN after ffill: {remaining_nans[remaining_nans > 0].to_dict()}")
                # Заповнюємо залишкові NaN медіаною
                df[fred_cols] = df[fred_cols].fillna(df[fred_cols].median())

        logger.info(f"✅ Macro features successfully added. Final shape: {df.shape}")
        return df

    def _load_macro_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        if self._is_cache_valid(start_date, end_date):
            logger.info(f"Loading macro data from cache: {self.cache_path}")
            return pd.read_parquet(self.cache_path)

        logger.info("Cache not found or outdated. Loading data from FRED...")
        series_ids = list(self.config.values())
        series_names = list(self.config.keys())

        try:
            fred_data = web.DataReader(series_ids, 'fred', start_date, end_date)
            fred_data.columns = series_names
            
            fred_data.to_parquet(self.cache_path)
            logger.info(f"Macro data saved to cache: {self.cache_path}")
            return fred_data
        except Exception as e:
            logger.error(f"Error loading data from FRED: {e}", exc_info=True)
            return pd.DataFrame()

    def _is_cache_valid(self, start_date: datetime, end_date: datetime) -> bool:
        if not self.cache_path.exists():
            return False
        
        try:
            cached_df = pd.read_parquet(self.cache_path)
            if cached_df.index.min() <= start_date and cached_df.index.max() >= end_date:
                logger.info("Cache fully covers the required date range.")
                return True
            else:
                logger.info("Date range in cache is insufficient. Refresh required.")
                return False
        except Exception as e:
            logger.warning(f"Error reading cache file {self.cache_path}: {e}. A reload will be performed.")
            return False