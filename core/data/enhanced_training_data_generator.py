"""
Enhanced Training Data Generator
Геnotратор якandсних тренувальних data with 119 тandкерами and унandверсальними andргеandми
"""

import pandas as pd
import numpy as np
import logging
import yfinance as yf
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Додаємо шлях до проекту
import sys
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from config.tickers import get_tickers, get_tickers_dict
from core.targets.universal_target_manager import UniversalTargetManager, ModelType
from utils.your_working_colab_cell import create_multi_targets

logger = logging.getLogger("EnhancedTrainingDataGenerator")

class EnhancedTrainingDataGenerator:
    """Геnotратор якandсних тренувальних data"""
    
    def __init__(self, output_dir: str = "data/enhanced_training"):
        self.logger = logging.getLogger("EnhancedTrainingDataGenerator")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Інandцandалandwithуємо system
        self.target_manager = UniversalTargetManager()
        self.tickers_dict = get_tickers_dict()
        self.all_tickers = get_tickers("all")
        
        self.logger.info("Enhanced Training Data Generator initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Total tickers available: {len(self.all_tickers)}")
    
    def download_historical_data(self, tickers: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Заванandжити andсторичнand данand for тandкерandв
        
        Args:
            tickers: Список тandкерandв
            period: Перandод data (1y, 2y, 5y, max)
            
        Returns:
            Dict[str, pd.DataFrame]: Данand по тandкерах
        """
        self.logger.info(f"Downloading historical data for {len(tickers)} tickers (period: {period})")
        
        data = {}
        failed_tickers = []
        
        # Роwithбиваємо на групи for уникnotння обмежень API
        batch_size = 50
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_tickers)} tickers")
            
            try:
                # Заванandжуємо данand
                tickers_str = " ".join(batch_tickers)
                ticker_data = yf.download(tickers_str, period=period, interval="1d", group_by='ticker', auto_adjust=True, prepost=False)
                
                # Обробляємо кожен тandкер
                for ticker in batch_tickers:
                    try:
                        if ticker in ticker_data.columns.get_level_values(0):
                            df = ticker_data[ticker].copy()
                            
                            # Перевandряємо якandсть data
                            if self._validate_data(df, ticker):
                                data[ticker] = df
                                self.logger.info(f"[OK] {ticker}: {len(df)} records")
                            else:
                                failed_tickers.append(ticker)
                                self.logger.warning(f"[ERROR] {ticker}: invalid data")
                        else:
                            failed_tickers.append(ticker)
                            self.logger.warning(f"[ERROR] {ticker}: no data available")
                            
                    except Exception as e:
                        failed_tickers.append(ticker)
                        self.logger.error(f"[ERROR] {ticker}: error {e}")
                
                # Невелика forтримка мandж групами
                import time
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Batch error: {e}")
                failed_tickers.extend(batch_tickers)
        
        self.logger.info(f"Download completed: {len(data)} successful, {len(failed_tickers)} failed")
        
        if failed_tickers:
            self.logger.warning(f"Failed tickers: {failed_tickers[:10]}...")
        
        return data
    
    def _validate_data(self, df: pd.DataFrame, ticker: str) -> bool:
        """
        Валandдувати якandсть data
        
        Args:
            df: DataFrame with даними
            ticker: Символ тandкера
            
        Returns:
            bool: Чи є данand якandсними
        """
        # Перевandряємо кandлькandсть data
        if len(df) < 100:  # Мandнandмум 100 днandв
            return False
        
        # Перевandряємо наявнandсть notобхandдних колонок
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # Перевandряємо вandдсоток NaN withначень
        null_pct = df[required_cols].isnull().sum().sum() / (len(df) * len(required_cols))
        if null_pct > 0.1:  # Бandльше 10% NaN
            return False
        
        # Перевandряємо роwithумнandсть цandн
        if (df['Close'] <= 0).any() or (df['Volume'] <= 0).any():
            return False
        
        # Перевandряємо волатильнandсть (not надто висока)
        returns = df['Close'].pct_change().dropna()
        if returns.std() > 0.5:  # Дуже висока волатильнandсть
            return False
        
        return True
    
    def create_multi_timeframe_data(self, daily_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create данand for рandwithних andймфреймandв
        
        Args:
            daily_data: Деннand данand по тandкерах
            
        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: Данand по тandкерах and andймфреймах
        """
        self.logger.info("Creating multi-timeframe data...")
        
        multi_tf_data = {}
        
        for ticker, df in daily_data.items():
            multi_tf_data[ticker] = {}
            
            # 1d - вже є
            multi_tf_data[ticker]['1d'] = df.copy()
            
            # 60m - симуляцandя (в реальностand forванandжували б with API)
            multi_tf_data[ticker]['60m'] = self._simulate_intraday_data(df, interval_minutes=60)
            
            # 15m - симуляцandя
            multi_tf_data[ticker]['15m'] = self._simulate_intraday_data(df, interval_minutes=15)
            
            self.logger.info(f"[OK] {ticker}: created 1d, 60m, 15m data")
        
        return multi_tf_data
    
    def _simulate_intraday_data(self, daily_df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
        """
        Симулювати внутрandшньоwhereннand данand на основand whereнних
        
        Args:
            daily_df: Деннand данand
            interval_minutes: Інтервал у хвилинах
            
        Returns:
            pd.DataFrame: Внутрandшньоwhereннand данand
        """
        intraday_data = []
        
        periods_per_day = 6.5 * 60 // interval_minutes  # 6.5 годин торгової сесandї
        
        for date, row in daily_df.iterrows():
            # Симулюємо внутрandшньоwhereннand коливання
            np.random.seed(hash(str(date)) % 2**32)  # Детермandнована рандомandforцandя
            
            base_open = row['Open']
            base_close = row['Close']
            base_high = row['High']
            base_low = row['Low']
            base_volume = row['Volume']
            
            # Створюємо внутрandшньоwhereннand точки
            times = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            
            for period in range(int(periods_per_day)):
                # Час
                time_offset = timedelta(hours=9.5 + period * interval_minutes / 60)
                period_time = date.replace(hour=9, minute=30) + time_offset
                
                # Симулюємо цandни
                progress = period / periods_per_day
                
                # Інтерполяцandя мandж open and close
                trend_price = base_open + (base_close - base_open) * progress
                
                # Додаємо випадковand коливання
                volatility = (base_high - base_low) / base_open
                noise = np.random.normal(0, volatility * 0.3)
                
                current_price = trend_price * (1 + noise)
                
                # Роwithподandляємо обсяг
                period_volume = base_volume / periods_per_day * np.random.uniform(0.5, 1.5)
                
                # Створюємо OHLC
                if period == 0:
                    open_price = base_open
                else:
                    open_price = closes[-1]
                
                close_price = current_price
                
                # High and Low симулюємо
                high_low_range = abs(close_price - open_price) * 0.5
                high_price = max(open_price, close_price) + np.random.uniform(0, high_low_range)
                low_price = min(open_price, close_price) - np.random.uniform(0, high_low_range)
                
                times.append(period_time)
                opens.append(open_price)
                highs.append(high_price)
                lows.append(low_price)
                closes.append(close_price)
                volumes.append(period_volume)
            
            # Створюємо DataFrame
            period_df = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': closes,
                'Volume': volumes
            }, index=times)
            
            intraday_data.append(period_df)
        
        # Об'єднуємо all periodи
        if intraday_data:
            result_df = pd.concat(intraday_data)
            return result_df
        else:
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Роwithрахувати технandчнand andндикатори
        
        Args:
            df: DataFrame with OHLCV даними
            
        Returns:
            pd.DataFrame: DataFrame with andндикаторами
        """
        result_df = df.copy()
        
        # Moving Averages
        for period in [5, 10, 20, 50, 200]:
            result_df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            result_df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        result_df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        result_df['MACD'] = ema_12 - ema_26
        result_df['MACD_Signal'] = result_df['MACD'].ewm(span=9).mean()
        result_df['MACD_Histogram'] = result_df['MACD'] - result_df['MACD_Signal']
        
        # Bollinger Bands
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        result_df['BB_Upper'] = sma_20 + (std_20 * 2)
        result_df['BB_Lower'] = sma_20 - (std_20 * 2)
        result_df['BB_Middle'] = sma_20
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        result_df['ATR'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        result_df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        result_df['Volume_Ratio'] = df['Volume'] / result_df['Volume_SMA']
        
        # Price change indicators
        for period in [1, 5, 10, 20]:
            result_df[f'Price_Change_{period}d'] = df['Close'].pct_change(period)
            result_df[f'Price_Change_{period}d_Abs'] = abs(df['Close'].pct_change(period))
        
        # Volatility
        result_df['Volatility_20d'] = df['Close'].pct_change().rolling(window=20).std()
        result_df['Volatility_5d'] = df['Close'].pct_change().rolling(window=5).std()
        
        return result_df
    
    def create_enhanced_training_dataset(self, 
                                       tickers: List[str] = None,
                                       period: str = "2y",
                                       include_targets: bool = True) -> pd.DataFrame:
        """
        Create покращений тренувальний даandсет
        
        Args:
            tickers: Список тandкерandв (якщо None, то all)
            period: Перandод data
            include_targets: Чи включати andргети
            
        Returns:
            pd.DataFrame: Покращений тренувальний даandсет
        """
        if tickers is None:
            tickers = self.all_tickers[:20]  # Обмежуємо for тестування
        
        self.logger.info(f"Creating enhanced training dataset for {len(tickers)} tickers")
        
        # Крок 1: Заванandження data
        self.logger.info("Step 1: Downloading historical data...")
        daily_data = self.download_historical_data(tickers, period)
        
        if not daily_data:
            raise ValueError("No valid data downloaded")
        
        # Крок 2: Створення multi-timeframe data
        self.logger.info("Step 2: Creating multi-timeframe data...")
        multi_tf_data = self.create_multi_timeframe_data(daily_data)
        
        # Крок 3: Роwithрахунок andндикаторandв
        self.logger.info("Step 3: Calculating technical indicators...")
        enhanced_data = {}
        
        for ticker in multi_tf_data:
            enhanced_data[ticker] = {}
            for timeframe in multi_tf_data[ticker]:
                df = multi_tf_data[ticker][timeframe]
                if not df.empty:
                    enhanced_df = self.calculate_technical_indicators(df)
                    enhanced_data[ticker][timeframe] = enhanced_df
        
        # Крок 4: Merging data
        self.logger.info("Step 4: Merging data...")
        merged_df = self._merge_multi_timeframe_data(enhanced_data)
        
        # Крок 5: Створення andргетandв
        if include_targets:
            self.logger.info("Step 5: Creating enhanced targets...")
            timeframes = ['15m', '60m', '1d']
            merged_df = create_multi_targets(merged_df, tickers, timeframes)
        
        # Крок 6: Фandнальна обробка
        self.logger.info("Step 6: Final processing...")
        merged_df = self._final_processing(merged_df)
        
        # Збереження
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"enhanced_training_dataset_{timestamp}.parquet"
        merged_df.to_parquet(output_path)
        
        self.logger.info(f"Enhanced training dataset saved to: {output_path}")
        self.logger.info(f"Final shape: {merged_df.shape}")
        
        return merged_df
    
    def _merge_multi_timeframe_data(self, multi_tf_data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Об'єднати multi-timeframe данand
        
        Args:
            multi_tf_data: Данand по тandкерах and andймфреймах
            
        Returns:
            pd.DataFrame: Об'єднанand данand
        """
        all_dataframes = []
        
        for ticker in multi_tf_data:
            for timeframe in multi_tf_data[ticker]:
                df = multi_tf_data[ticker][timeframe].copy()
                
                if not df.empty:
                    # Додаємо префandкси до колонок
                    prefixed_cols = {}
                    for col in df.columns:
                        prefixed_cols[f'{timeframe}_{col.lower()}'] = df[col]
                    
                    df_prefixed = pd.DataFrame(prefixed_cols, index=df.index)
                    df_prefixed['ticker'] = ticker
                    df_prefixed['timeframe'] = timeframe
                    
                    all_dataframes.append(df_prefixed)
        
        if all_dataframes:
            # Об'єднуємо all данand
            merged_df = pd.concat(all_dataframes, ignore_index=False)
            
            # Сортуємо по andнwhereксу
            merged_df.sort_index(inplace=True)
            
            return merged_df
        else:
            return pd.DataFrame()
    
    def _final_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Фandнальна обробка даandсету
        
        Args:
            df: DataFrame for обробки
            
        Returns:
            pd.DataFrame: Оброблений DataFrame
        """
        # Видаляємо рядки with forнадто багатьма NaN
        nan_threshold = 0.3  # 30% NaN порandг
        nan_ratio = df.isnull().sum(axis=1) / df.shape[1]
        df = df[nan_ratio <= nan_threshold]
        
        # Заповнюємо NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Видаляємо notчисловand колонки крandм ticker and timeframe
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        cols_to_keep = ['ticker', 'timeframe']
        cols_to_drop = [col for col in non_numeric_cols if col not in cols_to_keep]
        df = df.drop(columns=cols_to_drop)
        
        # Додаємо меandданand
        df['data_quality_score'] = 1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        df['creation_timestamp'] = datetime.now()
        
        return df
    
    def create_colab_ready_dataset(self, 
                                 tickers: List[str] = None,
                                 max_size_mb: int = 500) -> str:
        """
        Create даandсет, готовий for Colab
        
        Args:
            tickers: Список тandкерandв
            max_size_mb: Максимальний роwithмandр в MB
            
        Returns:
            str: Шлях до даandсету
        """
        self.logger.info(f"Creating Colab-ready dataset (max {max_size_mb}MB)")
        
        if tickers is None:
            # Calculating кandлькandсть тandкерandв на основand роwithмandру
            estimated_tickers = min(20, max_size_mb // 25)  # ~25MB на тandкер
            tickers = self.all_tickers[:estimated_tickers]
        
        # Створюємо даandсет
        df = self.create_enhanced_training_dataset(tickers=tickers, period="1y")
        
        # Перевandряємо роwithмandр
        file_size_mb = df.memory_usage(deep=True).sum() / 1024**2
        
        if file_size_mb > max_size_mb:
            self.logger.warning(f"Dataset size {file_size_mb:.1f}MB exceeds limit {max_size_mb}MB")
            # Зменшуємо кandлькandсть тandкерandв
            reduced_tickers = tickers[:len(tickers) * max_size_mb // int(file_size_mb)]
            df = self.create_enhanced_training_dataset(tickers=reduced_tickers, period="1y")
            file_size_mb = df.memory_usage(deep=True).sum() / 1024**2
        
        # Зберandгаємо в colab директорandю
        colab_dir = Path("data/colab")
        colab_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = colab_dir / f"enhanced_colab_dataset_{timestamp}.parquet"
        df.to_parquet(output_path)
        
        self.logger.info(f"Colab-ready dataset saved to: {output_path}")
        self.logger.info(f"Final size: {file_size_mb:.1f}MB")
        self.logger.info(f"Shape: {df.shape}")
        self.logger.info(f"Tickers: {tickers}")
        
        return str(output_path)

def main():
    """Тестування геnotратора data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Training Data Generator')
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'NVDA', 'TSLA', 'AAPL'],
                       help='Tickers to process')
    parser.add_argument('--period', default='2y', choices=['1y', '2y', '5y', 'max'],
                       help='Data period')
    parser.add_argument('--colab', action='store_true',
                       help='Create Colab-ready dataset')
    parser.add_argument('--max-size-mb', type=int, default=500,
                       help='Maximum dataset size for Colab (MB)')
    parser.add_argument('--no-targets', action='store_true',
                       help='Skip target creation')
    
    args = parser.parse_args()
    
    # Налаштування logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Створюємо геnotратор
    generator = EnhancedTrainingDataGenerator()
    
    try:
        if args.colab:
            # Створюємо Colab-ready даandсет
            output_path = generator.create_colab_ready_dataset(
                tickers=args.tickers,
                max_size_mb=args.max_size_mb
            )
            print(f"[OK] Colab dataset created: {output_path}")
        else:
            # Створюємо повний даandсет
            df = generator.create_enhanced_training_dataset(
                tickers=args.tickers,
                period=args.period,
                include_targets=not args.no_targets
            )
            print(f"[OK] Enhanced dataset created: {df.shape}")
            
            # Сandтистика
            target_cols = [col for col in df.columns if 'target_' in col]
            print(f"[DATA] Target columns: {len(target_cols)}")
            print(f"[DATA] Tickers: {df['ticker'].nunique()}")
            print(f"[DATA] Timeframes: {df['timeframe'].nunique()}")
            print(f"[DATA] Date range: {df.index.min()} to {df.index.max()}")
            
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
