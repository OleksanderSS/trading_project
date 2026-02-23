"""
Simple Data Rebuild
Просand перебудова data for 119 тandкерandв беwith складних forлежностей
"""

import pandas as pd
import numpy as np
import logging
import yfinance as yf
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Налаштування logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleDataRebuild:
    """Просand перебудова data"""
    
    def __init__(self):
        self.logger = logging.getLogger("SimpleDataRebuild")
        
        # Створюємо директорandї
        self.output_dir = Path("data/rebuild")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Прandоритетнand тandкери
        self.priority_tickers = [
            'SPY', 'QQQ', 'IWM', 'DIA',  # Основнand ETF
            'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA',  # Tech giants
            'JPM', 'BAC', 'WMT', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS'  # Blue chips
        ]
        
        self.logger.info("Simple Data Rebuild initialized")
        self.logger.info(f"Priority tickers: {len(self.priority_tickers)}")
    
    def download_data(self, tickers: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Заванandжити данand for тandкерandв
        
        Args:
            tickers: Список тandкерandв
            period: Перandод data
            
        Returns:
            Dict[str, pd.DataFrame]: Данand по тandкерах
        """
        self.logger.info(f"Downloading data for {len(tickers)} tickers (period: {period})")
        
        data = {}
        failed_tickers = []
        
        # Роwithбиваємо на групи
        batch_size = 20
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
                
                # Затримка мandж групами
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
        """Валandдувати якandсть data"""
        if len(df) < 100:  # Мandнandмум 100 днandв
            return False
        
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
        
        return True
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Роwithрахувати технandчнand andндикатори"""
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
    
    def create_simple_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create простand andргети"""
        result_df = df.copy()
        
        # Price change targets
        for period in [1, 5, 10, 20]:
            result_df[f'target_price_change_{period}d'] = df['Close'].pct_change(period).shift(-period)
        
        # Volatility targets
        for period in [5, 10, 20]:
            result_df[f'target_volatility_{period}d'] = df['Close'].pct_change().rolling(window=period).std().shift(-period)
        
        # Direction targets (classification)
        for period in [1, 5, 10, 20]:
            price_change = df['Close'].pct_change(period).shift(-period)
            result_df[f'target_direction_{period}d'] = np.where(price_change > 0, 1, 0)
            result_df[f'target_direction_{period}d_3class'] = np.where(price_change > 0.01, 2, 
                                                                   np.where(price_change < -0.01, 0, 1))
        
        # RSI targets
        for period in [5, 10, 20]:
            rsi_future = result_df['RSI'].shift(-period)
            result_df[f'target_rsi_{period}d'] = rsi_future
        
        # Volume targets
        for period in [5, 10, 20]:
            volume_future = df['Volume'].shift(-period)
            volume_ma = df['Volume'].rolling(window=20).mean()
            result_df[f'target_volume_ratio_{period}d'] = (volume_future / volume_ma).shift(-period)
        
        return result_df
    
    def create_multi_timeframe_data(self, daily_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Create данand for рandwithних andймфреймandв"""
        self.logger.info("Creating multi-timeframe data...")
        
        multi_tf_data = {}
        
        for ticker, df in daily_data.items():
            multi_tf_data[ticker] = {}
            
            # 1d - вже є
            multi_tf_data[ticker]['1d'] = df.copy()
            
            # 60m - симуляцandя
            multi_tf_data[ticker]['60m'] = self._simulate_intraday_data(df, interval_minutes=60)
            
            # 15m - симуляцandя
            multi_tf_data[ticker]['15m'] = self._simulate_intraday_data(df, interval_minutes=15)
            
            self.logger.info(f"[OK] {ticker}: created 1d, 60m, 15m data")
        
        return multi_tf_data
    
    def _simulate_intraday_data(self, daily_df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
        """Симулювати внутрandшньоwhereннand данand"""
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
    
    def rebuild_data(self, tickers: List[str] = None, max_tickers: int = 20) -> Dict[str, Any]:
        """
        Перебудувати данand
        
        Args:
            tickers: Список тandкерandв
            max_tickers: Максимальна кandлькandсть тandкерandв
            
        Returns:
            Dict[str, Any]: Реwithульandти перебудови
        """
        if tickers is None:
            tickers = self.priority_tickers[:max_tickers]
        
        self.logger.info(f"[START] Starting data rebuild for {len(tickers)} tickers")
        
        results = {
            "status": "in_progress",
            "start_time": datetime.now(),
            "tickers": tickers,
            "steps": {}
        }
        
        try:
            # Крок 1: Заванandження data
            self.logger.info("[DATA] Step 1: Downloading historical data...")
            daily_data = self.download_data(tickers, period="2y")
            
            if not daily_data:
                raise ValueError("No data downloaded")
            
            results["steps"]["download"] = {
                "status": "completed",
                "tickers_downloaded": len(daily_data),
                "failed_tickers": len(tickers) - len(daily_data)
            }
            
            # Крок 2: Створення multi-timeframe data
            self.logger.info("[DATA] Step 2: Creating multi-timeframe data...")
            multi_tf_data = self.create_multi_timeframe_data(daily_data)
            
            results["steps"]["multi_timeframe"] = {
                "status": "completed",
                "tickers": len(multi_tf_data),
                "timeframes": list(multi_tf_data[tickers[0]].keys()) if multi_tf_data else []
            }
            
            # Крок 3: Роwithрахунок andндикаторandв and andргетandв
            self.logger.info("[DATA] Step 3: Calculating indicators and targets...")
            enhanced_data = {}
            
            for ticker in multi_tf_data:
                enhanced_data[ticker] = {}
                for timeframe in multi_tf_data[ticker]:
                    df = multi_tf_data[ticker][timeframe]
                    if not df.empty:
                        # Calculating andндикатори
                        df_with_indicators = self.calculate_indicators(df)
                        
                        # Створюємо andргети
                        df_with_targets = self.create_simple_targets(df_with_indicators)
                        
                        enhanced_data[ticker][timeframe] = df_with_targets
            
            results["steps"]["indicators_targets"] = {
                "status": "completed",
                "tickers": len(enhanced_data),
                "timeframes": len(enhanced_data[tickers[0]]) if enhanced_data else 0
            }
            
            # Крок 4: Merging data
            self.logger.info("[DATA] Step 4: Merging data...")
            merged_df = self._merge_data(enhanced_data)
            
            results["steps"]["merge"] = {
                "status": "completed",
                "final_shape": merged_df.shape,
                "columns": len(merged_df.columns)
            }
            
            # Крок 5: Збереження
            self.logger.info("[DATA] Step 5: Saving data...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Зберandгаємо основний даandсет
            main_path = self.output_dir / f"rebuild_dataset_{timestamp}.parquet"
            merged_df.to_parquet(main_path)
            
            # Зберandгаємо for Colab
            colab_path = self.output_dir / f"colab_dataset_{timestamp}.parquet"
            colab_df = self._prepare_colab_dataset(merged_df)
            colab_df.to_parquet(colab_path)
            
            results["steps"]["save"] = {
                "status": "completed",
                "main_path": str(main_path),
                "colab_path": str(colab_path),
                "main_size_mb": main_path.stat().st_size / 1024**2,
                "colab_size_mb": colab_path.stat().st_size / 1024**2
            }
            
            # Фandнальнand реwithульandти
            results["status"] = "completed"
            results["end_time"] = datetime.now()
            results["duration"] = (results["end_time"] - results["start_time"]).total_seconds()
            results["final_dataset"] = str(main_path)
            results["colab_dataset"] = str(colab_path)
            
            # Сandтистика
            target_cols = [col for col in merged_df.columns if 'target_' in col]
            results["statistics"] = {
                "total_rows": len(merged_df),
                "total_columns": len(merged_df.columns),
                "target_columns": len(target_cols),
                "feature_columns": len(merged_df.columns) - len(target_cols),
                "unique_tickers": merged_df['ticker'].nunique() if 'ticker' in merged_df.columns else 0,
                "unique_timeframes": merged_df['timeframe'].nunique() if 'timeframe' in merged_df.columns else 0,
                "null_percentage": merged_df.isnull().sum().sum() / (merged_df.shape[0] * merged_df.shape[1]) * 100
            }
            
            self.logger.info(f"[OK] Data rebuild completed in {results['duration']:.1f} seconds")
            self.logger.info(f"[DATA] Final dataset: {merged_df.shape}")
            self.logger.info(f" Saved to: {main_path}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"[ERROR] Data rebuild failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            results["end_time"] = datetime.now()
            return results
    
    def _merge_data(self, enhanced_data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """Об'єднати данand"""
        all_dataframes = []
        
        for ticker in enhanced_data:
            for timeframe in enhanced_data[ticker]:
                df = enhanced_data[ticker][timeframe].copy()
                
                if not df.empty:
                    # Додаємо префandкси до колонок
                    prefixed_cols = {}
                    for col in df.columns:
                        if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            prefixed_cols[f'{timeframe}_{col.lower()}'] = df[col]
                        else:
                            prefixed_cols[col] = df[col]
                    
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
    
    def _prepare_colab_dataset(self, df: pd.DataFrame, max_size_mb: int = 300) -> pd.DataFrame:
        """Пandдготувати даandсет for Colab"""
        # Обмежуємо роwithмandр
        if len(df) > 50000:
            df = df.sample(n=50000, random_state=42)
        
        # Видаляємо рядки with forнадто багатьма NaN
        nan_threshold = 0.3
        nan_ratio = df.isnull().sum(axis=1) / df.shape[1]
        df = df[nan_ratio <= nan_threshold]
        
        # Заповнюємо NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df

def main():
    """Основна функцandя"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Data Rebuild')
    parser.add_argument('--tickers', nargs='+', help='Tickers to process')
    parser.add_argument('--max-tickers', type=int, default=20, help='Maximum tickers to process')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Створюємо перебудовувач
    rebuild = SimpleDataRebuild()
    
    try:
        # Запускаємо перебудову
        results = rebuild.rebuild_data(tickers=args.tickers, max_tickers=args.max_tickers)
        
        print("\n" + "="*60)
        print("DATA REBUILD RESULTS")
        print("="*60)
        
        if results["status"] == "completed":
            print("Status: COMPLETED")
            print(f"Duration: {results['duration']:.1f} seconds")
            print(f"Tickers: {len(results['tickers'])}")
            
            if "statistics" in results:
                stats = results["statistics"]
                print(f"Final shape: {stats['total_rows']} rows  {stats['total_columns']} columns")
                print(f"Target columns: {stats['target_columns']}")
                print(f"Feature columns: {stats['feature_columns']}")
                print(f"Unique tickers: {stats['unique_tickers']}")
                print(f"Unique timeframes: {stats['unique_timeframes']}")
                print(f"Null percentage: {stats['null_percentage']:.2f}%")
            
            if "final_dataset" in results:
                print(f"Main dataset: {results['final_dataset']}")
            
            if "colab_dataset" in results:
                print(f"Colab dataset: {results['colab_dataset']}")
            
            print("\nData rebuild completed successfully!")
            
        else:
            print("Status: FAILED")
            if "error" in results:
                print(f"Error: {results['error']}")
        
        print("="*60)
        
        return 0 if results["status"] == "completed" else 1
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
