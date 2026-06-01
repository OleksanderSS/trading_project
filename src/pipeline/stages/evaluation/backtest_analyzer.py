#!/usr/bin/env python3
"""
Backtest Analyzer - Backtest execution and analysis
Handles backtest execution and simulation data generation.
"""

import pandas as pd
import numpy as np
import asyncio
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import logging

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("BacktestAnalyzer")


class BacktestAnalyzer:
    """
    Backtest analyzer for strategy evaluation.
    
    Handles:
    - Backtest execution
    - Data preparation (pivot tables)
    - Simulation data generation
    - Backtest result validation
    """
    
    def __init__(self, backtester=None):
        """
        Initialize Backtest Analyzer.
        
        Args:
            backtester: Optional AdvancedBacktestEngine instance
        """
        self.logger = logger
        self.backtester = backtester
        self.logger.info("✅ BacktestAnalyzer initialized")
    
    def prepare_pivot(self, signals_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare pivot tables for backtesting.
        
        Args:
            signals_df: DataFrame with signals and price data
            
        Returns:
            Tuple of (price_pivot, signal_pivot)
        """
        try:
            if 'ticker' in signals_df.columns:
                if 'timestamp' in signals_df.columns and signals_df['timestamp'].notna().any():
                    self.logger.info('Pivoting by timestamp and ticker...')
                    price_pivot = signals_df.pivot_table(
                        index='timestamp', columns='ticker', values='price', aggfunc='mean'
                    )
                    sig_numeric = signals_df.copy()
                    sig_numeric['sig_val'] = sig_numeric['signal'].map({'BUY': 1, 'SELL': -1, 'HOLD': 0})
                    signal_pivot = sig_numeric.pivot_table(
                        index='timestamp', columns='ticker', values='sig_val', aggfunc='mean'
                    )
                else:
                    self.logger.info('No valid timestamps found. Aggregating as single snapshot...')
                    price_agg = signals_df.groupby('ticker')['price'].mean()
                    price_pivot = price_agg.to_frame().T
                    price_pivot.index = [pd.Timestamp.now()]
                    
                    sig_numeric = signals_df.copy()
                    sig_numeric['sig_val'] = sig_numeric['signal'].map({'BUY': 1, 'SELL': -1, 'HOLD': 0})
                    signal_agg = sig_numeric.groupby('ticker')['sig_val'].mean()
                    signal_pivot = signal_agg.to_frame().T
                    signal_pivot.index = price_pivot.index
            else:
                self.logger.warning("No 'ticker' column found in signals_df!")
                price_pivot = signals_df[['price']].copy()
                price_pivot.index = [pd.Timestamp.now()]
                sig_numeric = signals_df.copy()
                sig_numeric['sig_val'] = sig_numeric['signal'].map({'BUY': 1, 'SELL': -1, 'HOLD': 0})
                signal_pivot = sig_numeric[['sig_val']].copy()
                signal_pivot.index = price_pivot.index
            
            return price_pivot, signal_pivot
            
        except Exception as e:
            self.logger.error(f"Error preparing pivot tables: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def create_simulation_data(self, signals_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create simulation data for backtesting when real data is insufficient.
        
        Args:
            signals_df: DataFrame with signals
            
        Returns:
            Tuple of (price_df, signal_df)
        """
        try:
            self.logger.info('Creating simulation data for backtest...')
            
            if 'ticker' in signals_df.columns:
                tickers = signals_df['ticker'].unique()
            else:
                tickers = ['SPY', 'QQQ', 'AAPL']
            
            end_date = datetime.now()
            dates = pd.date_range(end=end_date, periods=30, freq='D')
            
            price_data = {}
            signal_data = {}
            
            for ticker in tickers:
                base_price = 100.0 + np.random.uniform(-50, 200)
                returns = np.random.normal(0.001, 0.02, len(dates))
                prices = [base_price]
                for ret in returns:
                    prices.append(prices[-1] * (1 + ret))
                prices = prices[1:]
                price_data[ticker] = prices
                
                signals = []
                for i, price in enumerate(prices):
                    if i == 0:
                        signals.append(0)
                    else:
                        price_change = (price - prices[i - 1]) / prices[i - 1]
                        if price_change > 0.02:
                            signals.append(1)
                        elif price_change < -0.02:
                            signals.append(-1)
                        else:
                            signals.append(0)
                signal_data[ticker] = signals
            
            price_df = pd.DataFrame(price_data, index=dates)
            signal_df = pd.DataFrame(signal_data, index=dates)
            
            self.logger.info(f'Created simulation data: {price_df.shape[0]} days, {len(tickers)} tickers')
            return price_df, signal_df
            
        except Exception as e:
            self.logger.error(f"Failed to create simulation data: {e}")
            dates = pd.date_range(end=datetime.now(), periods=2, freq='D')
            price_df = pd.DataFrame({'SPY': [100.0, 101.0]}, index=dates)
            signal_df = pd.DataFrame({'SPY': [0, 1]}, index=dates)
            return price_df, signal_df
    
    async def run_backtest(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest using the backtester.
        
        Args:
            signals_df: DataFrame with signals and price data
            
        Returns:
            Dictionary with backtest results
        """
        try:
            self.logger.info(f'Preparing data for backtest. Input shape: {signals_df.shape}')
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'Columns: {signals_df.columns.tolist()}')
            
            if signals_df.empty:
                self.logger.warning('⚠️ Empty signals DataFrame - cannot run backtest')
                return {}
            
            required_cols = ['price', 'signal']
            missing_cols = [col for col in required_cols if col not in signals_df.columns]
            if missing_cols:
                self.logger.warning(f'⚠️ Missing required columns for backtest: {missing_cols}')
                return {}
            
            if signals_df['price'].isna().all():
                self.logger.warning('⚠️ All price values are NaN - cannot run backtest')
                return {}
            
            # Prepare pivot tables
            price_pivot, signal_pivot = self.prepare_pivot(signals_df)
            
            if price_pivot.empty or signal_pivot.empty:
                self.logger.warning('⚠️ Empty pivoted data - cannot run backtest')
                return {}
            
            if not price_pivot.select_dtypes(include=[np.number]).columns.any():
                self.logger.warning('⚠️ No numeric price data - cannot run backtest')
                return {}
            
            if len(price_pivot) < 2:
                self.logger.warning('⚠️ Insufficient data points for backtest - creating simulation')
                price_pivot, signal_pivot = self.create_simulation_data(signals_df)
            
            self.logger.info(f'Pivoted data shape: {price_pivot.shape}')
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'Price data columns: {price_pivot.columns.tolist()}')
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'Signal data columns: {signal_pivot.columns.tolist()}')
            
            # Run backtest
            if self.backtester:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None, self.backtester.run_comprehensive_backtest, price_pivot, signal_pivot
                )
                
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'Backtest results keys: {list(results.keys())}')
                
                if not results or not isinstance(results, dict):
                    self.logger.warning('⚠️ Backtest returned invalid results')
                    return {}
                
                if 'error' in results:
                    self.logger.error(f"❌ Backtest error: {results['error']}")
                    return {}
                
                # Normalize performance metrics key
                if 'performance_metrics' in results:
                    results['performance'] = results['performance_metrics']
                    self.logger.info('✅ Backtest completed with performance metrics')
                elif 'performance' in results:
                    self.logger.info('✅ Backtest completed with legacy performance format')
                else:
                    self.logger.warning('⚠️ No performance metrics found in backtest results')
                
                # Create portfolio_history if missing
                if 'portfolio_history' not in results and 'performance_metrics' in results:
                    results['portfolio_history'] = self._create_portfolio_history(
                        results, price_pivot
                    )
                
                return results
            else:
                self.logger.warning('⚠️ No backtester available')
                return {}
                
        except Exception as e:
            self.logger.error(f'❌ Backtest execution failed: {e}')
            return {}
    
    def _create_portfolio_history(self, results: Dict[str, Any], 
                                  price_pivot: pd.DataFrame) -> pd.DataFrame:
        """
        Create portfolio history from performance metrics.
        
        Args:
            results: Backtest results dictionary
            price_pivot: Price pivot DataFrame
            
        Returns:
            DataFrame with portfolio history
        """
        try:
            initial_capital = results.get('initial_capital', 100000.0)
            performance_metrics = results.get('performance_metrics', {})
            total_return = performance_metrics.get('total_return', 0.0)
            final_value = initial_capital * (1 + total_return)
            dates = price_pivot.index
            equity_values = np.linspace(initial_capital, final_value, len(dates))
            portfolio_history = pd.DataFrame({'total_value': equity_values, 'date': dates})
            portfolio_history.set_index('date', inplace=True)
            results['portfolio_history'] = portfolio_history
            self.logger.info(f'✅ Created portfolio_history with {len(portfolio_history)} data points')
            return portfolio_history
        except Exception as e:
            self.logger.error(f'Failed to create portfolio_history: {e}')
            dates = pd.date_range(end=pd.Timestamp.now(), periods=2, freq='D')
            portfolio_history = pd.DataFrame({'total_value': [100000.0, 100000.0], 'date': dates})
            portfolio_history.set_index('date', inplace=True)
            return portfolio_history
    
    def can_run_backtest(self, signals_df: pd.DataFrame) -> bool:
        """
        Check if backtest can be launched.
        
        Args:
            signals_df: DataFrame with signals
            
        Returns:
            True if backtest can be run, False otherwise
        """
        return ('price' in signals_df.columns and 
                not signals_df['price'].isna().all() and 
                pd.api.types.is_numeric_dtype(signals_df['price']))


# Factory function
def get_backtest_analyzer(backtester=None) -> BacktestAnalyzer:
    """Factory function to get BacktestAnalyzer instance."""
    return BacktestAnalyzer(backtester)
