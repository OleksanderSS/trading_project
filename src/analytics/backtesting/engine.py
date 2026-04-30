import pandas as pd
import numpy as np
import logging
import copy
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Create numpy random generator
rng = np.random.default_rng(42)

from src.metrics.financial.portfolio_metrics import PortfolioMetricsCalculator
from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import get_current_config
from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine

# Attempting to import ExperienceDiary for integration
try:
    from src.meta_learning.memory.diary_engine import DiaryEngine
except ImportError:
    DiaryEngine = None

logger = ProjectLogger.get_logger("AdvancedBacktester")

class AdvancedBacktester:
    """Advanced backtester driven by centralized configuration."""

    def __init__(self, 
                 transaction_costs: Optional[Dict] = None, 
                 market_impact: Optional[Dict] = None, 
                 risk_params: Optional[Dict] = None):
        self.config_manager = get_current_config()
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        
        # Load configs using get for better granularity
        backtesting_config = self.config_manager.get('strategy.backtesting', {})
        risk_config = self.config_manager.get('strategy.risk_management', {})

        # Fallback to defaults if keys are missing
        self.transaction_costs = transaction_costs or backtesting_config.get('transaction_costs', {})
        if isinstance(self.transaction_costs, dict):
            self.commission_pct = self.transaction_costs.get('commission_pct', 0.001)
            self.spread_pct = self.transaction_costs.get('spread_pct', 0.0005)
            self.slippage_pct = self.transaction_costs.get('slippage_pct', 0.001)
        else:
            self.commission_pct = getattr(self.transaction_costs, 'commission_pct', 0.001)
            self.spread_pct = getattr(self.transaction_costs, 'spread_pct', 0.0005)
            self.slippage_pct = getattr(self.transaction_costs, 'slippage_pct', 0.001)

        self.market_impact = market_impact or backtesting_config.get('market_impact', {})
        self.sqrt_coefficient = self.market_impact.get('sqrt_coefficient', 0.1) if isinstance(self.market_impact, dict) else getattr(self.market_impact, 'sqrt_coefficient', 0.1)
        self.volatility_factor = self.market_impact.get('volatility_factor', 0.05) if isinstance(self.market_impact, dict) else getattr(self.market_impact, 'volatility_factor', 0.05)

        self.risk_params = risk_params or risk_config
        self.max_position_size = self.risk_params.get('max_position_size_pct', 1.0) if isinstance(self.risk_params, dict) else getattr(self.risk_params, 'max_position_size_pct', 1.0)
        
        self.minimum_trade_value = backtesting_config.get('minimum_trade_value', 10.0)
        self.initial_capital = backtesting_config.get('initial_capital', 100000.0)

        # Initialize analytics engine for reporting
        self.analytics_engine = UnifiedAnalyticsEngine(self.config_manager)
        
        # Initialize Experience Diary if available
        self.experience_diary = DiaryEngine() if DiaryEngine else None
        
        self.logger.info("AdvancedBacktester initialized with multi-ticker support and Meta-Learning integration.")

    def calculate_transaction_costs(self, trade_value: float, daily_volume_value: float = None, volatility: float = None) -> Dict[str, float]:
        """Calculates comprehensive transaction costs for a trade."""
        costs = {}
        costs['commission'] = abs(trade_value) * self.commission_pct
        costs['spread'] = abs(trade_value) * self.spread_pct
        
        slippage_impact = self.slippage_pct
        if daily_volume_value and volatility and daily_volume_value > 0:
            trade_ratio = abs(trade_value) / daily_volume_value
            sqrt_impact = self.sqrt_coefficient * np.sqrt(trade_ratio)
            vol_adjustment = 1 + volatility * self.volatility_factor
            slippage_impact += (sqrt_impact * vol_adjustment)
        
        costs['slippage'] = abs(trade_value) * slippage_impact
        costs['total'] = sum(costs.values())
        return costs

    def run_backtest(self, 
                     price_data: Union[pd.Series, pd.DataFrame], 
                     signal_data: Union[pd.Series, pd.DataFrame], 
                     initial_capital: Optional[float] = None, 
                     **kwargs) -> Dict[str, Any]:
        """
        Runs a backtest supporting multiple assets (portfolio backtesting).
        Tracks total portfolio value, combining cash and market positions for all tickers.
        """
        # Prepare data
        price_df, signal_df, tickers, capital = self._prepare_backtest_data(price_data, signal_data, initial_capital)
        combined_data = self._sync_data(price_df, signal_df)
        
        if combined_data.empty:
            logger.error("No valid synced data for backtesting.")
            return {}
        
        # Initialize portfolio
        portfolio = self._initialize_portfolio(combined_data, tickers, capital)
        trades = []
        
        # Run simulation
        portfolio, trades = self._run_simulation_loop(combined_data, portfolio, tickers, trades)
        
        # Calculate performance metrics
        return self._calculate_performance_metrics(portfolio, trades, capital)
    
    def _prepare_backtest_data(self, price_data, signal_data, initial_capital):
        """Prepare and convert data for backtesting"""
        # Convert Series to DataFrame for unified processing
        if isinstance(price_data, pd.Series):
            price_df = price_data.to_frame(name='default_ticker')
        else:
            price_df = price_data

        if isinstance(signal_data, pd.Series):
            signal_df = signal_data.to_frame(name='default_ticker')
        else:
            signal_df = signal_data

        tickers = price_df.columns.tolist()
        capital = initial_capital or self.initial_capital
        
        return price_df, signal_df, tickers, capital
    
    def _sync_data(self, price_df, signal_df):
        """Sync price and signal data"""
        return pd.concat([price_df.add_suffix('_price'), signal_df.add_suffix('_signal')], axis=1).dropna()
    
    def _initialize_portfolio(self, combined_data, tickers, capital):
        """Initialize portfolio DataFrame"""
        portfolio = pd.DataFrame(index=combined_data.index)
        portfolio['cash'] = capital
        for ticker in tickers:
            portfolio[f'{ticker}_pos'] = 0.0
            portfolio[f'{ticker}_val'] = 0.0
        portfolio['total_value'] = capital
        portfolio['returns'] = 0.0
        return portfolio
    
    def _run_simulation_loop(self, combined_data, portfolio, tickers, trades):
        """Run the main simulation loop"""
        # Rebalancing modifiers
        adaptive_multiplier = 1.0
        min_signal_threshold = 0.0
        
        for i in range(1, len(combined_data)):
            # Update positions
            portfolio = self._update_positions(combined_data, portfolio, tickers, i)
            
            # Rebalance based on signals
            portfolio, trades = self._rebalance_positions(combined_data, portfolio, tickers, i, trades, 
                                                     adaptive_multiplier, min_signal_threshold, **kwargs)
        
        return portfolio, trades
    
    def _update_positions(self, combined_data, portfolio, tickers, i):
        """Update existing positions based on new prices"""
        current_cash = portfolio.iloc[i-1]['cash']
        total_market_val = 0.0
        
        for ticker in tickers:
            pos = portfolio.iloc[i-1][f'{ticker}_pos']
            price = combined_data.iloc[i][f'{ticker}_price']
            val = pos * price
            portfolio.iloc[i, portfolio.columns.get_loc(f'{ticker}_pos')] = pos
            portfolio.iloc[i, portfolio.columns.get_loc(f'{ticker}_val')] = val
            total_market_val += val
        
        portfolio.iloc[i, portfolio.columns.get_loc('cash')] = current_cash
        new_total_value = current_cash + total_market_val
        portfolio.iloc[i, portfolio.columns.get_loc('total_value')] = new_total_value
        
        # Calculate portfolio returns
        prev_total = portfolio.iloc[i-1]['total_value']
        portfolio.iloc[i, portfolio.columns.get_loc('returns')] = (new_total_value / prev_total) - 1 if prev_total != 0 else 0.0
        
        return portfolio
    
    def _rebalance_positions(self, combined_data, portfolio, tickers, i, trades, adaptive_multiplier, min_signal_threshold, **kwargs):
        """Rebalance positions based on signals"""
        new_total_value = portfolio.iloc[i]['total_value']
        
        for ticker in tickers:
            trade_info = self._calculate_trade_info(combined_data, portfolio, ticker, i, new_total_value, tickers, adaptive_multiplier, min_signal_threshold, **kwargs)
            
            if trade_info and abs(trade_info['trade_value']) > self.minimum_trade_value:
                self._execute_trade(portfolio, trade_info, i)
                trades.append(trade_info)
        
        self._update_total_value(portfolio, tickers, i)
        return portfolio, trades
    
    def _calculate_trade_info(self, combined_data, portfolio, ticker, i, total_value, tickers, adaptive_multiplier, min_signal_threshold, **kwargs) -> Optional[Dict]:
        """Calculate trade information for a specific ticker"""
        signal = self._convert_signal(combined_data.iloc[i-1][f'{ticker}_signal'])
        
        # Apply adaptive filtering
        if abs(signal) < min_signal_threshold:
            return None
        
        # Calculate trade
        target_asset_value = total_value * signal * (self.max_position_size / len(tickers)) * adaptive_multiplier
        current_asset_value = portfolio.iloc[i][f'{ticker}_val']
        trade_value = target_asset_value - current_asset_value
        
        price = combined_data.iloc[i][f'{ticker}_price']
        vol = kwargs.get('volatility', 0.02)
        v_val = kwargs.get('volume', 1e9)
        
        costs = self.calculate_transaction_costs(trade_value, v_val, vol)
        
        return {
            'date': combined_data.index[i],
            'ticker': ticker,
            'signal': signal,
            'trade_value': trade_value,
            'price': price,
            'costs': costs['total'],
            'shares_to_trade': trade_value / price
        }
    
    def _execute_trade(self, portfolio, trade_info, i):
        """Execute a trade in the portfolio"""
        portfolio.iloc[i, portfolio.columns.get_loc(f"{trade_info['ticker']}_pos")] += trade_info['shares_to_trade']
        portfolio.iloc[i, portfolio.columns.get_loc(f"{trade_info['ticker']}_val")] += trade_info['trade_value']
        portfolio.iloc[i, portfolio.columns.get_loc('cash')] -= (trade_info['trade_value'] + trade_info['costs'])
    
    def _update_total_value(self, portfolio, tickers, i):
        """Update the total portfolio value"""
        final_total = portfolio.iloc[i]['cash'] + sum(portfolio.iloc[i][f'{t}_val'] for t in tickers)
        portfolio.iloc[i, portfolio.columns.get_loc('total_value')] = final_total
    
    def _convert_signal(self, signal):
        """Convert signal to numerical value"""
        if isinstance(signal, str):
            signal_map = {'BUY': 1.0, 'SELL': -1.0, 'HOLD': 0.0}
            signal = signal_map.get(signal.upper(), 0.0)
        else:
            signal = float(signal) if pd.notna(signal) else 0.0
        
        return signal
    
    def _calculate_performance_metrics(self, portfolio, trades, capital):
        """Calculate performance metrics"""
        # Calculate comprehensive performance metrics
        metrics_calculator = PortfolioMetricsCalculator(config_manager=self.config_manager)
        full_performance = metrics_calculator.calculate(portfolio['total_value'])
        
        return {
            'performance': full_performance,
            'portfolio_history': portfolio,
            'trades': trades,
            'initial_capital': capital
        }

    def run_robustness_test(self, 
                            price_data: Union[pd.Series, pd.DataFrame], 
                            signal_data: Union[pd.Series, pd.DataFrame], 
                            n_simulations: int = 10, 
                            noise_level: float = 0.005,
                            **kwargs) -> Dict[str, Any]:
        """
        Runs multiple backtest simulations with noise and signal dropping to assess strategy stability.
        """
        self.logger.info(f"Starting robustness test with {n_simulations} simulations (noise: {noise_level}).")
        
        # Initialize random generator
        rng = self._initialize_random_generator()
        
        # Run simulations
        sim_results = self._run_simulations(price_data, signal_data, n_simulations, noise_level, rng, **kwargs)
        
        if not sim_results:
            return {"error": "All simulations failed."}
        
        # Calculate and return summary
        return self._calculate_robustness_summary(sim_results, n_simulations)
    
    def _initialize_random_generator(self) -> np.random.Generator:
        """Initialize deterministic random generator"""
        seed = self.config_manager.get('performance.random_seed', 42)
        return np.random.default_rng(seed)
    
    def _run_simulations(self, price_data, signal_data, n_simulations, noise_level, rng, **kwargs) -> List[Dict]:
        """Run all robustness simulations"""
        sim_results = []
        
        for i in range(n_simulations):
            try:
                # Generate perturbed data
                perturbed_prices = self._perturb_prices(price_data, noise_level, rng)
                perturbed_signals = self._perturb_signals(signal_data, rng)
                
                # Run backtest with perturbed data
                res = self.run_backtest(perturbed_prices, perturbed_signals, **kwargs)
                if res and 'performance' in res:
                    sim_results.append(res['performance'])
                
                self.logger.info(f"Simulation {i+1}/{n_simulations} completed.")
            except Exception as e:
                self.logger.error(f"Simulation {i+1} failed: {e}")
        
        return sim_results
    
    def _perturb_prices(self, price_data, noise_level, rng) -> Union[pd.Series, pd.DataFrame]:
        """Add Gaussian noise to price data"""
        if isinstance(price_data, pd.DataFrame):
            perturbed_prices = price_data.copy()
            for col in perturbed_prices.columns:
                noise = rng.standard_normal(len(perturbed_prices))
                perturbed_prices[col] = perturbed_prices[col] * (1 + noise * noise_level)
            return perturbed_prices
        else:
            noise = rng.standard_normal(len(price_data))
            return price_data * (1 + noise * noise_level)
    
    def _perturb_signals(self, signal_data, rng) -> Union[pd.Series, pd.DataFrame]:
        """Randomly drop 5% of signals"""
        if isinstance(signal_data, pd.DataFrame):
            perturbed_signals = signal_data.copy()
            for col in perturbed_signals.columns:
                mask = rng.random(len(perturbed_signals)) < 0.05
                perturbed_signals.loc[mask, col] = 0.0
            return perturbed_signals
        else:
            perturbed_signals = signal_data.copy()
            mask = rng.random(len(perturbed_signals)) < 0.05
            perturbed_signals.loc[mask] = 0.0
            return perturbed_signals
    
    def _calculate_robustness_summary(self, sim_results: List[Dict], n_simulations: int) -> Dict[str, Any]:
        """Calculate robustness test summary statistics"""
        total_returns = [s.get('total_return', 0) for s in sim_results]
        sharpe_ratios = [s.get('sharpe_ratio', 0) for s in sim_results]
        
        mean_return = np.mean(total_returns)
        std_sharpe = np.std(sharpe_ratios)
        
        # Robustness score calculation
        robustness_score = np.mean(sharpe_ratios) / (1 + std_sharpe) if not np.isnan(std_sharpe) else 0

        summary = {
            'n_simulations': n_simulations,
            'mean_total_return': mean_return,
            'std_total_return': np.std(total_returns),
            'mean_sharpe_ratio': np.mean(sharpe_ratios),
            'std_sharpe_ratio': std_sharpe,
            'robustness_score': robustness_score,
            'simulations': sim_results
        }
        
        self.logger.info(f"Robustness test complete. Robustness Score: {robustness_score:.4f}")
        return summary

def example_usage():
    dates = pd.date_range('2022-01-01', periods=200)
    # Simulate two assets
    rng = np.random.default_rng(42)
    prices = pd.DataFrame({
        'AAPL': 150 + rng.standard_normal(200).cumsum(),
        'TSLA': 700 + rng.standard_normal(200).cumsum() * 5
    }, index=dates)
    
    signals = pd.DataFrame({
        'AAPL': rng.choice([-1, 0, 1], 200),
        'TSLA': rng.choice([-1, 0, 1], 200)
    }, index=dates)
    
    try:
        backtester = AdvancedBacktester(None, logging.getLogger())
        results = backtester.run_backtest(prices, signals)
        
        logging.info(f"Multi-ticker backtest complete for: {results['tickers']}")
        logging.info(f"Final Portfolio Return: {results['performance'].get('total_return', 0):.2%}")
        logging.info(f"Number of trades executed: {len(results['trades'])}")
        logger.info(f"Multi-ticker backtest complete for: {results['tickers']}")
        logger.info(f"Final Portfolio Return: {results['performance'].get('total_return', 0):.2%}")
        logger.info(f"Number of trades executed: {len(results['trades'])}")
            
    except Exception as e:
        logger.error(f"Error in multi-ticker example: {e}", exc_info=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    example_usage()