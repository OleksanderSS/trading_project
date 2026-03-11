import pandas as pd
import numpy as np
import logging
import copy
from typing import Dict, List, Optional, Any, Union

from src.metrics.financial_metrics import calculate_performance_metrics
from src.config.unified_config_manager import UnifiedConfigManager
from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine

# Attempting to import ExperienceDiary for integration
try:
    from src.meta_learning.experience_diary import ExperienceDiary
except ImportError:
    ExperienceDiary = None

logger = logging.getLogger(__name__)

class AdvancedBacktester:
    """Advanced backtester driven by centralized configuration."""

    def __init__(self, 
                 transaction_costs: Optional[Dict] = None, 
                 market_impact: Optional[Dict] = None, 
                 risk_params: Optional[Dict] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager = UnifiedConfigManager()
        
        # Load configs using get_specific_config for better granularity
        backtesting_config = self.config_manager.get_specific_config('strategy', 'backtesting') or {}
        risk_config = self.config_manager.get_specific_config('strategy', 'risk_management') or {}

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
        self.experience_diary = ExperienceDiary() if ExperienceDiary else None
        
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
                     volume_data: Optional[Union[pd.Series, pd.DataFrame]] = None, 
                     volatility_data: Optional[Union[pd.Series, pd.DataFrame]] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Runs a backtest supporting multiple assets (portfolio backtesting).
        Tracks total portfolio value, combining cash and market positions for all tickers.
        """
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

        # Sync data and drop NaNs
        combined_data = pd.concat([price_df.add_suffix('_price'), signal_df.add_suffix('_signal')], axis=1).dropna()
        if combined_data.empty:
            logger.error("No valid synced data for backtesting.")
            return {}

        portfolio = pd.DataFrame(index=combined_data.index)
        portfolio['cash'] = capital
        for ticker in tickers:
            portfolio[f'{ticker}_pos'] = 0.0
            portfolio[f'{ticker}_val'] = 0.0
        portfolio['total_value'] = capital
        portfolio['returns'] = 0.0
        trades = []

        # Rebalancing modifiers from Adaptive Thresholds (if available via context)
        adaptive_multiplier = 1.0
        min_signal_threshold = 0.0

        # Simulation loop
        for i in range(1, len(combined_data)):
            current_date = combined_data.index[i]
            
            # 1. Update existing positions based on new prices
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

            # 2. Rebalance based on signals
            for ticker in tickers:
                signal = combined_data.iloc[i-1][f'{ticker}_signal']
                
                # Apply adaptive filtering if thresholds were passed in advance or via pre-run analysis (conceptual)
                if abs(signal) < min_signal_threshold:
                    signal = 0.0

                # Max allocated value for this asset
                target_asset_value = new_total_value * signal * (self.max_position_size / len(tickers)) * adaptive_multiplier
                current_asset_value = portfolio.iloc[i][f'{ticker}_val']
                
                trade_value = target_asset_value - current_asset_value
                
                if abs(trade_value) > self.minimum_trade_value:
                    price = combined_data.iloc[i][f'{ticker}_price']
                    vol = 0.02
                    if volatility_data is not None:
                        if isinstance(volatility_data, pd.DataFrame) and ticker in volatility_data.columns:
                            vol = volatility_data.loc[current_date, ticker]
                        elif isinstance(volatility_data, pd.Series):
                            vol = volatility_data.loc[current_date]
                    
                    v_val = 1e9
                    if volume_data is not None:
                        if isinstance(volume_data, pd.DataFrame) and ticker in volume_data.columns:
                            v_val = volume_data.loc[current_date, ticker] * price
                        elif isinstance(volume_data, pd.Series):
                            v_val = volume_data.loc[current_date] * price

                    costs = self.calculate_transaction_costs(trade_value, v_val, vol)
                    
                    shares_to_trade = trade_value / price
                    portfolio.iloc[i, portfolio.columns.get_loc(f'{ticker}_pos')] += shares_to_trade
                    portfolio.iloc[i, portfolio.columns.get_loc(f'{ticker}_val')] += trade_value
                    portfolio.iloc[i, portfolio.columns.get_loc('cash')] -= (trade_value + costs['total'])
                    
                    trades.append({
                        'date': current_date,
                        'ticker': ticker,
                        'signal': signal,
                        'trade_value': trade_value,
                        'price': price,
                        'costs': costs['total']
                    })
            
            final_total = portfolio.iloc[i]['cash'] + sum(portfolio.iloc[i][f'{t}_val'] for t in tickers)
            portfolio.iloc[i, portfolio.columns.get_loc('total_value')] = final_total

        # 3. Comprehensive Analytics
        full_performance = calculate_performance_metrics(portfolio['returns'])
        
        # Enhanced data_map with news, macro, and events for analyzers
        data_map = {
            'price_data': price_df,
            'portfolio_data': portfolio,
            'returns': portfolio['returns'],
            'total_value': portfolio['total_value'],
            'news_data': kwargs.get('news_data'),
            'macro_data': kwargs.get('macro_data'),
            'events_data': kwargs.get('events_data')
        }
        
        try:
            advanced_analysis = self.analytics_engine.run_full_analysis(data_map)
            
            # Update rebalancing modifiers if adaptive thresholds analyzer produced results
            if 'adaptive_thresholds' in advanced_analysis:
                thresh = advanced_analysis['adaptive_thresholds']
                if isinstance(thresh, dict):
                    min_signal_threshold = thresh.get('min_prediction_prob', 0.0)
                    regime = thresh.get('market_regime', 'Normal')
                    if regime == 'High Volatility':
                        adaptive_multiplier = 0.5
                    elif regime == 'Low Volatility':
                        adaptive_multiplier = 1.2

        except Exception as e:
            logger.error(f"Portfolio advanced analysis failed: {e}")
            advanced_analysis = {"error": str(e)}

        results = {
            'performance': full_performance,
            'advanced_analysis': advanced_analysis,
            'portfolio_history': portfolio,
            'trades': trades,
            'tickers': tickers
        }
        
        # 4. Meta-Learning Integration: Log to Experience Diary
        if self.experience_diary:
            try:
                self.experience_diary.log_experience(
                    mode='backtest',
                    metrics=full_performance,
                    market_context={'tickers': tickers, 'period': f"{combined_data.index[0]} to {combined_data.index[-1]}"},
                    metadata={'trade_count': len(trades)}
                )
                logger.info("Backtest experience successfully logged to ExperienceDiary.")
            except Exception as e:
                logger.warning(f"Failed to log experience to diary: {e}")

        return results

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
        sim_results = []
        
        for i in range(n_simulations):
            try:
                # 1. Perturb prices with Gaussian noise
                if isinstance(price_data, pd.DataFrame):
                    perturbed_prices = price_data.copy()
                    for col in perturbed_prices.columns:
                        noise = np.random.normal(0, noise_level, len(perturbed_prices))
                        perturbed_prices[col] = perturbed_prices[col] * (1 + noise)
                else:
                    noise = np.random.normal(0, noise_level, len(price_data))
                    perturbed_prices = price_data * (1 + noise)

                # 2. Randomly drop signals (e.g., 5% of signals set to 0)
                if isinstance(signal_data, pd.DataFrame):
                    perturbed_signals = signal_data.copy()
                    for col in perturbed_signals.columns:
                        mask = np.random.rand(len(perturbed_signals)) < 0.05
                        perturbed_signals.loc[mask, col] = 0.0
                else:
                    perturbed_signals = signal_data.copy()
                    mask = np.random.rand(len(perturbed_signals)) < 0.05
                    perturbed_signals.loc[mask] = 0.0

                # 3. Run backtest with perturbed data
                res = self.run_backtest(perturbed_prices, perturbed_signals, **kwargs)
                if res and 'performance' in res:
                    sim_results.append(res['performance'])
                
                self.logger.info(f"Simulation {i+1}/{n_simulations} completed.")
            except Exception as e:
                self.logger.error(f"Simulation {i+1} failed: {e}")

        if not sim_results:
            return {"error": "All simulations failed."}

        # 4. Summarize results
        total_returns = [s.get('total_return', 0) for s in sim_results]
        sharpe_ratios = [s.get('sharpe_ratio', 0) for s in sim_results]
        
        mean_return = np.mean(total_returns)
        std_sharpe = np.std(sharpe_ratios)
        
        # Robustness score calculation: Higher mean return with lower volatility of Sharpe is better
        # Normalizing logic: (Mean Sharpe / (1 + Std Sharpe)) * (1 - Drawdown impact)
        # Simplified:
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
    prices = pd.DataFrame({
        'AAPL': 150 + np.random.randn(200).cumsum(),
        'TSLA': 700 + np.random.randn(200).cumsum() * 5
    }, index=dates)
    
    signals = pd.DataFrame({
        'AAPL': np.random.choice([-1, 0, 1], 200),
        'TSLA': np.random.choice([-1, 0, 1], 200)
    }, index=dates)
    
    try:
        backtester = AdvancedBacktester()
        results = backtester.run_backtest(prices, signals)
        
        logger.info(f"Multi-ticker backtest complete for: {results['tickers']}")
        logger.info(f"Final Portfolio Return: {results['performance'].get('total_return', 0):.2%}")
        logger.info(f"Number of trades executed: {len(results['trades'])}")
            
    except Exception as e:
        logger.error(f"Error in multi-ticker example: {e}", exc_info=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    example_usage()