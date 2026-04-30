# src/pipeline/stages/stage_7_evaluation.py

import pandas as pd
import matplotlib.pyplot as plt
import json
import aiofiles
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from src.pipeline.stages.base_stage import BaseStage
from src.config.unified_config_manager import UnifiedConfigManager
from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine
from src.core.logging.notifier import UniversalNotifier
from src.analytics.backtesting.engine import AdvancedBacktester
from src.metrics.financial.portfolio_metrics import PortfolioMetricsCalculator
from src.core.logging.logger import ProjectLogger

class EvaluationStage(BaseStage):
    """
    Stage 7: Strategy Evaluation
    Performs realistic backtesting, calculates professional financial metrics, and generates visualizations.
    """
    def __init__(self, config_manager: UnifiedConfigManager, brain: Dict[str, Any], **kwargs):
        super().__init__(config_manager, brain)
        self.logger = ProjectLogger.get_logger("EvaluationStage")
        self.results_dir = Path("data/results")
        self.reports_dir = Path("reports/evaluation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.analytics_engine = UnifiedAnalyticsEngine(self.config_manager)
        self.backtester = AdvancedBacktester()
        self.metrics_calculator = PortfolioMetricsCalculator()
        self.notifier = UniversalNotifier(config_manager)

    async def run(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Performs final performance evaluation and saves results.
        """
        signals, trading_activity, portfolio_summary = await self._load_signals(**kwargs)
        if not signals:
            return {}
        
        signals_df = self._prepare_signals(signals)
        return await self._perform_evaluation(signals_df, trading_activity, portfolio_summary)

    async def _load_signals(self, **kwargs) -> tuple:
        """Load signals from kwargs or disk."""
        signals = kwargs.get('signals')
        trading_activity = kwargs.get('trading_activity', [])
        portfolio_summary = kwargs.get('portfolio_summary', {})
        
        if not signals:
            self.logger.warning("⚠️ No 'signals' found in kwargs. Attempting to load from disk...")
            signals, trading_activity, portfolio_summary = await self._load_signals_from_disk(kwargs)
        
        return signals, trading_activity, portfolio_summary

    async def _load_signals_from_disk(self, kwargs: dict) -> tuple:
        """Load signals from disk files."""
        batch_name = kwargs.get('batch_name')
        output_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
        
        if batch_name:
            batch_dir = output_dir / batch_name
            stage_6_file = batch_dir / "stage_6_results.json"
            
            if stage_6_file.exists():
                try:
                    content = await self._read_file_async(stage_6_file)
                    stage_6_results = json.loads(content)
                    
                    signals = stage_6_results.get('predictions', [])
                    trading_activity = stage_6_results.get('trade_history', [])
                    portfolio_summary = stage_6_results.get('portfolio_summary', {})
                    
                    self.logger.info(f"✅ Loaded {len(signals)} signals from {stage_6_file.name}")
                    self.logger.info(f"✅ Loaded {len(trading_activity)} trades")
                    return signals, trading_activity, portfolio_summary
                except Exception as e:
                    self.logger.error(f"❌ Error loading {stage_6_file}: {e}")
                    return [], [], {}
            else:
                self.logger.warning(f"⚠️ File not found: {stage_6_file}")
        else:
            self.logger.warning("⚠️ Could not find batch_name")
        
        return [], [], {}

    async def _read_file_async(self, file_path: Path) -> str:
        """Read file asynchronously."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            return await f.read()

    def _prepare_signals(self, signals) -> pd.DataFrame:
        """Prepare signals DataFrame."""
        self.logger.info(f"📊 Received {len(signals)} signals for evaluation")
        
        # Convert signals to DataFrame if they are in list format
        if isinstance(signals, list):
            signals_df = pd.DataFrame(signals)
        else:
            signals_df = signals.copy()
        
        # Add signal column if missing
        if 'signal' not in signals_df.columns:
            signals_df['signal'] = signals_df['predictions'].apply(self._create_signal_from_prediction)
        
        # ELITE FIX: Ensure 'price' column exists for backtester
        if 'price' not in signals_df.columns and 'last_price' in signals_df.columns:
            signals_df['price'] = signals_df['last_price']
            self.logger.info("Mapped 'last_price' to 'price' for backtesting.")
        
        return signals_df

    def _create_signal_from_prediction(self, pred) -> str:
        """Create trading signal from prediction."""
        if isinstance(pred, (int, float)):
            return self._signal_from_value(pred)
        elif isinstance(pred, (list, tuple, np.ndarray)) and len(pred) > 0:
            val = pred[-1] if isinstance(pred[-1], (int, float)) else 0
            return self._signal_from_value(val)
        else:
            return 'HOLD'

    def _signal_from_value(self, value) -> str:
        """Convert numeric value to signal."""
        if value > 0:
            return 'BUY'
        elif value < 0:
            return 'SELL'
        else:
            return 'HOLD'

    async def _perform_evaluation(self, signals_df: pd.DataFrame, trading_activity: list, portfolio_summary: dict) -> Dict[str, Any]:
        """Perform the actual evaluation."""
        # Verify if backtest can be launched
        if not self._can_run_backtest(signals_df):
            self.logger.warning("⚠️ No valid numeric price data for backtesting. Using basic evaluation.")
            return await self._create_basic_evaluation(signals_df, trading_activity, portfolio_summary)

        self.logger.info("Starting evaluation stage with AdvancedBacktester...")
        return await self._run_advanced_evaluation(signals_df, trading_activity)

    def _can_run_backtest(self, signals_df: pd.DataFrame) -> bool:
        """Check if backtest can be launched."""
        return (
            'price' in signals_df.columns and
            not signals_df['price'].isna().all() and
            pd.api.types.is_numeric_dtype(signals_df['price'])
        )

    async def _run_advanced_evaluation(self, signals_df: pd.DataFrame, trading_activity: list) -> Dict[str, Any]:
        """Run advanced backtesting evaluation."""
        try:
            # 1. Run Realistic Backtest
            backtest_results = await self._run_backtest(signals_df)
            
            if not backtest_results or 'portfolio_history' not in backtest_results:
                self.logger.warning("⚠️ Backtest returned empty results. Using basic evaluation.")
                return await self._create_basic_evaluation(signals_df, trading_activity, {})

            # 2. Calculate Professional Financial Metrics
            portfolio_history = backtest_results['portfolio_history']
            financial_metrics = self._calculate_financial_metrics(portfolio_history)
            
            # 3. Deep Analysis via Unified Engine
            analysis_results = self._run_deep_analysis(signals_df, portfolio_history)
            
            # 4. Consolidate Summary
            summary = self._create_evaluation_summary(financial_metrics, backtest_results, analysis_results)

            # 5. Save Summary and Visualization
            await self._save_summary(summary)
            equity_path = self._plot_equity_curve(portfolio_history, financial_metrics)
            
            # 6. Send Notification
            await self._send_notification(financial_metrics, equity_path)

            self.logger.info(f"Evaluation complete. Total Return: {financial_metrics.get('total_return_pct', 0):.2%}")
            
            return {'evaluation_summary': summary}

        except Exception as e:
            self.handle_stage_error(e, context="EvaluationStage", severity="error")
            self.logger.error(f"Critical error during evaluation stage: {e}", exc_info=True)
            return await self._create_basic_evaluation(signals_df, trading_activity, {})

    async def _run_backtest(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """Runs the backtest using the AdvancedBacktester."""
        self.logger.info(f"Preparing data for AdvancedBacktester. Input shape: {signals_df.shape}")
        self.logger.debug(f"Columns: {signals_df.columns.tolist()}")
        
        try:
            # Pivot price and signal data for the backtester (needs tickers as columns)
            if 'ticker' in signals_df.columns:
                # If we have multiple timestamps and they are valid
                if 'timestamp' in signals_df.columns and signals_df['timestamp'].notna().any():
                    self.logger.info("Pivoting by timestamp and ticker...")
                    price_pivot = signals_df.pivot_table(index='timestamp', columns='ticker', values='price', aggfunc='mean')
                    
                    sig_numeric = signals_df.copy()
                    sig_numeric['sig_val'] = sig_numeric['signal'].map({'BUY': 1, 'SELL': -1, 'HOLD': 0})
                    signal_pivot = sig_numeric.pivot_table(index='timestamp', columns='ticker', values='sig_val', aggfunc='mean')
                else:
                    self.logger.info("No valid timestamps found. Aggregating as single snapshot...")
                    # Single snapshot - aggregate by ticker first to avoid duplicate columns
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
                price_pivot = signals_df[['price']]
                signal_pivot = signals_df[['signal']]

            self.logger.info(f"Pivoted data shape: {price_pivot.shape}")
            import asyncio
            loop = asyncio.get_event_loop()
            
            self.logger.info(f"Executing AdvancedBacktester on {len(price_pivot)} time points...")
            results = await loop.run_in_executor(
                None, 
                self.backtester.run_backtest, 
                price_pivot, 
                signal_pivot
            )
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Backtest execution failed: {e}")
            return {}

    async def _save_summary(self, summary: Dict):
        """Saves the evaluation summary to the results directory."""
        file_path = self.results_dir / f"summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(summary, indent=4, default=str))
        self.logger.info(f"Final summary saved to {file_path}")

    def _plot_equity_curve(self, history: pd.DataFrame, metrics: Dict) -> str:
        """Generates and saves the equity curve plot."""
        plt.figure(figsize=(12, 6))
        plt.plot(history.index, history['total_value'], label='Portfolio Value', color='green', linewidth=2)
        plt.title(f"Equity Curve | Return: {metrics.get('total_return_pct', 0):.2%} | Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
        plt.grid(True, alpha=0.3)
        plt.ylabel("Value ($)")
        
        plot_path = self.reports_dir / "equity_curve.png"
        plt.savefig(plot_path)
        plt.close()
        return str(plot_path)

    async def _create_basic_evaluation(self, signals_df: pd.DataFrame, trading_activity: list, portfolio_summary: dict) -> Dict[str, Any]:
        """Creates a basic evaluation summary when full backtest data is unavailable."""
        self.logger.info("📊 Creating basic evaluation from available data...")
        
        summary = {
            'metrics': {
                'total_signals': len(signals_df),
                'unique_tickers': signals_df['ticker'].nunique() if 'ticker' in signals_df.columns else 0,
                'avg_confidence': signals_df['confidence'].mean() if 'confidence' in signals_df.columns else 0,
                'trades_executed': len(trading_activity),
                'portfolio_value': portfolio_summary.get('total_value', 0),
                'cash_balance': portfolio_summary.get('cash', 0)
            },
            'signals_summary': {
                'total': len(signals_df),
                'by_ticker': signals_df.groupby('ticker').size().to_dict() if 'ticker' in signals_df.columns else {}
            },
            'trading_activity': trading_activity,
            'portfolio_summary': portfolio_summary,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        await self._save_summary(summary)
        self.logger.info(f"✅ Basic evaluation complete: {summary['metrics']['total_signals']} signals, {summary['metrics']['trades_executed']} trades")
        
        return {'evaluation_summary': summary}

    async def _send_notification(self, metrics: Dict, img_path: str):
        """Sends a final report notification via UniversalNotifier."""
        message = (
            f"🏁 **Pipeline Execution Finished**\n\n"
            f"📈 Total Return: {metrics.get('total_return_pct', 0):+.2%}\n"
            f"🛡 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
            f"📉 Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
            f"🗓 CAGR: {metrics.get('cagr', 0):.2%}"
        )
        await self.notifier.send_report(message, image_path=img_path)

    def _extract_market_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract market indicators from price data for analyzers."""
        try:
            if price_data is None or price_data.empty:
                return pd.DataFrame()
            
            indicators = pd.DataFrame(index=price_data.index)
            if 'close' in price_data.columns:
                indicators['volatility_5d'] = price_data['close'].pct_change().rolling(5).std()
                indicators['volatility_20d'] = price_data['close'].pct_change().rolling(20).std()
                indicators['volatility_ratio'] = indicators['volatility_5d'] / indicators['volatility_20d']
                indicators['trend_5d'] = price_data['close'].pct_change(5)
                indicators['trend_20d'] = price_data['close'].pct_change(20)
                indicators['trend_alignment'] = (indicators['trend_5d'] > 0).astype(int) == (indicators['trend_20d'] > 0).astype(int)
                indicators['price_to_ma20'] = price_data['close'] / price_data['close'].rolling(20).mean()
            
            if 'volume' in price_data.columns:
                indicators['volume_ratio'] = price_data['volume'] / price_data['volume'].rolling(20).mean()
            
            return indicators.dropna()
        except Exception as e:
            self.logger.warning(f"Failed to extract market indicators: {e}")
            return pd.DataFrame()

    def _extract_predictions_from_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Extract predictions from signals DataFrame."""
        try:
            predictions = {}
            if signals_df is not None and not signals_df.empty:
                if 'prediction' in signals_df.columns:
                    predictions['ensemble'] = self._get_last_prediction_value(signals_df)
                elif 'signal' in signals_df.columns:
                    predictions['ensemble'] = self._signal_to_numeric_value(signals_df)
            return pd.DataFrame(predictions, index=[0]) if predictions else pd.DataFrame()
        except Exception as e:
            self.logger.warning(f"Failed to extract predictions from signals: {e}")
            return pd.DataFrame()

    def _get_last_prediction_value(self, signals_df: pd.DataFrame) -> float:
        """Get the last prediction value from DataFrame."""
        return signals_df['prediction'].iloc[-1] if len(signals_df) > 0 else 0.0

    def _signal_to_numeric_value(self, signals_df: pd.DataFrame) -> float:
        """Convert last signal to numeric value."""
        if len(signals_df) == 0:
            return 0.0
        
        last_signal = signals_df['signal'].iloc[-1]
        if last_signal == 'BUY':
            return 1.0
        elif last_signal == 'SELL':
            return -1.0
        else:
            return 0.0

    def _extract_performance_metrics(self, portfolio_history: pd.DataFrame) -> Dict[str, Any]:
        """Extract performance metrics from portfolio history."""
        try:
            if portfolio_history is None or portfolio_history.empty:
                return {}
            
            metrics = {}
            if 'returns' in portfolio_history.columns:
                returns = portfolio_history['returns'].dropna()
                if not returns.empty:
                    metrics['total_return'] = (1 + returns).prod() - 1
                    metrics['volatility'] = returns.std()
                    metrics['sharpe_ratio'] = returns.mean() / (returns.std() + 1e-9)
            return metrics
        except Exception as e:
            self.logger.warning(f"Failed to extract performance metrics: {e}")
            return {}

    def _extract_features_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract features data from price data."""
        try:
            if price_data is None or price_data.empty:
                return pd.DataFrame()
            feature_cols = [col for col in price_data.columns if col not in ['timestamp', 'datetime']]
            return price_data[feature_cols].tail(100)
        except Exception as e:
            self.logger.warning(f"Failed to extract features data: {e}")
            return pd.DataFrame()

    def _calculate_financial_metrics(self, portfolio_history: pd.DataFrame) -> Dict[str, Any]:
        """Calculate professional financial metrics."""
        self.logger.info("Calculating professional financial metrics...")
        financial_metrics = self.metrics_calculator.calculate(portfolio_history['total_value'])
        return financial_metrics

    def _run_deep_analysis(self, signals_df: pd.DataFrame, portfolio_history: pd.DataFrame) -> Dict[str, Any]:
        """Run deep analysis via Unified Engine."""
        # Prepare price data
        price_data = signals_df[['price']] if 'price' in signals_df.columns else pd.DataFrame({'price': signals_df['price']})
        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame(name='price')
        
        if 'close' not in price_data.columns and 'price' in price_data.columns:
            price_data['close'] = price_data['price']
        if 'volume' not in price_data.columns:
            price_data['volume'] = 0
        
        market_data = pd.DataFrame({
            'price': signals_df['price'] if 'price' in signals_df.columns else 0,
            'volume': signals_df.get('volume', 0),
            'returns': portfolio_history['returns'].dropna() if 'returns' in portfolio_history else 0
        })
        
        data_map = self._create_analysis_data_map(signals_df, portfolio_history, price_data, market_data)
        data_map = {k: v for k, v in data_map.items() if v is not None and (not hasattr(v, 'empty') or not v.empty)}
        return self.analytics_engine.run_full_analysis(data_map)

    def _create_analysis_data_map(self, signals_df: pd.DataFrame, portfolio_history: pd.DataFrame, 
                                 price_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Create data map for analysis."""
        return {
            'price_data': price_data,
            'market_data': market_data,
            'signals': signals_df['signal'] if 'signal' in signals_df.columns else None,
            'returns': portfolio_history['returns'].dropna() if 'returns' in portfolio_history else pd.Series(),
            'portfolio_data': portfolio_history,
            'news_data': self.brain.get('news_data'),
            'macro_data': self.brain.get('macro_data'),
            'market_indicators': self._extract_market_indicators(price_data),
            'economic_data': self.brain.get('macro_data'),
            'historical_economic_data': self.brain.get('macro_data'),
            'predictions': self._extract_predictions_from_signals(signals_df),
            'performance_metrics': self._extract_performance_metrics(portfolio_history),
            'features_data': self._extract_features_data(price_data),
            'target_series': signals_df['signal'] if 'signal' in signals_df.columns else pd.Series(),
            'causal_series': price_data['close'] if 'close' in price_data.columns else pd.Series(),
        }

    def _create_evaluation_summary(self, financial_metrics: Dict[str, Any], backtest_results: Dict[str, Any], 
                                 analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create evaluation summary."""
        return {
            'metrics': financial_metrics,
            'backtest_stats': backtest_results.get('performance', {}),
            'analysis': analysis_results,
            'timestamp': pd.Timestamp.now().isoformat()
        }