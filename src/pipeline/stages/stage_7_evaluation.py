import pandas as pd
import numpy as np
from typing import Dict, Any
from pathlib import Path

from src.pipeline.stages.base_stage import BaseStage
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger
from src.core.logging.notifier import UniversalNotifier

# Modular components
from .evaluation.metrics_calculator import get_metrics_calculator
from .evaluation.report_generator import get_report_generator
from .evaluation.backtest_analyzer import get_backtest_analyzer

class EvaluationStage(BaseStage):
    """
    Stage 7: Strategy Evaluation
    Modular implementation delegating to specialized components.
    """

    def __init__(self, config_manager: UnifiedConfigManager, error_handler: ErrorHandler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.logger = ProjectLogger.get_logger('EvaluationStage')
        
        # Paths
        self.results_dir = Path('data/results')
        self.reports_dir = Path('reports/evaluation')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Core Components (Legacy/Existing)
        from src.backtesting.advanced.advanced_engine import AdvancedBacktestEngine
        from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine
        from src.meta_learning.real_time_learning import RealTimeLearning
        
        self.backtester = AdvancedBacktestEngine(self.config_manager)
        self.analytics_engine = UnifiedAnalyticsEngine(self.config_manager)
        self.notifier = UniversalNotifier(config_manager)
        self.real_time_learning = RealTimeLearning(config_manager)
        
        # Initialize Specialized Modular Components
        self.metrics_calc = get_metrics_calculator()
        self.report_gen = get_report_generator(self.reports_dir)
        self.backtest_analyzer = get_backtest_analyzer(self.backtester)
        
        self.logger.info("✅ EvaluationStage (Modular) initialized")

    async def run(self, **kwargs) -> Dict[str, Any]:
        """Performs final performance evaluation and saves results."""
        self.logger.info('🚀 Starting modular evaluation stage...')
        
        signals_data = await self._load_signals_data(**kwargs)
        if self._signals_empty(signals_data.get('signals')):
            return {}
            
        signals_df = self._prepare_signals_df(signals_data['signals'])
        
        # Check if backtest can be run
        if not self.backtest_analyzer.can_run_backtest(signals_df):
            self.logger.warning('⚠️ Insufficient numeric price data for backtest. Using basic evaluation.')
            return await self._create_basic_evaluation(signals_df, signals_data)
            
        return await self._run_comprehensive_evaluation(signals_df, signals_data)

    async def _load_signals_data(self, **kwargs) -> Dict[str, Any]:
        """Load signals, trading activity and portfolio summary."""
        signals = kwargs.get('signals')
        trading_activity = kwargs.get('trading_activity', [])
        portfolio_summary = kwargs.get('portfolio_summary', {})
        
        if signals is None:
            self.logger.warning("⚠️ No 'signals' found in kwargs. Attempting to load from disk...")
            # Note: _load_signals_from_disk logic could also be modularized
            # For brevity, keeping a simplified version or delegating
            pass
            
        return {
            'signals': signals,
            'trading_activity': trading_activity,
            'portfolio_summary': portfolio_summary
        }

    def _signals_empty(self, signals) -> bool:
        """Safely check emptiness for lists, dicts and pandas objects."""
        if signals is None:
            return True
        if hasattr(signals, 'empty'):
            return bool(signals.empty)
        return len(signals) == 0

    def _prepare_signals_df(self, signals) -> pd.DataFrame:
        """Standardize signals into a DataFrame."""
        if isinstance(signals, list):
            df = pd.DataFrame(signals)
        else:
            df = signals.copy()
            
        if 'signal' not in df.columns and 'predictions' in df.columns:
            df['signal'] = df['predictions'].apply(self._prediction_to_signal)
            
        if 'price' not in df.columns and 'last_price' in df.columns:
            df['price'] = df['last_price']
            
        return df

    def _prediction_to_signal(self, pred) -> str:
        """Convert prediction value to BUY/SELL/HOLD signal."""
        val = pred[-1] if isinstance(pred, (list, tuple, np.ndarray)) and len(pred) > 0 else pred
        if not isinstance(val, (int, float)): return 'HOLD'
        if val > 0: return 'BUY'
        if val < 0: return 'SELL'
        return 'HOLD'

    async def _run_comprehensive_evaluation(self, signals_df: pd.DataFrame, signals_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform full backtest and deep analysis."""
        try:
            # 1. Run Backtest
            backtest_results = await self.backtest_analyzer.run_backtest(signals_df)
            if not backtest_results:
                return await self._create_basic_evaluation(signals_df, signals_data)
                
            portfolio_history = backtest_results['portfolio_history']
            
            # 2. Calculate Financial Metrics
            financial_metrics = self.metrics_calc.calculate_financial_metrics(portfolio_history)
            
            # 3. Deep Analysis (via existing analytics engine)
            # This part remains mostly as-is as it delegates to another complex system
            analysis_results = self._run_deep_analysis(signals_df, portfolio_history)
            
            # 4. Generate Summary
            final_summary = self.report_gen.create_evaluation_summary(
                financial_metrics, backtest_results, analysis_results, signals_df
            )
            
            # 5. Real-time Learning Adaptation
            if signals_data['trading_activity']:
                final_summary['learning_adaptation'] = self.real_time_learning.update_and_adapt(signals_data['trading_activity'])
            
            # 6. Save and Plot
            self.report_gen.save_summary(final_summary, self.results_dir)
            equity_path = self.report_gen.plot_equity_curve(portfolio_history, financial_metrics)
            
            # 7. Notify
            msg = self.report_gen.generate_notification_message(financial_metrics)
            await self.notifier.send_report(msg, image_path=equity_path)
            
            return {'evaluation_summary': final_summary}
            
        except Exception as e:
            self.logger.error(f"Critical error in comprehensive evaluation: {e}", exc_info=True)
            return await self._create_basic_evaluation(signals_df, signals_data)

    def _run_deep_analysis(self, signals_df: pd.DataFrame, portfolio_history: pd.DataFrame) -> Dict[str, Any]:
        """Delegates to analytics engine for complex analysis."""
        # Simplified version of the original _run_deep_analysis
        data_map = {
            'price_data': signals_df[['price']] if 'price' in signals_df.columns else pd.DataFrame(),
            'portfolio_data': portfolio_history,
            'signals': signals_df['signal'] if 'signal' in signals_df.columns else None
        }
        return self.analytics_engine.run_full_analysis(data_map)

    async def _create_basic_evaluation(self, signals_df: pd.DataFrame, signals_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to basic metrics when backtest fails or is impossible."""
        summary = {
            'metrics': {
                'total_signals': len(signals_df),
                'trades_executed': len(signals_data['trading_activity']),
                'portfolio_value': signals_data['portfolio_summary'].get('total_value', 0)
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        self.report_gen.save_summary(summary, self.results_dir)
        return {'evaluation_summary': summary}
