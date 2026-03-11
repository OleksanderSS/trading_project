# src/pipeline/stages/stage_7_evaluation.py

import logging
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Optional, Dict, Any
from pathlib import Path

from src.pipeline.stages.base_stage import BaseStage
from src.config.unified_config_manager import UnifiedConfigManager
from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine
from src.core.logging.notifier import UniversalNotifier
from src.analytics.backtesting.engine import AdvancedBacktester
from src.metrics.financial.portfolio_metrics import PortfolioMetricsCalculator
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("EvaluationStage")

class EvaluationStage(BaseStage):
    """
    Етап 7: Оцінка Стратегії (Evaluation).
    Виконує реалістичний бектестинг, розрахунок професійних фінансових метрик та візуалізацію.
    """
    def __init__(self, config_manager: UnifiedConfigManager, brain: Dict[str, Any], **kwargs):
        super().__init__(config_manager, brain)
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
        Виконує фінальну оцінку ефективності та зберігає результати.
        """
        signals = kwargs.get('signals')
        if not signals:
            logger.warning("No signals found for evaluation. Skipping stage.")
            return {}

        signals_df = signals
        if signals_df.empty or 'price' not in signals_df.columns or 'signal' not in signals_df.columns:
            logger.warning("Signals DataFrame is missing required columns ('price', 'signal').")
            return {}

        logger.info("Starting evaluation stage with AdvancedBacktester...")

        try:
            # 1. Run Realistic Backtest
            backtest_results = self.backtester.run_backtest(
                price_data=signals_df['price'],
                signal_data=signals_df['signal'],
                volume_data=signals_df.get('volume'),
                volatility_data=self.brain.get('volatility_data')
            )

            if not backtest_results or 'portfolio_history' not in backtest_results:
                logger.error("Backtesting failed to return results.")
                return {}

            portfolio_history = backtest_results['portfolio_history']
            
            # 2. Calculate Professional Metrics
            logger.info("Calculating professional financial metrics...")
            financial_metrics = self.metrics_calculator.calculate(portfolio_history['total_value'])
            
            # 3. Deep Analysis via Unified Engine
            data_map = {
                'price_data': signals_df['price'],
                'signals': signals_df['signal'],
                'returns': portfolio_history['returns'].dropna(),
                'portfolio_data': portfolio_history,
                'news_data': self.brain.get('news_data'),
                'macro_data': self.brain.get('macro_data')
            }
            analysis_results = self.analytics_engine.run_full_analysis(data_map)

            # 4. Consolidate Summary
            summary = {
                'metrics': financial_metrics,
                'backtest_stats
                ': backtest_results.get('performance', {}),
                'analysis': analysis_results,
                'timestamp': pd.Timestamp.now().isoformat()
            }

            # 5. Save Summary and Visualization
            self._save_summary(summary)
            equity_path = self._plot_equity_curve(portfolio_history, financial_metrics)
            
            # 6. Send Notification
            self._send_notification(financial_metrics, equity_path)

            logger.info(f"Evaluation complete. Total Return: {financial_metrics.get('total_return_pct', 0):.2%}")
            
            return {'evaluation_summary': summary}

        except Exception as e:
            logger.error(f"Critical error during evaluation stage: {e}", exc_info=True)
            raise

    def _save_summary(self, summary: Dict):
        """Saves the evaluation summary to the results directory."""
        file_path = self.results_dir / f"summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        logger.info(f"Final summary saved to {file_path}")

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

    def _send_notification(self, metrics: Dict, img_path: str):
        """Sends a final report notification."""
        message = (
            f"🏁 **Pipeline Execution Finished**\n\n"
            f"📈 Total Return: {metrics.get('total_return_pct', 0):+.2%}\n"
            f"🛡 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
            f"📉 Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
            f"🗓 CAGR: {metrics.get('cagr', 0):.2%}"
        )
        self.notifier.send_report(message, image_path=img_path)