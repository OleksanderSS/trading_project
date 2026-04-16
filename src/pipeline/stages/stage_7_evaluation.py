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
        trading_activity = kwargs.get('trading_activity', [])
        portfolio_summary = kwargs.get('portfolio_summary', {})
        
        # ✅ NEW: Якщо signals не знайдені, спробуємо завантажити з диска (Stage 6 не запускалась)
        if not signals:
            logger.warning("⚠️ No 'signals' found in kwargs. Attempting to load from disk...")
            
            # Спробуємо знайти stage_6_results.json
            from pathlib import Path
            import json
            
            # Витягуємо batch_name з kwargs або шукаємо найновіший
            batch_name = kwargs.get('batch_name')
            output_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
            
            if not batch_name:
                # Шукаємо найновіший batch
                batch_dirs = list(output_dir.glob('test_ticker_*'))
                if batch_dirs:
                    batch_name = max(batch_dirs, key=lambda p: p.stat().st_mtime).name
                    logger.info(f"🔍 Знайдено найновіший batch: {batch_name}")
            
            if batch_name:
                batch_dir = output_dir / batch_name
                stage_6_file = batch_dir / "stage_6_results.json"
                
                if stage_6_file.exists():
                    try:
                        with open(stage_6_file, 'r') as f:
                            stage_6_results = json.load(f)
                        
                        signals = stage_6_results.get('predictions', [])
                        trading_activity = stage_6_results.get('trade_history', [])
                        portfolio_summary = stage_6_results.get('portfolio_summary', {})
                        
                        logger.info(f"✅ Завантажено {len(signals)} сигналів з {stage_6_file.name}")
                        logger.info(f"✅ Завантажено {len(trading_activity)} торгів")
                    except Exception as e:
                        logger.error(f"❌ Помилка завантаження {stage_6_file}: {e}")
                else:
                    logger.warning(f"⚠️ Файл не знайдено: {stage_6_file}")
            else:
                logger.warning(f"⚠️ Не вдалося знайти batch_name")
        
        if not signals:
            logger.warning("No signals found for evaluation. Skipping stage.")
            return {}
        
        logger.info(f"📊 Received {len(signals)} signals for evaluation")

        # ✅ Конвертуємо signals в DataFrame якщо це list
        if isinstance(signals, list):
            signals_df = pd.DataFrame(signals)
        else:
            signals_df = signals
        
        logger.info(f"📋 Signals DataFrame shape: {signals_df.shape}")
        logger.info(f"📋 Signals columns: {list(signals_df.columns)}")
        
        # ✅ Перевіряємо наявність необхідних колонок
        required_cols = ['ticker', 'predictions']
        missing_cols = [col for col in required_cols if col not in signals_df.columns]
        
        if missing_cols:
            logger.warning(f"⚠️ Missing required columns: {missing_cols}. Available: {list(signals_df.columns)}")
            # Спробуємо створити базову оцінку з наявних даних
            return self._create_basic_evaluation(signals_df, trading_activity, portfolio_summary)
        
        # ✅ Якщо немає price/signal колонок, створюємо їх з predictions
        if 'price' not in signals_df.columns and 'last_price' in signals_df.columns:
            signals_df['price'] = pd.to_numeric(signals_df['last_price'], errors='coerce')
        
        if 'signal' not in signals_df.columns:
            # Створюємо сигнал на основі predictions
            def create_signal(pred):
                if isinstance(pred, (int, float)):
                    return 'BUY' if pred > 0 else 'SELL' if pred < 0 else 'HOLD'
                elif isinstance(pred, (list, tuple)) and len(pred) > 0:
                    val = pred[-1] if isinstance(pred[-1], (int, float)) else 0
                    return 'BUY' if val > 0 else 'SELL' if val < 0 else 'HOLD'
                else:
                    return 'HOLD'
            
            signals_df['signal'] = signals_df['predictions'].apply(create_signal)
        
        # ✅ Перевіряємо чи можемо запустити backtest
        if 'price' not in signals_df.columns or signals_df['price'].isna().all() or not pd.api.types.is_numeric_dtype(signals_df['price']):
            logger.warning(f"⚠️ No valid price data for backtesting. Price dtype: {signals_df['price'].dtype if 'price' in signals_df.columns else 'N/A'}")
            return self._create_basic_evaluation(signals_df, trading_activity, portfolio_summary)

        logger.info("Starting evaluation stage with AdvancedBacktester...")

        try:
            # ✅ Додаткова перевірка перед backtesting
            if not pd.api.types.is_numeric_dtype(signals_df['price']):
                logger.warning(f"⚠️ Price column is not numeric: {signals_df['price'].dtype}")
                return self._create_basic_evaluation(signals_df, trading_activity, portfolio_summary)
            
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
            # ✅ FIX: Перевіряємо формат даних перед передачею аналізаторам
            # Аналізатори очікують DataFrame, а не Series
            
            # Конвертуємо Series в DataFrame якщо потрібно
            price_data = signals_df[['price']] if 'price' in signals_df.columns else pd.DataFrame({'price': signals_df['price']})
            if isinstance(price_data, pd.Series):
                price_data = price_data.to_frame(name='price')
            
            # Додаємо необхідні колонки для аналізаторів
            if 'close' not in price_data.columns and 'price' in price_data.columns:
                price_data['close'] = price_data['price']
            if 'volume' not in price_data.columns:
                price_data['volume'] = 0  # Placeholder якщо немає volume
            
            # Створюємо market_data для market_phase_analyzer
            market_data = pd.DataFrame({
                'price': signals_df['price'] if 'price' in signals_df.columns else 0,
                'volume': signals_df.get('volume', 0),
                'returns': portfolio_history['returns'].dropna() if 'returns' in portfolio_history else 0
            })
            
            data_map = {
                'price_data': price_data,  # DataFrame для critical_signal_detector
                'market_data': market_data,  # DataFrame для market_phase_analyzer
                'signals': signals_df['signal'] if 'signal' in signals_df.columns else None,
                'returns': portfolio_history['returns'].dropna() if 'returns' in portfolio_history else pd.Series(),
                'portfolio_data': portfolio_history,
                'news_data': self.brain.get('news_data'),
                'macro_data': self.brain.get('macro_data')
            }
            
            # Видаляємо None значення
            data_map = {k: v for k, v in data_map.items() if v is not None}
            
            analysis_results = self.analytics_engine.run_full_analysis(data_map)
            
            # ✅ Додаємо розширений Hedge Fund Analysis з factor exposures
            if 'returns' in data_map and not data_map['returns'].empty:
                try:
                    from src.analytics.analyzers.hedge_fund_analyzer import HedgeFundAnalyzer
                    hedge_fund_analyzer = HedgeFundAnalyzer()
                    
                    hedge_fund_results = hedge_fund_analyzer.analyze({
                        'returns': data_map['returns'],
                        'benchmark': None  # Можна додати benchmark якщо є
                    })
                    
                    # Додаємо factor analysis в результати
                    if 'factor_analysis' in hedge_fund_results:
                        analysis_results['hedge_fund_factor_analysis'] = hedge_fund_results['factor_analysis']
                        logger.info(f"✅ Hedge Fund Factor Analysis додано: {list(hedge_fund_results['factor_analysis'].keys())}")
                    
                    # Додаємо skill assessment
                    if 'skill_assessment' in hedge_fund_results:
                        analysis_results['manager_skill'] = hedge_fund_results['skill_assessment']
                        logger.info(f"✅ Manager Skill Assessment: {hedge_fund_results['skill_assessment'].get('rating', 'N/A')}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Hedge Fund Analysis failed: {e}")

            # 4. Consolidate Summary
            summary = {
                'metrics': financial_metrics,
                'backtest_stats': backtest_results.get('performance', {}),
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

        except (TypeError, ValueError, AttributeError) as e:
            logger.error(f"❌ Backtesting failed: {e}. Creating basic evaluation...")
            return self._create_basic_evaluation(signals_df, trading_activity, portfolio_summary)
        
        except Exception as e:
            logger.error(f"Critical error during evaluation stage: {e}", exc_info=True)
            return self._create_basic_evaluation(signals_df, trading_activity, portfolio_summary)

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

    def _create_basic_evaluation(self, signals_df: pd.DataFrame, trading_activity: list, portfolio_summary: dict) -> Dict[str, Any]:
        """
        Створює базову оцінку коли немає всіх необхідних даних для повного бектесту.
        """
        logger.info("📊 Creating basic evaluation from available data...")
        
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
        
        # Зберігаємо summary
        self._save_summary(summary)
        
        logger.info(f"✅ Basic evaluation complete: {summary['metrics']['total_signals']} signals, {summary['metrics']['trades_executed']} trades")
        
        return {'evaluation_summary': summary}

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