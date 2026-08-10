from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger
from src.core.logging.notifier import UniversalNotifier
from src.pipeline.stages.base_stage import BaseStage

from .backtest_analyzer import get_backtest_analyzer
from .io import save_evaluation_summary

# Modular components
from .metrics_calculator import get_evaluation_metrics_calculator as get_metrics_calculator
from .pipeline_control_artifacts import (
    build_evaluation_metric_candidate,
    write_evaluation_metric_artifact_candidate,
)
from .report_generator import get_report_generator


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
        from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine
        from src.backtesting.advanced.advanced_engine import AdvancedBacktestEngine
        self.backtester = AdvancedBacktestEngine(self.config_manager)
        self.analytics_engine = UnifiedAnalyticsEngine(self.config_manager)
        self.notifier = UniversalNotifier(config_manager)

        # Initialize Specialized Modular Components
        self.metrics_calc = get_metrics_calculator()
        self.report_gen = get_report_generator(self.reports_dir)
        self.backtest_analyzer = get_backtest_analyzer(self.backtester)

        self.logger.info("✅ EvaluationStage (Modular) initialized")

    async def run(self, **kwargs) -> dict[str, Any]:
        """Performs final performance evaluation and saves results."""
        self.logger.info('🚀 Starting modular evaluation stage...')

        signals_data = self._load_signals_data(**kwargs)
        if self._signals_empty(signals_data.get('signals')):
            return {}

        signals_df = self._prepare_signals_df(signals_data['signals'])

        # Check if backtest can be run
        if not self.backtest_analyzer.can_run_backtest(signals_df):
            self.logger.warning('⚠️ Insufficient numeric price data for backtest. Using basic evaluation.')
            return self._create_basic_evaluation(signals_df, signals_data)

        return await self._run_comprehensive_evaluation(signals_df, signals_data)

    def _load_signals_data(self, **kwargs) -> dict[str, Any]:
        """Load signals, trading activity and portfolio summary."""
        signals = kwargs.get('signals')
        if signals is None:
            # The normal review pipeline skips Stage 6, so Stage 5 predictions
            # become the evaluation signals directly.
            signals = kwargs.get('predictions')
        trading_activity = kwargs.get('trading_activity', [])
        portfolio_summary = kwargs.get('portfolio_summary', {})

        if signals is None:
            self.logger.warning("⚠️ No 'signals' found in kwargs. Attempting to load from disk...")
            pass

        return {
            'signals': signals,
            'trading_activity': trading_activity,
            'portfolio_summary': portfolio_summary,
            'analysis_inputs': self._collect_analysis_inputs(kwargs),
            'notification_authorized': kwargs.get(
                'evaluation_notification_authorized'
            ) is True,
        }

    def _collect_analysis_inputs(
        self,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        aliases = {
            'price_data': ('price_data', 'market_data'),
            'market_indicators': ('market_indicators',),
            'features_data': ('features_data', 'features_df'),
            'news_data': ('news_data',),
            'macro_data': ('macro_data', 'economic_data'),
            'model_results': ('model_results', 'models_metadata'),
            'training_metrics': ('training_metrics',),
            'validation_metrics': ('validation_metrics',),
            'target_series': ('target_series',),
            'causal_series': ('causal_series',),
        }
        collected: dict[str, Any] = {}
        for target_key, source_keys in aliases.items():
            for source_key in source_keys:
                value = kwargs.get(source_key)
                if value is not None:
                    collected[target_key] = value
                    break
        return collected

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
        if not isinstance(val, (int, float)):
            return 'HOLD'
        if val > 0:
            return 'BUY'
        if val < 0:
            return 'SELL'
        return 'HOLD'

    async def _run_comprehensive_evaluation(self, signals_df: pd.DataFrame, signals_data: dict[str, Any]) -> dict[str, Any]:
        """Perform full backtest and deep analysis with optional stress testing."""
        try:
            # 1. Run Backtest
            backtest_results = await self.backtest_analyzer.run_backtest(signals_df)
            if not backtest_results:
                return self._create_basic_evaluation(signals_df, signals_data)

            portfolio_history = backtest_results['portfolio_history']

            # 2. Calculate Financial Metrics
            financial_metrics = self.metrics_calc.calculate_financial_metrics(portfolio_history)

            # 3. Stress Testing (if enabled in config)
            stress_test_results = {}
            if self.config_manager.get('evaluation.enable_stress_testing', False):
                stress_test_results = self._run_stress_testing(portfolio_history, financial_metrics)

            # 4. Deep Analysis (via existing analytics engine)
            # This part remains mostly as-is as it delegates to another complex system
            analysis_results = self._run_deep_analysis(
                signals_df,
                portfolio_history,
                signals_data.get('analysis_inputs', {}),
            )

            # 5. Generate Summary
            final_summary = self.report_gen.create_evaluation_summary(
                financial_metrics, backtest_results, analysis_results, signals_df
            )
            if backtest_results.get('is_simulated_data'):
                final_summary['is_simulated_data'] = True
                self.logger.warning(
                    '⚠️ Evaluation metrics are based on randomly-generated '
                    'simulation data (input signals were too thin), not real market data'
                )

            # 6. Add stress testing results if available
            if stress_test_results:
                final_summary['stress_testing'] = stress_test_results

            # 7. Learning remains proposal-only and outside Stage 7.
            if signals_data['trading_activity']:
                final_summary['learning_review_candidate'] = (
                    self._build_learning_review_candidate(
                        signals_data['trading_activity']
                    )
                )

            notification_authorized = bool(
                signals_data.get('notification_authorized')
            )
            final_summary['notification_status'] = (
                'authorized_delivery_pending'
                if notification_authorized
                else 'review_only_not_sent'
            )

            # 8. Save and Plot
            summary_path = self.report_gen.save_summary(final_summary, self.results_dir)
            pipeline_control_paths = self._write_pipeline_control_evaluation_candidate(
                financial_metrics=financial_metrics,
                backtest_results=backtest_results,
                final_summary=final_summary,
                signals_df=signals_df,
                portfolio_history=portfolio_history,
                summary_path=summary_path,
            )
            if pipeline_control_paths:
                final_summary['pipeline_control_evaluation_metric_artifacts'] = pipeline_control_paths
                if summary_path:
                    save_evaluation_summary(Path(summary_path), final_summary)
            equity_path = self.report_gen.plot_equity_curve(portfolio_history, financial_metrics)

            # 9. External notification requires an explicit per-run opt-in.
            if notification_authorized:
                msg = self.report_gen.generate_notification_message(financial_metrics)
                await self.notifier.send_report(msg, image_path=equity_path)
                final_summary['notification_status'] = 'authorized_delivery_attempted'
                if summary_path:
                    save_evaluation_summary(Path(summary_path), final_summary)

            return {'evaluation_summary': final_summary}

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Critical error in comprehensive evaluation: {e}", exc_info=True)
            return self._create_basic_evaluation(signals_df, signals_data)

    def _build_learning_review_candidate(
        self,
        trading_activity: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            'status': 'proposal_only_pending_dean_os_review',
            'observed_trade_count': len(trading_activity),
            'learning_applied': False,
            'model_weights_changed': False,
            'risk_parameters_changed': False,
            'production_config_written': False,
            'learning_memory_written': False,
            'required_next_step': (
                'Route evaluated outcomes through the DEAN-OS outcome and '
                'learning-review workflow before any adaptation.'
            ),
        }

    def _write_pipeline_control_evaluation_candidate(
        self,
        *,
        financial_metrics: dict[str, Any],
        backtest_results: dict[str, Any],
        final_summary: dict[str, Any],
        signals_df: pd.DataFrame,
        portfolio_history: pd.DataFrame,
        summary_path: str | Path | None,
    ) -> dict[str, Any]:
        try:
            candidate = build_evaluation_metric_candidate(
                financial_metrics=financial_metrics,
                backtest_results=backtest_results,
                evaluation_summary=final_summary,
                signals_df=signals_df,
                portfolio_history=portfolio_history,
                summary_path=summary_path,
            )
            context_key = Path(summary_path).stem if summary_path else "stage_7_evaluation"
            return write_evaluation_metric_artifact_candidate(
                output_dir=self.results_dir,
                candidate=candidate,
                context_key=context_key,
            )
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError, OSError) as e:
            self.logger.warning(f"Could not write pipeline-control evaluation metric candidate: {e}")
            return {}

    def _run_deep_analysis(
        self,
        signals_df: pd.DataFrame,
        portfolio_history: pd.DataFrame,
        analysis_inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run only analyzers whose declared inputs are available."""
        data_map = dict(analysis_inputs or {})
        price_data_source = 'upstream_price_data'
        if 'price_data' not in data_map:
            features_data = data_map.get('features_data')
            derived_from_features = (
                isinstance(features_data, pd.DataFrame)
                and not features_data.empty
                and bool({'close', 'price'}.intersection(features_data.columns))
            )
            price_data = (
                features_data.copy()
                if derived_from_features
                else signals_df.copy()
            )
            price_data_source = (
                'derived_from_features_data'
                if derived_from_features
                else 'derived_from_stage5_signals'
            )
            if 'close' not in price_data.columns and 'price' in price_data.columns:
                price_data = price_data.rename(columns={'price': 'close'})
            if 'close' in price_data.columns and not price_data.empty:
                data_map['price_data'] = price_data
        data_map['portfolio_data'] = portfolio_history
        if 'signal' in signals_df.columns:
            data_map['signals'] = signals_df['signal']

        try:
            price_contexts = self._partition_price_contexts(
                data_map.get('price_data')
            )
            if len(price_contexts) > 1:
                results = self._run_context_partitioned_analysis(
                    data_map,
                    price_contexts,
                )
            else:
                results = self.analytics_engine.run_full_analysis(data_map)
            if isinstance(results, dict):
                results = dict(results)
                if len(price_contexts) == 1:
                    _, price_frame = next(iter(price_contexts.items()))
                    results['_stage7_context_window'] = (
                        self._price_context_window(price_frame)
                    )
                results['_stage7_analysis_contract'] = {
                    'status': 'supporting_review_context_only',
                    'input_keys': sorted(data_map),
                    'price_data_source': price_data_source,
                    'price_context_count': len(price_contexts),
                    'price_context_partitioned': len(price_contexts) > 1,
                    'can_clear_locked_evidence': False,
                    'can_promote_model': False,
                    'can_trade': False,
                }
            return results
        except Exception as e:
            self.logger.warning(f'Deep analysis unavailable (missing optional data): {e}')
            return {
                '_analysis_coverage': {
                    'status': 'analysis_engine_failed_closed',
                    'error_type': type(e).__name__,
                    'evidence_class': 'supporting_analysis_not_locked_evidence',
                    'can_promote_model': False,
                    'can_trade': False,
                }
            }

    def _partition_price_contexts(
        self,
        price_data: Any,
    ) -> dict[str, pd.DataFrame]:
        if not isinstance(price_data, pd.DataFrame) or price_data.empty:
            return {}

        frame = price_data
        index_context_names = [
            name for name in frame.index.names
            if name in {'ticker', 'symbol', 'interval', 'timeframe'}
            and name not in frame.columns
        ]
        if index_context_names:
            frame = frame.reset_index(index_context_names)

        context_columns = [
            name for name in ('ticker', 'symbol', 'interval', 'timeframe')
            if name in frame.columns
        ]
        if not context_columns:
            return {'all_prices': frame}

        contexts: dict[str, pd.DataFrame] = {}
        grouper: str | list[str] = (
            context_columns[0]
            if len(context_columns) == 1
            else context_columns
        )
        for raw_key, group in frame.groupby(
            grouper,
            sort=True,
            dropna=False,
        ):
            values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            context_key = '|'.join(
                f'{column}={value}'
                for column, value in zip(context_columns, values, strict=True)
            )
            contexts[context_key] = group.copy()
        return contexts

    def _context_invariant_analyzers(
        self,
        data_map: dict[str, Any],
        price_contexts: dict[str, pd.DataFrame],
    ) -> set[str]:
        """Analyzers that read nothing this loop varies.

        Derived from the engine's own data_mapping rather than named here,
        so an analyzer that starts reading price_data stops being counted
        automatically. Empty when there is only one context, where the
        distinction cannot matter.
        """
        if len(price_contexts) < 2:
            return set()
        varied = {'price_data'}
        engine = self.analytics_engine
        mapping = getattr(engine, 'analyzer_data_map', {}) or {}
        return {
            name for name, inputs in mapping.items()
            if inputs and not (set(inputs) & varied)
        }

    def _run_context_partitioned_analysis(
        self,
        data_map: dict[str, Any],
        price_contexts: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        context_results: dict[str, Any] = {}
        context_coverage: dict[str, Any] = {}
        executed_analyzers: set[str] = set()
        failed_analyzers: set[str] = set()
        disabled_analyzers: set[str] = set()

        # Analyzers whose inputs this loop does not vary are run ONCE.
        #
        # The loop swaps price_data and nothing else, so an analyzer reading
        # features_data sees the identical frame in all 66 contexts.
        # feature_drift does exactly that, and it also holds a single
        # project-wide baseline (reports/drift/reference_features.parquet),
        # so it compared one frame against itself 66 times and reported a
        # drift score of exactly 0.0 every time -- measured on the
        # 2026-08-10 run: 40 results, one distinct score, zero contexts
        # detecting drift. The other 26 spent the full 90s budget and timed
        # out reaching the same conclusion.
        #
        # Running it once is not a shortcut: 66 identical computations
        # cannot say more than one. Per-context drift would need a baseline
        # per context, which is a design decision and not this fix.
        invariant = self._context_invariant_analyzers(data_map, price_contexts)
        invariant_results: dict[str, Any] = {}

        for index, (context_key, price_frame) in enumerate(price_contexts.items()):
            context_data_map = dict(data_map)
            context_data_map['price_data'] = price_frame
            skip = invariant if index else set()
            # Per-context budget. 30s was chosen to stop a five-hour Stage 7,
            # and it did -- by timing out 54 of 66 contexts on 2026-08-09,
            # each recorded with an empty error message. A timeout that most
            # contexts cannot meet is not a budget, it is a silent skip.
            #
            # Configurable, because the right number depends on the feature
            # count and the machine; and now that a timeout says so in the
            # result, the next run reports whether this one is enough.
            result = self.analytics_engine.run_full_analysis(
                context_data_map,
                timeout=self.config_manager.get(
                    'analysis.engine.context_timeout_seconds', 90
                ),
                skip=skip,
            )
            if isinstance(result, dict):
                # Carry the one computed answer into every later context, so
                # the report still shows a result everywhere -- it just is
                # not recomputed 65 more times to reach the same number.
                for name in skip:
                    if name in invariant_results:
                        result[name] = invariant_results[name]
                for name in invariant:
                    if not index and name in result:
                        invariant_results[name] = result[name]
            if isinstance(result, dict):
                result = dict(result)
                result['_stage7_context_window'] = (
                    self._price_context_window(price_frame)
                )
            context_results[context_key] = result
            coverage = (
                result.get('_analysis_coverage', {})
                if isinstance(result, dict)
                else {}
            )
            context_coverage[context_key] = coverage
            executed_analyzers.update(coverage.get('executed', []))
            failed_analyzers.update(coverage.get('failed', []))
            disabled_analyzers.update(coverage.get('disabled', []))

        return {
            'analysis_by_context': context_results,
            '_analysis_coverage': {
                'status': 'stage7_context_partitioned_analysis_recorded',
                'context_count': len(price_contexts),
                'context_keys': sorted(price_contexts),
                'executed_analyzers': sorted(executed_analyzers),
                'failed_analyzers': sorted(failed_analyzers),
                'disabled_analyzers': sorted(disabled_analyzers),
                'context_coverage': context_coverage,
                'evidence_class': (
                    'supporting_analysis_not_locked_evidence'
                ),
                'can_promote_model': False,
                'can_trade': False,
            },
        }

    def _price_context_window(
        self,
        frame: pd.DataFrame,
    ) -> dict[str, Any]:
        window = {
            'row_count': int(len(frame)),
            'start': None,
            'end': None,
            'timestamp_source': None,
        }
        timestamps = None
        if isinstance(frame.index, pd.DatetimeIndex):
            timestamps = pd.Series(frame.index, index=frame.index)
            window['timestamp_source'] = 'datetime_index'
        else:
            for column in (
                'timestamp',
                'datetime',
                'date',
                'time',
            ):
                if column not in frame.columns:
                    continue
                parsed = pd.to_datetime(
                    frame[column],
                    errors='coerce',
                    utc=True,
                ).dropna()
                if not parsed.empty:
                    timestamps = parsed
                    window['timestamp_source'] = column
                    break
        if timestamps is None or len(timestamps) == 0:
            return window
        start = pd.Timestamp(timestamps.min())
        end = pd.Timestamp(timestamps.max())
        if start.tzinfo is None:
            start = start.tz_localize('UTC')
        else:
            start = start.tz_convert('UTC')
        if end.tzinfo is None:
            end = end.tz_localize('UTC')
        else:
            end = end.tz_convert('UTC')
        window['start'] = start.isoformat()
        window['end'] = end.isoformat()
        return window

    def _run_stress_testing(self, portfolio_history: pd.DataFrame, financial_metrics: dict) -> dict[str, Any]:
        """Run stress testing scenarios on the portfolio."""
        stress_results = {
            'scenarios': {},
            'summary': {}
        }

        try:
            # Scenario 1: High Volatility Stress
            if 'total_return_pct' in financial_metrics:
                stress_results['scenarios']['high_volatility'] = {
                    'description': 'Portfolio performance under high volatility conditions',
                    'impact': financial_metrics['total_return_pct'] * 0.5,  # Assume 50% reduction
                    'status': 'passed' if financial_metrics['total_return_pct'] > 0 else 'failed'
                }

            # Scenario 2: Market Crash Stress
            # EvaluationMetricsCalculator._calculate_basic_metrics() only
            # ever produces 'max_drawdown' (a fraction, e.g. -0.47), never
            # 'max_drawdown_pct' - this scenario silently never ran before.
            if 'max_drawdown' in financial_metrics:
                max_drawdown_pct = abs(financial_metrics['max_drawdown']) * 100
                stress_results['scenarios']['market_crash'] = {
                    'description': 'Portfolio performance during market crash',
                    'max_drawdown_stress': max_drawdown_pct * 1.5,
                    'status': 'passed' if max_drawdown_pct < 20 else 'warning'
                }

            # Scenario 3: Low Liquidity Stress
            if 'sharpe_ratio' in financial_metrics:
                stress_results['scenarios']['low_liquidity'] = {
                    'description': 'Portfolio performance under low liquidity conditions',
                    'sharpe_stress': financial_metrics['sharpe_ratio'] * 0.7,
                    'status': 'passed' if financial_metrics['sharpe_ratio'] > 0.5 else 'warning'
                }

            stress_results['summary'] = {
                'total_scenarios': len(stress_results['scenarios']),
                'passed': sum(1 for s in stress_results['scenarios'].values() if s['status'] == 'passed'),
                'warnings': sum(1 for s in stress_results['scenarios'].values() if s['status'] == 'warning'),
                'failed': sum(1 for s in stress_results['scenarios'].values() if s['status'] == 'failed')
            }

            self.logger.info(f"✅ Stress testing completed: {stress_results['summary']['passed']}/{stress_results['summary']['total_scenarios']} passed")

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error in stress testing: {e}", exc_info=True)
            stress_results['error'] = str(e)  # type: ignore

        return stress_results

    def _create_basic_evaluation(self, signals_df: pd.DataFrame, signals_data: dict[str, Any]) -> dict[str, Any]:
        """Fallback to basic metrics when backtest fails or is impossible."""
        summary = {
            'metrics': {
                'total_signals': len(signals_df),
                'trades_executed': len(signals_data['trading_activity']),
                'portfolio_value': signals_data['portfolio_summary'].get('total_value', 0)
            },
            'timestamp': pd.Timestamp.now().isoformat(),
            'notification_status': 'basic_evaluation_not_sent',
        }
        if signals_data['trading_activity']:
            summary['learning_review_candidate'] = (
                self._build_learning_review_candidate(
                    signals_data['trading_activity']
                )
            )
        self.report_gen.save_summary(summary, self.results_dir)
        return {'evaluation_summary': summary}
