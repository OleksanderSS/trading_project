"""
Orchestrates the entire trading process for Stage 6.

This module connects the different components of the trading pipeline:
- Aggregates signals through the Consensus Engine.
- Resolves risk-compliant trade orders via the Portfolio Manager.
- Executes orders via the Trader.
- Synchronizes the state of the Virtual Portfolio.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.trading.consensus_engine import ConsensusEngine
from src.trading.portfolio_manager import PortfolioManager
from src.trading.post_inference_filter import PostInferenceFilter
from src.trading.trader import TradeOrder, Trader
from src.trading.virtual_portfolio import VirtualPortfolio


class TradingOrchestrator:
    """
    Manages the data flow and operational logic from signal generation to order execution.
    """

    def __init__(self, consensus_engine: ConsensusEngine | None,
        portfolio_manager: PortfolioManager, virtual_portfolio:
        VirtualPortfolio, trader: Trader, post_inference_filter: PostInferenceFilter | None=None, risk_metrics: Any | None=None,
        param_manager: Any | None=None, regime_detector: Any | None=
        None, knn_finder: Any | None=None, macro_analyzer: Any | None
        =None):
        """
        Initializes the orchestrator with its core and elite trading dependencies.
        """
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        self.consensus_engine = consensus_engine
        self.portfolio_manager = portfolio_manager
        self.portfolio = virtual_portfolio
        self.trader = trader
        self.filter = post_inference_filter
        self.risk_metrics = risk_metrics
        self.param_manager = param_manager
        self.regime_detector = regime_detector
        self.knn_finder = knn_finder
        self.macro_analyzer = macro_analyzer
        self.error_handler = None  # Optional; set externally if needed
        self.logger.info(
            'TradingOrchestrator initialized with full Elite-stack support (KNN + Macro).'
            )

    def process_signals(self, raw_predictions: list[dict[str, Any]],
        current_prices: dict[str, float], enriched_data: pd.DataFrame | None=None):
        """
        The main pipeline entry point for processing a batch of new model predictions.

        Args:
            raw_predictions: A list of prediction dictionaries output from Stage 5.
            current_prices: A map of tickers to their current realized market price.
            enriched_data: Optional full feature dataset for deep analysis.
        """
        if self.macro_analyzer and enriched_data is not None:
            self.logger.info('Executing Macro Context Analysis...')
            self.macro_analyzer.analyze(enriched_data)
        regime = self._detect_regime(enriched_data)
        predictions_to_process = self._apply_pre_filtering(raw_predictions)
        consensus_signals = self._synthesize_consensus_signals(
            predictions_to_process, regime=regime, enriched_data=enriched_data)
        if not consensus_signals:
            self.logger.info(
                'Cycle complete: No actionable signals identified by Consensus protocol.'
                )
            return
        
        self.logger.info(
            f'Consensus Engine identified {len(consensus_signals)} actionable trading opportunities.'
            )
            
        filtered_signals = self._apply_veto_committee(consensus_signals)
        if not filtered_signals:
             self.logger.info('Veto Committee rejected all signals. Cycle finished.')
             return

        self._handle_risk_exits(current_prices)
        trade_orders = self._generate_trade_orders(filtered_signals,
            current_prices)
        if not trade_orders:
            self.logger.info(
                'Portfolio Manager declined order generation based on risk limits. Cycle finished.'
                )
            return
        self.logger.info(
            f'Portfolio Manager authorized {len(trade_orders)} new trade orders.'
            )
        self._execute_orders(trade_orders)
        self.portfolio.update_performance(current_prices)
        self.logger.info(
            'Trading cycle concluded. Portfolio metrics and state successfully synchronized.'
            )

    def _apply_veto_committee(self, consensus_signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Passes the mathematical consensus signals through the AgenticVetoSystem (Investment Committee).
        Filters out signals that the agent vetoed.
        """
        try:
            import asyncio
            from src.agents.archive.veto_system import veto_system
            
            # 1. Adapt data for the agent
            agent_payload = []
            for sig in consensus_signals:
                fingerprint = "unknown"
                if 'report' in sig and hasattr(sig['report'], 'context_data'):
                    fingerprint = sig['report'].context_data.get('fingerprint', 'unknown')
                    
                agent_payload.append({
                    'ticker': sig['ticker'],
                    'action': sig['final_signal'],
                    'confidence': sig.get('confidence', 0.5),
                    'context_fingerprint': fingerprint
                })
            
            # 2. Run async agent synchronously (safely handle existing event loops)
            self.logger.info("Passing signals to AgenticVetoSystem for review...")
            
            def run_async_in_thread(coro):
                import threading
                result = []
                err = []
                def target():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result.append(loop.run_until_complete(coro))
                    except Exception as ex:
                        err.append(ex)
                    finally:
                        loop.close()
                t = threading.Thread(target=target)
                t.start()
                t.join()
                if err:
                    raise err[0]
                return result[0]
                
            try:
                asyncio.get_running_loop()
                # Loop exists, run in thread
                reviewed_recs = run_async_in_thread(veto_system.review_recommendations(agent_payload, latest_news=""))
            except RuntimeError:
                # No loop exists, safe to use asyncio.run
                reviewed_recs = asyncio.run(veto_system.review_recommendations(agent_payload, latest_news=""))
            
            # 3. Filter and map back
            approved_signals = []
            for orig_sig, rev_rec in zip(consensus_signals, reviewed_recs):
                if rev_rec.get('vetoed'):
                    self.logger.warning(f"VETOED {orig_sig['ticker']}: {rev_rec.get('veto_reason')}")
                else:
                    self.logger.info(f"APPROVED {orig_sig['ticker']}: {rev_rec.get('veto_reason', 'OK')}")
                    orig_sig['veto_causal_graph'] = rev_rec.get('causal_graph', [])
                    approved_signals.append(orig_sig)
            
            return approved_signals

        except Exception as e:
            self.logger.error(f"AgenticVetoSystem failed: {e}. Falling back to mathematical consensus.", exc_info=True)
            return consensus_signals

    def _detect_regime(self, enriched_data: pd.DataFrame | None) ->str:
        """Detect the market regime and hand it to the risk layer.

        This used to read:

            regime = 'ranging'
            if self.regime_detector:
                self.logger.info('Detecting market regime...')

        -- the detector was injected, the log line was printed, and the
        detector was never called, so every cycle ran as 'ranging'. The
        message made the logs look like detection was happening.

        The result is also pushed into PortfolioManager.current_regime so the
        kill switch can tighten with the regime (PipelinePolicyManager treats
        the configured limit as a ceiling, so this can only ever tighten).
        """
        regime = 'ranging'
        if self.regime_detector is not None and enriched_data is not None:
            try:
                closes = enriched_data['close'].astype(float)
                returns = closes.pct_change(fill_method=None).replace(
                    [np.inf, -np.inf], np.nan).dropna().values
                if len(returns) > 30:
                    result = self.regime_detector.detect_regime(
                        returns, data_bundle={'prices': closes.values})
                    detected = str(result.get('regime', '')).lower()
                    regime = self._map_detected_regime(detected)
                    self.logger.info(
                        f"Market regime detected: '{detected}' -> '{regime}'")
                else:
                    self.logger.info(
                        f'Not enough returns ({len(returns)}) to detect a '
                        f"regime; defaulting to '{regime}'.")
            except (KeyError, ValueError, TypeError, AttributeError) as e:
                self.logger.warning(
                    f"Regime detection failed ({e}); defaulting to '{regime}'.")

        if self.portfolio_manager is not None:
            self.portfolio_manager.current_regime = regime
        return regime

    @staticmethod
    def _map_detected_regime(detected: str) ->str:
        """Map the detector's vocabulary onto MarketRegime member names.

        AdaptiveParameterManager knows trending_up / trending_down / ranging /
        volatile / dead. Anything unrecognised stays 'ranging', and
        PipelinePolicyManager falls back to the configured ceiling for names
        it does not know, so a mismatch can never widen a limit.
        """
        if 'trend' in detected and 'up' in detected:
            return 'trending_up'
        if 'trend' in detected and 'down' in detected:
            return 'trending_down'
        if 'crisis' in detected or 'volatil' in detected:
            return 'volatile'
        if 'dead' in detected or 'stagnant' in detected:
            return 'dead'
        return 'ranging'

    def _apply_pre_filtering(self, raw_predictions: list[dict[str, Any]]
        ) ->list[dict[str, Any]]:
        """Apply optional pre-filtering to reduce noise in predictions."""
        if not self.filter:
            return raw_predictions
        import pandas as pd
        predictions_df = pd.DataFrame(raw_predictions)
        filtered_df = self.filter.apply(predictions_df)
        from typing import cast
        return cast(list[dict[str, Any]], filtered_df.to_dict('records'))

    def _synthesize_consensus_signals(self, predictions_to_process: list[
        dict[str, Any]], regime: str='neutral', enriched_data: pd.DataFrame | None=None) ->list[dict[str, Any]]:
        """Synthesize consensus signals from predictions."""
        consensus_signals = []
        for prediction in predictions_to_process:
            ticker = prediction.get('ticker')
            if not ticker:
                self.logger.warning(
                    "Skipping prediction payload: missing required field 'ticker'"
                    )
                continue
            signal_data = self._process_single_prediction(prediction,
                ticker, regime=regime, enriched_data=enriched_data)
            if signal_data:
                consensus_signals.append(signal_data)
        return consensus_signals

    def _process_single_prediction(self, prediction: dict[str, Any], ticker:
        str, regime: str='neutral', enriched_data: pd.DataFrame | None=None
        ) ->dict[str, Any] | None:
        """Process a single prediction and generate consensus signal."""
        pred_value = self._extract_prediction_value(prediction)
        model_predictions = self._build_model_predictions(prediction,
            pred_value)
        context_data = self._build_context_data(prediction, ticker)
        context_data['regime'] = regime
        knn_results = None
        if self.knn_finder and enriched_data is not None:
            try:
                historical_features = enriched_data[enriched_data['ticker'] ==
                    ticker]
                target_features = historical_features.tail(1)
                knn_payload = {'historical_features': historical_features,
                    'target_features': target_features}
                knn_analysis = self.knn_finder.analyze(knn_payload)
                knn_results = knn_analysis.get('similarities', {}).get(
                    target_features.index[-1], [])
            except (ValueError, TypeError, Exception) as e:
                self.logger.error(f'KNN analysis failed for {ticker}: {e}', exc_info=True)
                if self.error_handler:
                    self.error_handler.handle_error(e, context={'ticker': ticker})
                raise RuntimeError(f"KNN analysis failed for {ticker}: {e}") from e
        # The DEAN critic needs the engineered feature row for its meta-model
        # and a volatility reading for its rules. `enriched_data` is already
        # in hand here; previously nothing was forwarded, so the critic could
        # only ever see the flat 7-key context dict and never actually ran.
        critic_features = None
        if enriched_data is not None and 'ticker' in enriched_data.columns:
            ticker_rows = enriched_data[enriched_data['ticker'] == ticker]
            if not ticker_rows.empty:
                critic_features = ticker_rows.tail(1)
                context_data.setdefault(
                    'volatility', self._latest_volatility(critic_features)
                )

        try:
            if self.consensus_engine is not None:
                report = self.consensus_engine.generate_consensus(
                    model_predictions=model_predictions, context_data=
                    context_data, knn_results=knn_results,
                    features=critic_features)
                if report.final_signal != 'HOLD':
                    return {'ticker': ticker, 'final_signal': report.
                        final_signal, 'confidence': report.confidence,
                        'report': report}
                else:
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            f'Consensus for {ticker} resulted in HOLD. Skipping execution.'
                            )
            else:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f'Consensus engine not available for {ticker}. Skipping consensus synthesis.'
                        )
        except (ValueError, TypeError, Exception) as e:
            self.logger.error(f'Consensus synthesis failed for {ticker}: {e}', exc_info=True)
            if self.error_handler:
                self.error_handler.handle_error(e, context={'ticker': ticker})
            raise RuntimeError(f"Consensus synthesis failed for {ticker}: {e}") from e
        return None

    def _extract_prediction_value(self, prediction: dict[str, Any]) ->float:
        """Extract scalar prediction value from diverse source formats."""
        pred_value = prediction.get('predictions')
        if isinstance(pred_value, (list, tuple, np.ndarray)):
            return float(pred_value[-1]) if len(pred_value) > 0 else 0.0
        elif pred_value is not None and hasattr(pred_value, 'item'):
            return float(pred_value.item())
        else:
            return float(pred_value) if pred_value is not None else 0.0

    def _build_model_predictions(self, prediction: dict[str, Any],
        pred_value: float) ->dict[str, float]:
        """Reconstruct architecture-specific prediction matrix."""
        predictions_by_model = prediction.get('predictions_by_model', {})
        if predictions_by_model:
            return {model_name: float(pred) for model_name, pred in
                predictions_by_model.items()}
        else:
            primary_model = prediction.get('selected_primary_model', 'unknown')
            return {primary_model: pred_value}

    @staticmethod
    def _latest_volatility(feature_row: pd.DataFrame) -> float | None:
        """Pull a volatility reading out of the enriched feature row.

        Feature columns carry an interval suffix (feature_orchestrator appends
        `_{interval}`), so the exact name varies -- match on the prefix the
        technical enricher uses rather than hardcoding one column.
        """
        for col in feature_row.columns:
            if str(col).upper().startswith('VOLATILITY_'):
                value = feature_row.iloc[-1][col]
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if value == value:  # not NaN
                    return value
        return None

    def _build_context_data(self, prediction: dict[str, Any], ticker: str
        ) ->dict[str, Any]:
        """Build context data for regime-aware decision making."""
        return {'ticker': ticker, 'fingerprint': prediction.get(
            'context_fingerprint', '0|0|0'), 'regime': prediction.get(
            'market_regime', 'neutral'), 'tf': prediction.get('timeframe',
            '1d'), 'last_price': prediction.get('last_price'),
            'anomaly_score': prediction.get('anomaly_score', 0.0),
            'timestamp': prediction.get('timestamp')}

    def _handle_risk_exits(self, current_prices: dict[str, float]) ->None:
        """Handle high-priority risk exits (Stop-Loss / Take-Profit)."""
        exit_orders = self.portfolio_manager.check_risk_exits(current_prices)
        if exit_orders:
            self.logger.info(
                f'Risk Protocol: Generated {len(exit_orders)} exit orders (SL/TP triggers).'
                )
            self._execute_orders(exit_orders)

    def _generate_trade_orders(self, consensus_signals: list[dict[str, Any]
        ], current_prices: dict[str, float]) ->list[TradeOrder]:
        """Generate trade orders from consensus signals."""
        return self.portfolio_manager.generate_orders_from_signals(
            consensus_signals, current_prices)

    def _execute_orders(self, orders: list[TradeOrder]):
        """
        Dispatches orders to the trade execution interface and synchronizes the virtual portfolio state.
        """
        for order in orders:
            success = self.trader.execute_order(order)
            if success:
                self.logger.info(
                    f'Execution Success: {order}. Synchronizing internal ledger.'
                    )
                if order.action == 'BUY':
                    order_params = {'ticker': order.ticker, 'quantity':
                        order.quantity, 'price': order.price, 'reason':
                        getattr(order, 'reason', ''), 'confidence': getattr
                        (order, 'confidence', 0.8)}
                    self.portfolio.buy_stock(order_params)
                elif order.action == 'SELL':
                    self.portfolio.sell_stock(order.ticker, order.quantity,
                        order.price, reason=order.reason)
            else:
                self.logger.error(
                    f'Execution Failure: {order}. Order blocked or rejected by broker interface.'
                    )
