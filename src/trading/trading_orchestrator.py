"""
Orchestrates the entire trading process for Stage 6.

This module connects the different components of the trading pipeline:
- Aggregates signals through the Consensus Engine.
- Resolves risk-compliant trade orders via the Portfolio Manager.
- Executes orders via the Trader.
- Synchronizes the state of the Virtual Portfolio.
"""
import logging
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from src.core.logging.logger import ProjectLogger
from src.trading.consensus_engine import ConsensusEngine
from src.trading.portfolio_manager import PortfolioManager
from src.trading.virtual_portfolio import VirtualPortfolio
from src.trading.trader import Trader, TradeOrder
from src.trading.post_inference_filter import PostInferenceFilter


class TradingOrchestrator:
    """
    Manages the data flow and operational logic from signal generation to order execution.
    """

    def __init__(self, consensus_engine: Optional[ConsensusEngine],
        portfolio_manager: PortfolioManager, virtual_portfolio:
        VirtualPortfolio, trader: Trader, post_inference_filter: Optional[
        PostInferenceFilter]=None, risk_metrics: Optional[Any]=None,
        param_manager: Optional[Any]=None, regime_detector: Optional[Any]=
        None, knn_finder: Optional[Any]=None, macro_analyzer: Optional[Any]
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
        self.logger.info(
            'TradingOrchestrator initialized with full Elite-stack support (KNN + Macro).'
            )

    def process_signals(self, raw_predictions: List[Dict[str, Any]],
        current_prices: Dict[str, float], enriched_data: Optional[pd.
        DataFrame]=None):
        """
        The main pipeline entry point for processing a batch of new model predictions.
        
        Args:
            raw_predictions: A list of prediction dictionaries output from Stage 5.
            current_prices: A map of tickers to their current realized market price.
            enriched_data: Optional full feature dataset for deep analysis.
        """
        macro_results = None
        if self.macro_analyzer and enriched_data is not None:
            self.logger.info('Executing Macro Context Analysis...')
            macro_results = self.macro_analyzer.analyze(enriched_data)
        regime = 'ranging'
        if self.regime_detector:
            self.logger.info(
                'Detecting market regime for decision optimization...')
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
        self._handle_risk_exits(current_prices)
        trade_orders = self._generate_trade_orders(consensus_signals,
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

    def _apply_pre_filtering(self, raw_predictions: List[Dict[str, Any]]
        ) ->List[Dict[str, Any]]:
        """Apply optional pre-filtering to reduce noise in predictions."""
        if not self.filter:
            return raw_predictions
        import pandas as pd
        predictions_df = pd.DataFrame(raw_predictions)
        filtered_df = self.filter.apply(predictions_df)
        from typing import cast
        return cast(list[dict[str, Any]], filtered_df.to_dict('records'))

    def _synthesize_consensus_signals(self, predictions_to_process: List[
        Dict[str, Any]], regime: str='neutral', enriched_data: Optional[pd.
        DataFrame]=None) ->List[Dict[str, Any]]:
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

    def _process_single_prediction(self, prediction: Dict[str, Any], ticker:
        str, regime: str='neutral', enriched_data: Optional[pd.DataFrame]=None
        ) ->Optional[Dict[str, Any]]:
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
                self.error_handler.handle_error(e, context={'ticker': ticker})
                raise RuntimeError(f"KNN analysis failed for {ticker}: {e}") from e
        try:
            if self.consensus_engine is not None:
                report = self.consensus_engine.generate_consensus(
                    model_predictions=model_predictions, context_data=
                    context_data, knn_results=knn_results)
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
            self.error_handler.handle_error(e, context={'ticker': ticker})
            raise RuntimeError(f"Consensus synthesis failed for {ticker}: {e}") from e
        return None

    def _extract_prediction_value(self, prediction: Dict[str, Any]) ->float:
        """Extract scalar prediction value from diverse source formats."""
        pred_value = prediction.get('predictions')
        if isinstance(pred_value, (list, tuple, np.ndarray)):
            return float(pred_value[-1]) if len(pred_value) > 0 else 0.0
        elif pred_value is not None and hasattr(pred_value, 'item'):
            return float(pred_value.item())
        else:
            return float(pred_value) if pred_value is not None else 0.0

    def _build_model_predictions(self, prediction: Dict[str, Any],
        pred_value: float) ->Dict[str, float]:
        """Reconstruct architecture-specific prediction matrix."""
        predictions_by_model = prediction.get('predictions_by_model', {})
        if predictions_by_model:
            return {model_name: float(pred) for model_name, pred in
                predictions_by_model.items()}
        else:
            primary_model = prediction.get('selected_primary_model', 'unknown')
            return {primary_model: pred_value}

    def _build_context_data(self, prediction: Dict[str, Any], ticker: str
        ) ->Dict[str, Any]:
        """Build context data for regime-aware decision making."""
        return {'ticker': ticker, 'fingerprint': prediction.get(
            'context_fingerprint', '0|0|0'), 'regime': prediction.get(
            'market_regime', 'neutral'), 'tf': prediction.get('timeframe',
            '1d'), 'last_price': prediction.get('last_price'),
            'anomaly_score': prediction.get('anomaly_score', 0.0),
            'timestamp': prediction.get('timestamp')}

    def _handle_risk_exits(self, current_prices: Dict[str, float]) ->None:
        """Handle high-priority risk exits (Stop-Loss / Take-Profit)."""
        exit_orders = self.portfolio_manager.check_risk_exits(current_prices)
        if exit_orders:
            self.logger.info(
                f'Risk Protocol: Generated {len(exit_orders)} exit orders (SL/TP triggers).'
                )
            self._execute_orders(exit_orders)

    def _generate_trade_orders(self, consensus_signals: List[Dict[str, Any]
        ], current_prices: Dict[str, float]) ->List[TradeOrder]:
        """Generate trade orders from consensus signals."""
        return self.portfolio_manager.generate_orders_from_signals(
            consensus_signals, current_prices)

    def _execute_orders(self, orders: List[TradeOrder]):
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
