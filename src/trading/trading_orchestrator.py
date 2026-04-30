# src/trading/trading_orchestrator.py
"""
Orchestrates the entire trading process for Stage 6.

This module connects the different components of the trading pipeline:
- Aggregates signals through the Consensus Engine.
- Resolves risk-compliant trade orders via the Portfolio Manager.
- Executes orders via the Trader.
- Synchronizes the state of the Virtual Portfolio.
"""

from typing import List, Dict, Optional
import numpy as np
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

    def __init__(self, 
                 consensus_engine: ConsensusEngine,
                 portfolio_manager: PortfolioManager,
                 virtual_portfolio: VirtualPortfolio,
                 trader: Trader,
                 post_inference_filter: Optional[PostInferenceFilter] = None):
        """
        Initializes the orchestrator with its core trading dependencies.
        """
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        self.consensus_engine = consensus_engine
        self.portfolio_manager = portfolio_manager
        self.portfolio = virtual_portfolio
        self.trader = trader
        self.filter = post_inference_filter
        self.logger.info("TradingOrchestrator initialized and awaiting inference cycles.")

    def process_signals(self, 
                        raw_predictions: List[Dict[str, any]], 
                        current_prices: Dict[str, float]):
        """
        The main pipeline entry point for processing a batch of new model predictions.
        
        Args:
            raw_predictions: A list of prediction dictionaries output from Stage 5.
            current_prices: A map of tickers to their current realized market price.
        """
        self.logger.info(f"Initiating trading cycle with {len(raw_predictions)} raw predictions.")

        # 1. OPTIONAL: Inference pre-filtering (noise reduction)
        predictions_to_process = self._apply_pre_filtering(raw_predictions)

        # 2. Consensus Synthesis
        consensus_signals = self._synthesize_consensus_signals(predictions_to_process)
        
        if not consensus_signals:
            self.logger.info("Cycle complete: No actionable signals identified by Consensus protocol.")
            return

        self.logger.info(f"Consensus Engine identified {len(consensus_signals)} actionable trading opportunities.")

        # 3. HIGH PRIORITY: Hard-risk exits (Stop-Loss / Take-Profit)
        self._handle_risk_exits(current_prices)

        # 4. Entry Generation: Convert opportunities into risk-compliant orders
        trade_orders = self._generate_trade_orders(consensus_signals, current_prices)
        if not trade_orders:
            self.logger.info("Portfolio Manager declined order generation based on risk limits. Cycle finished.")
            return

        self.logger.info(f"Portfolio Manager authorized {len(trade_orders)} new trade orders.")

        # 5. Execution Pipeline
        self._execute_orders(trade_orders)

        # 6. Post-Trade Synchronization
        self.portfolio.update_performance(current_prices)
        self.logger.info("Trading cycle concluded. Portfolio metrics and state successfully synchronized.")
    
    def _apply_pre_filtering(self, raw_predictions: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Apply optional pre-filtering to reduce noise in predictions."""
        if not self.filter:
            return raw_predictions
        
        import pandas as pd
        predictions_df = pd.DataFrame(raw_predictions)
        filtered_df = self.filter.apply(predictions_df)
        return filtered_df.to_dict('records')
    
    def _synthesize_consensus_signals(self, predictions_to_process: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Synthesize consensus signals from predictions."""
        consensus_signals = []
        
        for prediction in predictions_to_process:
            ticker = prediction.get('ticker')
            if not ticker:
                self.logger.warning("Skipping prediction payload: missing required field 'ticker'")
                continue
            
            signal_data = self._process_single_prediction(prediction, ticker)
            if signal_data:
                consensus_signals.append(signal_data)
        
        return consensus_signals
    
    def _process_single_prediction(self, prediction: Dict[str, any], ticker: str) -> Optional[Dict[str, any]]:
        """Process a single prediction and generate consensus signal."""
        # Extract scalar prediction value from diverse source formats
        pred_value = self._extract_prediction_value(prediction)
        
        # Reconstruct architecture-specific prediction matrix
        model_predictions = self._build_model_predictions(prediction, pred_value)
        
        # Context construction for regime-aware decision making
        context_data = self._build_context_data(prediction, ticker)
        
        # Execute consensus decision logic
        try:
            report = self.consensus_engine.generate_consensus(
                model_predictions=model_predictions,
                context_data=context_data
            )
            
            # Filter out passive signals
            if report.final_signal != 'HOLD':
                return {
                    'ticker': ticker,
                    'final_signal': report.final_signal,
                    'confidence': report.confidence,
                    'report': report 
                }
            else:
                self.logger.debug(f"Consensus for {ticker} resulted in HOLD. Skipping execution.")
        except Exception as e:
            self.logger.error(f"Consensus synthesis failed for {ticker}: {e}")
        
        return None
    
    def _extract_prediction_value(self, prediction: Dict[str, any]) -> float:
        """Extract scalar prediction value from diverse source formats."""
        pred_value = prediction.get('predictions')
        if isinstance(pred_value, (list, tuple, np.ndarray)):
            return float(pred_value[-1]) if len(pred_value) > 0 else 0.0
        elif hasattr(pred_value, 'item'):  # Case for numpy scalar
            return float(pred_value.item())
        else:
            return float(pred_value) if pred_value is not None else 0.0
    
    def _build_model_predictions(self, prediction: Dict[str, any], pred_value: float) -> Dict[str, float]:
        """Reconstruct architecture-specific prediction matrix."""
        predictions_by_model = prediction.get('predictions_by_model', {})
        if predictions_by_model:
            # Use multi-architecture predictions for weighted ensembling
            return {
                model_name: float(pred) for model_name, pred in predictions_by_model.items()
            }
        else:
            # Use primary model output if architecture breakdown is unavailable
            primary_model = prediction.get('selected_primary_model', 'unknown')
            return {primary_model: pred_value}
    
    def _build_context_data(self, prediction: Dict[str, any], ticker: str) -> Dict[str, any]:
        """Build context data for regime-aware decision making."""
        return {
            'ticker': ticker,
            'fingerprint': prediction.get('context_fingerprint', '0|0|0'),
            'regime': prediction.get('market_regime', 'neutral'),
            'tf': prediction.get('timeframe', '1d'),
            'last_price': prediction.get('last_price'),
            'timestamp': prediction.get('timestamp')
        }
    
    def _handle_risk_exits(self, current_prices: Dict[str, float]) -> None:
        """Handle high-priority risk exits (Stop-Loss / Take-Profit)."""
        exit_orders = self.portfolio_manager.check_risk_exits(current_prices)
        if exit_orders:
            self.logger.info(f"Risk Protocol: Generated {len(exit_orders)} exit orders (SL/TP triggers).")
            self._execute_orders(exit_orders)
    
    def _generate_trade_orders(self, consensus_signals: List[Dict[str, any]], current_prices: Dict[str, float]) -> List[TradeOrder]:
        """Generate trade orders from consensus signals."""
        return self.portfolio_manager.generate_orders_from_signals(consensus_signals, current_prices)

    def _execute_orders(self, orders: List[TradeOrder]):
        """
        Dispatches orders to the trade execution interface and synchronizes the virtual portfolio state.
        """
        for order in orders:
            success = self.trader.execute_order(order)
            if success:
                self.logger.info(f"Execution Success: {order}. Synchronizing internal ledger.")
                if order.action == 'BUY':
                    order_params = {
                        'ticker': order.ticker,
                        'quantity': order.quantity,
                        'price': order.price,
                        'reason': getattr(order, 'reason', ''),
                        'confidence': getattr(order, 'confidence', 0.8)
                    }
                    self.portfolio.buy_stock(order_params)
                elif order.action == 'SELL':
                    self.portfolio.sell_stock(order.ticker, order.quantity, order.price, reason=order.reason)
            else:
                self.logger.error(f"Execution Failure: {order}. Order blocked or rejected by broker interface.")
