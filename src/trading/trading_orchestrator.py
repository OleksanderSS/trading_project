"""
Orchestrates the entire trading process for Stage 6.

This module connects the different components of the trading pipeline:
- Gets signals from the Consensus Engine.
- Gets trade orders from the Portfolio Manager.
- Executes orders via the Trader.
- Updates the state of the Virtual Portfolio.
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
    Manages the flow of information from signal generation to execution.
    """

    def __init__(self, 
                 consensus_engine: ConsensusEngine,
                 portfolio_manager: PortfolioManager,
                 virtual_portfolio: VirtualPortfolio,
                 trader: Trader,
                 post_inference_filter: Optional[PostInferenceFilter] = None):
        """
        Initializes the orchestrator with all necessary trading components.
        """
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        self.consensus_engine = consensus_engine
        self.portfolio_manager = portfolio_manager
        self.portfolio = virtual_portfolio
        self.trader = trader
        self.filter = post_inference_filter
        self.logger.info("TradingOrchestrator initialized and ready.")

    def process_signals(self, 
                        raw_predictions: List[Dict[str, any]], 
                        current_prices: Dict[str, float]):
        """
        The main entry point to process a batch of new model predictions.
        
        Args:
            raw_predictions: A list of prediction dicts from Stage 5.
            current_prices: A dict mapping tickers to their current market prices.
        """
        self.logger.info(f"Starting new trading cycle with {len(raw_predictions)} raw predictions.")

        # 1. (Optional) Pre-filter raw predictions
        if self.filter:
            # Assuming predictions are in a DataFrame-compatible format
            import pandas as pd
            predictions_df = pd.DataFrame(raw_predictions)
            filtered_df = self.filter.apply(predictions_df)
            predictions_to_process = filtered_df.to_dict('records')
        else:
            predictions_to_process = raw_predictions

        # 2. Generate consensus from (filtered) predictions
        consensus_signals = []
        for prediction in predictions_to_process:
            ticker = prediction.get('ticker')
            if not ticker:
                self.logger.warning("Prediction missing 'ticker', skipping")
                continue
            
            # ✅ FIX: Витягуємо prediction values та створюємо правильний формат для ConsensusEngine
            # ConsensusEngine.generate_consensus() очікує:
            # - model_predictions: Dict[str, float] - {model_name: prediction_value}
            # - context_data: Dict[str, Any] - контекстні дані
            
            # Витягуємо predictions (може бути list, array, або scalar)
            pred_value = prediction.get('predictions')
            if isinstance(pred_value, (list, tuple, np.ndarray)):
                pred_value = float(pred_value[-1]) if len(pred_value) > 0 else 0.0
            elif hasattr(pred_value, 'item'):  # numpy scalar
                pred_value = float(pred_value.item())
            else:
                pred_value = float(pred_value) if pred_value is not None else 0.0
            
            # Створюємо model_predictions dict
            predictions_by_model = prediction.get('predictions_by_model', {})
            if predictions_by_model:
                # ✅ FIX: predictions_by_model містить реальні прогнози, а не ваги
                # Використовуємо їх як model_predictions для ConsensusEngine
                model_predictions = {
                    model_name: float(pred) for model_name, pred in predictions_by_model.items()
                }
            else:
                # Інакше використовуємо primary model з основним прогнозом
                primary_model = prediction.get('selected_primary_model', 'unknown')
                model_predictions = {primary_model: pred_value}
            
            # Створюємо context_data
            # ✅ Витягуємо context_fingerprint з prediction (якщо є) або з features
            context_fingerprint = prediction.get('context_fingerprint', '0|0|0')
            market_regime = prediction.get('market_regime', 'neutral')
            
            context_data = {
                'ticker': ticker,
                'fingerprint': context_fingerprint,
                'regime': market_regime,
                'tf': prediction.get('timeframe', '1d'),
                'last_price': prediction.get('last_price'),
                'timestamp': prediction.get('timestamp')
            }
            
            # Викликаємо ConsensusEngine з правильними параметрами
            try:
                report = self.consensus_engine.generate_consensus(
                    model_predictions=model_predictions,
                    context_data=context_data
                )
                
                if report.final_signal != 'HOLD':
                    signal_data = {
                        'ticker': ticker,
                        'final_signal': report.final_signal,
                        'confidence': report.confidence,
                        'report': report  # Pass the full report for richer logging/decisions
                    }
                    consensus_signals.append(signal_data)
                else:
                    self.logger.debug(f"Signal for {ticker} is HOLD, skipping")
            except Exception as e:
                self.logger.error(f"Consensus generation failed for {ticker}: {e}", exc_info=True)
        
        if not consensus_signals:
            self.logger.info("No actionable signals after consensus. Cycle finished.")
            return

        self.logger.info(f"{len(consensus_signals)} actionable signals generated by Consensus Engine.")

        # 3. Check for risk-based exits (SL/TP) FIRST
        exit_orders = self.portfolio_manager.check_risk_exits(current_prices)
        if exit_orders:
            self.logger.info(f"Generated {len(exit_orders)} exit orders due to SL/TP triggers.")
            self._execute_orders(exit_orders)

        # 4. Generate new entry/exit orders from consensus signals
        trade_orders = self.portfolio_manager.generate_orders_from_signals(consensus_signals, current_prices)

        if not trade_orders:
            self.logger.info("No trade orders generated by Portfolio Manager. Cycle finished.")
            self.logger.info(f"  - consensus_signals count: {len(consensus_signals)}")
            self.logger.info(f"  - current_prices count: {len(current_prices)}")
            if consensus_signals:
                self.logger.info(f"  - first signal: {consensus_signals[0]}")
            return

        self.logger.info(f"Generated {len(trade_orders)} new trade orders.")

        # 5. Execute all generated orders
        self._execute_orders(trade_orders)

        # 6. Update portfolio performance metrics and save state
        self.portfolio.update_performance(current_prices)
        self.logger.info("Portfolio performance updated and state saved.")

    def _execute_orders(self, orders: List[TradeOrder]):
        """
        Sends orders to the trader and updates the portfolio upon success.
        """
        for order in orders:
            success = self.trader.execute_order(order)
            if success:
                self.logger.info(f"Successfully executed order: {order}. Updating portfolio state.")
                # The trader only simulates. Now we tell the portfolio to update its state.
                if order.action == 'BUY':
                    self.portfolio.buy_stock(order.ticker, order.quantity, order.price, reason=order.reason)
                elif order.action == 'SELL':
                    self.portfolio.sell_stock(order.ticker, order.quantity, order.price, reason=order.reason)
            else:
                self.logger.error(f"Failed to execute order: {order}. Portfolio not updated.")

