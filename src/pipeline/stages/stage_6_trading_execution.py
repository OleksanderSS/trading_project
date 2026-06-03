"""
Stage 6: Trading Execution

This stage takes the final predictions from Stage 5 and orchestrates the entire
trading process using the refactored trading module.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Dict, List
from src.core.logging.logger import ProjectLogger
from src.meta_learning.memory.diary_engine import (
    DecisionOutcome,
    DecisionRecord,
    DecisionType,
    DiaryEngine,
)
from src.pipeline.stages.base_stage import BaseStage
from src.risk.elite_risk_metrics import EliteRiskMetrics
from src.trading.adaptive_parameter_manager import AdaptiveParameterManager
from src.trading.consensus_engine import EnhancedConsensusEngine
from src.trading.elite_risk_sizer import EliteRiskSizer
from src.trading.portfolio_manager import PortfolioManager
from src.trading.post_inference_filter import PostInferenceFilter
from src.trading.trader import Trader
from src.trading.trading_orchestrator import TradingOrchestrator
from src.trading.virtual_portfolio import VirtualPortfolio


class TradingExecutionStage(BaseStage):
    """
    🎯 CONTEXT-AWARE EXECUTION:
    - Реалізує 'Anxiety Kill-Switch' на основі швидкості зміни контексту.
    - Використовує Патерн-Експертів для фінального вибору моделі.
    - Адаптує розмір позиції до ринкового хаосу.
    """
    def __init__(self, config_manager, error_handler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        self._initialize_trading_stack()

    def _initialize_trading_stack(self):
        """Initializes components with pattern-awareness."""
        self.portfolio = VirtualPortfolio()
        self.post_inference_filter = PostInferenceFilter()
        self.diary_engine = DiaryEngine()
        self.enhanced_consensus = EnhancedConsensusEngine()
        self.risk_sizer = EliteRiskSizer(logger=self.logger)
        self.risk_metrics = EliteRiskMetrics(logger=self.logger)
        self.param_manager = AdaptiveParameterManager(logger=self.logger)
        
        self.portfolio_manager = PortfolioManager(
            virtual_portfolio=self.portfolio, 
            elite_risk_sizer=self.risk_sizer, 
            config=self.config_manager.get('strategy.risk_management', {})
        )
        self.trader = Trader(paper_trading=True)
        
        self.trading_orchestrator = TradingOrchestrator(
            consensus_engine=None, 
            portfolio_manager=self.portfolio_manager,
            virtual_portfolio=self.portfolio, 
            trader=self.trader,
            post_inference_filter=self.post_inference_filter
        )
        self.logger.info('✅ Pattern-Aware Trading Stack initialized.')

    async def run(self, **kwargs) -> dict[str, Any]:
        """Runs the context-aware trading cycle."""
        if not hasattr(self, 'trading_orchestrator'):
            self._initialize_trading_stack()
            
        predictions, current_prices = await self._load_or_extract_data(kwargs)
        if not predictions:
            return {}

        # 1. Адаптивне виконання на основі контексту
        processed_signals = self._apply_context_rules(predictions)
        existing_tx_count = len(getattr(self.portfolio, 'transactions', []))
        
        # 2. Торгівля
        self.trading_orchestrator.process_signals(
            raw_predictions=processed_signals, 
            current_prices=current_prices
        )
        
        new_transactions = getattr(self.portfolio, 'transactions', [])[existing_tx_count:]
        diary_records_written = self._record_transactions_to_diary(
            new_transactions, processed_signals)
        result = self._finalize_results(processed_signals, current_prices, kwargs)
        result['diary_records_written'] = diary_records_written
        return result

    def _apply_context_rules(self, predictions: List[Dict]) -> List[Dict]:
        """
        🎯 ANXIETY & EXPERT SYNC:
        Застосовує правила "тривожності" та синхронізує прогнози з Чемпіонами патернів.
        """
        filtered_signals = []
        for pred in predictions:
            ticker = pred.get('ticker')
            velocity = self._safe_float(pred.get('context_velocity')) or 0.0
            
            # --- Rule 1: Anxiety Kill-Switch ---
            # Якщо ринок занадто швидко змінюється (velocity > 0.7), знижуємо впевненість
            if velocity > 0.7:
                pred['confidence'] = self._safe_float(pred.get('confidence')) or 0.0
                self.logger.warning(f"🚨 High Context Velocity ({velocity:.2f}) for {ticker}. Reducing exposure.")
                pred['confidence'] *= 0.5 # Штраф 50%
                
            # --- Rule 2: Panic Block ---
            # Якщо швидкість критична, блокуємо нові BUY (Anxiety Index proxy)
            if velocity > 0.85:
                if (self._extract_model_prediction(pred) or 0.0) > 0:
                     self.logger.error(f"🛑 CRITICAL ANXIETY for {ticker}. Blocking BUY signal.")
                     pred['confidence'] = 0.0 # Повністю анулюємо сигнал
            
            filtered_signals.append(pred)
            
        return filtered_signals

    def _record_transactions_to_diary(
        self,
        transactions: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
    ) -> int:
        if not transactions:
            return 0

        predictions_by_ticker = {
            str(pred.get('ticker', '')).upper(): pred
            for pred in predictions
            if pred.get('ticker')
        }
        records_written = 0

        for transaction in transactions:
            ticker = str(transaction.get('ticker', '')).upper()
            if not ticker:
                self.logger.warning(
                    "Skipping diary write for transaction without ticker: %s",
                    transaction,
                )
                continue

            prediction = predictions_by_ticker.get(ticker, {})
            try:
                record = self._build_decision_record(transaction, prediction)
                self.diary_engine.record_decision(record)
                records_written += 1
            except Exception as e:
                self.logger.error(
                    f"Failed to record Stage 6 transaction in diary for {ticker}: {e}",
                    exc_info=True,
                )

        return records_written

    def _build_decision_record(
        self,
        transaction: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> DecisionRecord:
        ticker = str(transaction.get('ticker') or prediction.get('ticker') or '').upper()
        tx_type = str(transaction.get('type', '')).upper()
        pnl = self._safe_float(transaction.get('pnl'))
        pnl_pct = self._safe_float(transaction.get('pnl_pct'))
        entry_price, exit_price = self._transaction_prices(transaction)

        predictions_by_model = prediction.get('predictions_by_model') or {}
        model_name = (
            prediction.get('selected_primary_model')
            or next(iter(predictions_by_model), None)
            or 'stage6_execution'
        )
        confidence = (
            self._safe_float(transaction.get('confidence'))
            if tx_type == 'BUY'
            else self._safe_float(prediction.get('confidence'))
        )
        profit_loss = (pnl_pct / 100.0) if pnl_pct is not None else pnl

        market_context = {
            'transaction_type': tx_type,
            'reason': transaction.get('reason', ''),
            'quantity': self._safe_float(transaction.get('quantity')),
            'price': self._safe_float(transaction.get('price')),
            'trade_value': self._safe_float(transaction.get('trade_value')),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'context_pattern_id': prediction.get('context_pattern_id'),
            'context_pattern_seq': prediction.get('context_pattern_seq'),
            'context_velocity': self._safe_float(prediction.get('context_velocity')),
            'confidence': confidence,
            'raw_forecast': self._safe_float(prediction.get('raw_forecast')),
            'selected_primary_model': prediction.get('selected_primary_model'),
        }

        return DecisionRecord(
            agent_id=str(model_name),
            ticker=ticker,
            decision_type=DecisionType.BUY if tx_type == 'BUY' else DecisionType.SELL,
            reasoning=str(transaction.get('reason') or f"Stage 6 {tx_type} execution"),
            market_context=market_context,
            context_fingerprint=str(
                prediction.get('context_fingerprint')
                or prediction.get('context_pattern_id')
                or 'unknown_context'
            ),
            context_pattern_seq=prediction.get('context_pattern_seq'),
            model_prediction=self._extract_model_prediction(prediction),
            model_confidence=confidence,
            entry_price=entry_price,
            exit_price=exit_price,
            outcome=self._transaction_outcome(tx_type, pnl),
            profit_loss=profit_loss,
            decision_timestamp=self._transaction_timestamp(transaction.get('timestamp')),
        )

    def _transaction_outcome(
        self,
        tx_type: str,
        pnl: Optional[float],
    ) -> DecisionOutcome:
        if tx_type == 'BUY' or pnl is None:
            return DecisionOutcome.PENDING
        if pnl > 0:
            return DecisionOutcome.PROFITABLE
        if pnl < 0:
            return DecisionOutcome.UNPROFITABLE
        return DecisionOutcome.BREAK_EVEN

    def _transaction_timestamp(self, value: Any) -> int:
        if isinstance(value, datetime):
            timestamp = value
        elif isinstance(value, (int, float)):
            numeric = float(value)
            return int(numeric if abs(numeric) > 10_000_000_000 else numeric * 1000)
        elif isinstance(value, str) and value:
            try:
                timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return int(timestamp.timestamp() * 1000)

    def _transaction_prices(
        self,
        transaction: Dict[str, Any],
    ) -> tuple[Optional[float], Optional[float]]:
        tx_type = str(transaction.get('type', '')).upper()
        price = self._safe_float(transaction.get('price'))
        if tx_type == 'BUY':
            return price, None

        quantity = self._safe_float(transaction.get('quantity'))
        net_revenue = self._safe_float(transaction.get('net_revenue'))
        pnl = self._safe_float(transaction.get('pnl'))
        if quantity and quantity != 0 and net_revenue is not None and pnl is not None:
            return (net_revenue - pnl) / quantity, price
        return None, price

    def _extract_model_prediction(self, prediction: Dict[str, Any]) -> Optional[float]:
        for key in ('predictions', 'raw_forecast', 'prediction'):
            value = self._safe_float(prediction.get(key))
            if value is not None:
                return value
        return None

    def _safe_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return self._safe_float(value[-1])
        if (
            not isinstance(value, (str, bytes))
            and hasattr(value, '__len__')
            and hasattr(value, '__getitem__')
        ):
            try:
                if len(value) == 0:
                    return None
                return self._safe_float(value[-1])
            except (TypeError, ValueError, IndexError):
                return None
        if hasattr(value, 'item'):
            try:
                return self._safe_float(value.item())
            except (TypeError, ValueError, AttributeError):
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    async def _load_or_extract_data(self, kwargs: dict) -> tuple:
        predictions = kwargs.get('predictions')
        current_prices = kwargs.get('current_prices')
        if not predictions:
            load_result = await self._load_predictions_from_disk(kwargs)
            predictions, current_prices = load_result[0], load_result[1]
        return predictions, current_prices

    async def _load_predictions_from_disk(self, kwargs: dict) -> tuple:
        batch_name = kwargs.get('batch_name') or self._find_latest_batch_name()
        output_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
        
        if batch_name:
            batch_dir = output_dir / batch_name
            file = batch_dir / 'stage_5_results.json'
            if file.exists():
                import aiofiles
                async with aiofiles.open(file, encoding='utf-8') as f:
                    data = json.loads(await f.read())
                    return data.get('predictions', []), data.get('current_prices', {}), kwargs
        return [], {}, kwargs

    def _find_latest_batch_name(self) -> Optional[str]:
        output_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
        batch_dirs = list(output_dir.glob('test_ticker_*'))
        return max(batch_dirs, key=lambda p: p.stat().st_mtime).name if batch_dirs else None

    def _finalize_results(self, predictions, current_prices, kwargs):
        portfolio_summary = self.portfolio.get_portfolio_summary(current_prices)
        trade_history = getattr(self.portfolio, 'transactions', [])
        
        self.logger.info(f"📊 Portfolio Final Value: {portfolio_summary.get('total_value', 0):.2f}")
        return {
            'trading_activity': trade_history[-5:],
            'portfolio_summary': portfolio_summary,
            'signals': predictions
        }
