"""
Stage 6: Trading Execution

This stage takes the final predictions from Stage 5 and orchestrates the entire
trading process using the refactored trading module.
"""
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.logging.logger import ProjectLogger
from src.meta_learning.memory.diary_engine import (
    DecisionOutcome,
    DecisionRecord,
    DecisionType,
    DiaryEngine,
    diary_timestamp,
)
from src.pipeline.stages.base_stage import BaseStage
from src.risk.elite_risk_metrics import EliteRiskMetrics
from src.risk.max_exposure_monitor import MaxExposureMonitor
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
        self._trading_stack_initialized = False

    def _initialize_trading_stack(self):
        """Initialize the stateful paper stack only after explicit approval."""
        if getattr(self, '_trading_stack_initialized', False):
            return
        self.portfolio = VirtualPortfolio()
        self.post_inference_filter = PostInferenceFilter()
        self.diary_engine = DiaryEngine()
        self.enhanced_consensus = EnhancedConsensusEngine()
        self.risk_sizer = EliteRiskSizer(logger=self.logger)
        self.risk_metrics = EliteRiskMetrics(logger=self.logger)
        self.param_manager = AdaptiveParameterManager(logger=self.logger)
        # ✅ Integrated: multi-layer exposure monitoring
        self.exposure_monitor = MaxExposureMonitor(
            config=self.config_manager.get('strategy.risk_management', {})
        )

        self.portfolio_manager = PortfolioManager(
            virtual_portfolio=self.portfolio,
            elite_risk_sizer=self.risk_sizer,
            config=self.config_manager.get('strategy.risk_management', {})
        )
        self.trader = Trader(paper_trading=True)

        self.trading_orchestrator = TradingOrchestrator(
            consensus_engine=self.enhanced_consensus,  # Увімкнено консенсус
            portfolio_manager=self.portfolio_manager,
            virtual_portfolio=self.portfolio,
            trader=self.trader,
            post_inference_filter=self.post_inference_filter
        )
        self.logger.info('✅ Pattern-Aware Trading Stack initialized.')

        self._trading_stack_initialized = True

    async def run(self, **kwargs) -> dict[str, Any]:
        """Run review-only by default or an explicitly approved paper cycle."""
        requested_mode, execution_mode = self._resolve_execution_mode(kwargs)
        predictions, current_prices = await self._load_or_extract_data(kwargs)
        if not predictions:
            return self._execution_boundary_result(
                predictions=[],
                requested_mode=requested_mode,
                execution_mode=execution_mode,
                status='no_predictions',
                reason='Stage 6 received no predictions.',
            )

        # 1. Адаптивне виконання на основі контексту
        processed_signals = self._apply_context_rules(predictions)
        if execution_mode != 'paper':
            status = (
                'blocked_live_execution_disabled'
                if execution_mode == 'live'
                else (
                    'blocked_invalid_execution_mode'
                    if execution_mode == 'invalid'
                    else 'review_only_no_execution'
                )
            )
            reason = {
                'live': 'Live trading is intentionally unavailable in the active pipeline.',
                'invalid': f"Unsupported execution mode: {requested_mode!r}.",
            }.get(
                execution_mode,
                'Predictions remain review-only; Stage 6 performed no paper or live action.',
            )
            return self._execution_boundary_result(
                predictions=processed_signals,
                requested_mode=requested_mode,
                execution_mode=execution_mode,
                status=status,
                reason=reason,
            )

        return self._execution_boundary_result(
            predictions=processed_signals,
            requested_mode=requested_mode,
            execution_mode=execution_mode,
            status='blocked_paper_execution_requires_isolated_executor',
            reason=(
                'The active pipeline cannot execute paper trades. Use the '
                'review receipt -> paper simulation plan -> isolated external '
                'executor -> paper result review workflow.'
            ),
        )

    def _resolve_execution_mode(self, kwargs: dict[str, Any]) -> tuple[str, str]:
        requested = str(kwargs.get('execution_mode') or 'review_only').strip().lower()
        aliases = {
            'review': 'review_only',
            'review_only': 'review_only',
            'dry_run': 'review_only',
            'paper': 'paper',
            'paper_only': 'paper',
            'paper_trading': 'paper',
            'live': 'live',
            'live_trading': 'live',
        }
        return requested, aliases.get(requested, 'invalid')

    def _execution_boundary_result(
        self,
        *,
        predictions: list[dict[str, Any]],
        requested_mode: str,
        execution_mode: str,
        status: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            'trading_activity': [],
            'portfolio_summary': {},
            'signals': predictions,
            'diary_records_written': 0,
            'execution_mode': execution_mode,
            'execution_status': status,
            'execution_authorized': False,
            'execution_boundary': {
                'requested_mode': requested_mode,
                'effective_mode': execution_mode,
                'paper_execution_authorized': False,
                'live_execution_supported': False,
                'portfolio_mutated': False,
                'diary_records_written': 0,
                'reason': reason,
            },
        }

    def _apply_context_rules(self, predictions: list[dict]) -> list[dict]:
        """
        🎯 ANXIETY & EXPERT SYNC:
        Застосовує правила "тривожності" та синхронізує прогнози з Чемпіонами патернів.
        """
        filtered_signals = []
        for source_prediction in predictions:
            pred = dict(source_prediction)
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
        transactions: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> int:
        """
        RESERVED for the isolated paper-executor workflow.

        The review-only run() path never reaches this: paper execution is
        deferred to the external executor (review receipt -> paper simulation
        plan -> isolated executor -> paper result review). Kept here so the
        isolated executor can reuse the diary-mapping contract without
        duplicating it.
        """
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
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(
                    f"Failed to record Stage 6 transaction in diary for {ticker}: {e}",
                    exc_info=True,
                )

        return records_written

    def _build_decision_record(
        self,
        transaction: dict[str, Any],
        prediction: dict[str, Any],
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
        pnl: float | None,
    ) -> DecisionOutcome:
        if tx_type == 'BUY' or pnl is None:
            return DecisionOutcome.PENDING
        if pnl > 0:
            return DecisionOutcome.PROFITABLE
        if pnl < 0:
            return DecisionOutcome.UNPROFITABLE
        return DecisionOutcome.BREAK_EVEN

    def _transaction_timestamp(self, value: Any) -> int:
        """Transaction time as a diary timestamp (UNIX SECONDS).

        This used to return milliseconds while every other writer of
        experience_diary.decision_timestamp wrote seconds -- see the note on
        diary_timestamp() in diary_engine. It also substituted "now" for an
        unparseable or missing value without saying so, which stamps a
        historical trade with the moment the pipeline happened to run.
        """
        if isinstance(value, datetime):
            timestamp = value
        elif isinstance(value, (int, float)):
            numeric = float(value)
            # Accept either unit on the way in: a value past ~2286 in seconds
            # is really milliseconds.
            return int(numeric / 1000 if abs(numeric) > 10_000_000_000 else numeric)
        elif isinstance(value, str) and value:
            try:
                timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                self.logger.warning(
                    "Transaction timestamp %r could not be parsed; recording "
                    "this decision at the current time instead, which is not "
                    "when it happened.", value
                )
                timestamp = datetime.now(UTC)
        else:
            if value is not None:
                self.logger.warning(
                    "Transaction timestamp %r is of unusable type %s; "
                    "recording this decision at the current time instead.",
                    value, type(value).__name__,
                )
            timestamp = datetime.now(UTC)

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return diary_timestamp(timestamp)

    def _transaction_prices(
        self,
        transaction: dict[str, Any],
    ) -> tuple[float | None, float | None]:
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

    def _extract_model_prediction(self, prediction: dict[str, Any]) -> float | None:
        for key in ('predictions', 'raw_forecast', 'prediction'):
            value = self._safe_float(prediction.get(key))
            if value is not None:
                return value
        return None

    def _safe_float(self, value: Any) -> float | None:
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

    def _find_latest_batch_name(self) -> str | None:
        output_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
        # 'main_database' is the project-wide default batch name (used by
        # system_orchestrator.py, colab_environment.py, batch_manager.py,
        # hybrid_orchestrator.py) - check it first, matching the sibling
        # TradingDataIO._load_predictions_from_disk's already-correct
        # logic. Without this, any invocation without an explicit
        # batch_name (e.g. CLI runs where args.batch_name is None) would
        # only ever look for test_ticker_* dirs and silently miss the real
        # main_database/stage_5_results.json even when it exists.
        if (output_dir / 'main_database').exists():
            return 'main_database'
        batch_dirs = list(output_dir.glob('test_ticker_*'))
        return max(batch_dirs, key=lambda p: p.stat().st_mtime).name if batch_dirs else None

    def _finalize_results(self, predictions, current_prices, kwargs):
        """
        RESERVED for the isolated paper-executor workflow.

        Not called by the review-only run() path. The isolated paper executor
        owns portfolio mutation and calls this to assemble its result packet.
        """
        portfolio_summary = self.portfolio.get_portfolio_summary(current_prices)
        trade_history = getattr(self.portfolio, 'transactions', [])

        self.logger.info(f"📊 Portfolio Final Value: {portfolio_summary.get('total_value', 0):.2f}")
        return {
            'trading_activity': trade_history[-5:],
            'portfolio_summary': portfolio_summary,
            'signals': predictions
        }
