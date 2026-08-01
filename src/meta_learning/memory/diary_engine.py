#!/usr/bin/env python3
"""
Experience Diary - Decision Learning & Memory System

ARCHITECTURAL PURPOSE:
This module is the "memory" of the entire trading system. It acts like an airplane's "black box".
Its sole purpose is to meticulously record every trading decision made, its reasons,
the context, and, most importantly, its final outcome (profit or loss).

This diary is the single source of truth for the Meta-Analyzer ("The Critic"). It records
the actions of both the main "Champion" and the parallel "Pretender" simulations, allowing
for an objective comparison of their effectiveness.
"""

import json
import logging
import sys
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.meta_learning.base import BaseMetaComponent

# ---------------------------------------------------------------------------
# experience_diary.decision_timestamp is UNIX SECONDS.
#
# Three writers filled this one BIGINT column in two different units: this
# module's DecisionRecord default and log_training_event wrote seconds, while
# the consensus-metadata writer and Stage 6's _transaction_timestamp wrote
# milliseconds. The column is used for `ORDER BY decision_timestamp DESC`
# (knn_context_finder, get_recent_decisions) and as part of the upsert key
# ["agent_id", "decision_timestamp", "ticker"], so mixing units means every
# millisecond row (~1.7e12) sorts above every second row (~1.7e9) forever,
# regardless of when either actually happened.
#
# Seconds wins because all 19,305 rows already in the table are seconds, so
# no data migration is needed. Nothing interprets this column as an absolute
# instant -- there is no fromtimestamp/date filtering anywhere -- so the unit
# only has to be consistent.
#
# Sub-second resolution is deliberately not required: the upsert key would
# collide only if the same agent logged two decisions for the same ticker
# within one second, and decisions are made per bar (15m at the finest).
# ---------------------------------------------------------------------------


def diary_timestamp(moment: datetime | None = None) -> int:
    """The one way to produce a decision_timestamp. UNIX seconds."""
    return int((moment or datetime.now(UTC)).timestamp())


class DecisionType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    TRAINING = "training"
    METADATA = "metadata"

class DecisionOutcome(Enum):
    PROFITABLE = "profitable"
    UNPROFITABLE = "unprofitable"
    BREAK_EVEN = "break_even"
    PENDING = "pending"
    NEUTRAL = "neutral"
    NOT_APPLICABLE = "not_applicable"

@dataclass
class DecisionRecord:
    """
    A structure describing ONE trading decision. This is the "data contract" for diary entries.
    """
    agent_id: str  # ID of the agent that made the decision (e.g., "champion", "pretender_v1.1")
    ticker: str
    decision_type: DecisionType
    reasoning: str  # Why the decision was made (e.g., "Signal from CatBoost model")

    # Decision Context
    market_context: dict[str, Any]
    context_fingerprint: str # Long string (30+ drivers) from Context Map 2.0
    context_pattern_seq: str | None = None
    model_prediction: float | None = None
    model_confidence: float | None = None

    # Execution Details
    entry_price: float | None = None
    exit_price: float | None = None

    # Outcome
    outcome: DecisionOutcome = DecisionOutcome.PENDING
    profit_loss: float | None = None

    # Other Metadata
    decision_timestamp: int = field(default_factory=diary_timestamp)
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Stable UUID string instead of random 31-bit int


class DiaryEngine(BaseMetaComponent):
    """
    The main class that implements the logic for recording, reading, and analyzing trading experience.
    Migrated to DuckDB for high-performance meta-analysis. Supporting Context Map 2.0.
    Acts as the system's memory engine for tracking trade performance and context.
    
    Note: DataManager is lazy-imported to allow DiaryEngine to work without immediate DB initialization.
    Falls back to in-memory-only mode if DuckDB is not available.
    """
    def __init__(self, data_manager=None, maxsize: int = 10000):
        self.config = get_current_config()
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)

        # Lazy import DataManager with fallback for missing DuckDB
        try:
            from src.data.management.data_manager import DataManager
            self.data_manager = data_manager or DataManager(self.config, None)
            self._db_available = True
        except ImportError as e:
            self.logger.warning(f"DataManager/DuckDB not available: {e}. Using in-memory fallback mode.")
            self.data_manager = None
            self._db_available = False

        self.table_name = "experience_diary"

        # ✅ Phase 4 Quality: Add memory-limited in-memory buffer
        self.maxsize = maxsize
        self.entries: deque[DecisionRecord] = deque(maxlen=maxsize)  # Auto-evict oldest entries

        # Initialize contextual components only if DB is available
        self.weight_calculator = None
        self.knn_finder = None
        if self._db_available and self.data_manager:
            try:
                from src.meta_learning.memory.contextual_weight_calculator import ContextualWeightCalculator
                from src.meta_learning.memory.knn_context_finder import KnnContextFinder
                self.weight_calculator = ContextualWeightCalculator(self.data_manager, self.logger)
                self.knn_finder = KnnContextFinder(self.data_manager, self.weight_calculator, self.logger)
            except ImportError as e:
                self.logger.warning(f"Contextual components not available: {e}. Using basic mode.")

        # Lazy database initialization - only initialize when first needed
        self._db_initialized = False

    @property
    def name(self) -> str:
        """Unique identifier for the meta-component."""
        return "diary"

    def update(self, data: DecisionRecord | list[DecisionRecord]) -> None:
        """
        Updates the memory engine with new experience data.

        Args:
            data: A single DecisionRecord or a list of records to be recorded.
        """
        if isinstance(data, list):
            for record in data:
                self.record_decision(record)
        else:
            self.record_decision(data)

    def get_state(self) -> dict[str, Any]:
        """
        Returns the current internal state of the diary.
        """
        try:
            # Ensure DB is initialized before querying
            self._ensure_db_initialized()

            if not self._db_available or not self.data_manager:
                # Fallback to in-memory state
                return {
                    "total_trades_recorded": len(self.entries),
                    "table_name": self.table_name,
                    "mode": "in-memory"
                }

            # Use literal table name instead of f-string for security
            query = "SELECT COUNT(*) as total_trades FROM experience_diary"
            result_list = self.data_manager.fetch_all(query)
            result = pd.DataFrame(result_list)
            total_trades = int(result.iloc[0]['total_trades']) if not result.empty else 0

            return {
                "total_trades_recorded": total_trades,
                "table_name": self.table_name
            }
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Failed to retrieve diary state: {e}",
                exc_info=True)
            return {"error": str(e)}

    def _ensure_db_initialized(self):
        """Lazily initialize the database only when first needed."""
        if not self._db_initialized and self._db_available and self.data_manager:
            self._initialize_database()
            self._db_initialized = True

    def _initialize_database(self):
        """Initializes the DuckDB table for the experience diary."""
        # Ensured context_fingerprint is VARCHAR to handle long strings (30+ drivers)
        # Changed id from INTEGER to VARCHAR to support stable UUID strings
        query = """
        CREATE TABLE IF NOT EXISTS experience_diary (
            id VARCHAR PRIMARY KEY,
            agent_id VARCHAR NOT NULL,
            decision_timestamp BIGINT NOT NULL,
            ticker VARCHAR NOT NULL,
            decision_type VARCHAR NOT NULL,
            reasoning VARCHAR,
            market_context VARCHAR, -- Saved as JSON string
            context_fingerprint VARCHAR, -- Tri-state drivers map
            context_pattern_seq VARCHAR, -- Raw rolling sequence used for KNN pattern matching
            model_prediction DOUBLE,
            model_confidence DOUBLE,
            entry_price DOUBLE,
            exit_price DOUBLE,
            outcome VARCHAR NOT NULL,
            profit_loss DOUBLE
        )
        """
        self.data_manager.execute_query(query)
        self._ensure_context_pattern_seq_column()
        self.logger.info(f"ExperienceDiary initialized in DuckDB table '{self.table_name}'.")

    def _ensure_context_pattern_seq_column(self) -> None:
        """Bring an older diary table up to the schema declared in _initialize_database.

        `CREATE TABLE IF NOT EXISTS` cannot migrate a table that already
        exists, so every change to the DDL above needs a matching step here.
        Two are known:

        - `context_pattern_seq` was added to the DDL after tables existed.
        - `id` was changed from INTEGER to VARCHAR (see DecisionRecord.
          decision_id: "Stable UUID string instead of random 31-bit int").
          Tables created before that keep INTEGER, so every real
          `record_decision()` fails with "Could not convert string '<uuid>'
          to INT32" -- which took down the whole ModelingStage, since
          training writes to the diary.
        """
        try:
            schema = self.data_manager.get_table_schema(self.table_name)
        except Exception as e:
            self.logger.warning(f"Could not read {self.table_name} schema: {e}", exc_info=True)
            return

        if not schema:
            return

        try:
            if 'context_pattern_seq' not in schema:
                self.data_manager.execute_query(
                    f'ALTER TABLE {self.table_name} ADD COLUMN context_pattern_seq VARCHAR'
                )
                self.logger.info(f"Added context_pattern_seq column to {self.table_name}.")
        except Exception as e:
            self.logger.warning(
                f"Could not ensure context_pattern_seq column: {e}", exc_info=True
            )

        id_type = str(schema.get('id', '')).upper()
        if id_type and 'CHAR' not in id_type and 'STRING' not in id_type:
            self._migrate_id_to_varchar(id_type)

    def _migrate_id_to_varchar(self, id_type: str) -> None:
        """Rebuild the diary table with `id` as VARCHAR, preserving every row.

        A plain ALTER is rejected -- `id` carries a PRIMARY KEY constraint and
        DuckDB refuses to retype a constrained column -- so the table is
        rebuilt through a temporary copy. Widening INTEGER to VARCHAR is
        lossless: existing integer ids keep their exact value as text.
        """
        tmp = f"{self.table_name}__migrating"
        try:
            self.logger.warning(
                f"Migrating {self.table_name}.id from {id_type} to VARCHAR "
                f"(legacy schema predates UUID decision ids)."
            )
            self.data_manager.execute_query(f'DROP TABLE IF EXISTS {tmp}')
            self.data_manager.execute_query(
                f'CREATE TABLE {tmp} AS SELECT * FROM {self.table_name}'
            )
            copied_cols = list(self.data_manager.get_table_schema(tmp))
            self.data_manager.execute_query(f'DROP TABLE {self.table_name}')
            self._initialize_database()
            # Name the columns explicitly: ALTER ADD COLUMN appends at the end,
            # so the old table's column ORDER need not match the fresh DDL's.
            select_list = ', '.join(
                'CAST(id AS VARCHAR)' if col == 'id' else col for col in copied_cols
            )
            self.data_manager.execute_query(
                f'INSERT INTO {self.table_name} ({", ".join(copied_cols)}) '
                f'SELECT {select_list} FROM {tmp}'
            )
            self.data_manager.execute_query(f'DROP TABLE {tmp}')
            self.logger.info(f"{self.table_name}.id migrated to VARCHAR.")
        except Exception as e:
            self.logger.error(
                f"Failed to migrate {self.table_name}.id to VARCHAR: {e}. "
                f"A copy of the original rows may remain in '{tmp}'.",
                exc_info=True,
            )

    def log_event(self, ticker: str, model_name: str, target: str, metrics: float, context_fingerprint: str = 'default', context_pattern_seq: str | None = None):
        """
        Logs a non-trading event (e.g., training result) to the experience diary.
        """
        record = DecisionRecord(
            agent_id=model_name,
            decision_timestamp=diary_timestamp(),
            ticker=ticker,
            decision_type=DecisionType.TRAINING,
            reasoning=f"Model training for target {target}",
            market_context={'target': target, 'score': float(metrics)},
            context_fingerprint=context_fingerprint,
            context_pattern_seq=context_pattern_seq,
            model_prediction=float(metrics),
            model_confidence=1.0,
            entry_price=0.0,
            exit_price=0.0,
            outcome=DecisionOutcome.NEUTRAL,
            profit_loss=0.0
        )
        self.record_decision(record)

    def record_decision(self, decision: DecisionRecord):
        """Records a single trading decision in the database."""
        df = pd.DataFrame([{
            "id": decision.decision_id,
            "agent_id": decision.agent_id,
            "decision_timestamp": decision.decision_timestamp,
            "ticker": decision.ticker,
            "decision_type": decision.decision_type.value,
            "reasoning": decision.reasoning,
            "market_context": json.dumps(decision.market_context),
            "context_fingerprint": decision.context_fingerprint,
            "context_pattern_seq": decision.context_pattern_seq,
            "model_prediction": decision.model_prediction,
            "model_confidence": decision.model_confidence,
            "entry_price": decision.entry_price,
            "exit_price": decision.exit_price,
            "outcome": decision.outcome.value,
            "profit_loss": decision.profit_loss
        }])
        self.data_manager.upsert(self.table_name, df, unique_on=["agent_id", "decision_timestamp", "ticker"])
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Recorded decision for {decision.ticker} by {decision.agent_id}")

    def record_decision_metadata(self, metadata: dict[str, Any]):
        """Records consensus decision metadata for analysis."""
        try:
            # Store metadata in a separate table or extend existing one
            # For now, we'll log it as a special record
            df = pd.DataFrame([{
                # Must match DecisionRecord.decision_id's type (UUID string).
                # This used to emit `uuid.uuid4().int & 0x7FFFFFFF`, so the two
                # writers of this same table disagreed on the type of `id`.
                "id": str(uuid.uuid4()),
                "agent_id": "consensus_engine",
                # Was `* 1000`: this writer put milliseconds in a column every
                # other writer filled with seconds.
                "decision_timestamp": diary_timestamp(),
                "ticker": "CONSENSUS",
                "decision_type": "metadata",
                "reasoning": json.dumps(metadata),
                "market_context": json.dumps(metadata),
                "context_fingerprint": metadata.get('fingerprint', ''),
                "context_pattern_seq": metadata.get('context_pattern_seq', ''),
                "model_prediction": metadata.get('raw_score', 0.0),
                "model_confidence": metadata.get('critic_score', 0.0),
                "entry_price": 0.0,
                "exit_price": 0.0,
                "outcome": "metadata",
                "profit_loss": 0.0
            }])
            self.data_manager.upsert(self.table_name, df, unique_on=["agent_id", "decision_timestamp", "ticker"])
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Recorded consensus metadata")
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Failed to record decision metadata: {e}",
                exc_info=True)

    def get_history_by_agent(self, agent_id: str) -> pd.DataFrame:
        """Retrieves the decision history for a specific agent."""
        # Use parameterized query to prevent SQL injection
        query = "SELECT * FROM experience_diary WHERE agent_id = ?"
        return pd.DataFrame(self.data_manager.fetch_all(query, params=[agent_id]))

    def get_recent_trades(self, window: int = 500) -> pd.DataFrame:
        """
        Retrieves recent trades for calibration.

        Args:
            window: Number of recent trades to retrieve

        Returns:
            DataFrame with trade history including confidence and outcome signs
        """
        # Use parameterized query to prevent SQL injection
        query = """
        SELECT
            model_confidence as confidence,
            CASE WHEN model_prediction > 0.5 THEN 1 ELSE -1 END as prediction_sign,
            CASE WHEN outcome = ? THEN (CASE WHEN model_prediction > 0.5 THEN 1 ELSE -1 END)
                 ELSE (CASE WHEN model_prediction > 0.5 THEN -1 ELSE 1 END) END as actual_sign
        FROM experience_diary
        WHERE outcome != ?
        ORDER BY decision_timestamp DESC
        LIMIT ?
        """
        try:
            df = pd.DataFrame(self.data_manager.fetch_all(query, params=[
                DecisionOutcome.PROFITABLE.value,
                DecisionOutcome.PENDING.value,
                window
            ]))
            if df.empty:
                self.logger.warning("No historical trades found for calibration")
            return df
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Failed to retrieve recent trades: {e}")
            return pd.DataFrame()

    def get_context_vulnerability(self, agent_id: str) -> dict[str, Any]:
        """
        Performs statistical analysis of unprofitable trades to find failure patterns
        within the 30+ driver context fingerprint.
        """
        # Use parameterized query to prevent SQL injection
        query = """
        SELECT context_fingerprint, COUNT(*) as loss_count
        FROM experience_diary
        WHERE agent_id = ? AND outcome = ?
        GROUP BY context_fingerprint
        ORDER BY loss_count DESC
        LIMIT 10
        """
        loss_patterns = pd.DataFrame(self.data_manager.fetch_all(query, params=[
            agent_id,
            DecisionOutcome.UNPROFITABLE.value
        ]))

        if loss_patterns.empty:
            return {"status": "No failure patterns detected"}

        # Analyze which specific elements of the string are most frequent in losses
        vulnerabilities = self._analyze_fingerprint_components(loss_patterns)

        return {
            "agent_id": agent_id,
            "total_unprofitable": int(loss_patterns['loss_count'].sum()),
            "top_loss_fingerprints": loss_patterns.to_dict('records'),
            "component_vulnerabilities": vulnerabilities
        }

    def get_context_success_analysis(self, agent_id: str) -> dict[str, Any]:
        """Identifies the 'Ideal Context' fingerprints where the model excels."""
        # Use parameterized query to prevent SQL injection
        query = """
        SELECT context_fingerprint, COUNT(*) as win_count, AVG(profit_loss) as avg_pnl
        FROM experience_diary
        WHERE agent_id = ? AND outcome = ?
        GROUP BY context_fingerprint
        HAVING win_count >= 2
        ORDER BY avg_pnl DESC
        LIMIT 10
        """
        success_patterns = pd.DataFrame(self.data_manager.fetch_all(query, params=[
            agent_id,
            DecisionOutcome.PROFITABLE.value
        ]))
        if success_patterns.empty:
            return {"status": "No consistent success patterns detected"}

        ideal_conditions = self._analyze_fingerprint_components(success_patterns, col='win_count')

        return {
            "agent_id": agent_id,
            "top_success_fingerprints": success_patterns.to_dict('records'),
            "ideal_components": ideal_conditions
        }

    def _analyze_fingerprint_components(self, df: pd.DataFrame, col: str = 'loss_count') -> dict[int, dict[str, float]]:
        """Internal helper to decompose fingerprints into individual tri-state driver stats."""
        component_stats: dict[int, dict[str, float]] = {}
        for _, row in df.iterrows():
            fp = str(row['context_fingerprint'])
            # Context Map 2.0 uses '|' for drivers and '__' for time
            drivers_part = fp.split('__')[0] if '__' in fp else fp
            drivers = drivers_part.split('|')

            for idx, val in enumerate(drivers):
                if idx not in component_stats:
                    component_stats[idx] = {'-1': 0.0, '0': 0.0, '1': 0.0}
                if val in component_stats[idx]:
                    component_stats[idx][val] += float(row[col])

        return component_stats

    def export_context_heatmap_data(self, agent_id: str) -> pd.DataFrame:
        """
        Exports data structured for context-performance heatmaps.
        Aggregates Win Rate by Time Components (from fingerprint).
        """
        # Use parameterized query to prevent SQL injection
        query = """
        SELECT
            split_part(split_part(context_fingerprint, '__', 2), '|', 1) as day_of_week,
            split_part(split_part(context_fingerprint, '__', 2), '|', 2) as hour,
            AVG(CASE WHEN outcome = 'profitable' THEN 1.0 ELSE 0.0 END) as win_rate,
            COUNT(*) as trade_count
        FROM experience_diary
        WHERE agent_id = ?
        GROUP BY day_of_week, hour
        """
        return pd.DataFrame(self.data_manager.fetch_all(query, params=[agent_id]))

    def compare_agents(self, agent_ids: list[str]) -> dict[str, Any]:
        """
        Performs performance comparison and context-specific promotion analysis.
        """
        comparison_results = self._calculate_agent_performance(agent_ids)
        recommendations = self._generate_promotion_recommendations(agent_ids, comparison_results)

        return {
            "agents": comparison_results,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }

    def _calculate_agent_performance(self, agent_ids: list[str]) -> dict[str, Any]:
        """Розраховує продуктивність для кожного агента."""
        comparison_results: dict[str, Any] = {}

        for agent_id in agent_ids:
            df = self.get_history_by_agent(agent_id)
            if df.empty:
                comparison_results[agent_id] = {"error": "No data"}
                continue

            returns = df['profit_loss'].dropna().values
            if len(returns) == 0:
                comparison_results[agent_id] = {"error": "No valid returns"}
                continue

            performance_metrics = self._calculate_performance_metrics(returns)
            comparison_results[agent_id] = {
                **performance_metrics,
                "vulnerabilities": self.get_context_vulnerability(agent_id),
                "success_zones": self.get_context_success_analysis(agent_id)
            }

        return comparison_results

    def _calculate_performance_metrics(self, returns: np.ndarray) -> dict[str, Any]:
        """Розраховує метрики продуктивності для масиву повернень."""
        clean_returns = np.asarray(returns, dtype=float)
        clean_returns = clean_returns[np.isfinite(clean_returns)]
        if clean_returns.size == 0:
            return {
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "total_trades": 0
            }

        total_pnl = np.sum(clean_returns)
        win_rate = (clean_returns > 0).mean()
        return_std = float(np.std(clean_returns))
        sharpe = (
            np.mean(clean_returns) / return_std * np.sqrt(252)
            if np.isfinite(return_std) and return_std > 1e-12
            else 0.0
        )

        return {
            "total_pnl": float(total_pnl),
            "win_rate": float(win_rate),
            "sharpe_ratio": float(sharpe),
            "total_trades": int(len(clean_returns))
        }

    def _generate_promotion_recommendations(self, agent_ids: list[str],
                                       comparison_results: dict[str, Any]) -> list[dict[str, Any]]:
        """Генерує рекомендації щодо просування на основі продуктивності."""
        recommendations = []
        champion_id = next((aid for aid in agent_ids if 'champion' in aid.lower()),
                          agent_ids[0] if agent_ids else None)

        if champion_id and len(agent_ids) > 1:
            recommendations = self._check_promotion_criteria(agent_ids, champion_id, comparison_results)

        return recommendations

    def _check_promotion_criteria(self, agent_ids: list[str], champion_id: str,
                               comparison_results: dict[str, Any]) -> list[dict[str, Any]]:
        """Перевіряє критерії просування для агентів."""
        recommendations = []
        champion_sharpe = comparison_results.get(champion_id, {}).get('sharpe_ratio', 0)

        for agent_id in agent_ids:
            if agent_id == champion_id:
                continue

            agent_sharpe = comparison_results.get(agent_id, {}).get('sharpe_ratio', 0)

            # Check for regime-specific excellence
            if agent_sharpe > champion_sharpe * 1.15:
                recommendations.append({
                    "type": "PROMOTION",
                    "agent_id": agent_id,
                    "context": "Global (General Performance)",
                    "reason": "Significantly higher Sharpe ratio"
                })

        return recommendations

    def suggest_threshold_adjustments(self, agent_id: str) -> dict[str, Any]:
        """Suggests adjustments for AdaptiveThresholds based on recent performance."""
        df = self.get_history_by_agent(agent_id).tail(20)
        if len(df) < 5:
            return {"adjustment": 0.0, "reason": "Insufficient data"}

        win_rate = (df['outcome'] == DecisionOutcome.PROFITABLE.value).mean()

        if win_rate < 0.45:
            return {"adjustment": 0.05, "action": "tighten", "reason": f"Low win rate: {win_rate:.2%}"}
        elif win_rate > 0.65:
            return {"adjustment": -0.05, "action": "loosen", "reason": f"High win rate: {win_rate:.2%}"}

        return {"adjustment": 0.0, "action": "maintain", "reason": "Stable performance"}

    def log_entry(self, entry: DecisionRecord) -> None:
        """
        Log entry to in-memory buffer with automatic eviction.

        ✅ Phase 4 Quality: Memory-limited buffer prevents unbounded growth.
        """
        if len(self.entries) == self.maxsize:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Diary buffer full ({self.maxsize}), evicting oldest entry")
        self.entries.append(entry)

    def get_recent_entries(self, limit: int = 100) -> list[DecisionRecord]:
        """Get most recent entries from in-memory buffer."""
        return list(self.entries)[-limit:]

    def memory_usage(self) -> float:
        """Return memory usage of in-memory buffer in MB."""
        return sys.getsizeof(self.entries) / 1024 / 1024

    def get_contextual_model_weights(self, context_fingerprint: str) -> dict[str, float]:
        """
        Повертає ваги моделей на основі їх історичної ефективності в даному контексті.

        Args:
            context_fingerprint: Fingerprint контексту

        Returns:
            Dict з вагами моделей (model_name -> weight)
        """
        return self.weight_calculator.get_contextual_model_weights(context_fingerprint)

    def get_contextual_model_weights_by_pattern_seq(
        self, context_pattern_seq: str
    ) -> dict[str, float]:
        """Return model weights for an exact rolling context-pattern sequence."""
        return self.weight_calculator.get_contextual_model_weights_by_pattern_seq(context_pattern_seq)

    def get_knn_contextual_model_weights(
        self,
        context_fingerprint: str,
        *,
        context_pattern_seq: str | None = None,
        n_neighbors: int = 5,
        window: int = 5000,
        min_neighbors: int = 3,
    ) -> dict[str, float]:
        """
        KNN expansion for contextual weights.

        If we don't have enough history for the exact fingerprint, we search for similar
        fingerprints (based on tri-state vector tokens) and average their contextual weights.
        """
        return self.knn_finder.get_knn_contextual_model_weights(
            context_fingerprint,
            context_pattern_seq=context_pattern_seq,
            n_neighbors=n_neighbors,
            window=window,
            min_neighbors=min_neighbors,
        )

