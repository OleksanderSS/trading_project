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
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager
from src.meta_learning.base import BaseMetaComponent


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
    model_prediction: float | None = None
    model_confidence: float | None = None

    # Execution Details
    entry_price: float | None = None
    exit_price: float | None = None

    # Outcome
    outcome: DecisionOutcome = DecisionOutcome.PENDING
    profit_loss: float | None = None

    # Other Metadata
    decision_timestamp: int = field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp()))
    decision_id: int | None = None


class DiaryEngine(BaseMetaComponent):
    """
    The main class that implements the logic for recording, reading, and analyzing trading experience.
    Migrated to DuckDB for high-performance meta-analysis. Supporting Context Map 2.0.
    Acts as the system's memory engine for tracking trade performance and context.
    """
    def __init__(self, data_manager: DataManager | None = None, maxsize: int = 10000):
        self.config = get_current_config()
        self.data_manager = data_manager or DataManager(self.config, None)
        self.table_name = "experience_diary"
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)

        # ✅ Phase 4 Quality: Add memory-limited in-memory buffer
        self.maxsize = maxsize
        self.entries: deque[DecisionRecord] = deque(maxlen=maxsize)  # Auto-evict oldest entries

        self._initialize_database()

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
            query = f"SELECT COUNT(*) as total_trades FROM {self.table_name}"
            result_list = self.data_manager.fetch_all(query)
            result = pd.DataFrame(result_list)
            total_trades = int(result.iloc[0]['total_trades']) if not result.empty else 0

            return {
                "total_trades_recorded": total_trades,
                "table_name": self.table_name
            }
        except Exception as e:
            self.logger.error(f"Failed to retrieve diary state: {e}")
            return {"error": str(e)}

    def _initialize_database(self):
        """Initializes the DuckDB table for the experience diary."""
        # Ensured context_fingerprint is VARCHAR to handle long strings (30+ drivers)
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id INTEGER PRIMARY KEY,
            agent_id VARCHAR NOT NULL,
            decision_timestamp BIGINT NOT NULL,
            ticker VARCHAR NOT NULL,
            decision_type VARCHAR NOT NULL,
            reasoning VARCHAR,
            market_context VARCHAR, -- Saved as JSON string
            context_fingerprint VARCHAR, -- Tri-state drivers map
            model_prediction DOUBLE,
            model_confidence DOUBLE,
            entry_price DOUBLE,
            exit_price DOUBLE,
            outcome VARCHAR NOT NULL,
            profit_loss DOUBLE
        )
        """
        self.data_manager.execute_query(query)
        # ✅ ENHANCED: Create index on context_fingerprint for performance
        index_query = f"CREATE INDEX IF NOT EXISTS idx_context_fingerprint ON {self.table_name}(context_fingerprint)"
        self.data_manager.execute_query(index_query)
        self.logger.info(f"ExperienceDiary initialized in DuckDB table '{self.table_name}' with index.")

    def log_event(self, ticker: str, model_name: str, target: str, metrics: float, context_fingerprint: str = 'default'):
        """
        Logs a non-trading event (e.g., training result) to the experience diary.
        """
        record = DecisionRecord(
            agent_id=model_name,
            decision_timestamp=int(datetime.now(timezone.utc).timestamp()),
            ticker=ticker,
            decision_type=DecisionType.TRAINING,
            reasoning=f"Model training for target {target}",
            market_context={'target': target, 'score': float(metrics)},
            context_fingerprint=context_fingerprint,
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
        import uuid
        df = pd.DataFrame([{
            "id": uuid.uuid4().int & 0x7FFFFFFF,
            "agent_id": decision.agent_id,
            "decision_timestamp": decision.decision_timestamp,
            "ticker": decision.ticker,
            "decision_type": decision.decision_type.value,
            "reasoning": decision.reasoning,
            "market_context": json.dumps(decision.market_context),
            "context_fingerprint": decision.context_fingerprint,
            "model_prediction": decision.model_prediction,
            "model_confidence": decision.model_confidence,
            "entry_price": decision.entry_price,
            "exit_price": decision.exit_price,
            "outcome": decision.outcome.value,
            "profit_loss": decision.profit_loss
        }])
        self.data_manager.upsert(self.table_name, df, unique_on=["agent_id", "decision_timestamp", "ticker"])
        self.logger.debug(f"Recorded decision for {decision.ticker} by {decision.agent_id}")

    def record_decision_metadata(self, metadata: dict[str, Any]):
        """Records consensus decision metadata for analysis."""
        try:
            # Store metadata in a separate table or extend existing one
            # For now, we'll log it as a special record
            import uuid
            df = pd.DataFrame([{
                "id": uuid.uuid4().int & 0x7FFFFFFF,
                "agent_id": "consensus_engine",
                "decision_timestamp": int(pd.Timestamp.now().timestamp() * 1000),
                "ticker": "CONSENSUS",
                "decision_type": "metadata",
                "reasoning": json.dumps(metadata),
                "market_context": json.dumps(metadata),
                "context_fingerprint": metadata.get('fingerprint', ''),
                "model_prediction": metadata.get('raw_score', 0.0),
                "model_confidence": metadata.get('critic_score', 0.0),
                "entry_price": 0.0,
                "exit_price": 0.0,
                "outcome": "metadata",
                "profit_loss": 0.0
            }])
            self.data_manager.upsert(self.table_name, df, unique_on=["agent_id", "decision_timestamp", "ticker"])
            self.logger.debug("Recorded consensus metadata")
        except Exception as e:
            self.logger.error(f"Failed to record decision metadata: {e}")

    def get_history_by_agent(self, agent_id: str) -> pd.DataFrame:
        """Retrieves the decision history for a specific agent."""
        query = f"SELECT * FROM {self.table_name} WHERE agent_id = '{agent_id}'"
        return pd.DataFrame(self.data_manager.fetch_all(query))

    def get_recent_trades(self, window: int = 500) -> pd.DataFrame:
        """
        Retrieves recent trades for calibration.

        Args:
            window: Number of recent trades to retrieve

        Returns:
            DataFrame with trade history including confidence and outcome signs
        """
        query = f"""
        SELECT
            model_confidence as confidence,
            CASE WHEN model_prediction > 0.5 THEN 1 ELSE -1 END as prediction_sign,
            CASE WHEN outcome = '{DecisionOutcome.PROFITABLE.value}' THEN (CASE WHEN model_prediction > 0.5 THEN 1 ELSE -1 END)
                 ELSE (CASE WHEN model_prediction > 0.5 THEN -1 ELSE 1 END) END as actual_sign
        FROM {self.table_name}
        WHERE outcome != '{DecisionOutcome.PENDING.value}'
        ORDER BY decision_timestamp DESC
        LIMIT {window}
        """
        try:
            df = pd.DataFrame(self.data_manager.fetch_all(query))
            if df.empty:
                self.logger.warning("No historical trades found for calibration")
            return df
        except Exception as e:
            self.logger.error(f"Failed to retrieve recent trades: {e}")
            return pd.DataFrame()

    def get_context_vulnerability(self, agent_id: str) -> dict[str, Any]:
        """
        Performs statistical analysis of unprofitable trades to find failure patterns
        within the 30+ driver context fingerprint.
        """
        query = f"""
        SELECT context_fingerprint, COUNT(*) as loss_count
        FROM {self.table_name}
        WHERE agent_id = '{agent_id}' AND outcome = '{DecisionOutcome.UNPROFITABLE.value}'
        GROUP BY context_fingerprint
        ORDER BY loss_count DESC
        LIMIT 10
        """
        loss_patterns = pd.DataFrame(self.data_manager.fetch_all(query))

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
        query = f"""
        SELECT context_fingerprint, COUNT(*) as win_count, AVG(profit_loss) as avg_pnl
        FROM {self.table_name}
        WHERE agent_id = '{agent_id}' AND outcome = '{DecisionOutcome.PROFITABLE.value}'
        GROUP BY context_fingerprint
        HAVING win_count >= 2
        ORDER BY avg_pnl DESC
        LIMIT 10
        """
        success_patterns = pd.DataFrame(self.data_manager.fetch_all(query))
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
        # We extract Time features from the fingerprint: DayOfWeek__Hour__MarketOpen
        query = f"""
        SELECT
            split_part(split_part(context_fingerprint, '__', 2), '|', 1) as day_of_week,
            split_part(split_part(context_fingerprint, '__', 2), '|', 2) as hour,
            AVG(CASE WHEN outcome = 'profitable' THEN 1.0 ELSE 0.0 END) as win_rate,
            COUNT(*) as trade_count
        FROM {self.table_name}
        WHERE agent_id = '{agent_id}'
        GROUP BY day_of_week, hour
        """
        return pd.DataFrame(self.data_manager.fetch_all(query))

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
        total_pnl = np.sum(returns)
        win_rate = (returns > 0).mean()
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) != 0 else 0

        return {
            "total_pnl": float(total_pnl),
            "win_rate": float(win_rate),
            "sharpe_ratio": float(sharpe),
            "total_trades": int(len(returns))
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
        if len(df) < 5: return {"adjustment": 0.0, "reason": "Insufficient data"}

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
        # Запит для отримання історичної ефективності моделей в цьому контексті
        query = f"""
        SELECT
            agent_id,
            COUNT(*) as total_decisions,
            AVG(CASE WHEN outcome = '{DecisionOutcome.PROFITABLE.value}' THEN 1.0 ELSE 0.0 END) as win_rate,
            AVG(profit_loss) as avg_pnl
        FROM {self.table_name}
        WHERE context_fingerprint = '{context_fingerprint}'
        GROUP BY agent_id
        HAVING total_decisions >= 2
        ORDER BY win_rate DESC, avg_pnl DESC
        """

        try:
            # Виконуємо запит через DuckDB
            result_df = self.data_manager.con.execute(query).fetchdf()

            if result_df.empty:
                # Якщо немає історії для цього контексту, повертаємо рівні ваги
                self.logger.debug(f"No historical data for context {context_fingerprint}, using equal weights")
                return {}

            # Розраховуємо ваги на основі win_rate та avg_pnl
            weights = {}
            total_score = 0.0

            for _, row in result_df.iterrows():
                agent_id = row['agent_id']
                win_rate = row['win_rate']
                avg_pnl = row['avg_pnl']

                # Комбінована метрика: win_rate * (1 + normalized_pnl)
                # Нормалізуємо avg_pnl до діапазону [0, 1]
                normalized_pnl = max(0, min(1, (avg_pnl + 1) / 2))  # Припускаємо pnl в діапазоні [-1, 1]
                score = win_rate * (1 + normalized_pnl)

                weights[agent_id] = score
                total_score += score

            # Нормалізуємо ваги до суми 1.0
            if total_score > 0:
                weights = {k: v / total_score for k, v in weights.items()}

            self.logger.debug(f"Contextual weights for {context_fingerprint}: {weights}")
            return weights

        except Exception as e:
            self.logger.error(f"Error getting contextual model weights: {e}")
            return {}
