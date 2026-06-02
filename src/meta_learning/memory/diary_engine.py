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
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from collections import deque
import sys

from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager
from src.config.unified_config_manager import get_current_config
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
    market_context: Dict[str, Any]
    context_fingerprint: str # Long string (30+ drivers) from Context Map 2.0
    context_pattern_seq: Optional[str] = None
    model_prediction: Optional[float] = None
    model_confidence: Optional[float] = None
    
    # Execution Details
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    
    # Outcome
    outcome: DecisionOutcome = DecisionOutcome.PENDING
    profit_loss: Optional[float] = None
    
    # Other Metadata
    decision_timestamp: int = field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp()))
    decision_id: Optional[int] = None


class DiaryEngine(BaseMetaComponent):
    """
    The main class that implements the logic for recording, reading, and analyzing trading experience.
    Migrated to DuckDB for high-performance meta-analysis. Supporting Context Map 2.0.
    Acts as the system's memory engine for tracking trade performance and context.
    """
    def __init__(self, data_manager: Optional[DataManager] = None, maxsize: int = 10000):
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

    def update(self, data: Union[DecisionRecord, List[DecisionRecord]]) -> None:
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

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the current internal state of the diary.
        """
        try:
            # Use literal table name instead of f-string for security
            query = "SELECT COUNT(*) as total_trades FROM experience_diary"
            result_list = self.data_manager.fetch_all(query)
            result = pd.DataFrame(result_list)
            total_trades = int(result.iloc[0]['total_trades']) if not result.empty else 0
            
            return {
                "total_trades_recorded": total_trades,
                "table_name": self.table_name
            }
        except Exception as e:
            self.logger.error(f"Failed to retrieve diary state: {e}",
                exc_info=True)
            return {"error": str(e)}

    def _initialize_database(self):
        """Initializes the DuckDB table for the experience diary."""
        # Ensured context_fingerprint is VARCHAR to handle long strings (30+ drivers)
        query = """
        CREATE TABLE IF NOT EXISTS experience_diary (
            id INTEGER PRIMARY KEY,
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
        """Migrate older diary tables that were created before pattern sequences."""
        try:
            schema = self.data_manager.get_table_schema(self.table_name)
            if 'context_pattern_seq' not in schema:
                self.data_manager.execute_query(
                    'ALTER TABLE experience_diary ADD COLUMN context_pattern_seq VARCHAR'
                )
                self.logger.info("Added context_pattern_seq column to experience_diary.")
        except Exception as e:
            self.logger.warning(
                f"Could not ensure context_pattern_seq column: {e}", exc_info=True
            )

    def log_event(self, ticker: str, model_name: str, target: str, metrics: float, context_fingerprint: str = 'default', context_pattern_seq: Optional[str] = None):
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

    def record_decision_metadata(self, metadata: Dict[str, Any]):
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
        except Exception as e:
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
        except Exception as e:
            self.logger.error(f"Failed to retrieve recent trades: {e}")
            return pd.DataFrame()

    def get_context_vulnerability(self, agent_id: str) -> Dict[str, Any]:
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

    def get_context_success_analysis(self, agent_id: str) -> Dict[str, Any]:
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

    def _analyze_fingerprint_components(self, df: pd.DataFrame, col: str = 'loss_count') -> Dict[int, Dict[str, float]]:
        """Internal helper to decompose fingerprints into individual tri-state driver stats."""
        component_stats: Dict[int, Dict[str, float]] = {}
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

    def compare_agents(self, agent_ids: List[str]) -> Dict[str, Any]:
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

    def _calculate_agent_performance(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Розраховує продуктивність для кожного агента."""
        comparison_results: Dict[str, Any] = {}
        
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

    def _calculate_performance_metrics(self, returns: np.ndarray) -> Dict[str, Any]:
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

    def _generate_promotion_recommendations(self, agent_ids: List[str], 
                                       comparison_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерує рекомендації щодо просування на основі продуктивності."""
        recommendations = []
        champion_id = next((aid for aid in agent_ids if 'champion' in aid.lower()), 
                          agent_ids[0] if agent_ids else None)
        
        if champion_id and len(agent_ids) > 1:
            recommendations = self._check_promotion_criteria(agent_ids, champion_id, comparison_results)
        
        return recommendations

    def _check_promotion_criteria(self, agent_ids: List[str], champion_id: str, 
                               comparison_results: Dict[str, Any]) -> List[Dict[str, Any]]:
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

    def suggest_threshold_adjustments(self, agent_id: str) -> Dict[str, Any]:
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
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Diary buffer full ({self.maxsize}), evicting oldest entry")
        self.entries.append(entry)
    
    def get_recent_entries(self, limit: int = 100) -> List[DecisionRecord]:
        """Get most recent entries from in-memory buffer."""
        return list(self.entries)[-limit:]
    
    def memory_usage(self) -> float:
        """Return memory usage of in-memory buffer in MB."""
        return sys.getsizeof(self.entries) / 1024 / 1024

    def get_contextual_model_weights(self, context_fingerprint: str) -> Dict[str, float]:
        """
        Повертає ваги моделей на основі їх історичної ефективності в даному контексті.
        
        Args:
            context_fingerprint: Fingerprint контексту
            
        Returns:
            Dict з вагами моделей (model_name -> weight)
        """
        # Запит для отримання історичної ефективності моделей в цьому контексті
        # Use parameterized query to prevent SQL injection
        query = """
        SELECT 
            agent_id,
            COUNT(*) as total_decisions,
            AVG(
                CASE
                    WHEN decision_type = 'training' THEN COALESCE(model_prediction, 0.0)
                    WHEN outcome = ? THEN 1.0
                    ELSE 0.0
                END
            ) as performance_score,
            COALESCE(AVG(profit_loss), 0.0) as avg_pnl
        FROM experience_diary
        WHERE context_fingerprint = ?
        GROUP BY agent_id
        HAVING total_decisions >= 2
        ORDER BY performance_score DESC, avg_pnl DESC
        """
        
        try:
            # Виконуємо запит через DuckDB
            result_df = self.data_manager.con.execute(query, [DecisionOutcome.PROFITABLE.value, context_fingerprint]).fetchdf()
            
            if result_df.empty:
                # Якщо немає історії для цього контексту, повертаємо рівні ваги
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"No historical data for context {context_fingerprint}, using equal weights")
                return {}
            
            # Розраховуємо ваги на основі win_rate та avg_pnl
            weights = {}
            total_score = 0.0
            
            for _, row in result_df.iterrows():
                agent_id = row['agent_id']
                performance_score = row['performance_score']
                avg_pnl = row['avg_pnl']
                
                # Комбінована метрика: win_rate * (1 + normalized_pnl)
                # Нормалізуємо avg_pnl до діапазону [0, 1]
                normalized_pnl = max(0, min(1, (avg_pnl + 1) / 2))  # Припускаємо pnl в діапазоні [-1, 1]
                score = performance_score * (1 + normalized_pnl)
                
                weights[agent_id] = score
                total_score += score
            
            # Нормалізуємо ваги до суми 1.0
            if total_score > 0:
                weights = {k: v / total_score for k, v in weights.items()}
            
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Contextual weights for {context_fingerprint}: {weights}")
            return weights
            
        except Exception as e:
            self.logger.error(f"Error getting contextual model weights: {e}",
                exc_info=True)
            raise RuntimeError(
                f"Failed to get contextual model weights for {context_fingerprint}"
            ) from e

    def get_contextual_model_weights_by_pattern_seq(
        self, context_pattern_seq: str
    ) -> Dict[str, float]:
        """Return model weights for an exact rolling context-pattern sequence."""
        if not context_pattern_seq:
            return {}
        query = """
        SELECT 
            agent_id,
            COUNT(*) as total_decisions,
            AVG(
                CASE
                    WHEN decision_type = 'training' THEN COALESCE(model_prediction, 0.0)
                    WHEN outcome = ? THEN 1.0
                    ELSE 0.0
                END
            ) as performance_score,
            COALESCE(AVG(profit_loss), 0.0) as avg_pnl
        FROM experience_diary
        WHERE context_pattern_seq = ?
        GROUP BY agent_id
        HAVING total_decisions >= 2
        ORDER BY performance_score DESC, avg_pnl DESC
        """
        try:
            result_df = self.data_manager.con.execute(
                query, [DecisionOutcome.PROFITABLE.value, context_pattern_seq]
            ).fetchdf()
            return self._weights_from_context_rows(result_df)
        except Exception as e:
            self.logger.error(
                f"Error getting pattern-sequence model weights: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to get contextual model weights for pattern sequence {context_pattern_seq}"
            ) from e

    def _weights_from_context_rows(self, result_df: pd.DataFrame) -> Dict[str, float]:
        if result_df.empty:
            return {}
        weights = {}
        total_score = 0.0
        for _, row in result_df.iterrows():
            avg_pnl = row['avg_pnl']
            normalized_pnl = max(0, min(1, (avg_pnl + 1) / 2))
            score = row['performance_score'] * (1 + normalized_pnl)
            weights[row['agent_id']] = score
            total_score += score
        return {k: v / total_score for k, v in weights.items()} if total_score > 0 else weights

    def get_knn_contextual_model_weights(
        self,
        context_fingerprint: str,
        *,
        context_pattern_seq: Optional[str] = None,
        n_neighbors: int = 5,
        window: int = 5000,
        min_neighbors: int = 3,
    ) -> Dict[str, float]:
        """
        KNN expansion for contextual weights.

        If we don't have enough history for the exact fingerprint, we search for similar
        fingerprints (based on tri-state vector tokens) and average their contextual weights.
        """
        # 1) Fast path: exact fingerprint has some history
        exact = self.get_contextual_model_weights(context_fingerprint)
        if exact:
            return exact
        if context_pattern_seq:
            exact_seq = self.get_contextual_model_weights_by_pattern_seq(
                context_pattern_seq)
            if exact_seq:
                return exact_seq

        try:
            if context_pattern_seq:
                pattern_weights = self._get_knn_weights_for_pattern_sequence(
                    context_pattern_seq,
                    n_neighbors=n_neighbors,
                    window=window,
                    min_neighbors=min_neighbors,
                )
                if pattern_weights:
                    return pattern_weights
            # Load recent rows that have a fingerprint and a resolved outcome.
            query = """
            SELECT context_fingerprint
            FROM experience_diary
            WHERE context_fingerprint IS NOT NULL
              AND context_fingerprint != ''
              AND outcome NOT IN (?, ?)
            ORDER BY decision_timestamp DESC
            LIMIT ?
            """
            df = self.data_manager.con.execute(
                query,
                [DecisionOutcome.PENDING.value, DecisionOutcome.NOT_APPLICABLE.value, int(window)],
            ).fetchdf()
            if df.empty:
                return {}

            hist_fps = df["context_fingerprint"].astype(str).dropna().unique().tolist()
            if len(hist_fps) < min_neighbors:
                return {}

            # Build numeric fingerprint vectors for KNN.
            def fp_to_vec(fp: str) -> list[float]:
                parts = [p for p in str(fp).split("|") if p != ""]
                vec: list[float] = []
                for p in parts:
                    try:
                        vec.append(float(p))
                    except (TypeError, ValueError):
                        # Non-numeric token; ignore (keeps vectors comparable).
                        continue
                return vec

            target_vec = fp_to_vec(context_fingerprint)
            if not target_vec:
                return {}

            hist_vecs = [(fp, fp_to_vec(fp)) for fp in hist_fps]
            # Keep only same-length vectors for stability.
            hist_vecs = [(fp, v) for fp, v in hist_vecs if len(v) == len(target_vec)]
            if len(hist_vecs) < min_neighbors:
                return {}

            import pandas as pd
            from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder

            cols = [f"fp_{i}" for i in range(len(target_vec))]
            hist_df = pd.DataFrame([v for _, v in hist_vecs], columns=cols)
            hist_df.index = [fp for fp, _ in hist_vecs]
            target_df = pd.DataFrame([target_vec], columns=cols, index=["target"])

            finder = KnnSimilarityFinder(config={"n_neighbors": int(n_neighbors)})
            res = finder.analyze({"historical_features": hist_df, "target_features": target_df})
            sims = res.get("similarities", {}).get("target", [])
            neighbor_fps = [s.get("id") for s in sims if s.get("id")]
            if len(neighbor_fps) < min_neighbors:
                return {}

            # Aggregate neighbor weights.
            agg: Dict[str, float] = {}
            for fp in neighbor_fps:
                w = self.get_contextual_model_weights(str(fp))
                for k, v in w.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
            if not agg:
                return {}
            total = sum(agg.values())
            if total > 0:
                agg = {k: v / total for k, v in agg.items()}
            return agg
        except Exception as e:
            self.logger.error(
                f"Error getting KNN contextual model weights: {e}", exc_info=True
            )
            raise RuntimeError(
                f"Failed to get KNN contextual model weights for {context_fingerprint}"
            ) from e

    def _get_knn_weights_for_pattern_sequence(
        self,
        context_pattern_seq: str,
        *,
        n_neighbors: int,
        window: int,
        min_neighbors: int,
    ) -> Dict[str, float]:
        query = """
        SELECT context_pattern_seq
        FROM experience_diary
        WHERE context_pattern_seq IS NOT NULL
          AND context_pattern_seq != ''
          AND outcome NOT IN (?, ?)
        ORDER BY decision_timestamp DESC
        LIMIT ?
        """
        df = self.data_manager.con.execute(
            query,
            [DecisionOutcome.PENDING.value, DecisionOutcome.NOT_APPLICABLE.value, int(window)],
        ).fetchdf()
        if df.empty:
            return {}
        hist_patterns = df["context_pattern_seq"].astype(str).dropna().unique().tolist()
        if len(hist_patterns) < min_neighbors:
            return {}

        target_vec = self._pattern_sequence_to_vec(context_pattern_seq)
        if not target_vec:
            return {}
        hist_vecs = [
            (pattern, self._pattern_sequence_to_vec(pattern))
            for pattern in hist_patterns
        ]
        hist_vecs = [
            (pattern, vec) for pattern, vec in hist_vecs
            if len(vec) == len(target_vec)
        ]
        if len(hist_vecs) < min_neighbors:
            return {}

        import pandas as pd
        from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder

        cols = [f"pattern_{i}" for i in range(len(target_vec))]
        hist_df = pd.DataFrame([v for _, v in hist_vecs], columns=cols)
        hist_df.index = [pattern for pattern, _ in hist_vecs]
        target_df = pd.DataFrame([target_vec], columns=cols, index=["target"])

        finder = KnnSimilarityFinder(config={"n_neighbors": int(n_neighbors)})
        res = finder.analyze({"historical_features": hist_df, "target_features": target_df})
        sims = res.get("similarities", {}).get("target", [])
        neighbor_patterns = [s.get("id") for s in sims if s.get("id")]
        if len(neighbor_patterns) < min_neighbors:
            return {}

        agg: Dict[str, float] = {}
        for pattern in neighbor_patterns:
            weights = self.get_contextual_model_weights_by_pattern_seq(str(pattern))
            for model_name, value in weights.items():
                agg[model_name] = agg.get(model_name, 0.0) + float(value)
        if not agg:
            return {}
        total = sum(agg.values())
        return {k: v / total for k, v in agg.items()} if total > 0 else agg

    @staticmethod
    def _fingerprint_to_vec(fingerprint: str) -> List[float]:
        vec: List[float] = []
        for token in str(fingerprint).split("|"):
            if token == "":
                continue
            try:
                vec.append(float(token))
            except (TypeError, ValueError):
                continue
        return vec

    @classmethod
    def _pattern_sequence_to_vec(cls, context_pattern_seq: str) -> List[float]:
        parts = [part for part in str(context_pattern_seq).split(">>") if part]
        width = 0
        parsed_parts: List[List[float]] = []
        for part in parts:
            if part == "START":
                parsed_parts.append([])
                continue
            vec = cls._fingerprint_to_vec(part)
            parsed_parts.append(vec)
            if vec and width == 0:
                width = len(vec)
        if width == 0:
            return []
        flattened: List[float] = []
        for vec in parsed_parts:
            if not vec:
                flattened.extend([0.0] * width)
            elif len(vec) < width:
                flattened.extend(vec + [0.0] * (width - len(vec)))
            else:
                flattened.extend(vec[:width])
        return flattened
