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
from datetime import datetime
from enum import Enum
import logging

from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager
from src.config.unified_config_manager import get_current_config
from src.meta_learning.base import BaseMetaComponent

class DecisionType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class DecisionOutcome(Enum):
    PROFITABLE = "profitable"
    UNPROFITABLE = "unprofitable"
    BREAK_EVEN = "break_even"
    PENDING = "pending"

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
    model_prediction: Optional[float] = None
    model_confidence: Optional[float] = None
    
    # Execution Details
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    
    # Outcome
    outcome: DecisionOutcome = DecisionOutcome.PENDING
    profit_loss: Optional[float] = None
    
    # Other Metadata
    decision_timestamp: int = field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    decision_id: Optional[int] = None


class DiaryEngine(BaseMetaComponent):
    """
    The main class that implements the logic for recording, reading, and analyzing trading experience.
    Migrated to DuckDB for high-performance meta-analysis. Supporting Context Map 2.0.
    Acts as the system's memory engine for tracking trade performance and context.
    """
    def __init__(self, data_manager: Optional[DataManager] = None):
        self.config = get_current_config()
        self.data_manager = data_manager or DataManager(self.config, None)
        self.table_name = "experience_diary"
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
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
            query = f"SELECT COUNT(*) as total_trades FROM {self.table_name}"
            result = self.data_manager.load_data(query)
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
            id SEQUENCE PRIMARY KEY,
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
        self.logger.info(f"ExperienceDiary initialized in DuckDB table '{self.table_name}'.")

    def record_decision(self, decision: DecisionRecord):
        """Records a single trading decision in the database."""
        df = pd.DataFrame([{
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
        self.data_manager.upsert(self.table_name, df)
        self.logger.debug(f"Recorded decision for {decision.ticker} by {decision.agent_id}")

    def get_history_by_agent(self, agent_id: str) -> pd.DataFrame:
        """Retrieves the decision history for a specific agent."""
        query = f"SELECT * FROM {self.table_name} WHERE agent_id = '{agent_id}'"
        return self.data_manager.load_data(query)

    def get_context_vulnerability(self, agent_id: str) -> Dict[str, Any]:
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
        loss_patterns = self.data_manager.load_data(query)
        
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
        query = f"""
        SELECT context_fingerprint, COUNT(*) as win_count, AVG(profit_loss) as avg_pnl
        FROM {self.table_name}
        WHERE agent_id = '{agent_id}' AND outcome = '{DecisionOutcome.PROFITABLE.value}'
        GROUP BY context_fingerprint
        HAVING win_count >= 2
        ORDER BY avg_pnl DESC
        LIMIT 10
        """
        success_patterns = self.data_manager.load_data(query)
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
        component_stats = {}
        for _, row in df.iterrows():
            fp = str(row['context_fingerprint'])
            # Context Map 2.0 uses '|' for drivers and '__' for time
            drivers_part = fp.split('__')[0] if '__' in fp else fp
            drivers = drivers_part.split('|')
            
            for idx, val in enumerate(drivers):
                if idx not in component_stats:
                    component_stats[idx] = {'-1': 0, '0': 0, '1': 0}
                if val in component_stats[idx]:
                    component_stats[idx][val] += row[col]
        
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
        return self.data_manager.load_data(query)

    def compare_agents(self, agent_ids: List[str]) -> Dict[str, Any]:
        """
        Performs performance comparison and context-specific promotion analysis.
        """
        comparison_results = {}
        
        for agent_id in agent_ids:
            df = self.get_history_by_agent(agent_id)
            if df.empty:
                comparison_results[agent_id] = {"error": "No data"}
                continue

            returns = df['profit_loss'].dropna().values
            if len(returns) == 0:
                comparison_results[agent_id] = {"error": "No valid returns"}
                continue

            total_pnl = np.sum(returns)
            win_rate = (returns > 0).mean()
            sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) != 0 else 0
            
            comparison_results[agent_id] = {
                "total_pnl": float(total_pnl),
                "win_rate": float(win_rate),
                "sharpe_ratio": float(sharpe),
                "total_trades": int(len(returns)),
                "vulnerabilities": self.get_context_vulnerability(agent_id),
                "success_zones": self.get_context_success_analysis(agent_id)
            }

        # Context-Specific Promotion Logic
        recommendations = []
        champion_id = next((aid for aid in agent_ids if 'champion' in aid.lower()), agent_ids[0] if agent_ids else None)
        
        if champion_id and len(agent_ids) > 1:
            for agent_id in agent_ids:
                if agent_id == champion_id: continue
                
                # Check for regime-specific excellence
                # This could be expanded to look at specific drivers, simplified to High/Low Vol for now
                if comparison_results.get(agent_id, {}).get('sharpe_ratio', 0) > comparison_results.get(champion_id, {}).get('sharpe_ratio', 0) * 1.15:
                    recommendations.append({
                        "type": "PROMOTION",
                        "agent_id": agent_id,
                        "context": "Global (General Performance)",
                        "reason": "Significantly higher Sharpe ratio"
                    })

        return {
            "agents": comparison_results,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }

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

    def close(self):
        """DataManager handles lifecycle."""
        pass