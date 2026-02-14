#!/usr/bin/env python3
"""
Experience Diary - Decision Learning & Memory System
Щоwhereнник досвandду - система навчання на рandшеннях and пам'ять
"""

import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Типи рandшень"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    ADJUST_POSITION = "adjust_position"
    RISK_MANAGEMENT = "risk_management"

class DecisionOutcome(Enum):
    """Реwithульandти рandшень"""
    PROFITABLE = "profitable"
    LOSING = "losing"
    BREAKEVEN = "breakeven"
    PARTIAL_PROFIT = "partial_profit"
    PARTIAL_LOSS = "partial_loss"

class LearningCategory(Enum):
    """Категорandї навчання"""
    PATTERN_RECOGNITION = "pattern_recognition"
    RISK_MANAGEMENT = "risk_management"
    MARKET_TIMING = "market_timing"
    POSITION_SIZING = "position_sizing"
    EXIT_STRATEGY = "exit_strategy"
    MARKET_REGIME = "market_regime"

@dataclass
class DecisionRecord:
    """Запис рandшення"""
    id: Optional[int]
    timestamp: datetime
    decision_type: DecisionType
    ticker: str
    position_size: float
    entry_price: float
    exit_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str
    confidence: float
    market_context: Dict[str, Any]
    agent_role: str
    outcome: Optional[DecisionOutcome]
    profit_loss: Optional[float]
    profit_loss_pct: Optional[float]
    holding_period_days: Optional[int]
    max_drawdown: Optional[float]
    max_runup: Optional[float]
    lessons_learned: List[str]
    mistakes_made: List[str]
    success_factors: List[str]
    market_regime: str
    volatility_regime: str
    correlation_context: Dict[str, float]
    liquidity_metrics: Dict[str, float]

@dataclass
class LearningInsight:
    """Навчальний andнсайт"""
    id: Optional[int]
    timestamp: datetime
    category: LearningCategory
    insight: str
    confidence: float
    supporting_decisions: List[int]
    contradicting_decisions: List[int]
    success_rate: float
    applicable_conditions: List[str]
    action_recommendation: str
    validation_status: str  # validated, testing, hypothesis

class ExperienceDiaryEngine:
    """
    Двигун щоwhereнника досвandду
    Реалandwithує систему навчання на рandшеннях and пам'ять
    """
    
    def __init__(self, db_path: str = "experience_diary.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
        
    def _initialize_database(self):
        """Інandцandалandwithуємо баwithу data"""
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Таблиця рandшень
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                decision_type TEXT NOT NULL,
                ticker TEXT NOT NULL,
                position_size REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                reasoning TEXT NOT NULL,
                confidence REAL NOT NULL,
                market_context TEXT NOT NULL,
                agent_role TEXT NOT NULL,
                outcome TEXT,
                profit_loss REAL,
                profit_loss_pct REAL,
                holding_period_days INTEGER,
                max_drawdown REAL,
                max_runup REAL,
                lessons_learned TEXT,
                mistakes_made TEXT,
                success_factors TEXT,
                market_regime TEXT,
                volatility_regime TEXT,
                correlation_context TEXT,
                liquidity_metrics TEXT
            )
        """)
        
        # Таблиця andнсайтandв
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                category TEXT NOT NULL,
                insight TEXT NOT NULL,
                confidence REAL NOT NULL,
                supporting_decisions TEXT,
                contradicting_decisions TEXT,
                success_rate REAL NOT NULL,
                applicable_conditions TEXT,
                action_recommendation TEXT NOT NULL,
                validation_status TEXT NOT NULL
            )
        """)
        
        # Таблиця патернandв
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                pattern_name TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                description TEXT NOT NULL,
                success_rate REAL NOT NULL,
                occurrence_count INTEGER NOT NULL,
                avg_profit_loss REAL,
                conditions TEXT,
                action_recommendation TEXT
            )
        """)
        
        # Таблиця меand-навчання
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS meta_learning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                learning_type TEXT NOT NULL,
                improvement_area TEXT NOT NULL,
                before_performance REAL,
                after_performance REAL,
                improvement_pct REAL,
                validation_method TEXT,
                confidence REAL NOT NULL,
                implementation_status TEXT
            )
        """)
        
        self.conn.commit()
    
    def record_decision(self, decision: DecisionRecord) -> int:
        """Записуємо рandшення в щоwhereнник"""
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO decisions (
                timestamp, decision_type, ticker, position_size, entry_price,
                exit_price, stop_loss, take_profit, reasoning, confidence,
                market_context, agent_role, outcome, profit_loss, profit_loss_pct,
                holding_period_days, max_drawdown, max_runup, lessons_learned,
                mistakes_made, success_factors, market_regime, volatility_regime,
                correlation_context, liquidity_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            decision.timestamp,
            decision.decision_type.value,
            decision.ticker,
            decision.position_size,
            decision.entry_price,
            decision.exit_price,
            decision.stop_loss,
            decision.take_profit,
            decision.reasoning,
            decision.confidence,
            json.dumps(decision.market_context),
            decision.agent_role,
            decision.outcome.value if decision.outcome else None,
            decision.profit_loss,
            decision.profit_loss_pct,
            decision.holding_period_days,
            decision.max_drawdown,
            decision.max_runup,
            json.dumps(decision.lessons_learned),
            json.dumps(decision.mistakes_made),
            json.dumps(decision.success_factors),
            decision.market_regime,
            decision.volatility_regime,
            json.dumps(decision.correlation_context),
            json.dumps(decision.liquidity_metrics)
        ))
        
        decision_id = cursor.lastrowid
        self.conn.commit()
        
        logger.info(f"[NOTE] Decision recorded: {decision.decision_type.value} {decision.ticker} (ID: {decision_id})")
        return decision_id
    
    def update_decision_outcome(self, decision_id: int, outcome: DecisionOutcome,
                              exit_price: float, additional_data: Dict[str, Any] = None):
        """Оновлюємо реwithульandт рandшення"""
        
        cursor = self.conn.cursor()
        
        # Calculating метрики
        cursor.execute("SELECT entry_price, position_size FROM decisions WHERE id = ?", (decision_id,))
        result = cursor.fetchone()
        
        if not result:
            logger.warning(f"Decision {decision_id} not found")
            return
        
        entry_price, position_size = result
        profit_loss = (exit_price - entry_price) * position_size
        profit_loss_pct = (exit_price - entry_price) / entry_price
        
        # Оновлюємо forпис
        update_data = {
            'outcome': outcome.value,
            'exit_price': exit_price,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct
        }
        
        if additional_data:
            update_data.update(additional_data)
        
        set_clause = ", ".join([f"{k} = ?" for k in update_data.keys()])
        values = list(update_data.values()) + [decision_id]
        
        cursor.execute(f"UPDATE decisions SET {set_clause} WHERE id = ?", values)
        self.conn.commit()
        
        logger.info(f"[DATA] Decision {decision_id} updated: {outcome.value}, P&L: {profit_loss_pct:.2%}")
        
        # Автоматично геnotруємо andнсайти
        self._generate_insights_from_decision(decision_id)
    
    def get_decision_history(self, limit: int = 100, 
                           filters: Dict[str, Any] = None) -> List[DecisionRecord]:
        """Отримуємо andсторandю рandшень"""
        
        query = "SELECT * FROM decisions"
        params = []
        
        if filters:
            conditions = []
            if 'decision_type' in filters:
                conditions.append("decision_type = ?")
                params.append(filters['decision_type'])
            if 'ticker' in filters:
                conditions.append("ticker = ?")
                params.append(filters['ticker'])
            if 'outcome' in filters:
                conditions.append("outcome = ?")
                params.append(filters['outcome'])
            if 'date_from' in filters:
                conditions.append("timestamp >= ?")
                params.append(filters['date_from'])
            if 'date_to' in filters:
                conditions.append("timestamp <= ?")
                params.append(filters['date_to'])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        
        decisions = []
        for row in cursor.fetchall():
            decision = self._row_to_decision_record(row)
            decisions.append(decision)
        
        return decisions
    
    def analyze_performance_patterns(self) -> Dict[str, Any]:
        """Аналandwithуємо патерни продуктивностand"""
        
        cursor = self.conn.cursor()
        
        # Загальна сandтистика
        cursor.execute("""
            SELECT 
                COUNT(*) as total_decisions,
                COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) as profitable_decisions,
                COUNT(CASE WHEN outcome = 'losing' THEN 1 END) as losing_decisions,
                AVG(profit_loss_pct) as avg_return,
                AVG(confidence) as avg_confidence,
                AVG(holding_period_days) as avg_holding_period
            FROM decisions 
            WHERE outcome IS NOT NULL
        """)
        
        stats = cursor.fetchone()
        
        # Продуктивнandсть for типами рandшень
        cursor.execute("""
            SELECT decision_type, COUNT(*) as count, 
                   AVG(profit_loss_pct) as avg_return,
                   COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) * 100.0 / COUNT(*) as success_rate
            FROM decisions 
            WHERE outcome IS NOT NULL
            GROUP BY decision_type
        """)
        
        by_type = cursor.fetchall()
        
        # Продуктивнandсть for тandкерами
        cursor.execute("""
            SELECT ticker, COUNT(*) as count,
                   AVG(profit_loss_pct) as avg_return,
                   COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) * 100.0 / COUNT(*) as success_rate
            FROM decisions 
            WHERE outcome IS NOT NULL
            GROUP BY ticker
            HAVING count >= 5
            ORDER BY success_rate DESC
        """)
        
        by_ticker = cursor.fetchall()
        
        # Продуктивнandсть for ринковими режимами
        cursor.execute("""
            SELECT market_regime, COUNT(*) as count,
                   AVG(profit_loss_pct) as avg_return,
                   COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) * 100.0 / COUNT(*) as success_rate
            FROM decisions 
            WHERE outcome IS NOT NULL AND market_regime IS NOT NULL
            GROUP BY market_regime
        """)
        
        by_regime = cursor.fetchall()
        
        return {
            'overall_stats': {
                'total_decisions': stats[0],
                'profitable_decisions': stats[1],
                'losing_decisions': stats[2],
                'avg_return': stats[3] or 0,
                'avg_confidence': stats[4] or 0,
                'avg_holding_period': stats[5] or 0,
                'success_rate': (stats[1] / stats[0] * 100) if stats[0] > 0 else 0
            },
            'by_decision_type': [
                {
                    'type': row[0],
                    'count': row[1],
                    'avg_return': row[2] or 0,
                    'success_rate': row[3] or 0
                }
                for row in by_type
            ],
            'by_ticker': [
                {
                    'ticker': row[0],
                    'count': row[1],
                    'avg_return': row[2] or 0,
                    'success_rate': row[3] or 0
                }
                for row in by_ticker
            ],
            'by_market_regime': [
                {
                    'regime': row[0],
                    'count': row[1],
                    'avg_return': row[2] or 0,
                    'success_rate': row[3] or 0
                }
                for row in by_regime
            ]
        }
    
    def generate_learning_insights(self) -> List[LearningInsight]:
        """Геnotруємо навчальнand andнсайти"""
        
        insights = []
        
        # Аналandwithуємо успandшнand патерни
        successful_patterns = self._analyze_successful_patterns()
        insights.extend(successful_patterns)
        
        # Аналandwithуємо notвдачand
        failure_patterns = self._analyze_failure_patterns()
        insights.extend(failure_patterns)
        
        # Аналandwithуємо риwithик-меnotджмент
        risk_insights = self._analyze_risk_management()
        insights.extend(risk_insights)
        
        # Аналandwithуємо andймandнг
        timing_insights = self._analyze_market_timing()
        insights.extend(timing_insights)
        
        # Зберandгаємо andнсайти
        for insight in insights:
            self._save_insight(insight)
        
        return insights
    
    def get_recommendations_for_current_context(self, market_context: Dict[str, Any]) -> List[str]:
        """Отримуємо рекомендацandї for поточного контексту"""
        
        cursor = self.conn.cursor()
        
        # Знаходимо релевантнand andнсайти
        cursor.execute("""
            SELECT insight, success_rate, applicable_conditions, action_recommendation
            FROM insights 
            WHERE validation_status = 'validated' AND success_rate > 0.6
            ORDER BY success_rate DESC, confidence DESC
            LIMIT 10
        """)
        
        insights = cursor.fetchall()
        
        recommendations = []
        current_regime = market_context.get('market_regime', 'unknown')
        current_volatility = market_context.get('volatility_regime', 'normal')
        
        for insight in insights:
            insight_text, success_rate, conditions, recommendation = insight
            
            # Перевandряємо релевантнandсть умов
            if current_regime in conditions or current_volatility in conditions:
                recommendations.append(f"{recommendation} (Success rate: {success_rate:.1%})")
        
        return recommendations
    
    def _generate_insights_from_decision(self, decision_id: int):
        """Геnotруємо andнсайти with конкретного рandшення"""
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,))
        decision = self._row_to_decision_record(cursor.fetchone())
        
        if not decision.outcome:
            return
        
        # Аналandwithуємо успandшнand рandшення
        if decision.outcome == DecisionOutcome.PROFITABLE:
            self._extract_success_factors(decision)
        
        # Аналandwithуємо notвдалand рandшення
        elif decision.outcome == DecisionOutcome.LOSING:
            self._extract_failure_factors(decision)
    
    def _extract_success_factors(self, decision: DecisionRecord):
        """Видandляємо фактори успandху"""
        
        success_factors = []
        
        # Аналandwithуємо reasoning
        if 'RSI oversold' in decision.reasoning and decision.outcome == DecisionOutcome.PROFITABLE:
            success_factors.append("RSI oversold condition is profitable")
        
        # Аналandwithуємо роwithмandр поwithицandї
        if decision.position_size <= 0.02 and decision.outcome == DecisionOutcome.PROFITABLE:
            success_factors.append("Small position size (2% rule) works well")
        
        # Аналandwithуємо ринковий режим
        if decision.market_regime == 'bull_market' and decision.decision_type == DecisionType.BUY:
            success_factors.append("Buying in bull market is effective")
        
        # Зберandгаємо фактори
        if success_factors:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE decisions SET success_factors = ? WHERE id = ?
            """, (json.dumps(success_factors), decision.id))
            self.conn.commit()
    
    def _extract_failure_factors(self, decision: DecisionRecord):
        """Видandляємо фактори notвдачand"""
        
        mistakes = []
        
        # Аналandwithуємо reasoning
        if 'high confidence' in decision.reasoning.lower() and decision.outcome == DecisionOutcome.LOSING:
            mistakes.append("Overconfidence leads to losses")
        
        # Аналandwithуємо роwithмandр поwithицandї
        if decision.position_size > 0.05 and decision.outcome == DecisionOutcome.LOSING:
            mistakes.append("Large position size increases risk")
        
        # Аналandwithуємо вandдсутнandсть стоп-лоссу
        if decision.stop_loss is None and decision.outcome == DecisionOutcome.LOSING:
            mistakes.append("Missing stop-loss increases losses")
        
        # Зберandгаємо помилки
        if mistakes:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE decisions SET mistakes_made = ? WHERE id = ?
            """, (json.dumps(mistakes), decision.id))
            self.conn.commit()
    
    def _analyze_successful_patterns(self) -> List[LearningInsight]:
        """Аналandwithуємо успandшнand патерни"""
        
        cursor = self.conn.cursor()
        
        # Знаходимо успandшнand патерни for reasoning
        cursor.execute("""
            SELECT reasoning, COUNT(*) as count, AVG(profit_loss_pct) as avg_return
            FROM decisions 
            WHERE outcome = 'profitable' AND reasoning IS NOT NULL
            GROUP BY reasoning
            HAVING count >= 3
            ORDER BY avg_return DESC
        """)
        
        patterns = cursor.fetchall()
        insights = []
        
        for pattern in patterns[:5]:  # Топ-5 патернandв
            reasoning, count, avg_return = pattern
            
            insight = LearningInsight(
                id=None,
                timestamp=datetime.now(),
                category=LearningCategory.PATTERN_RECOGNITION,
                insight=f"Pattern '{reasoning}' shows {avg_return:.1%} average return",
                confidence=min(1.0, count / 10),  # Чим бandльше прикладandв, тим впевnotнandше
                supporting_decisions=[],
                contradicting_decisions=[],
                success_rate=1.0,  # This успandшнand патерни
                applicable_conditions=[],
                action_recommendation=f"Look for opportunities with: {reasoning}",
                validation_status="validated"
            )
            
            insights.append(insight)
        
        return insights
    
    def _analyze_failure_patterns(self) -> List[LearningInsight]:
        """Аналandwithуємо патерни notвдач"""
        
        cursor = self.conn.cursor()
        
        # Знаходимо патерни notвдач
        cursor.execute("""
            SELECT reasoning, COUNT(*) as count, AVG(profit_loss_pct) as avg_return
            FROM decisions 
            WHERE outcome = 'losing' AND reasoning IS NOT NULL
            GROUP BY reasoning
            HAVING count >= 3
            ORDER BY avg_return ASC
        """)
        
        patterns = cursor.fetchall()
        insights = []
        
        for pattern in patterns[:5]:  # Топ-5 найгandрших патернandв
            reasoning, count, avg_return = pattern
            
            insight = LearningInsight(
                id=None,
                timestamp=datetime.now(),
                category=LearningCategory.PATTERN_RECOGNITION,
                insight=f"Pattern '{reasoning}' shows {avg_return:.1%} average loss",
                confidence=min(1.0, count / 10),
                supporting_decisions=[],
                contradicting_decisions=[],
                success_rate=0.0,  # This патерни notвдач
                applicable_conditions=[],
                action_recommendation=f"Avoid situations with: {reasoning}",
                validation_status="validated"
            )
            
            insights.append(insight)
        
        return insights
    
    def _analyze_risk_management(self) -> List[LearningInsight]:
        """Аналandwithуємо риwithик-меnotджмент"""
        
        cursor = self.conn.cursor()
        
        # Аналandwithуємо ефективнandсть стоп-лоссandв
        cursor.execute("""
            SELECT 
                CASE WHEN stop_loss IS NOT NULL THEN 'with_stop_loss' ELSE 'without_stop_loss' END as has_stop,
                COUNT(*) as count,
                AVG(profit_loss_pct) as avg_return,
                COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) * 100.0 / COUNT(*) as success_rate
            FROM decisions 
            WHERE outcome IS NOT NULL
            GROUP BY has_stop
        """)
        
        results = cursor.fetchall()
        insights = []
        
        for result in results:
            has_stop, count, avg_return, success_rate = result
            
            if has_stop == 'with_stop_loss' and success_rate > 0.6:
                insight = LearningInsight(
                    id=None,
                    timestamp=datetime.now(),
                    category=LearningCategory.RISK_MANAGEMENT,
                    insight=f"Stop-loss usage improves success rate to {success_rate:.1%}",
                    confidence=min(1.0, count / 20),
                    supporting_decisions=[],
                    contradicting_decisions=[],
                    success_rate=success_rate / 100,
                    applicable_conditions=["all_trades"],
                    action_recommendation="Always use stop-loss orders",
                    validation_status="validated"
                )
                insights.append(insight)
        
        return insights
    
    def _analyze_market_timing(self) -> List[LearningInsight]:
        """Аналandwithуємо andймandнг ринку"""
        
        cursor = self.conn.cursor()
        
        # Аналandwithуємо продуктивнandсть for годиною дня
        cursor.execute("""
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as count,
                AVG(profit_loss_pct) as avg_return,
                COUNT(CASE WHEN outcome = 'profitable' THEN 1 END) * 100.0 / COUNT(*) as success_rate
            FROM decisions 
            WHERE outcome IS NOT NULL
            GROUP BY hour
            HAVING count >= 5
            ORDER BY success_rate DESC
        """)
        
        results = cursor.fetchall()
        insights = []
        
        for result in results[:3]:  # Топ-3 години
            hour, count, avg_return, success_rate = result
            
            if success_rate > 0.7:
                insight = LearningInsight(
                    id=None,
                    timestamp=datetime.now(),
                    category=LearningCategory.MARKET_TIMING,
                    insight=f"Trading at {hour}:00 shows {success_rate:.1%} success rate",
                    confidence=min(1.0, count / 15),
                    supporting_decisions=[],
                    contradicting_decisions=[],
                    success_rate=success_rate / 100,
                    applicable_conditions=[f"hour_{hour}"],
                    action_recommendation=f"Consider more trading around {hour}:00",
                    validation_status="testing"
                )
                insights.append(insight)
        
        return insights
    
    def _save_insight(self, insight: LearningInsight):
        """Зберandгаємо andнсайт"""
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO insights (
                timestamp, category, insight, confidence, supporting_decisions,
                contradicting_decisions, success_rate, applicable_conditions,
                action_recommendation, validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            insight.timestamp,
            insight.category.value,
            insight.insight,
            insight.confidence,
            json.dumps(insight.supporting_decisions),
            json.dumps(insight.contradicting_decisions),
            insight.success_rate,
            json.dumps(insight.applicable_conditions),
            insight.action_recommendation,
            insight.validation_status
        ))
        
        self.conn.commit()
    
    def _row_to_decision_record(self, row) -> DecisionRecord:
        """Конвертуємо рядок баwithи data в DecisionRecord"""
        
        return DecisionRecord(
            id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            decision_type=DecisionType(row[2]),
            ticker=row[3],
            position_size=row[4],
            entry_price=row[5],
            exit_price=row[6],
            stop_loss=row[7],
            take_profit=row[8],
            reasoning=row[9],
            confidence=row[10],
            market_context=json.loads(row[11]),
            agent_role=row[12],
            outcome=DecisionOutcome(row[13]) if row[13] else None,
            profit_loss=row[14],
            profit_loss_pct=row[15],
            holding_period_days=row[16],
            max_drawdown=row[17],
            max_runup=row[18],
            lessons_learned=json.loads(row[19]) if row[19] else [],
            mistakes_made=json.loads(row[20]) if row[20] else [],
            success_factors=json.loads(row[21]) if row[21] else [],
            market_regime=row[22],
            volatility_regime=row[23],
            correlation_context=json.loads(row[24]) if row[24] else {},
            liquidity_metrics=json.loads(row[25]) if row[25] else {}
        )
    
    def close(self):
        """Закриваємо with'єднання with баwithою data"""
        if self.conn:
            self.conn.close()

def main():
    """Тестування щоwhereнника досвandду"""
    print(" EXPERIENCE DIARY - Decision Learning & Memory System")
    print("=" * 60)
    
    diary = ExperienceDiaryEngine()
    
    # Тестуємо forпис рandшення
    print(f"\n[NOTE] TESTING DECISION RECORDING")
    print("-" * 40)
    
    # Створюємо тестове рandшення
    decision = DecisionRecord(
        id=None,
        timestamp=datetime.now(),
        decision_type=DecisionType.BUY,
        ticker="AAPL",
        position_size=0.015,
        entry_price=150.0,
        exit_price=None,
        stop_loss=145.0,
        take_profit=160.0,
        reasoning="RSI oversold at 28, MACD bullish crossover",
        confidence=0.75,
        market_context={"vix": 18.5, "trend": "up", "volume_ratio": 1.2},
        agent_role="Worker",
        outcome=None,
        profit_loss=None,
        profit_loss_pct=None,
        holding_period_days=None,
        max_drawdown=None,
        max_runup=None,
        lessons_learned=[],
        mistakes_made=[],
        success_factors=[],
        market_regime="bull_market",
        volatility_regime="normal",
        correlation_context={"SPY": 0.7, "QQQ": 0.8},
        liquidity_metrics={"daily_volume": 50000000, "bid_ask_spread": 0.01}
    )
    
    decision_id = diary.record_decision(decision)
    print(f"[OK] Decision recorded with ID: {decision_id}")
    
    # Тестуємо оновлення реwithульandту
    print(f"\n[DATA] TESTING OUTCOME UPDATE")
    print("-" * 40)
    
    diary.update_decision_outcome(
        decision_id, 
        DecisionOutcome.PROFITABLE, 
        155.5,
        {
            'holding_period_days': 5,
            'max_drawdown': 0.02,
            'max_runup': 0.05
        }
    )
    print(f"[OK] Outcome updated: PROFITABLE")
    
    # Тестуємо аналandwith продуктивностand
    print(f"\n[UP] TESTING PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    performance = diary.analyze_performance_patterns()
    
    print(f"[DATA] Total decisions: {performance['overall_stats']['total_decisions']}")
    print(f"[MONEY] Success rate: {performance['overall_stats']['success_rate']:.1f}%")
    print(f"[UP] Average return: {performance['overall_stats']['avg_return']:.2%}")
    
    # Тестуємо геnotрацandю andнсайтandв
    print(f"\n[BRAIN] TESTING INSIGHT GENERATION")
    print("-" * 40)
    
    insights = diary.generate_learning_insights()
    print(f"[IDEA] Generated {len(insights)} insights")
    
    for insight in insights[:3]:
        print(f"    {insight.insight}")
        print(f"     Recommendation: {insight.action_recommendation}")
    
    # Тестуємо рекомендацandї for контексту
    print(f"\n[TARGET] TESTING CONTEXT RECOMMENDATIONS")
    print("-" * 40)
    
    current_context = {
        'market_regime': 'bull_market',
        'volatility_regime': 'normal',
        'vix': 16.5
    }
    
    recommendations = diary.get_recommendations_for_current_context(current_context)
    print(f"[IDEA] Generated {len(recommendations)} recommendations")
    
    for rec in recommendations[:3]:
        print(f"    {rec}")
    
    # Тестуємо andсторandю рandшень
    print(f"\n TESTING DECISION HISTORY")
    print("-" * 40)
    
    history = diary.get_decision_history(limit=5)
    print(f"[DATA] Retrieved {len(history)} decisions")
    
    for decision in history:
        print(f"    {decision.decision_type.value} {decision.ticker} - {decision.outcome.value if decision.outcome else 'PENDING'}")
    
    # Закриваємо with'єднання
    diary.close()
    
    print(f"\n[TARGET] EXPERIENCE DIARY READY!")
    print(f"[NOTE] Decision recording and tracking")
    print(f"[BRAIN] Automatic insight generation")
    print(f"[UP] Performance pattern analysis")
    print(f"[IDEA] Context-aware recommendations")
    print(f" Learning from experience")

if __name__ == "__main__":
    main()
