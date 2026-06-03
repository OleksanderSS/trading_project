import sqlite3
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from .models import MarketEvent, MarketContext, EventType, EventImpact, MarketRegime
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ContextStorage")

class ContextStorage:
    """Handles persistence of market events and context snapshots."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()

    def _ensure_connection(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def _initialize_database(self):
        conn = self._ensure_connection()
        cursor = conn.cursor()
        
        # Market Events Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                event_type TEXT,
                title TEXT,
                description TEXT,
                source TEXT,
                impact_level TEXT,
                affected_tickers TEXT,
                affected_sectors TEXT,
                keywords TEXT,
                sentiment_score REAL,
                confidence REAL,
                relevance_score REAL,
                expiration_time DATETIME,
                processed BOOLEAN,
                impact_assessment TEXT
            )
        ''')
        
        # Market Context Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                market_regime TEXT,
                volatility_regime TEXT,
                sentiment_index REAL,
                fear_greed_index REAL,
                vix_level REAL,
                major_events TEXT,
                sector_performance TEXT,
                macro_indicators TEXT,
                risk_factors TEXT,
                opportunities TEXT,
                pattern_memory_insight TEXT
            )
        ''')
        conn.commit()

    def save_market_event(self, event: MarketEvent):
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO market_events (
                timestamp, event_type, title, description, source, 
                impact_level, affected_tickers, affected_sectors, keywords,
                sentiment_score, confidence, relevance_score, expiration_time,
                processed, impact_assessment
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.timestamp.isoformat(), event.event_type.value, event.title,
            event.description, event.source, event.impact_level.value,
            json.dumps(event.affected_tickers), json.dumps(event.affected_sectors),
            json.dumps(event.keywords), event.sentiment_score, event.confidence,
            event.relevance_score, event.expiration_time.isoformat() if event.expiration_time else None,
            event.processed, json.dumps(event.impact_assessment)
        ))
        conn.commit()

    def save_market_context(self, context: MarketContext):
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO market_context (
                timestamp, market_regime, volatility_regime, sentiment_index,
                fear_greed_index, vix_level, major_events, sector_performance,
                macro_indicators, risk_factors, opportunities, pattern_memory_insight
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            context.timestamp.isoformat(), context.market_regime.value,
            context.volatility_regime, context.sentiment_index,
            context.fear_greed_index, context.vix_level,
            json.dumps([e.id for e in context.major_events if e.id]),
            json.dumps(context.sector_performance),
            json.dumps(context.macro_indicators),
            json.dumps(context.risk_factors),
            json.dumps(context.opportunities),
            context.pattern_memory_insight
        ))
        conn.commit()

    def get_recent_events(self, hours: int = 24) -> List[MarketEvent]:
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        cursor.execute('SELECT * FROM market_events WHERE timestamp > ? ORDER BY timestamp DESC', (cutoff,))
        return [self._row_to_event(row) for row in cursor.fetchall()]

    def _row_to_event(self, row: sqlite3.Row) -> MarketEvent:
        return MarketEvent(
            id=row['id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            event_type=EventType(row['event_type']),
            title=row['title'],
            description=row['description'],
            source=row['source'],
            impact_level=EventImpact(row['impact_level']),
            affected_tickers=json.loads(row['affected_tickers']),
            affected_sectors=json.loads(row['affected_sectors']),
            keywords=json.loads(row['keywords']),
            sentiment_score=row['sentiment_score'],
            confidence=row['confidence'],
            relevance_score=row['relevance_score'],
            expiration_time=datetime.fromisoformat(row['expiration_time']) if row['expiration_time'] else None,
            processed=bool(row['processed']),
            impact_assessment=json.loads(row['impact_assessment'])
        )

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
