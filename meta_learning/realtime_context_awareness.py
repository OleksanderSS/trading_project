#!/usr/bin/env python3
"""
Real-time Context Awareness - News & Events Integration
Контекстна обandwithнанandсть в реальному часand - Інтеграцandя новин and подandй
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
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Типи подandй"""
    ECONOMIC_RELEASE = "economic_release"
    CORPORATE_NEWS = "corporate_news"
    MARKET_EVENT = "market_event"
    GEOPOLITICAL = "geopolitical"
    REGULATORY = "regulatory"
    WEATHER = "weather"
    SOCIAL_SENTIMENT = "social_sentiment"

class EventImpact(Enum):
    """Рandвень впливу подandї"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MarketRegime(Enum):
    """Ринковand режими"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRISIS = "crisis"

@dataclass
class MarketEvent:
    """Ринкова подandя"""
    id: Optional[int]
    timestamp: datetime
    event_type: EventType
    title: str
    description: str
    source: str
    impact_level: EventImpact
    affected_tickers: List[str]
    affected_sectors: List[str]
    keywords: List[str]
    sentiment_score: float
    confidence: float
    relevance_score: float
    expiration_time: Optional[datetime]
    processed: bool
    impact_assessment: Dict[str, Any]

@dataclass
class MarketContext:
    """Ринковий контекст"""
    timestamp: datetime
    market_regime: MarketRegime
    volatility_regime: str
    sentiment_index: float
    fear_greed_index: Optional[float]
    vix_level: Optional[float]
    major_events: List[MarketEvent]
    sector_performance: Dict[str, float]
    macro_indicators: Dict[str, float]
    risk_factors: List[str]
    opportunities: List[str]

class RealtimeContextAwarenessEngine:
    """
    Двигун контекстної обandwithнаностand в реальному часand
    Реалandwithує монandторинг новин, подandй and ринкового контексту
    """
    
    def __init__(self, db_path: str = "realtime_context_awareness.db"):
        self.db_path = db_path
        self.conn = None
        
        # Конфandгурацandя джерел data
        self.news_sources = {
            "financial_times": "https://www.ft.com",
            "reuters": "https://www.reuters.com",
            "bloomberg": "https://www.bloomberg.com",
            "yahoo_finance": "https://finance.yahoo.com"
        }
        
        # Keywords for фandльтрацandї
        self.keywords = {
            "economic": ["gdp", "inflation", "interest rates", "employment", "fed", "ecb"],
            "market": ["bull", "bear", "crash", "rally", "correction", "volatility"],
            "corporate": ["earnings", "merger", "acquisition", "bankruptcy", "ipo"],
            "geopolitical": ["war", "sanctions", "election", "trade", "tensions"]
        }
        
        # Сектори for вandдстеження
        self.sectors = [
            "technology", "healthcare", "finance", "energy", 
            "consumer", "industrial", "materials", "utilities"
        ]
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Інandцandалandwithуємо баwithу data"""
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Таблиця подandй
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                event_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                source TEXT NOT NULL,
                impact_level TEXT NOT NULL,
                affected_tickers TEXT,
                affected_sectors TEXT,
                keywords TEXT,
                sentiment_score REAL NOT NULL,
                confidence REAL NOT NULL,
                relevance_score REAL NOT NULL,
                expiration_time DATETIME,
                processed BOOLEAN NOT NULL,
                impact_assessment TEXT
            )
        """)
        
        # Таблиця ринкового контексту
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                market_regime TEXT NOT NULL,
                volatility_regime TEXT NOT NULL,
                sentiment_index REAL NOT NULL,
                fear_greed_index REAL,
                vix_level REAL,
                major_events TEXT,
                sector_performance TEXT,
                macro_indicators TEXT,
                risk_factors TEXT,
                opportunities TEXT
            )
        """)
        
        # Таблиця новинних джерел
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS news_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                last_fetch DATETIME,
                active BOOLEAN NOT NULL,
                reliability_score REAL NOT NULL
            )
        """)
        
        # Таблиця настроїв
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                source TEXT NOT NULL,
                ticker TEXT,
                sentiment_score REAL NOT NULL,
                confidence REAL NOT NULL,
                text_sample TEXT
            )
        """)
        
        # Інandцandалandwithуємо джерела новин
        self._initialize_news_sources()
        
        self.conn.commit()
    
    def _initialize_news_sources(self):
        """Інandцandалandwithуємо джерела новин"""
        
        cursor = self.conn.cursor()
        
        for source_name, source_url in self.news_sources.items():
            cursor.execute("""
                INSERT OR IGNORE INTO news_sources (name, url, active, reliability_score)
                VALUES (?, ?, ?, ?)
            """, (source_name, source_url, True, 0.8))
        
        self.conn.commit()
    
    def scan_news_sources(self) -> List[MarketEvent]:
        """Скануємо джерела новин"""
        
        events = []
        
        for source_name, source_url in self.news_sources.items():
            try:
                source_events = self._fetch_news_from_source(source_name, source_url)
                events.extend(source_events)
                logger.info(f" Fetched {len(source_events)} events from {source_name}")
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to fetch from {source_name}: {e}")
        
        # Зберandгаємо подandї
        for event in events:
            self._save_market_event(event)
        
        logger.info(f"[DATA] Total events scanned: {len(events)}")
        return events
    
    def analyze_market_context(self) -> MarketContext:
        """Аналandwithуємо поточний ринковий контекст"""
        
        # Отримуємо осandннand подandї
        recent_events = self._get_recent_events(hours=24)
        
        # Виwithначаємо ринковий режим
        market_regime = self._detect_market_regime(recent_events)
        
        # Аналandwithуємо волатильнandсть
        volatility_regime = self._detect_volatility_regime(recent_events)
        
        # Calculating andнwhereкс настроїв
        sentiment_index = self._calculate_sentiment_index(recent_events)
        
        # Отримуємо Fear & Greed Index
        fear_greed_index = self._get_fear_greed_index()
        
        # Отримуємо VIX
        vix_level = self._get_vix_level()
        
        # Аналandwithуємо секторну продуктивнandсть
        sector_performance = self._analyze_sector_performance(recent_events)
        
        # Отримуємо макроandндикатори
        macro_indicators = self._get_macro_indicators()
        
        # Виявляємо фактори риwithику
        risk_factors = self._identify_risk_factors(recent_events)
        
        # Виявлямо можливостand
        opportunities = self._identify_opportunities(recent_events)
        
        # Створюємо контекст
        context = MarketContext(
            timestamp=datetime.now(),
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            sentiment_index=sentiment_index,
            fear_greed_index=fear_greed_index,
            vix_level=vix_level,
            major_events=recent_events[:10],  # Топ-10 подandй
            sector_performance=sector_performance,
            macro_indicators=macro_indicators,
            risk_factors=risk_factors,
            opportunities=opportunities
        )
        
        # Зберandгаємо контекст
        self._save_market_context(context)
        
        return context
    
    def get_contextual_recommendations(self, ticker: str) -> Dict[str, Any]:
        """Отримуємо контекстуальнand рекомендацandї for тandкера"""
        
        # Отримуємо поточний контекст
        context = self._get_latest_context()
        if not context:
            return {"error": "No context available"}
        
        # Отримуємо релевантнand подandї
        relevant_events = self._get_ticker_events(ticker, hours=48)
        
        # Аналandwithуємо вплив подandй
        event_impact = self._analyze_event_impact(ticker, relevant_events)
        
        # Геnotруємо рекомендацandї
        recommendations = self._generate_contextual_recommendations(
            ticker, context, relevant_events, event_impact
        )
        
        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "market_context": {
                "regime": context.market_regime.value,
                "volatility": context.volatility_regime,
                "sentiment": context.sentiment_index,
                "fear_greed": context.fear_greed_index,
                "vix": context.vix_level
            },
            "relevant_events": [
                {
                    "title": event.title,
                    "impact": event.impact_level.value,
                    "sentiment": event.sentiment_score,
                    "relevance": event.relevance_score
                }
                for event in relevant_events[:5]
            ],
            "event_impact": event_impact,
            "recommendations": recommendations,
            "risk_factors": [
                factor for factor in context.risk_factors 
                if ticker.lower() in factor.lower() or any(
                    sector in factor.lower() 
                    for sector in self._get_ticker_sectors(ticker)
                )
            ],
            "opportunities": [
                opp for opp in context.opportunities 
                if ticker.lower() in opp.lower() or any(
                    sector in opp.lower() 
                    for sector in self._get_ticker_sectors(ticker)
                )
            ]
        }
    
    def update_sentiment_analysis(self, ticker: str, text_data: List[str]) -> Dict[str, float]:
        """Оновлюємо аналandwith настроїв"""
        
        cursor = self.conn.cursor()
        
        sentiment_scores = []
        
        for text in text_data:
            sentiment = self._analyze_text_sentiment(text)
            
            # Зберandгаємо реwithульandт
            cursor.execute("""
                INSERT INTO sentiment_data (
                    timestamp, source, ticker, sentiment_score, confidence, text_sample
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                "user_input",
                ticker,
                sentiment["score"],
                sentiment["confidence"],
                text[:200]  # Зберandгаємо тandльки першand 200 символandв
            ))
            
            sentiment_scores.append(sentiment["score"])
        
        self.conn.commit()
        
        # Calculating середнandй настрandй
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        
        logger.info(f"[DATA] Updated sentiment for {ticker}: {avg_sentiment:.3f}")
        
        return {
            "ticker": ticker,
            "average_sentiment": avg_sentiment,
            "sample_count": len(sentiment_scores),
            "sentiment_distribution": {
                "positive": sum(1 for s in sentiment_scores if s > 0.1),
                "neutral": sum(1 for s in sentiment_scores if -0.1 <= s <= 0.1),
                "negative": sum(1 for s in sentiment_scores if s < -0.1)
            }
        }
    
    def get_event_history(self, event_type: Optional[EventType] = None,
                         limit: int = 50) -> List[MarketEvent]:
        """Отримуємо andсторandю подandй"""
        
        query = "SELECT * FROM market_events"
        params = []
        
        if event_type:
            query += " WHERE event_type = ?"
            params.append(event_type.value)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        
        events = []
        for row in cursor.fetchall():
            event = self._row_to_market_event(row)
            events.append(event)
        
        return events
    
    def analyze_context_effectiveness(self) -> Dict[str, Any]:
        """Аналandwithуємо ефективнandсть контекстної обandwithнаностand"""
        
        cursor = self.conn.cursor()
        
        # Сandтистика подandй
        cursor.execute("""
            SELECT 
                event_type,
                COUNT(*) as total_events,
                AVG(sentiment_score) as avg_sentiment,
                AVG(relevance_score) as avg_relevance,
                COUNT(CASE WHEN impact_level = 'high' THEN 1 END) as high_impact_events
            FROM market_events
            GROUP BY event_type
        """)
        
        by_type = cursor.fetchall()
        
        # Ефективнandсть джерел
        cursor.execute("""
            SELECT 
                ns.name,
                COUNT(me.id) as events_count,
                AVG(me.relevance_score) as avg_relevance,
                AVG(me.confidence) as avg_confidence
            FROM news_sources ns
            LEFT JOIN market_events me ON ns.name = me.source
            WHERE ns.active = 1
            GROUP BY ns.name
            ORDER BY events_count DESC
        """)
        
        by_source = cursor.fetchall()
        
        # Тренди настроїв
        cursor.execute("""
            SELECT 
                DATE(timestamp) as date,
                AVG(sentiment_score) as daily_sentiment
            FROM sentiment_data
            WHERE timestamp >= date('now', '-30 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        """)
        
        sentiment_trend = cursor.fetchall()
        
        return {
            'by_event_type': [
                {
                    'type': row[0],
                    'total_events': row[1],
                    'avg_sentiment': row[2] or 0,
                    'avg_relevance': row[3] or 0,
                    'high_impact_events': row[4],
                    'high_impact_ratio': (row[4] / row[1] * 100) if row[1] > 0 else 0
                }
                for row in by_type
            ],
            'by_source': [
                {
                    'source': row[0],
                    'events_count': row[1],
                    'avg_relevance': row[2] or 0,
                    'avg_confidence': row[3] or 0
                }
                for row in by_source
            ],
            'sentiment_trend': [
                {
                    'date': row[0],
                    'sentiment': row[1] or 0
                }
                for row in sentiment_trend
            ]
        }
    
    def _fetch_news_from_source(self, source_name: str, source_url: str) -> List[MarketEvent]:
        """Отримуємо новини with джерела"""
        
        # Симуляцandя отримання новин (в реальностand тут був би HTTP forпит)
        events = []
        
        # Геnotруємо тестовand подandї
        event_types = list(EventType)
        impact_levels = list(EventImpact)
        
        for i in range(np.random.randint(3, 10)):
            event = MarketEvent(
                id=None,
                timestamp=datetime.now() - timedelta(hours=np.random.randint(0, 24)),
                event_type=np.random.choice(event_types),
                title=f"Sample news {i+1} from {source_name}",
                description=f"This is a sample news article about market events",
                source=source_name,
                impact_level=np.random.choice(impact_levels),
                affected_tickers=[f"TICKER_{np.random.randint(1, 100)}"],
                affected_sectors=[np.random.choice(self.sectors)],
                keywords=["market", "news", "analysis"],
                sentiment_score=np.random.uniform(-1, 1),
                confidence=np.random.uniform(0.6, 1.0),
                relevance_score=np.random.uniform(0.5, 1.0),
                expiration_time=datetime.now() + timedelta(hours=48),
                processed=False,
                impact_assessment={}
            )
            events.append(event)
        
        return events
    
    def _save_market_event(self, event: MarketEvent):
        """Зберandгаємо подandю"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO market_events (
                timestamp, event_type, title, description, source, impact_level,
                affected_tickers, affected_sectors, keywords, sentiment_score,
                confidence, relevance_score, expiration_time, processed, impact_assessment
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.timestamp,
            event.event_type.value,
            event.title,
            event.description,
            event.source,
            event.impact_level.value,
            json.dumps(event.affected_tickers),
            json.dumps(event.affected_sectors),
            json.dumps(event.keywords),
            event.sentiment_score,
            event.confidence,
            event.relevance_score,
            event.expiration_time,
            event.processed,
            json.dumps(event.impact_assessment)
        ))
        
        self.conn.commit()
    
    def _save_market_context(self, context: MarketContext):
        """Зберandгаємо ринковий контекст"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO market_context (
                timestamp, market_regime, volatility_regime, sentiment_index,
                fear_greed_index, vix_level, major_events, sector_performance,
                macro_indicators, risk_factors, opportunities
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            context.timestamp,
            context.market_regime.value,
            context.volatility_regime,
            context.sentiment_index,
            context.fear_greed_index,
            context.vix_level,
            json.dumps([asdict(event) for event in context.major_events]),
            json.dumps(context.sector_performance),
            json.dumps(context.macro_indicators),
            json.dumps(context.risk_factors),
            json.dumps(context.opportunities)
        ))
        
        self.conn.commit()
    
    def _get_recent_events(self, hours: int = 24) -> List[MarketEvent]:
        """Отримуємо осandннand подandї"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM market_events 
            WHERE timestamp >= datetime('now', '-{} hours')
            ORDER BY relevance_score DESC, timestamp DESC
        """.format(hours))
        
        events = []
        for row in cursor.fetchall():
            event = self._row_to_market_event(row)
            events.append(event)
        
        return events
    
    def _detect_market_regime(self, events: List[MarketEvent]) -> MarketRegime:
        """Виwithначаємо ринковий режим"""
        
        if not events:
            return MarketRegime.SIDEWAYS
        
        # Аналandwithуємо настрої подandй
        positive_events = sum(1 for e in events if e.sentiment_score > 0.2)
        negative_events = sum(1 for e in events if e.sentiment_score < -0.2)
        
        # Аналandwithуємо рandвень впливу
        high_impact_events = sum(1 for e in events if e.impact_level in [EventImpact.HIGH, EventImpact.CRITICAL])
        
        # Виwithначаємо режим
        if high_impact_events >= 3:
            return MarketRegime.VOLATILE
        elif high_impact_events >= 5:
            return MarketRegime.CRISIS
        elif positive_events > negative_events * 1.5:
            return MarketRegime.BULL_MARKET
        elif negative_events > positive_events * 1.5:
            return MarketRegime.BEAR_MARKET
        else:
            return MarketRegime.SIDEWAYS
    
    def _detect_volatility_regime(self, events: List[MarketEvent]) -> str:
        """Виwithначаємо режим волатильностand"""
        
        if not events:
            return "normal"
        
        high_impact_count = sum(1 for e in events if e.impact_level in [EventImpact.HIGH, EventImpact.CRITICAL])
        
        if high_impact_count >= 5:
            return "extreme"
        elif high_impact_count >= 3:
            return "high"
        elif high_impact_count >= 1:
            return "elevated"
        else:
            return "normal"
    
    def _calculate_sentiment_index(self, events: List[MarketEvent]) -> float:
        """Calculating andнwhereкс настроїв"""
        
        if not events:
            return 0.0
        
        # Вwithважуємо настрої for релевантнandстю
        weighted_sentiment = sum(
            e.sentiment_score * e.relevance_score * e.confidence 
            for e in events
        )
        total_weight = sum(e.relevance_score * e.confidence for e in events)
        
        return weighted_sentiment / total_weight if total_weight > 0 else 0.0
    
    def _get_fear_greed_index(self) -> Optional[float]:
        """Отримуємо Fear & Greed Index"""
        
        # Симуляцandя отримання Fear & Greed Index
        return np.random.uniform(0, 100)
    
    def _get_vix_level(self) -> Optional[float]:
        """Отримуємо VIX рandвень"""
        
        # Симуляцandя отримання VIX
        return np.random.uniform(10, 40)
    
    def _analyze_sector_performance(self, events: List[MarketEvent]) -> Dict[str, float]:
        """Аналandwithуємо секторну продуктивнandсть"""
        
        sector_performance = {}
        
        for sector in self.sectors:
            # Симуляцandя роwithрахунку продуктивностand сектору
            sector_events = [e for e in events if sector in e.affected_sectors]
            
            if sector_events:
                avg_sentiment = np.mean([e.sentiment_score for e in sector_events])
                sector_performance[sector] = avg_sentiment
            else:
                sector_performance[sector] = 0.0
        
        return sector_performance
    
    def _get_macro_indicators(self) -> Dict[str, float]:
        """Отримуємо макроandндикатори"""
        
        # Симуляцandя макроandндикаторandв
        return {
            "gdp_growth": np.random.uniform(-2, 5),
            "inflation_rate": np.random.uniform(1, 6),
            "unemployment_rate": np.random.uniform(3, 10),
            "interest_rate": np.random.uniform(0, 5),
            "consumer_confidence": np.random.uniform(50, 150)
        }
    
    def _identify_risk_factors(self, events: List[MarketEvent]) -> List[str]:
        """Виявляємо фактори риwithику"""
        
        risk_factors = []
        
        for event in events:
            if event.impact_level in [EventImpact.HIGH, EventImpact.CRITICAL]:
                if event.sentiment_score < -0.3:
                    risk_factors.append(f"High impact negative event: {event.title}")
        
        # Додаємо forгальнand фактори риwithику
        if len([e for e in events if e.event_type == EventType.GEOPOLITICAL]) > 2:
            risk_factors.append("Elevated geopolitical tensions")
        
        if len([e for e in events if e.event_type == EventType.REGULATORY]) > 1:
            risk_factors.append("Regulatory changes detected")
        
        return risk_factors
    
    def _identify_opportunities(self, events: List[MarketEvent]) -> List[str]:
        """Виявлямо можливостand"""
        
        opportunities = []
        
        for event in events:
            if event.impact_level in [EventImpact.HIGH, EventImpact.CRITICAL]:
                if event.sentiment_score > 0.3:
                    opportunities.append(f"Positive high impact event: {event.title}")
        
        # Додаємо forгальнand можливостand
        if len([e for e in events if e.event_type == EventType.ECONOMIC_RELEASE]) > 0:
            opportunities.append("Economic data releases present trading opportunities")
        
        return opportunities
    
    def _get_latest_context(self) -> Optional[MarketContext]:
        """Отримуємо осandннandй контекст"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM market_context 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        if row:
            return self._row_to_market_context(row)
        
        return None
    
    def _get_ticker_events(self, ticker: str, hours: int = 48) -> List[MarketEvent]:
        """Отримуємо подandї for тandкера"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM market_events 
            WHERE timestamp >= datetime('now', '-{} hours')
            AND (affected_tickers LIKE ? OR affected_sectors LIKE ?)
            ORDER BY relevance_score DESC, timestamp DESC
        """.format(hours), (f'%{ticker}%', f'%{ticker}%'))
        
        events = []
        for row in cursor.fetchall():
            event = self._row_to_market_event(row)
            events.append(event)
        
        return events
    
    def _analyze_event_impact(self, ticker: str, events: List[MarketEvent]) -> Dict[str, float]:
        """Аналandwithуємо вплив подandй на тandкер"""
        
        if not events:
            return {"overall_impact": 0.0, "sentiment_impact": 0.0, "volatility_impact": 0.0}
        
        # Calculating forгальний вплив
        overall_impact = np.mean([
            e.relevance_score * e.confidence * (1 if e.sentiment_score > 0 else -1)
            for e in events
        ])
        
        # Вплив на настрої
        sentiment_impact = np.mean([e.sentiment_score * e.relevance_score for e in events])
        
        # Вплив на волатильнandсть
        volatility_impact = np.mean([
            e.relevance_score * (1 if e.impact_level in [EventImpact.HIGH, EventImpact.CRITICAL] else 0.5)
            for e in events
        ])
        
        return {
            "overall_impact": overall_impact,
            "sentiment_impact": sentiment_impact,
            "volatility_impact": volatility_impact
        }
    
    def _generate_contextual_recommendations(self, ticker: str, context: MarketContext,
                                           events: List[MarketEvent], impact: Dict[str, float]) -> List[str]:
        """Геnotруємо контекстуальнand рекомендацandї"""
        
        recommendations = []
        
        # Рекомендацandї на основand ринкового режиму
        if context.market_regime == MarketRegime.BULL_MARKET:
            recommendations.append("Consider increasing exposure in bull market conditions")
        elif context.market_regime == MarketRegime.BEAR_MARKET:
            recommendations.append("Reduce position sizes in bear market conditions")
        elif context.market_regime == MarketRegime.VOLATILE:
            recommendations.append("Use tighter stop-losses in volatile conditions")
        
        # Рекомендацandї на основand настроїв
        if impact["sentiment_impact"] > 0.3:
            recommendations.append("Positive sentiment suggests potential upside")
        elif impact["sentiment_impact"] < -0.3:
            recommendations.append("Negative sentiment indicates caution advised")
        
        # Рекомендацandї на основand волатильностand
        if impact["volatility_impact"] > 0.7:
            recommendations.append("High volatility detected - consider smaller positions")
        
        # Рекомендацandї на основand подandй
        high_impact_events = [e for e in events if e.impact_level in [EventImpact.HIGH, EventImpact.CRITICAL]]
        if high_impact_events:
            recommendations.append(f"Monitor {len(high_impact_events)} high-impact events closely")
        
        return recommendations
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Аналandwithуємо настрandй тексту"""
        
        # Просand евристика for аналandwithу настроїв
        positive_words = ["good", "great", "excellent", "positive", "bullish", "growth", "profit"]
        negative_words = ["bad", "terrible", "negative", "bearish", "decline", "loss", "risk"]
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return {"score": 0.0, "confidence": 0.0}
        
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        confidence = min(1.0, total_sentiment_words / 10)  # Чим бandльше слandв, тим впевnotнandше
        
        return {"score": sentiment_score, "confidence": confidence}
    
    def _get_ticker_sectors(self, ticker: str) -> List[str]:
        """Отримуємо сектори for тandкера"""
        
        # Симуляцandя виvalues секторandв
        sector_mapping = {
            "AAPL": ["technology"],
            "MSFT": ["technology"],
            "JPM": ["finance"],
            "JNJ": ["healthcare"],
            "XOM": ["energy"]
        }
        
        return sector_mapping.get(ticker, ["technology"])  # За forмовчуванням technology
    
    def _row_to_market_event(self, row) -> MarketEvent:
        """Конвертуємо рядок баwithи data в MarketEvent"""
        
        return MarketEvent(
            id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            event_type=EventType(row[2]),
            title=row[3],
            description=row[4],
            source=row[5],
            impact_level=EventImpact(row[6]),
            affected_tickers=json.loads(row[7]) if row[7] else [],
            affected_sectors=json.loads(row[8]) if row[8] else [],
            keywords=json.loads(row[9]) if row[9] else [],
            sentiment_score=row[10],
            confidence=row[11],
            relevance_score=row[12],
            expiration_time=datetime.fromisoformat(row[13]) if row[13] else None,
            processed=bool(row[14]),
            impact_assessment=json.loads(row[15]) if row[15] else {}
        )
    
    def _row_to_market_context(self, row) -> MarketContext:
        """Конвертуємо рядок баwithи data в MarketContext"""
        
        major_events_data = json.loads(row[6]) if row[6] else []
        major_events = [MarketEvent(**event_data) for event_data in major_events_data]
        
        return MarketContext(
            timestamp=datetime.fromisoformat(row[0]),
            market_regime=MarketRegime(row[1]),
            volatility_regime=row[2],
            sentiment_index=row[3],
            fear_greed_index=row[4],
            vix_level=row[5],
            major_events=major_events,
            sector_performance=json.loads(row[7]) if row[7] else {},
            macro_indicators=json.loads(row[8]) if row[8] else {},
            risk_factors=json.loads(row[9]) if row[9] else [],
            opportunities=json.loads(row[10]) if row[10] else []
        )
    
    def close(self):
        """Закриваємо with'єднання with баwithою data"""
        if self.conn:
            self.conn.close()

def main():
    """Тестування контекстної обandwithнаностand в реальному часand"""
    print(" REAL-TIME CONTEXT AWARENESS - News & Events Integration")
    print("=" * 60)
    
    context_engine = RealtimeContextAwarenessEngine()
    
    # Тестуємо сканування новин
    print(f"\n TESTING NEWS SCANNING")
    print("-" * 40)
    
    events = context_engine.scan_news_sources()
    print(f"[DATA] Scanned {len(events)} events from news sources")
    
    for event in events[:3]:
        print(f"    {event.title} ({event.impact_level.value})")
    
    # Тестуємо аналandwith контексту
    print(f"\n[SEARCH] TESTING CONTEXT ANALYSIS")
    print("-" * 40)
    
    context = context_engine.analyze_market_context()
    print(f"[UP] Market regime: {context.market_regime.value}")
    print(f"[DATA] Volatility regime: {context.volatility_regime}")
    print(f" Sentiment index: {context.sentiment_index:.3f}")
    print(f" Fear & Greed: {context.fear_greed_index:.1f}")
    print(f"[DOWN] VIX: {context.vix_level:.1f}")
    print(f"[WARN] Risk factors: {len(context.risk_factors)}")
    print(f"[TARGET] Opportunities: {len(context.opportunities)}")
    
    # Тестуємо контекстуальнand рекомендацandї
    print(f"\n[IDEA] TESTING CONTEXTUAL RECOMMENDATIONS")
    print("-" * 40)
    
    recommendations = context_engine.get_contextual_recommendations("AAPL")
    print(f"[DATA] Recommendations for AAPL:")
    for rec in recommendations.get("recommendations", [])[:3]:
        print(f"    {rec}")
    
    print(f"\n[UP] Market context:")
    ctx = recommendations.get("market_context", {})
    for key, value in ctx.items():
        print(f"    {key}: {value}")
    
    # Тестуємо оновлення настроїв
    print(f"\n TESTING SENTIMENT UPDATE")
    print("-" * 40)
    
    sentiment_result = context_engine.update_sentiment_analysis(
        "AAPL",
        ["Great earnings report, positive outlook", "Strong growth expected", "Market bullish on tech"]
    )
    print(f"[DATA] Sentiment analysis for AAPL:")
    print(f"    Average sentiment: {sentiment_result['average_sentiment']:.3f}")
    print(f"    Sample count: {sentiment_result['sample_count']}")
    
    # Тестуємо аналandwith ефективностand
    print(f"\n[DATA] TESTING EFFECTIVENESS ANALYSIS")
    print("-" * 40)
    
    effectiveness = context_engine.analyze_context_effectiveness()
    
    print(f"[UP] Event statistics by type:")
    for type_data in effectiveness['by_event_type']:
        print(f"    {type_data['type']}: {type_data['total_events']} events")
    
    print(f"\n Source effectiveness:")
    for source_data in effectiveness['by_source']:
        print(f"    {source_data['source']}: {source_data['events_count']} events")
    
    # Тестуємо andсторandю подandй
    print(f"\n TESTING EVENT HISTORY")
    print("-" * 40)
    
    history = context_engine.get_event_history(limit=5)
    print(f" Recent events: {len(history)}")
    
    for event in history[:3]:
        print(f"    {event.title}: {event.impact_level.value}")
    
    context_engine.close()
    print(f"\n[OK] Real-time Context Awareness test completed!")

if __name__ == "__main__":
    main()
