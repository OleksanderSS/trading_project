#!/usr/bin/env python3
"""
Real-time Context Awareness - News & Events Integration

This module implements the system's real-time context and regime detection engine.
It monitors news, events, and market context to provide situational awareness.
"""

import json
import sqlite3
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import requests
from bs4 import BeautifulSoup

from src.core.logging.logger import ProjectLogger
from src.meta_learning.base import BaseMetaComponent

class EventType(Enum):
    """Event types"""
    ECONOMIC_RELEASE = "economic_release"
    CORPORATE_NEWS = "corporate_news"
    MARKET_EVENT = "market_event"
    GEOPOLITICAL = "geopolitical"
    REGULATORY = "regulatory"
    WEATHER = "weather"
    SOCIAL_SENTIMENT = "social_sentiment"

class EventImpact(Enum):
    """Event impact level"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MarketRegime(Enum):
    """Market regimes"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRISIS = "crisis"

@dataclass
class MarketEvent:
    """Market event"""
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
    """Market context"""
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

class ContextAwarenessEngine(BaseMetaComponent):
    """
    Real-time Context Awareness Engine
    Implements monitoring of news, events, and market context.
    Inherits from BaseMetaComponent.
    """
    
    def __init__(self, db_path: str = "realtime_context_awareness.db"):
        self.logger = ProjectLogger.get_logger("ContextAwarenessEngine")
        self.db_path = db_path
        self.conn = None
        
        # Configuration of data sources
        self.news_sources = {
            "financial_times": "https://www.ft.com",
            "reuters": "https://www.reuters.com",
            "bloomberg": "https://www.bloomberg.com",
            "yahoo_finance": "https://finance.yahoo.com"
        }
        
        # Keywords for filtering
        self.keywords = {
            "economic": ["gdp", "inflation", "interest rates", "employment", "fed", "ecb"],
            "market": ["bull", "bear", "crash", "rally", "correction", "volatility"],
            "corporate": ["earnings", "merger", "acquisition", "bankruptcy", "ipo"],
            "geopolitical": ["war", "sanctions", "election", "trade", "tensions"]
        }
        
        # Sectors for tracking
        self.sectors = [
            "technology", "healthcare", "finance", "energy", 
            "consumer", "industrial", "materials", "utilities"
        ]
        
        self._initialize_database()

    @property
    def name(self) -> str:
        """Unique identifier for the component."""
        return "context_awareness"

    def update(self, data: Any = None) -> None:
        """
        Updates the engine by scanning news and analyzing market context.
        If data is provided (e.g., specific news items), it could be processed directly.
        """
        self.logger.info("Triggering context awareness update...")
        try:
            self.scan_news_sources()
            self.analyze_market_context()
            self.logger.info("Context awareness update completed.")
        except Exception as e:
            self.logger.error(f"Failed to update context awareness: {e}")

    def get_state(self) -> Dict[str, Any]:
        """Returns the current market regime and sentiment index."""
        context = self._get_latest_context()
        if context:
            return {
                "market_regime": context.market_regime.value,
                "sentiment_index": context.sentiment_index,
                "volatility_regime": context.volatility_regime,
                "timestamp": context.timestamp.isoformat()
            }
        return {"status": "no_context_available"}
    
    def _initialize_database(self):
        """Initialize the database"""
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Use dictionary-like row factory for robustness
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Events table
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
        
        # Market context table
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
        
        # News sources table
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
        
        # Sentiment table
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
        
        # Initialize news sources
        self._initialize_news_sources()
        
        self.conn.commit()
    
    def _initialize_news_sources(self):
        """Initialize news sources"""
        
        cursor = self.conn.cursor()
        
        for source_name, source_url in self.news_sources.items():
            cursor.execute("""
                INSERT OR IGNORE INTO news_sources (name, url, active, reliability_score)
                VALUES (?, ?, ?, ?)
            """, (source_name, source_url, True, 0.8))
        
        self.conn.commit()
    
    def scan_news_sources(self) -> List[MarketEvent]:
        """Scan news sources"""
        
        events = []
        
        for source_name, source_url in self.news_sources.items():
            try:
                source_events = self._fetch_news_from_source(source_name)
                events.extend(source_events)
                self.logger.info(f" Fetched {len(source_events)} events from {source_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to fetch from {source_name}: {e}")
        
        # Save events
        for event in events:
            self._save_market_event(event)
        
        self.logger.info(f"Total events scanned: {len(events)}")
        return events
    
    def analyze_market_context(self) -> MarketContext:
        """Analyze the current market context"""
        
        # Get recent events
        recent_events = self._get_recent_events(hours=24)
        
        # Detect market regime
        market_regime = self._detect_market_regime(recent_events)
        
        # Analyze volatility
        volatility_regime = self._detect_volatility_regime(recent_events)
        
        # Calculate sentiment index
        sentiment_index = self._calculate_sentiment_index(recent_events)
        
        # Get Fear & Greed Index
        fear_greed_index = self._get_fear_greed_index() or 50.0  # Default neutral value
        
        # Get VIX level
        vix_level = self._get_vix_level() or 20.0  # Default normal VIX level
        
        # Analyze sector performance
        sector_performance = self._analyze_sector_performance(recent_events)
        
        # Get macro indicators
        macro_indicators = self._get_macro_indicators()
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(recent_events)
        
        # Identify opportunities
        opportunities = self._identify_opportunities(recent_events)
        
        # Create context
        context = MarketContext(
            timestamp=datetime.now(),
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            sentiment_index=sentiment_index,
            fear_greed_index=fear_greed_index,
            vix_level=vix_level,
            major_events=recent_events[:10],  # Top 10 events
            sector_performance=sector_performance,
            macro_indicators=macro_indicators,
            risk_factors=risk_factors,
            opportunities=opportunities
        )
        
        # Save context
        self._save_market_context(context)
        
        return context
    
    def get_contextual_recommendations(self, ticker: str) -> Dict[str, Any]:
        """Get contextual recommendations for a ticker"""
        
        # Get current context
        context = self._get_latest_context()
        if not context:
            return {"error": "No context available"}
        
        # Get relevant events
        relevant_events = self._get_ticker_events(ticker, hours=48)
        
        # Analyze event impact
        event_impact = self._analyze_event_impact(ticker, relevant_events)
        
        # Generate recommendations
        recommendations = self._generate_contextual_recommendations(
            context, relevant_events, event_impact
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
        """Update sentiment analysis"""
        
        cursor = self.conn.cursor()
        
        sentiment_scores = []
        
        for text in text_data:
            sentiment = self._analyze_text_sentiment(text)
            
            # Save result
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
                text[:200]  # Save only the first 200 characters
            ))
            
            sentiment_scores.append(sentiment["score"])
        
        self.conn.commit()
        
        # Calculate average sentiment
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        
        self.logger.info(f"Updated sentiment for {ticker}: {avg_sentiment:.3f}")
        
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
        """Get event history"""
        
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
        """Analyze the effectiveness of context awareness"""
        
        cursor = self.conn.cursor()
        
        # Event statistics
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
        
        # Source effectiveness
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
        
        # Sentiment trends
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
    
    def _fetch_news_from_source(self, source_name: str) -> List[MarketEvent]:
        """
        Fetch news from a source.
        TODO: Integrate with real news collectors.
        """
        self.logger.warning(f"Real news fetching not yet implemented for {source_name}. Returning empty list.")
        return []
    
    def _save_market_event(self, event: MarketEvent):
        """Save a market event"""
        
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
        """Save the market context"""
        
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
            json.dumps([asdict(event) for event in context.major_events], default=str),
            json.dumps(context.sector_performance),
            json.dumps(context.macro_indicators),
            json.dumps(context.risk_factors),
            json.dumps(context.opportunities)
        ))
        
        self.conn.commit()
    
    def _get_recent_events(self, hours: int = 24) -> List[MarketEvent]:
        """Get recent events"""
        
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
        """Detect the market regime"""
        
        if not events:
            return MarketRegime.SIDEWAYS
        
        # Analyze event sentiment
        positive_events = sum(1 for e in events if e.sentiment_score > 0.2)
        negative_events = sum(1 for e in events if e.sentiment_score < -0.2)
        
        # Analyze impact level
        high_impact_events = sum(1 for e in events if e.impact_level in [EventImpact.HIGH, EventImpact.CRITICAL])
        
        # Detect regime
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
        """Detect the volatility regime"""
        
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
        """Calculate the sentiment index"""
        
        if not events:
            return 0.0
        
        # Weight sentiment by relevance
        weighted_sentiment = sum(
            e.sentiment_score * e.relevance_score * e.confidence 
            for e in events
        )
        total_weight = sum(e.relevance_score * e.confidence for e in events)
        
        return weighted_sentiment / total_weight if total_weight > 0 else 0.0
    
    def _get_fear_greed_index(self) -> Optional[float]:
        """Get the Fear & Greed Index from external source."""
        try:
            # Implement basic Fear & Greed Index fetching
            # Using CNN's Fear & Greed Index API endpoint
            import requests
            response = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and "fear_and_greed" in data and "history" in data["fear_and_greed"]:
                    latest = data["fear_and_greed"]["history"][-1]
                    return float(latest.get("value", 50.0))
        except Exception as e:
            logger.debug(f"Failed to fetch Fear & Greed Index: {e}")
        
        # Return neutral value if fetching fails
        return 50.0
    
    def _get_vix_level(self) -> Optional[float]:
        """Get the VIX level from market data provider."""
        try:
            # Implement basic VIX level fetching using yfinance
            import yfinance as yf
            vix_data = yf.download("^VIX", period="1d", interval="1d", progress=False)
            if not vix_data.empty:
                return float(vix_data["Close"].iloc[-1])
        except Exception as e:
            logger.debug(f"Failed to fetch VIX level: {e}")
        
        # Return default VIX level if fetching fails
        return 20.0
    
    def _analyze_sector_performance(self, events: List[MarketEvent]) -> Dict[str, float]:
        """Analyze sector performance"""
        
        sector_performance = {}
        
        for sector in self.sectors:
            sector_events = [e for e in events if sector in e.affected_sectors]
            
            if sector_events:
                avg_sentiment = np.mean([e.sentiment_score for e in sector_events])
                sector_performance[sector] = avg_sentiment
            else:
                sector_performance[sector] = 0.0
        
        return sector_performance
    
    def _get_macro_indicators(self) -> Dict[str, float]:
        """Get macro indicators from FRED collector (TODO)"""
        return {
            "gdp_growth": 0.0,
            "inflation_rate": 0.0,
            "unemployment_rate": 0.0,
            "interest_rate": 0.0,
            "consumer_confidence": 0.0
        }
    
    def _identify_risk_factors(self, events: List[MarketEvent]) -> List[str]:
        """Identify risk factors"""
        
        risk_factors = []
        
        for event in events:
            if event.impact_level in [EventImpact.HIGH, EventImpact.CRITICAL]:
                if event.sentiment_score < -0.3:
                    risk_factors.append(f"High impact negative event: {event.title}")
        
        # Add general risk factors
        if len([e for e in events if e.event_type == EventType.GEOPOLITICAL]) > 2:
            risk_factors.append("Elevated geopolitical tensions")
        
        if len([e for e in events if e.event_type == EventType.REGULATORY]) > 1:
            risk_factors.append("Regulatory changes detected")
        
        return risk_factors
    
    def _identify_opportunities(self, events: List[MarketEvent]) -> List[str]:
        """Identify opportunities"""
        
        opportunities = []
        
        for event in events:
            if event.impact_level in [EventImpact.HIGH, EventImpact.CRITICAL]:
                if event.sentiment_score > 0.3:
                    opportunities.append(f"Positive high impact event: {event.title}")
        
        # Add general opportunities
        if len([e for e in events if e.event_type == EventType.ECONOMIC_RELEASE]) > 0:
            opportunities.append("Economic data releases present trading opportunities")
        
        return opportunities
    
    def _get_latest_context(self) -> Optional[MarketContext]:
        """Get the latest context"""
        
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
        """Get events for a ticker"""
        
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
        """Analyze the impact of events on a ticker"""
        
        if not events:
            return {"overall_impact": 0.0, "sentiment_impact": 0.0, "volatility_impact": 0.0}
        
        # Calculate overall impact
        overall_impact = np.mean([
            e.relevance_score * e.confidence * (1 if e.sentiment_score > 0 else -1)
            for e in events
        ])
        
        # Impact on sentiment
        sentiment_impact = np.mean([e.sentiment_score * e.relevance_score for e in events])
        
        # Impact on volatility
        volatility_impact = np.mean([
            e.relevance_score * (1 if e.impact_level in [EventImpact.HIGH, EventImpact.CRITICAL] else 0.5)
            for e in events
        ])
        
        return {
            "overall_impact": overall_impact,
            "sentiment_impact": sentiment_impact,
            "volatility_impact": volatility_impact
        }
    
    def _generate_contextual_recommendations(self, context: MarketContext,
                                           events: List[MarketEvent], impact: Dict[str, float]) -> List[str]:
        """Generate contextual recommendations"""
        
        recommendations = []
        
        # Recommendations based on market regime
        recommendations.extend(self._get_regime_recommendations(context.market_regime))
        
        # Recommendations based on sentiment
        recommendations.extend(self._get_sentiment_recommendations(impact["sentiment_impact"]))
        
        # Recommendations based on volatility
        recommendations.extend(self._get_volatility_recommendations(impact["volatility_impact"]))
        
        # Recommendations based on events
        recommendations.extend(self._get_event_recommendations(events))
        
        return recommendations

    def _get_regime_recommendations(self, regime: MarketRegime) -> List[str]:
        """Get recommendations based on market regime."""
        if regime == MarketRegime.BULL_MARKET:
            return ["Consider increasing exposure in bull market conditions"]
        elif regime == MarketRegime.BEAR_MARKET:
            return ["Reduce position sizes in bear market conditions"]
        elif regime == MarketRegime.VOLATILE:
            return ["Use tighter stop-losses in volatile conditions"]
        return []

    def _get_sentiment_recommendations(self, sentiment_impact: float) -> List[str]:
        """Get recommendations based on sentiment impact."""
        if sentiment_impact > 0.3:
            return ["Positive sentiment suggests potential upside"]
        elif sentiment_impact < -0.3:
            return ["Negative sentiment indicates caution advised"]
        return []

    def _get_volatility_recommendations(self, volatility_impact: float) -> List[str]:
        """Get recommendations based on volatility impact."""
        if volatility_impact > 0.7:
            return ["High volatility detected - consider smaller positions"]
        return []

    def _get_event_recommendations(self, events: List[MarketEvent]) -> List[str]:
        """Get recommendations based on events."""
        high_impact_events = [e for e in events if e.impact_level in [EventImpact.HIGH, EventImpact.CRITICAL]]
        if high_impact_events:
            return [f"Monitor {len(high_impact_events)} high-impact events closely"]
        return []
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze the sentiment of a text"""
        
        # Simple heuristics for sentiment analysis
        positive_words = ["good", "great", "excellent", "positive", "bullish", "growth", "profit"]
        negative_words = ["bad", "terrible", "negative", "bearish", "decline", "loss", "risk"]
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return {"score": 0.0, "confidence": 0.0}
        
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        confidence = min(1.0, total_sentiment_words / 10)  # The more words, the more confident
        
        return {"score": sentiment_score, "confidence": confidence}
    
    def _get_ticker_sectors(self, ticker: str) -> List[str]:
        """Get sectors for a ticker"""
        
        # Simulation of getting sectors (in production, this would be fetched from a database or API)
        sector_mapping = {
            "AAPL": ["technology"],
            "MSFT": ["technology"],
            "JPM": ["finance"],
            "JNJ": ["healthcare"],
            "XOM": ["energy"]
        }
        
        return sector_mapping.get(ticker, ["technology"])  # Default to technology
    
    def _row_to_market_event(self, row: Union[sqlite3.Row, tuple, list]) -> MarketEvent:
        """
        Convert a database row to a MarketEvent object.
        Supports both sqlite3.Row and indexed sequences.
        """
        if isinstance(row, sqlite3.Row):
            data = dict(row)
        else:
            # Fallback to indexed mapping if row factory is not used
            column_names = [
                'id', 'timestamp', 'event_type', 'title', 'description', 'source', 'impact_level',
                'affected_tickers', 'affected_sectors', 'keywords', 'sentiment_score',
                'confidence', 'relevance_score', 'expiration_time', 'processed', 'impact_assessment'
            ]
            if len(row) < len(column_names):
                 raise ValueError(f"Database row has insufficient columns: {len(row)} < {len(column_names)}")
            data = dict(zip(column_names, row))
        
        return MarketEvent(
            id=data['id'],
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            event_type=EventType(data['event_type']),
            title=data['title'],
            description=data['description'],
            source=data['source'],
            impact_level=EventImpact(data['impact_level']),
            affected_tickers=json.loads(data['affected_tickers']) if isinstance(data['affected_tickers'], str) else (data['affected_tickers'] or []),
            affected_sectors=json.loads(data['affected_sectors']) if isinstance(data['affected_sectors'], str) else (data['affected_sectors'] or []),
            keywords=json.loads(data['keywords']) if isinstance(data['keywords'], str) else (data['keywords'] or []),
            sentiment_score=data['sentiment_score'],
            confidence=data['confidence'],
            relevance_score=data['relevance_score'],
            expiration_time=datetime.fromisoformat(data['expiration_time']) if isinstance(data['expiration_time'], str) else data['expiration_time'],
            processed=bool(data['processed']),
            impact_assessment=json.loads(data['impact_assessment']) if isinstance(data['impact_assessment'], str) else (data['impact_assessment'] or {})
        )
    
    def _row_to_market_context(self, row: Union[sqlite3.Row, tuple, list]) -> MarketContext:
        """
        Convert a database row to a MarketContext object.
        Supports both sqlite3.Row and indexed sequences.
        """
        data = self._extract_row_data(row, self._get_context_columns())
        
        major_events = self._parse_major_events(data.get('major_events'))
        
        return MarketContext(
            timestamp=self._parse_datetime(data['timestamp']),
            market_regime=MarketRegime(data['market_regime']),
            volatility_regime=data['volatility_regime'],
            sentiment_index=data['sentiment_index'],
            fear_greed_index=data['fear_greed_index'],
            vix_level=data['vix_level'],
            major_events=major_events,
            sector_performance=self._parse_json_field(data, 'sector_performance', {}),
            macro_indicators=self._parse_json_field(data, 'macro_indicators', {}),
            risk_factors=self._parse_json_field(data, 'risk_factors', []),
            opportunities=self._parse_json_field(data, 'opportunities', [])
        )

    def _get_context_columns(self) -> List[str]:
        """Get column names for market context."""
        return [
            'id', 'timestamp', 'market_regime', 'volatility_regime', 'sentiment_index',
            'fear_greed_index', 'vix_level', 'major_events', 'sector_performance',
            'macro_indicators', 'risk_factors', 'opportunities'
        ]

    def _extract_row_data(self, row: Union[sqlite3.Row, tuple, list], column_names: List[str]) -> Dict[str, Any]:
        """Extract data from database row."""
        if isinstance(row, sqlite3.Row):
            return dict(row)
        else:
            if len(row) < len(column_names):
                raise ValueError(f"Database row has insufficient columns: {len(row)} < {len(column_names)}")
            return dict(zip(column_names, row))

    def _parse_major_events(self, major_events_raw) -> List[MarketEvent]:
        """Parse major events from raw data."""
        major_events_data = json.loads(major_events_raw) if isinstance(major_events_raw, str) else (major_events_raw or [])
        
        major_events = []
        for event_data in major_events_data:
            if isinstance(event_data, dict):
                processed_event = self._process_event_data(event_data)
                if processed_event:
                    major_events.append(processed_event)
        
        return major_events

    def _process_event_data(self, event_data: Dict[str, Any]) -> Optional[MarketEvent]:
        """Process individual event data."""
        try:
            event_data_copy = event_data.copy()
            
            # Convert string fields to proper types
            self._convert_event_field(event_data_copy, 'event_type', EventType)
            self._convert_event_field(event_data_copy, 'impact_level', EventImpact)
            self._convert_event_field(event_data_copy, 'timestamp', datetime)
            self._convert_event_field(event_data_copy, 'expiration_time', datetime)
            
            return MarketEvent(**event_data_copy)
        except Exception as e:
            self.logger.warning(f"Failed to process event data: {e}")
            return None

    def _convert_event_field(self, event_data: Dict[str, Any], field_name: str, target_type):
        """Convert event field to target type if it's a string."""
        if field_name in event_data and isinstance(event_data[field_name], str):
            if target_type in [EventType, EventImpact]:
                event_data[field_name] = target_type(event_data[field_name])
            elif target_type == datetime:
                event_data[field_name] = datetime.fromisoformat(event_data[field_name])

    def _parse_datetime(self, timestamp_value) -> datetime:
        """Parse datetime from string or return as-is."""
        return datetime.fromisoformat(timestamp_value) if isinstance(timestamp_value, str) else timestamp_value

    def _parse_json_field(self, data: Dict[str, Any], field_name: str, default: Any) -> Any:
        """Parse JSON field or return default."""
        field_value = data.get(field_name)
        if isinstance(field_value, str):
            try:
                return json.loads(field_value)
            except json.JSONDecodeError:
                return default
        return field_value or default
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()