"""
NEWS TICKER DETECTOR
NLP for виявлення релевантних тandкерandв у новинах
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pandas as pd

logger = logging.getLogger(__name__)

class NewsTickerDetector:
    """
    NLP whereтектор for виявлення тandкерandв у новинах
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        
        # Словник компанandй and them тandкерandв
        self.company_tickers = {
            # Tech Giants
            'apple': 'AAPL',
            'microsoft': 'MSFT', 
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'amazon': 'AMZN',
            'meta': 'META',
            'facebook': 'META',
            
            # Finance
            'jpmorgan': 'JPM',
            'bank of america': 'BAC',
            'wells fargo': 'WFC',
            'goldman sachs': 'GS',
            
            # Healthcare
            'johnson & johnson': 'JNJ',
            'pfizer': 'PFE',
            'unitedhealth': 'UNH',
            
            # Energy
            'exxon': 'XOM',
            'exxonmobil': 'XOM',
            'chevron': 'CVX',
            
            # Consumer
            'procter & gamble': 'PG',
            'coca-cola': 'KO',
            'walmart': 'WMT',
            'home depot': 'HD',
            
            # Industrial
            'general electric': 'GE',
            '3m': 'MMM',
            'caterpillar': 'CAT',
            
            # ETFs
            'spdr': 'SPY',
            'ishares': 'IWM',
            'vanguard': 'VTI',
            'gld': 'GLD',
            'tlts': 'TLT'
        }
        
        # Прямand тandкери
        self.direct_tickers = {
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'BAC', 'WFC', 'GS',
            'JNJ', 'PFE', 'UNH', 'XOM', 'CVX', 'PG', 'KO', 'WMT', 'HD', 'GE', 'MMM',
            'CAT', 'SPY', 'QQQ', 'NVDA', 'TSLA', 'DIA', 'IWM', 'VTI', 'GLD', 'TLT'
        }
        
        # Keywords for фandнансових новин
        self.financial_keywords = {
            'market', 'stock', 'trade', 'trading', 'investment', 'investor',
            'share', 'equity', 'portfolio', 'dividend', 'earnings', 'revenue',
            'profit', 'loss', 'bull', 'bear', 'rally', 'crash', 'volatility'
        }
        
        # Порог релевантностand
        self.relevance_threshold = config.get('news_relevance_threshold', 0.7) if config else 0.7
        self.enable_nlp_detection = config.get('enable_nlp_ticker_detection', True) if config else True
        
    def extract_tickers_from_text(self, text: str) -> List[Tuple[str, float]]:
        """
        Витягти тandкери with тексту новини
        
        Args:
            text: Текст новини
            
        Returns:
            List[Tuple[str, float]]: Список (ticker, confidence_score)
        """
        if not self.enable_nlp_detection:
            return []
            
        text_lower = text.lower()
        found_tickers = []
        
        # 1. Прямий пошук тandкерandв
        for ticker in self.direct_tickers:
            if ticker in text_upper:
                # Рахуємо кandлькandсть withгадок
                count = text_upper.count(ticker)
                confidence = min(0.9, 0.5 + count * 0.1)  # Бandльше withгадок = вища довandра
                found_tickers.append((ticker, confidence))
        
        # 2. Пошук наwithв компанandй
        for company, ticker in self.company_tickers.items():
            if company in text_lower:
                # Перевandряємо чи це not частина andншого слова
                words = text_lower.split()
                for word in words:
                    if company in word and len(word) <= len(company) + 3:
                        confidence = 0.7  # Середня довandра for наwithв компанandй
                        found_tickers.append((ticker, confidence))
                        break
        
        # 3. Фandнансова релевантнandсть
        financial_score = self._calculate_financial_relevance(text_lower)
        
        # 4. Сортування for довandрою
        found_tickers.sort(key=lambda x: x[1], reverse=True)
        
        # Фandльтрацandя по порогу
        relevant_tickers = [
            (ticker, min(confidence, financial_score)) 
            for ticker, confidence in found_tickers 
            if confidence >= self.relevance_threshold
        ]
        
        return relevant_tickers
    
    def _calculate_financial_relevance(self, text: str) -> float:
        """
        Роwithрахувати фandнансову релевантнandсть тексту
        
        Args:
            text: Текст новини
            
        Returns:
            float: Score вandд 0.0 до 1.0
        """
        words = text.split()
        financial_words = sum(1 for word in words if word in self.financial_keywords)
        
        if len(words) == 0:
            return 0.0
            
        relevance = financial_words / len(words)
        return min(1.0, relevance * 5)  # Масшandбування
    
    def get_primary_ticker(self, text: str, fallback_symbol: str = 'SPY') -> str:
        """
        Отримати основний тandкер for новини
        
        Args:
            text: Текст новини
            fallback_symbol: Запасний символ
            
        Returns:
            str: Основний тandкер
        """
        tickers = self.extract_tickers_from_text(text)
        
        if tickers:
            # Поверandємо тandкер with найвищою довandрою
            return tickers[0][0]
        
        # Якщо тandкери not withнайwhereно, перевandряємо фandнансову релевантнandсть
        financial_score = self._calculate_financial_relevance(text.lower())
        
        if financial_score >= self.relevance_threshold:
            # Фandнансово релевантна новина беwith конкретного тandкера
            return fallback_symbol
        
        # Не фandнансова новина
        return None
    
    def analyze_news_batch(self, news_data: List[Dict]) -> List[Dict]:
        """
        Проаналandwithувати пакет новин and add тandкери
        
        Args:
            news_data: Список новин
            
        Returns:
            List[Dict]: Новини with доданими тandкерами
        """
        analyzed_news = []
        
        for news in news_data:
            text = news.get('title', '') + ' ' + news.get('content', '')
            
            # Витягуємо тandкери
            tickers = self.extract_tickers_from_text(text)
            primary_ticker = self.get_primary_ticker(text)
            
            # Оновлюємо новину
            enhanced_news = news.copy()
            enhanced_news['detected_tickers'] = [ticker for ticker, _ in tickers]
            enhanced_news['ticker_confidence'] = {ticker: confidence for ticker, confidence in tickers}
            enhanced_news['primary_ticker'] = primary_ticker
            enhanced_news['financial_relevance'] = self._calculate_financial_relevance(text.lower())
            
            analyzed_news.append(enhanced_news)
        
        return analyzed_news
    
    def get_ticker_distribution(self, news_data: List[Dict]) -> Dict[str, int]:
        """
        Отримати роwithподandл тandкерandв у новинах
        
        Args:
            news_data: Список новин
            
        Returns:
            Dict[str, int]: Кandлькandсть новин по тandкерах
        """
        ticker_counts = defaultdict(int)
        
        for news in news_data:
            primary_ticker = news.get('primary_ticker')
            if primary_ticker:
                ticker_counts[primary_ticker] += 1
        
        return dict(ticker_counts)
    
    def filter_relevant_news(self, news_data: List[Dict], 
                           min_relevance: float = None) -> List[Dict]:
        """
        Вandдфandльтрувати релевантнand новини
        
        Args:
            news_data: Список новин
            min_relevance: Мandнandмальна релевантнandсть
            
        Returns:
            List[Dict]: Релевантнand новини
        """
        if min_relevance is None:
            min_relevance = self.relevance_threshold
        
        relevant_news = []
        
        for news in news_data:
            financial_relevance = news.get('financial_relevance', 0.0)
            has_ticker = news.get('primary_ticker') is not None
            
            if financial_relevance >= min_relevance or has_ticker:
                relevant_news.append(news)
        
        return relevant_news


def create_news_ticker_detector(config: Dict = None) -> NewsTickerDetector:
    """
    Factory function for створення whereтектора
    
    Args:
        config: Конфandгурацandя
        
    Returns:
        NewsTickerDetector: Екwithемпляр whereтектора
    """
    return NewsTickerDetector(config)
