"""
News Impact Classifier
Класифікує вплив новин на тікери та таймфрейми
"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("NewsImpactClassifier")


@dataclass
class NewsImpact:
    """Дані про вплив новини"""
    impact_type: str  # market_wide, sector_specific, ticker_specific
    affected_tickers: list[str]  # Список уражених тікерів
    affected_sectors: list[str]  # Список уражених секторів
    timeframes: list[str]  # Релевантні таймфрейми
    impact_strength: str  # high, medium, low
    priority: str  # immediate, normal, low
    retention_hours: int  # Скільки годин вплив актуальний
    description: str  # Опис типу новини


class NewsImpactClassifier:
    """
    Класифікатор впливу новин на ринок
    """

    def __init__(self, config_manager: UnifiedConfigManager):
        self.config_manager = config_manager
        self.impact_config = self._load_impact_config()
        self.sector_mapping = self._load_sector_mapping()

        logger.info(f"NewsImpactClassifier initialized with {len(self.impact_config)} impact types")

    def _load_impact_config(self) -> dict:
        """Завантажити конфігурацію впливу новин"""
        try:
            config_path = Path(__file__).parent.parent / "config" / "news_impact_classification.yaml"
            with open(config_path, encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config['news_impact_classification']
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Failed to load impact config: {e}")
            raise RuntimeError("Failed to load news impact classification config") from e

    def _load_sector_mapping(self) -> dict[str, list[str]]:
        """Завантажити маппінг секторів та тікерів"""
        try:
            assets_config = self.config_manager.get_config('assets', default={})
            sectors = assets_config.get('sectors', {})

            sector_mapping = {}
            for sector_name, sector_data in sectors.items():
                if isinstance(sector_data, dict) and 'assets' in sector_data:
                    sector_mapping[sector_name] = sector_data['assets']

            logger.info(f"Loaded {len(sector_mapping)} sectors with ticker mappings")
            return sector_mapping
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Failed to load sector mapping: {e}")
            raise RuntimeError("Failed to load news impact sector mapping") from e

    def classify_impact(self, news_text: str, news_type: str = "general") -> NewsImpact:
        """
        Класифікувати вплив новини

        Args:
            news_text: Текст новини (title + content)
            news_type: Тип новини (general, company_specific, etc.)

        Returns:
            NewsImpact: Об'єкт з інформацією про вплив
        """
        # Нормалізація тексту
        normalized_text = self._normalize_text(news_text)

        # Знайти відповідний тип впливу
        impact_type = self._find_impact_type(normalized_text, news_type)

        # Отримати конфігурацію для цього типу
        impact_config = self.impact_config.get(impact_type, {})

        # Визначити уражені тікери
        affected_tickers = self._get_affected_tickers(normalized_text, impact_config)

        # Визначити уражені сектори
        affected_sectors = self._get_affected_sectors(affected_tickers, impact_config)

        # Отримати релевантні таймфрейми
        timeframes = impact_config.get('timeframes', ['1d'])

        # Отримати силу впливу
        impact_strength = impact_config.get('impact_strength', 'medium')

        # Отримати пріоритет та час актуальності
        priority, retention_hours = self._get_priority_and_retention(impact_strength)

        # Опис
        description = impact_config.get('description', impact_type)

        return NewsImpact(
            impact_type=impact_type,
            affected_tickers=affected_tickers,
            affected_sectors=affected_sectors,
            timeframes=timeframes,
            impact_strength=impact_strength,
            priority=priority,
            retention_hours=retention_hours,
            description=description
        )

    def _normalize_text(self, text: str) -> str:
        """Нормалізація тексту для аналізу"""
        # Конвертація в нижній регістр
        text = text.lower()

        # Видалення спеціальних символів
        text = re.sub(r'[^\w\s]', ' ', text)

        # Видалення зайвих пробілів
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _find_impact_type(self, normalized_text: str, news_type: str) -> str:
        """Знайти тип впливу на основі тексту та типу новини"""

        # Спочатку перевіряємо специфічні ключові слова
        for impact_type, config in self.impact_config.items():
            keywords = config.get('keywords', [])

            # Перевіряємо чи є ключові слова в тексті
            for keyword in keywords:
                if keyword.lower() in normalized_text:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Found impact type: {impact_type} (keyword: {keyword})")
                    return impact_type

        # Якщо не знайдено ключових слів, використовуємо тип новини
        if news_type == "company_specific":
            return "company_specific"
        elif news_type == "general":
            return "market_wide"
        else:
            return "market_wide"  # Default

    def _get_affected_tickers(self, normalized_text: str, impact_config: dict) -> list[str]:
        """Визначити уражені тікери"""
        affected_tickers = impact_config.get('affected_tickers', 'all')

        if affected_tickers == "all":
            # Повертаємо всі тікери з конфігурації
            all_tickers = []
            for sector_tickers in self.sector_mapping.values():
                all_tickers.extend(sector_tickers)
            return list(set(all_tickers))  # Унікальні значення

        elif affected_tickers == "dynamic":
            # Динамічно визначаємо тікери з тексту
            return self._extract_tickers_from_text(normalized_text)

        else:
            # Повертаємо список з конфігурації
            return affected_tickers

    def _extract_tickers_from_text(self, normalized_text: str) -> list[str]:
        """Витягти тікери з тексту новини"""
        found_tickers = []

        # Маппінг назв компаній до тікерів
        company_to_ticker = {
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'amazon': 'AMZN',
            'nvidia': 'NVDA',
            'tesla': 'TSLA',
            'meta': 'META',
            'facebook': 'META',
            'amd': 'AMD',
            'intel': 'INTC',
            'tsmc': 'TSM',
            'jpmorgan': 'JPM',
            'bank of america': 'BAC',
            'goldman sachs': 'GS',
            'coca-cola': 'KO',
            'walmart': 'WMT',
            'exxonmobil': 'XOM',
            'chevron': 'CVX'
        }

        # Шукаємо назви компаній в тексті
        for company, ticker in company_to_ticker.items():
            if company in normalized_text:
                found_tickers.append(ticker)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Found ticker {ticker} from company name '{company}'")

        # Шукаємо прямі тікери
        ticker_pattern = r'\b[A-Z]{1,4}\b'
        potential_tickers = re.findall(ticker_pattern, normalized_text.upper())

        # Фільтруємо тільки відомі тікери
        all_known_tickers = []
        for sector_tickers in self.sector_mapping.values():
            all_known_tickers.extend(sector_tickers)

        for ticker in potential_tickers:
            if ticker in all_known_tickers and ticker not in found_tickers:
                found_tickers.append(ticker)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Found ticker {ticker} directly in text")

        return found_tickers

    def _get_affected_sectors(self, affected_tickers: list[str], impact_config: dict) -> list[str]:
        """Визначити уражені сектори"""
        affected_sectors = impact_config.get('affected_sectors', 'all')

        if affected_sectors == "all":
            return list(self.sector_mapping.keys())

        elif affected_sectors == "dynamic":
            # Визначаємо сектори на основі тікерів
            sectors = set()
            for sector, tickers in self.sector_mapping.items():
                if any(ticker in affected_tickers for ticker in tickers):
                    sectors.add(sector)
            return list(sectors)

        else:
            return affected_sectors

    def _get_priority_and_retention(self, impact_strength: str) -> tuple[str, int]:
        """Отримати пріоритет та час актуальності"""
        mapping = {
            'high': ('immediate', 48),
            'medium': ('normal', 24),
            'low': ('low', 12)
        }
        return mapping.get(impact_strength, ('normal', 24))

    def get_relevant_combinations(self, news_impact: NewsImpact) -> list[tuple[str, str]]:
        """
        Отримати релевантні комбінації тікер-таймфрейм

        Returns:
            List[Tuple[ticker, timeframe]]: Список релевантних комбінацій
        """
        combinations = []

        for ticker in news_impact.affected_tickers:
            for timeframe in news_impact.timeframes:
                combinations.append((ticker, timeframe))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Generated {len(combinations)} relevant combinations for {news_impact.impact_type}")
        return combinations

    def log_impact_analysis(self, news_text: str, news_impact: NewsImpact):
        """Залогувати аналіз впливу"""
        logger.info("📰 News Impact Analysis:")
        logger.info(f"   Type: {news_impact.impact_type}")
        logger.info(f"   Tickers: {len(news_impact.affected_tickers)} -> {news_impact.affected_tickers[:5]}...")
        logger.info(f"   Sectors: {len(news_impact.affected_sectors)} -> {news_impact.affected_sectors[:3]}...")
        logger.info(f"   Timeframes: {news_impact.timeframes}")
        logger.info(f"   Strength: {news_impact.impact_strength}")
        logger.info(f"   Priority: {news_impact.priority}")
        logger.info(f"   Combinations: {len(self.get_relevant_combinations(news_impact))}")
