#!/usr/bin/env python3
"""
Enhanced Sector-Based Ticker Configuration
Оптимізовані тікери по секторах для максимальної волатильності та прибутковості
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class SectorConfig:
    """Конфігурація сектору"""
    name: str
    tickers: List[str]
    volatility_score: float  # 1-10, 10 = найвища волатильність
    profit_potential: float  # 1-10, 10 = найвищий потенціал
    risk_level: str  # low, medium, high, extreme
    correlation_with_market: float  # 0-1, кореляція з ринком
    recommended_position_size: float  # 0.05-0.2, розмір позиції
    optimal_timeframes: List[str]  # оптимальні таймфрейми


class EnhancedSectorTickerManager:
    """Менеджер тікерів по секторах з оптимізацією під волатильність"""
    
    def __init__(self):
        self.sectors = self._create_enhanced_sectors()
        self.volatility_ranking = self._create_volatility_ranking()
        self.profitability_ranking = self._create_profitability_ranking()
    
    def _create_enhanced_sectors(self) -> Dict[str, SectorConfig]:
        """Створити оптимізовані сектори"""
        return {
            # Екстремальна волатильність - максимальні можливості
            "extreme_volatility": SectorConfig(
                name="Extreme Volatility Tech",
                tickers=['TSLA', 'NVDA', 'AMD', 'COIN', 'MARA', 'RIOT', 'PLTR', 'GME', 'SNAP', 'ROKU'],
                volatility_score=9.5,
                profit_potential=9.0,
                risk_level="extreme",
                correlation_with_market=0.7,
                recommended_position_size=0.05,
                optimal_timeframes=['15m', '1h', '4h']
            ),
            
            # AI та Big Tech - висока волатильність з сильними трендами
            "ai_big_tech": SectorConfig(
                name="AI & Big Tech",
                tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'CRM', 'ADBE'],
                volatility_score=7.5,
                profit_potential=8.5,
                risk_level="high",
                correlation_with_market=0.8,
                recommended_position_size=0.08,
                optimal_timeframes=['15m', '1h', '1d']  # Повертаємо 15m (60 днів)
            ),
            
            # Напівпровідники - циклічна волатильність
            "semiconductors": SectorConfig(
                name="Semiconductors",
                tickers=['INTC', 'MU', 'SOXX', 'SMH', 'TSM', 'ASML'],
                volatility_score=8.0,
                profit_potential=8.0,
                risk_level="high",
                correlation_with_market=0.75,
                recommended_position_size=0.07,
                optimal_timeframes=['15m', '1h', '1d']  # Повертаємо 15m (60 днів)
            ),
            
            # Крипто-пов'язані - екстремальна волатильність
            "crypto_related": SectorConfig(
                name="Crypto-Related",
                tickers=['SQ', 'PYPL', 'MSTR', 'BKNG', 'EBAY'],
                volatility_score=8.5,
                profit_potential=8.5,
                risk_level="extreme",
                correlation_with_market=0.6,
                recommended_position_size=0.06,
                optimal_timeframes=['15m', '1h', '1d']  # Повертаємо 15m (60 днів)
            ),
            
            # Енергія - висока волатильність залежно від новин
            "energy": SectorConfig(
                name="Energy Sector",
                tickers=['XOM', 'CVX', 'CLF', 'HAL', 'SLB', 'COP'],
                volatility_score=7.0,
                profit_potential=7.5,
                risk_level="high",
                correlation_with_market=0.5,
                recommended_position_size=0.08,
                optimal_timeframes=['1h', '4h', '1d']
            ),
            
            # Фінанси - помірна волатильність з новинами
            "finance": SectorConfig(
                name="Financial Sector",
                tickers=['JPM', 'BAC', 'WFC', 'GS', 'V'],
                volatility_score=6.0,
                profit_potential=6.5,
                risk_level="medium",
                correlation_with_market=0.8,
                recommended_position_size=0.1,
                optimal_timeframes=['1h', '4h', '1d']
            ),
            
            # Біотех - висока волатильність на новинах
            "biotech": SectorConfig(
                name="Biotechnology",
                tickers=['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'BMY', 'AMGN', 'GILD'],
                volatility_score=7.5,
                profit_potential=8.0,
                risk_level="high",
                correlation_with_market=0.4,
                recommended_position_size=0.06,
                optimal_timeframes=['15m', '1h', '4h']
            ),
            
            # Споживчий сектор - стабільний з вибуховими моментами
            "consumer": SectorConfig(
                name="Consumer Sector",
                tickers=['PG', 'KO', 'HD', 'MCD', 'WMT', 'COST', 'NKE', 'SBUX'],
                volatility_score=5.0,
                profit_potential=5.5,
                risk_level="low",
                correlation_with_market=0.6,
                recommended_position_size=0.12,
                optimal_timeframes=['4h', '1d']
            ),
            
            # Промисловість - циклічна волатильність
            "industrial": SectorConfig(
                name="Industrial Sector",
                tickers=['GE', 'MMM', 'HON', 'CAT', 'DE', 'UPS', 'RTX', 'BA'],
                volatility_score=6.5,
                profit_potential=6.0,
                risk_level="medium",
                correlation_with_market=0.7,
                recommended_position_size=0.09,
                optimal_timeframes=['1h', '4h', '1d']
            ),
            
            # ETF для хеджування та диверсифікації
            "etf_strategic": SectorConfig(
                name="Strategic ETFs",
                tickers=['QQQ', 'SPY', 'IWM', 'GLD', 'TLT', 'XLE', 'XLK', 'ARKK', 'SOXX', 'SMH'],
                volatility_score=4.5,
                profit_potential=5.0,
                risk_level="low",
                correlation_with_market=0.9,
                recommended_position_size=0.15,
                optimal_timeframes=['1h', '4h', '1d']
            )
        }
    
    def _create_volatility_ranking(self) -> List[str]:
        """Рейтинг волатильності секторів"""
        sectors_by_volatility = sorted(
            self.sectors.items(),
            key=lambda x: x[1].volatility_score,
            reverse=True
        )
        return [sector[0] for sector in sectors_by_volatility]
    
    def _create_profitability_ranking(self) -> List[str]:
        """Рейтинг прибутковості секторів"""
        sectors_by_profit = sorted(
            self.sectors.items(),
            key=lambda x: x[1].profit_potential,
            reverse=True
        )
        return [sector[0] for sector in sectors_by_profit]
    
    def get_tickers_by_strategy(self, strategy: str, limit: Optional[int] = None) -> List[str]:
        """
        Отримати тікери за стратегією
        
        Args:
            strategy: Стратегія вибору
            limit: Обмеження кількості тікерів
            
        Returns:
            List[str]: Список тікерів
        """
        if strategy == "extreme_volatility":
            # Найбільш волатильні для максимального прибутку
            tickers = []
            for sector_name in self.volatility_ranking[:3]:  # Топ-3 волатильні
                sector = self.sectors[sector_name]
                tickers.extend(sector.tickers)
            return tickers[:limit] if limit else tickers
            
        elif strategy == "balanced_growth":
            # Балансований ріст з помірною волатильністю
            tickers = []
            for sector_name in ["ai_big_tech", "semiconductors", "crypto_related"]:
                tickers.extend(self.sectors[sector_name].tickers)
            return tickers[:limit] if limit else tickers
            
        elif strategy == "news_driven":
            # Сектори, чутливі до новин
            tickers = []
            for sector_name in ["crypto_related", "energy", "biotech"]:
                tickers.extend(self.sectors[sector_name].tickers)
            return tickers[:limit] if limit else tickers
            
        elif strategy == "momentum":
            # Моментум стратегія - сильні трендові сектори
            tickers = []
            for sector_name in ["ai_big_tech", "semiconductors", "extreme_volatility"]:
                tickers.extend(self.sectors[sector_name].tickers)
            return tickers[:limit] if limit else tickers
            
        elif strategy == "conservative":
            # Консервативний підхід
            tickers = []
            for sector_name in ["consumer", "finance", "etf_strategic"]:
                tickers.extend(self.sectors[sector_name].tickers)
            return tickers[:limit] if limit else tickers
            
        else:
            # За замовчуванням - збалансований підхід
            return self.get_tickers_by_strategy("balanced_growth", limit)
    
    def get_optimal_configuration(self, risk_tolerance: str, capital: float) -> Dict[str, any]:
        """
        Отримати оптимальну конфігурацію
        
        Args:
            risk_tolerance: Рівень ризику (low, medium, high, extreme)
            capital: Капітал
            
        Returns:
            Dict: Оптимальна конфігурація
        """
        risk_mapping = {
            "low": ["consumer", "finance", "etf_strategic"],
            "medium": ["industrial", "energy", "ai_big_tech"],
            "high": ["semiconductors", "crypto_related", "biotech"],
            "extreme": ["extreme_volatility"]
        }
        
        selected_sectors = risk_mapping.get(risk_tolerance, risk_mapping["medium"])
        
        # Формуємо конфігурацію
        config = {
            "risk_tolerance": risk_tolerance,
            "selected_sectors": selected_sectors,
            "tickers": [],
            "position_sizes": {},
            "timeframes": [],
            "expected_volatility": 0,
            "expected_return": 0
        }
        
        # Додаємо тікери з обраних секторів
        for sector_name in selected_sectors:
            sector = self.sectors[sector_name]
            config["tickers"].extend(sector.tickers)
            
            # Розраховуємо розміри позицій
            for ticker in sector.tickers:
                config["position_sizes"][ticker] = sector.recommended_position_size
            
            # Додаємо таймфрейми
            config["timeframes"].extend(sector.optimal_timeframes)
            
            # Накопичуємо очікувані показники
            config["expected_volatility"] += sector.volatility_score
            config["expected_return"] += sector.profit_potential
        
        # Усереднюємо показники
        config["expected_volatility"] /= len(selected_sectors)
        config["expected_return"] /= len(selected_sectors)
        
        # Унікальні таймфрейми
        config["timeframes"] = list(set(config["timeframes"]))
        
        # Розраховуємо максимальну кількість позицій
        max_positions = int(capital * 0.1 / 1000)  # 10% капіталу, мінімум $1000 на позицію
        config["max_positions"] = min(max_positions, len(config["tickers"]))
        
        return config
    
    def get_sector_analysis(self) -> pd.DataFrame:
        """Отримати аналіз секторів у вигляді DataFrame"""
        data = []
        for sector_name, sector in self.sectors.items():
            data.append({
                'sector': sector_name,
                'name': sector.name,
                'tickers_count': len(sector.tickers),
                'volatility_score': sector.volatility_score,
                'profit_potential': sector.profit_potential,
                'risk_level': sector.risk_level,
                'correlation': sector.correlation_with_market,
                'position_size': sector.recommended_position_size,
                'optimal_timeframes': ', '.join(sector.optimal_timeframes)
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('volatility_score', ascending=False)
    
    def recommend_tickers_for_capital(self, capital: float, max_positions: int = 10) -> Dict[str, any]:
        """
        Рекомендувати тікери для конкретного капіталу
        
        Args:
            capital: Доступний капітал
            max_positions: Максимальна кількість позицій
            
        Returns:
            Dict: Рекомендації
        """
        # Розраховуємо мінімальний розмір позиції
        min_position_size = capital * 0.02  # 2% мінімум
        max_position_size = capital * 0.15  # 15% максимум
        
        # Вибираємо тікери за волатильністю та прибутковістю
        recommendations = {
            "capital": capital,
            "max_positions": max_positions,
            "min_position_size": min_position_size,
            "max_position_size": max_position_size,
            "selected_tickers": [],
            "position_allocations": {},
            "sector_allocation": {},
            "expected_metrics": {
                "portfolio_volatility": 0,
                "expected_return": 0,
                "risk_score": 0
            }
        }
        
        # Беремо тікери з топ секторів
        selected_tickers = []
        sector_weights = {}
        
        for sector_name in self.volatility_ranking[:4]:  # Топ-4 сектори
            sector = self.sectors[sector_name]
            
            # Вибираємо топ тікери з сектору
            sector_tickers = sector.tickers[:min(3, max_positions // 4)]
            selected_tickers.extend(sector_tickers)
            
            # Розподіляємо ваги секторів
            sector_weights[sector_name] = len(sector_tickers) / max_positions
            
            if len(selected_tickers) >= max_positions:
                break
        
        selected_tickers = selected_tickers[:max_positions]
        
        # Розраховуємо розподіл капіталу
        equal_allocation = capital / len(selected_tickers)
        
        for ticker in selected_tickers:
            recommendations["selected_tickers"].append(ticker)
            recommendations["position_allocations"][ticker] = equal_allocation
            
            # Знаходимо сектор тікера
            for sector_name, sector in self.sectors.items():
                if ticker in sector.tickers:
                    if sector_name not in recommendations["sector_allocation"]:
                        recommendations["sector_allocation"][sector_name] = 0
                    recommendations["sector_allocation"][sector_name] += equal_allocation
                    break
        
        # Розраховуємо очікувані метрики
        total_volatility = 0
        total_return = 0
        
        for ticker in selected_tickers:
            for sector_name, sector in self.sectors.items():
                if ticker in sector.tickers:
                    weight = equal_allocation / capital
                    total_volatility += sector.volatility_score * weight
                    total_return += sector.profit_potential * weight
                    break
        
        recommendations["expected_metrics"]["portfolio_volatility"] = total_volatility
        recommendations["expected_metrics"]["expected_return"] = total_return
        recommendations["expected_metrics"]["risk_score"] = total_volatility / 10  # Нормалізуємо
        
        return recommendations


# Глобальний екземпляр
enhanced_sector_manager = EnhancedSectorTickerManager()


def get_enhanced_tickers(strategy: str = "balanced_growth", limit: Optional[int] = None) -> List[str]:
    """Отримати оптимізовані тікери за стратегією"""
    return enhanced_sector_manager.get_tickers_by_strategy(strategy, limit)


def get_sector_config_for_risk(risk_tolerance: str, capital: float) -> Dict[str, any]:
    """Отримати конфігурацію секторів для рівня ризику"""
    return enhanced_sector_manager.get_optimal_configuration(risk_tolerance, capital)


def analyze_sectors() -> pd.DataFrame:
    """Отримати аналіз секторів"""
    return enhanced_sector_manager.get_sector_analysis()


def recommend_portfolio(capital: float, max_positions: int = 10) -> Dict[str, any]:
    """Рекомендувати портфель"""
    return enhanced_sector_manager.recommend_tickers_for_capital(capital, max_positions)


if __name__ == "__main__":
    # Приклад використання
    print("=== Enhanced Sector Ticker Analysis ===")
    
    # Аналіз секторів
    sector_df = analyze_sectors()
    print("\n[DATA] Sector Analysis:")
    print(sector_df[['sector', 'volatility_score', 'profit_potential', 'risk_level']].to_string())
    
    # Рекомендації для різних стратегій
    strategies = ["extreme_volatility", "balanced_growth", "conservative"]
    
    for strategy in strategies:
        tickers = get_enhanced_tickers(strategy, limit=10)
        print(f"\n[TARGET] {strategy.upper()} Strategy ({len(tickers)} tickers):")
        print(f"   {', '.join(tickers)}")
    
    # Рекомендації для портфеля
    capital = 50000
    portfolio = recommend_portfolio(capital, max_positions=8)
    
    print(f"\n[MONEY] Portfolio Recommendations for ${capital:,}:")
    print(f"   Selected tickers: {len(portfolio['selected_tickers'])}")
    print(f"   Position size: ${portfolio['position_allocations'][portfolio['selected_tickers'][0]]:,.2f}")
    print(f"   Expected volatility: {portfolio['expected_metrics']['portfolio_volatility']:.1f}/10")
    print(f"   Expected return: {portfolio['expected_metrics']['expected_return']:.1f}/10")
