#!/usr/bin/env python3
"""
Economic Context Mapper - Оптимальний вибір економічних показників
Порівняння останньої публікації з попередньою з урахуванням шуму
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger("EconomicContextMapper")

def time_series_cross_validation(model, X, y, cv=5):
    """
    Time series cross-validation to prevent data leakage.
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=cv)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
        
        logger.info(f"Fold {fold + 1}: score = {score:.4f}")
    
    return np.mean(scores), np.std(scores)



import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EconomicIndicator:
    """Конфігурація економічного показника"""
    name: str
    source: str  # FRED, Yahoo, API
    frequency: str  # daily, weekly, monthly
    threshold: float  # порог для детекції змін
    noise_filter: float  # фільтр шуму (0-1)
    weight: float  # вага в контексті
    direction: str  # 'higher_better', 'lower_better', 'neutral'


class EconomicContextMapper:
    """
    Мапер економічного контексту з порівнянням показників
    Логіка: останній показник vs попередній з урахуванням шуму
    """
    
    def __init__(self):
        self.logger = logging.getLogger("EconomicContextMapper")
        
        # Розширені економічні показники для широкого контексту
        self.economic_indicators = {
            # Процентні ставки та дохідність
            'fedfunds': EconomicIndicator(
                name='Federal Funds Rate',
                source='FRED',
                frequency='daily',
                threshold=0.25,
                noise_filter=0.1,
                weight=0.08,
                direction='higher_better'
            ),
            
            't10y2y': EconomicIndicator(
                name='10Y-2Y Treasury Spread',
                source='FRED',
                frequency='daily',
                threshold=0.5,
                noise_filter=0.15,
                weight=0.07,
                direction='higher_better'
            ),
            
            't30y10y': EconomicIndicator(
                name='30Y-10Y Treasury Spread',
                source='FRED',
                frequency='daily',
                threshold=0.3,
                noise_filter=0.12,
                weight=0.05,
                direction='higher_better'
            ),
            
            'real_yield': EconomicIndicator(
                name='Real Yield (10Y-TIPS)',
                source='FRED',
                frequency='daily',
                threshold=0.2,
                noise_filter=0.1,
                weight=0.06,
                direction='higher_better'
            ),
            
            # Ринкові індикатори
            'vix': EconomicIndicator(
                name='VIX Index',
                source='Yahoo',
                frequency='daily',
                threshold=2.0,
                noise_filter=0.2,
                weight=0.09,
                direction='lower_better'
            ),
            
            'vix3m': EconomicIndicator(
                name='VIX 3M',
                source='Yahoo',
                frequency='daily',
                threshold=1.5,
                noise_filter=0.18,
                weight=0.04,
                direction='lower_better'
            ),
            
            'put_call_ratio': EconomicIndicator(
                name='Put/Call Ratio',
                source='CBOE',
                frequency='daily',
                threshold=0.1,
                noise_filter=0.15,
                weight=0.05,
                direction='higher_better'
            ),
            
            'advance_decline': EconomicIndicator(
                name='Advance/Decline Ratio',
                source='NYSE',
                frequency='daily',
                threshold=0.2,
                noise_filter=0.12,
                weight=0.06,
                direction='higher_better'
            ),
            
            'new_highs_lows': EconomicIndicator(
                name='New Highs - New Lows',
                source='NYSE',
                frequency='daily',
                threshold=100,
                noise_filter=0.2,
                weight=0.05,
                direction='higher_better'
            ),
            
            # Часові та психологічні показники
            'day_of_week': EconomicIndicator(
                name='Day of Week',
                source='Calendar',
                frequency='daily',
                threshold=0.1,
                noise_filter=0.05,
                weight=0.02,
                direction='neutral'
            ),
            
            'day_of_month': EconomicIndicator(
                name='Day of Month',
                source='Calendar',
                frequency='daily',
                threshold=0.15,
                noise_filter=0.1,
                weight=0.02,
                direction='neutral'
            ),
            
            'week_of_year': EconomicIndicator(
                name='Week of Year',
                source='Calendar',
                frequency='weekly',
                threshold=0.2,
                noise_filter=0.1,
                weight=0.03,
                direction='neutral'
            ),
            
            'month_of_year': EconomicIndicator(
                name='Month of Year',
                source='Calendar',
                frequency='monthly',
                threshold=0.25,
                noise_filter=0.15,
                weight=0.03,
                direction='neutral'
            ),
            
            'quarter': EconomicIndicator(
                name='Quarter',
                source='Calendar',
                frequency='quarterly',
                threshold=0.3,
                noise_filter=0.2,
                weight=0.04,
                direction='neutral'
            ),
            
            # Технічні показники (агреговані з тіперами)
            'market_breadth': EconomicIndicator(
                name='Market Breadth',
                source='Technical',
                frequency='daily',
                threshold=0.05,
                noise_filter=0.1,
                weight=0.06,
                direction='higher_better'
            ),
            
            'market_sentiment': EconomicIndicator(
                name='Market Sentiment Index',
                source='News/VIX',
                frequency='daily',
                threshold=0.1,
                noise_filter=0.15,
                weight=0.07,
                direction='higher_better'
            ),
            
            'volatility_index': EconomicIndicator(
                name='Volatility Index (VIX)',
                source='Technical',
                frequency='daily',
                threshold=2.0,
                noise_filter=0.2,
                weight=0.08,
                direction='higher_better'
            ),
            
            'trend_strength': EconomicIndicator(
                name='Trend Strength Index',
                source='Technical',
                frequency='daily',
                threshold=0.3,
                noise_filter=0.15,
                weight=0.05,
                direction='higher_better'
            ),
            
            'momentum_index': EconomicIndicator(
                name='Momentum Index',
                source='Technical',
                frequency='daily',
                threshold=0.2,
                noise_filter=0.1,
                weight=0.06,
                direction='higher_better'
            ),
            
            # Психологічні показники (оптимізмізм)
            'fear_index': EconomicIndicator(
                name='Fear Index (CNN Fear & Greed)',
                source='CNN Fear & Greed',
                frequency='daily',
                threshold=10.0,
                noise_filter=0.2,
                weight=0.08,
                direction='lower_better'
            ),
            
            'greed_index': EconomicIndicator(
                name='Greed Index (CNN Fear & Greed)',
                source='CNN Fear & Greed',
                frequency='daily',
                threshold=10.0,
                noise_filter=0.2,
                weight=0.08,
                direction='higher_better'
            ),
            
            'put_call_ratio': EconomicIndicator(
                name='Put/Call Ratio',
                source='CBOE',
                frequency='daily',
                threshold=0.1,
                noise_filter=0.15,
                weight=0.05,
                direction='higher_better'
            ),
            
            'advance_decline': EconomicIndicator(
                name='Advance/Decline Ratio',
                source='NYSE',
                frequency='daily',
                threshold=0.2,
                noise_filter=0.12,
                weight=0.06,
                direction='higher_better'
            ),
            
            # Додаткові ринкові показники
            'market_momentum': EconomicIndicator(
                name='Market Momentum Index',
                source='S&P 500',
                frequency='daily',
                threshold=0.02,
                noise_filter=0.1,
                weight=0.07,
                direction='higher_better'
            ),
            
            'sector_rotation': EconomicIndicator(
                name='Sector Rotation Index',
                source='Sector ETFs',
                frequency='weekly',
                threshold=0.15,
                noise_filter=0.2,
                weight=0.05,
                direction='neutral'
            ),
            
            'liquidity_index': EconomicIndicator(
                name='Liquidity Index',
                source='Volume/Spread',
                frequency='daily',
                threshold=0.05,
                noise_filter=0.1,
                weight=0.04,
                direction='higher_better'
            ),
            
            'institutional_flow': EconomicIndicator(
                name='Institutional Flow Index',
                source='CFTC',
                frequency='weekly',
                threshold=0.1,
                noise_filter=0.15,
                weight=0.06,
                direction='higher_better'
            ),
            
            # Сировинні показники
            'oil_price': EconomicIndicator(
                name='Crude Oil Price',
                source='Yahoo',
                frequency='daily',
                threshold=5.0,
                noise_filter=0.15,
                weight=0.06,
                direction='neutral'
            ),
            
            'gold_price': EconomicIndicator(
                name='Gold Price',
                source='Yahoo',
                frequency='daily',
                threshold=3.0,
                noise_filter=0.15,
                weight=0.05,
                direction='higher_better'
            ),
            
            'usd_index': EconomicIndicator(
                name='USD Index (DXY)',
                source='Yahoo',
                frequency='daily',
                threshold=1.0,
                noise_filter=0.1,
                weight=0.04,
                direction='neutral'
            ),
            
            'crypto_index': EconomicIndicator(
                name='Crypto Index (BTC Dominance)',
                source='CoinMarketCap',
                frequency='daily',
                threshold=5.0,
                noise_filter=0.25,
                weight=0.03,
                direction='higher_better'
            ),
            
            # Геополітичні показники
            'geopolitical_risk': EconomicIndicator(
                name='Geopolitical Risk Index',
                source='News Analysis',
                frequency='daily',
                threshold=0.1,
                noise_filter=0.2,
                weight=0.07,
                direction='lower_better'
            ),
            
            'trade_war_impact': EconomicIndicator(
                name='Trade War Impact Index',
                source='News Analysis',
                frequency='daily',
                threshold=0.15,
                noise_filter=0.2,
                weight=0.06,
                direction='lower_better'
            ),
            
            'election_cycle': EconomicIndicator(
                name='Election Cycle Index',
                source='Political Analysis',
                frequency='monthly',
                threshold=0.2,
                noise_filter=0.15,
                weight=0.05,
                direction='neutral'
            ),
            'gdp_growth': EconomicIndicator(
                name='GDP Growth',
                source='FRED',
                frequency='quarterly',
                threshold=0.5,
                noise_filter=0.1,
                weight=0.09,
                direction='higher_better'
            ),
            
            'gdp_nowcast': EconomicIndicator(
                name='GDP Nowcast',
                source='Atlanta Fed',
                frequency='weekly',
                threshold=0.3,
                noise_filter=0.12,
                weight=0.05,
                direction='higher_better'
            ),
            
            # Сировинні ринки
            'oil': EconomicIndicator(
                name='Crude Oil Price',
                source='Yahoo',
                frequency='daily',
                threshold=5.0,
                noise_filter=0.15,
                weight=0.06,
                direction='neutral'
            ),
            
            'gold': EconomicIndicator(
                name='Gold Price',
                source='Yahoo',
                frequency='daily',
                threshold=50.0,
                noise_filter=0.2,
                weight=0.04,
                direction='neutral'
            ),
            
            'copper': EconomicIndicator(
                name='Copper Price',
                source='Yahoo',
                frequency='daily',
                threshold=0.5,
                noise_filter=0.18,
                weight=0.03,
                direction='higher_better'
            ),
            
            'natural_gas': EconomicIndicator(
                name='Natural Gas Price',
                source='Yahoo',
                frequency='daily',
                threshold=1.0,
                noise_filter=0.2,
                weight=0.02,
                direction='neutral'
            ),
            
            # Валютні ринки
            'dxy': EconomicIndicator(
                name='US Dollar Index',
                source='Yahoo',
                frequency='daily',
                threshold=1.0,
                noise_filter=0.1,
                weight=0.05,
                direction='neutral'
            ),
            
            'eur_usd': EconomicIndicator(
                name='EUR/USD',
                source='Yahoo',
                frequency='daily',
                threshold=0.01,
                noise_filter=0.12,
                weight=0.03,
                direction='neutral'
            ),
            
            'usd_jpy': EconomicIndicator(
                name='USD/JPY',
                source='Yahoo',
                frequency='daily',
                threshold=1.0,
                noise_filter=0.15,
                weight=0.03,
                direction='neutral'
            ),
            
            # Ділові та споживчі показники
            'confidence': EconomicIndicator(
                name='Consumer Confidence',
                source='FRED',
                frequency='monthly',
                threshold=5.0,
                noise_filter=0.12,
                weight=0.06,
                direction='higher_better'
            ),
            
            'pmi': EconomicIndicator(
                name='Manufacturing PMI',
                source='FRED',
                frequency='monthly',
                threshold=2.0,
                noise_filter=0.1,
                weight=0.07,
                direction='higher_better'
            ),
            
            'services_pmi': EconomicIndicator(
                name='Services PMI',
                source='FRED',
                frequency='monthly',
                threshold=2.0,
                noise_filter=0.1,
                weight=0.05,
                direction='higher_better'
            ),
            
            'retail_sales': EconomicIndicator(
                name='Retail Sales',
                source='FRED',
                frequency='monthly',
                threshold=1.0,
                noise_filter=0.08,
                weight=0.04,
                direction='higher_better'
            ),
            
            'durable_goods': EconomicIndicator(
                name='Durable Goods Orders',
                source='FRED',
                frequency='monthly',
                threshold=2.0,
                noise_filter=0.15,
                weight=0.03,
                direction='higher_better'
            ),
            
            # Ринок нерухомості
            'housing_starts': EconomicIndicator(
                name='Housing Starts',
                source='FRED',
                frequency='monthly',
                threshold=50,
                noise_filter=0.1,
                weight=0.04,
                direction='higher_better'
            ),
            
            'building_permits': EconomicIndicator(
                name='Building Permits',
                source='FRED',
                frequency='monthly',
                threshold=50,
                noise_filter=0.1,
                weight=0.03,
                direction='higher_better'
            ),
            
            'existing_home_sales': EconomicIndicator(
                name='Existing Home Sales',
                source='FRED',
                frequency='monthly',
                threshold=100,
                noise_filter=0.12,
                weight=0.03,
                direction='higher_better'
            ),
            
            'case_shiller': EconomicIndicator(
                name='Case-Shiller Index',
                source='FRED',
                frequency='monthly',
                threshold=2.0,
                noise_filter=0.08,
                weight=0.04,
                direction='higher_better'
            ),
            
            # Виробництво та промисловість
            'industrial_production': EconomicIndicator(
                name='Industrial Production',
                source='FRED',
                frequency='monthly',
                threshold=0.5,
                noise_filter=0.08,
                weight=0.05,
                direction='higher_better'
            ),
            
            'capacity_utilization': EconomicIndicator(
                name='Capacity Utilization',
                source='FRED',
                frequency='monthly',
                threshold=1.0,
                noise_filter=0.1,
                weight=0.04,
                direction='higher_better'
            ),
            
            'factory_orders': EconomicIndicator(
                name='Factory Orders',
                source='FRED',
                frequency='monthly',
                threshold=1.0,
                noise_filter=0.12,
                weight=0.03,
                direction='higher_better'
            ),
            
            # Ринок праці детальніше
            'job_openings': EconomicIndicator(
                name='Job Openings',
                source='FRED',
                frequency='monthly',
                threshold=100,
                noise_filter=0.15,
                weight=0.04,
                direction='higher_better'
            ),
            
            'job_quits': EconomicIndicator(
                name='Job Quits Rate',
                source='FRED',
                frequency='monthly',
                threshold=0.1,
                noise_filter=0.1,
                weight=0.03,
                direction='higher_better'
            ),
            
            'weekly_claims': EconomicIndicator(
                name='Weekly Jobless Claims',
                source='FRED',
                frequency='weekly',
                threshold=10,
                noise_filter=0.12,
                weight=0.05,
                direction='lower_better'
            ),
            
            # Фінансові ринки
            'sp500_pe': EconomicIndicator(
                name='S&P 500 P/E Ratio',
                source='Yahoo',
                frequency='daily',
                threshold=2.0,
                noise_filter=0.15,
                weight=0.04,
                direction='lower_better'
            ),
            
            'sp500_dividend': EconomicIndicator(
                name='S&P 500 Dividend Yield',
                source='Yahoo',
                frequency='daily',
                threshold=0.5,
                noise_filter=0.1,
                weight=0.03,
                direction='higher_better'
            ),
            
            'credit_spread': EconomicIndicator(
                name='Investment Grade Credit Spread',
                source='FRED',
                frequency='daily',
                threshold=0.5,
                noise_filter=0.12,
                weight=0.05,
                direction='higher_better'
            ),
            
            'high_yield_spread': EconomicIndicator(
                name='High Yield Credit Spread',
                source='FRED',
                frequency='daily',
                threshold=1.0,
                noise_filter=0.15,
                weight=0.04,
                direction='higher_better'
            ),
            
            # Геополітичні та ризикові показники
            'geopolitical_risk': EconomicIndicator(
                name='Geopolitical Risk Index',
                source='Caldara',
                frequency='monthly',
                threshold=10,
                noise_filter=0.2,
                weight=0.03,
                direction='lower_better'
            ),
            
            'policy_uncertainty': EconomicIndicator(
                name='Economic Policy Uncertainty',
                source='Baker',
                frequency='monthly',
                threshold=10,
                noise_filter=0.18,
                weight=0.03,
                direction='lower_better'
            ),
            
            'emerging_markets_risk': EconomicIndicator(
                name='Emerging Markets Risk Premium',
                source='JPMorgan',
                frequency='daily',
                threshold=0.5,
                noise_filter=0.15,
                weight=0.02,
                direction='higher_better'
            )
        }
        
        # Контекстні показники часу
        self.temporal_indicators = {
            'weekday': {'range': (0, 6), 'weight': 0.03},
            'hour_of_day': {'range': (0, 23), 'weight': 0.02},
            'is_market_hours': {'range': (0, 1), 'weight': 0.02},
            'month': {'range': (1, 12), 'weight': 0.02},
            'quarter': {'range': (1, 4), 'weight': 0.02}
        }
        
        # Кешування результатів
        self.context_cache = {}
        self.cache_duration = timedelta(hours=1)
        
        self.logger.info(f"EconomicContextMapper initialized with {len(self.economic_indicators)} economic indicators")
    
    def compare_indicator_values(self, current_value: float, previous_value: float, 
                               indicator: EconomicIndicator) -> int:
        """
        Порівняння показників з урахуванням шуму
        
        Args:
            current_value: Поточне значення
            previous_value: Попереднє значення
            indicator: Конфігурація показника
            
        Returns:
            int: 1 (вище), -1 (нижче), 0 (without змін)
        """
        if pd.isna(current_value) or pd.isna(previous_value):
            return 0
        
        # Розраховуємо різницю
        diff = current_value - previous_value
        abs_diff = abs(diff)
        
        # Застосовуємо фільтр шуму
        noise_threshold = indicator.threshold * indicator.noise_filter
        
        if abs_diff < noise_threshold:
            return 0  # Різниця в межах шуму
        
        # Визначаємо напрямок з урахуванням бажаної динаміки
        if indicator.direction == 'higher_better':
            return 1 if diff > 0 else -1
        elif indicator.direction == 'lower_better':
            return -1 if diff > 0 else 1
        else:  # neutral
            return 1 if diff > 0 else -1
    
    def get_technical_context(self, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Отримати технічний контекст з показників
        
        Args:
            technical_data: Технічні показники (RSI, MACD, etc.)
            
        Returns:
            Dict: Технічний контекст
        """
        technical_context = {
            'timestamp': datetime.now().isoformat(),
            'technical_indicators': {},
            'overall_technical_score': 0,
            'technical_regime': 'neutral',
            'technical_strength': 'medium'
        }
        
        total_score = 0
        total_weight = 0
        
        # Обробляємо технічні показники
        for indicator_name, indicator_data in technical_data.items():
            if indicator_name in self.economic_indicators:
                indicator_config = self.economic_indicators[indicator_name]
                
                # Для технічних показників використовуємо іншу логіку
                if indicator_config.source in ['Technical', 'News/VIX', 'CBOE', 'NYSE', 'S&P 500']:
                    # Технічні показники порівнюємо з історичними значеннями
                    current_value = indicator_data.get('current_value', 0)
                    historical_values = indicator_data.get('historical_values', [])
                    
                    if historical_values and len(historical_values) > 1:
                        # Розраховуємо порівняння
                        previous_value = historical_values[-2]  # Попереднє значення
                        comparison = self._compare_technical_values(
                            current_value, previous_value, indicator_config
                        )
                        
                        # Розраховуємо скор
                        indicator_score = comparison * indicator_config.weight
                        
                        technical_context['technical_indicators'][indicator_name] = {
                            'current_value': current_value,
                            'previous_value': previous_value,
                            'comparison': comparison,
                            'score': indicator_score,
                            'weight': indicator_config.weight,
                            'direction': indicator_config.direction,
                            'signal_strength': abs(indicator_score) / indicator_config.weight
                        }
                        
                        total_score += indicator_score
                        total_weight += indicator_config.weight
        
        # Розраховуємо загальний технічний скор
        if total_weight > 0:
            technical_context['overall_technical_score'] = total_score / total_weight
        
        # Визначаємо технічний режим
        technical_context['technical_regime'] = self._determine_technical_regime(
            technical_context['overall_technical_score']
        )
        
        # Визначаємо силу технічних сигналів
        technical_context['technical_strength'] = self._determine_technical_strength(
            technical_context['technical_indicators']
        )
        
        return technical_context
    
    def get_psychological_context(self, current_date: datetime) -> Dict[str, Any]:
        """
        Отримати психологічний контекст (день тижня, місяця, etc.)
        
        Args:
            current_date: Поточна дата
            
        Returns:
            Dict: Психологічний контекст
        """
        psychological_context = {
            'timestamp': current_date.isoformat(),
            'psychological_indicators': {},
            'overall_psychological_score': 0,
            'psychological_bias': 'neutral',
            'seasonal_factor': 'normal'
        }
        
        total_score = 0
        total_weight = 0
        
        # День тижня (0=Monday, 6=Sunday)
        day_of_week = current_date.weekday()
        day_of_week_score = self._calculate_day_of_week_score(day_of_week)
        
        psychological_context['psychological_indicators']['day_of_week'] = {
            'current_value': day_of_week,
            'day_name': current_date.strftime('%A'),
            'score': day_of_week_score,
            'weight': 0.02,
            'psychological_impact': self._get_day_of_week_impact(day_of_week)
        }
        
        total_score += day_of_week_score * 0.02
        total_weight += 0.02
        
        # День місяця
        day_of_month = current_date.day
        day_of_month_score = self._calculate_day_of_month_score(day_of_month)
        
        psychological_context['psychological_indicators']['day_of_month'] = {
            'current_value': day_of_month,
            'score': day_of_month_score,
            'weight': 0.02,
            'psychological_impact': self._get_day_of_month_impact(day_of_month)
        }
        
        total_score += day_of_month_score * 0.02
        total_weight += 0.02
        
        # Тиждень року
        week_of_year = current_date.isocalendar()[1]
        week_of_year_score = self._calculate_week_of_year_score(week_of_year)
        
        psychological_context['psychological_indicators']['week_of_year'] = {
            'current_value': week_of_year,
            'score': week_of_year_score,
            'weight': 0.03,
            'psychological_impact': self._get_week_of_year_impact(week_of_year)
        }
        
        total_score += week_of_year_score * 0.03
        total_weight += 0.03
        
        # Місяць року
        month_of_year = current_date.month
        month_of_year_score = self._calculate_month_of_year_score(month_of_year)
        
        psychological_context['psychological_indicators']['month_of_year'] = {
            'current_value': month_of_year,
            'month_name': current_date.strftime('%B'),
            'score': month_of_year_score,
            'weight': 0.03,
            'psychological_impact': self._get_month_of_year_impact(month_of_year)
        }
        
        total_score += month_of_year_score * 0.03
        total_weight += 0.03
        
        # Квартал
        quarter = (current_date.month - 1) // 3 + 1
        quarter_score = self._calculate_quarter_score(quarter)
        
        psychological_context['psychological_indicators']['quarter'] = {
            'current_value': quarter,
            'quarter_name': f'Q{quarter}',
            'score': quarter_score,
            'weight': 0.04,
            'psychological_impact': self._get_quarter_impact(quarter)
        }
        
        total_score += quarter_score * 0.04
        total_weight += 0.04
        
        # Розраховуємо загальний психологічний скор
        if total_weight > 0:
            psychological_context['overall_psychological_score'] = total_score / total_weight
        
        # Визначаємо психологічний bias
        psychological_context['psychological_bias'] = self._determine_psychological_bias(
            psychological_context['overall_psychological_score']
        )
        
        # Визначаємо сезонний фактор
        psychological_context['seasonal_factor'] = self._determine_seasonal_factor(
            month_of_year, quarter
        )
        
        return psychological_context
    
    def get_comprehensive_context(self, economic_data: Dict[str, float], 
                                 technical_data: Dict[str, Any],
                                 current_date: datetime) -> Dict[str, Any]:
        """
        Отримати комплексний контекст (економічний + технічний + психологічний)
        
        Args:
            economic_data: Економічні дані
            technical_data: Технічні дані
            current_date: Поточна дата
            
        Returns:
            Dict: Комплексний контекст
        """
        # Отримуємо окремі контексти
        economic_context = self.get_economic_context(economic_data)
        technical_context = self.get_technical_context(technical_data)
        psychological_context = self.get_psychological_context(current_date)
        
        # Створюємо комплексний контекст
        comprehensive_context = {
            'timestamp': datetime.now().isoformat(),
            'economic_context': economic_context,
            'technical_context': technical_context,
            'psychological_context': psychological_context,
            'overall_score': 0,
            'dominant_regime': 'neutral',
            'confidence_level': 'medium',
            'key_drivers': [],
            'risk_assessment': 'medium'
        }
        
        # Розраховуємо загальний скор з урахуванням ваг
        economic_weight = 0.5
        technical_weight = 0.3
        psychological_weight = 0.2
        
        overall_score = (
            economic_context.get('overall_score', 0) * economic_weight +
            technical_context.get('overall_technical_score', 0) * technical_weight +
            psychological_context.get('overall_psychological_score', 0) * psychological_weight
        )
        
        comprehensive_context['overall_score'] = overall_score
        
        # Визначаємо домінуючий режим
        comprehensive_context['dominant_regime'] = self._determine_dominant_regime(
            economic_context.get('market_regime', 'neutral'),
            technical_context.get('technical_regime', 'neutral'),
            psychological_context.get('psychological_bias', 'neutral')
        )
        
        # Визначаємо рівень впевненості
        comprehensive_context['confidence_level'] = self._determine_confidence_level(
            comprehensive_context['overall_score']
        )
        
        # Визначаємо ключові драйвери
        comprehensive_context['key_drivers'] = self._identify_key_drivers(
            economic_context, technical_context, psychological_context
        )
        
        # Оцінка ризику
        comprehensive_context['risk_assessment'] = self._assess_overall_risk(
            comprehensive_context
        )
        
        return comprehensive_context
    
    def _compare_technical_values(self, current_value: float, previous_value: float,
                                 indicator: EconomicIndicator) -> int:
        """
        Порівняння технічних значень
        
        Args:
            current_value: Поточне значення
            previous_value: Попереднє значення
            indicator: Конфігурація показника
            
        Returns:
            int: 1 (вище), -1 (нижче), 0 (без змін)
        """
        if pd.isna(current_value) or pd.isna(previous_value):
            return 0
        
        # Розраховуємо різницю
        diff = current_value - previous_value
        abs_diff = abs(diff)
        
        # Застосовуємо фільтр шуму
        noise_threshold = indicator.threshold * indicator.noise_filter
        
        if abs_diff < noise_threshold:
            return 0  # Різниця в межах шуму
        
        # Визначаємо напрямок з урахуванням типу показника
        if indicator.name in ['VIX Index', 'Volatility Index', 'Fear Index']:
            # Для показників волатильності/страху - нижче краще
            return -1 if diff > 0 else 1 if diff < 0 else 0
        elif indicator.name in ['Market Sentiment Index', 'Trend Strength Index', 'Momentum Index']:
            # Для показників сентименту/тренду - вище краще
            return 1 if diff > 0 else -1 if diff < 0 else 0
        else:
            # Для інших показників - нейтрально
            return 1 if diff > 0 else -1 if diff < 0 else 0
    
    def _calculate_day_of_week_score(self, day_of_week: int) -> float:
        """Розрахувати скор для дня тижня"""
        # Monday (0) - негативний (Monday effect)
        # Friday (4) - позитивний (weekend optimism)
        # Wednesday (2) - нейтральний
        scores = [ -0.1, -0.05, 0.0, 0.05, 0.1, 0.0, -0.05 ]  # Mon-Sun
        return scores[day_of_week]
    
    def _get_day_of_week_impact(self, day_of_week: int) -> str:
        """Отримати психологічний вплив дня тижня"""
        impacts = {
            0: "Monday effect - negative sentiment",
            1: "Tuesday recovery",
            2: "Wednesday neutral",
            3: "Thursday momentum",
            4: "Friday optimism",
            5: "Saturday weekend mode",
            6: "Sunday weekend effect"
        }
        return impacts.get(day_of_week, "Unknown")
    
    def _calculate_day_of_month_score(self, day_of_month: int) -> float:
        """Розрахувати скор для дня місяця"""
        # Початок місяця (1-5) - позитивний (payday effect)
        # Кінець місяця (25-31) - негативний (bill pressure)
        if day_of_month <= 5:
            return 0.05
        elif day_of_month >= 25:
            return -0.05
        else:
            return 0.0
    
    def _get_day_of_month_impact(self, day_of_month: int) -> str:
        """Отримати психологічний вплив дня місяця"""
        if day_of_month <= 5:
            return "Payday effect - positive sentiment"
        elif day_of_month >= 25:
            return "Bill pressure - negative sentiment"
        else:
            return "Normal period"
    
    def _calculate_week_of_year_score(self, week_of_year: int) -> float:
        """Розрахувати скор для тижня року"""
        # Початок року (1-5) - позитивний (New Year optimism)
        # Кінець року (48-52) - негативний (tax loss harvesting)
        if week_of_year <= 5:
            return 0.03
        elif week_of_year >= 48:
            return -0.03
        else:
            return 0.0
    
    def _get_week_of_year_impact(self, week_of_year: int) -> str:
        """Отримати психологічний вплив тижня року"""
        if week_of_year <= 5:
            return "New Year optimism"
        elif week_of_year >= 48:
            return "Tax loss harvesting period"
        else:
            return "Normal period"
    
    def _calculate_month_of_year_score(self, month_of_year: int) -> float:
        """Розрахувати скор для місяця року"""
        # Сезонні ефекти
        seasonal_scores = {
            1: 0.02,   # January - New Year optimism
            2: 0.01,   # February - Valentine's effect
            3: 0.03,   # March - Spring optimism
            4: 0.02,   # April - Tax season
            5: 0.01,   # May - Summer approach
            6: 0.00,   # June - Summer start
            7: -0.01,  # July - Summer lull
            8: -0.02,  # August - Summer end
            9: 0.02,   # September - Back to school
            10: 0.03,  # October - Halloween effect
            11: -0.01, # November - Pre-holiday
            12: 0.01   # December - Holiday season
        }
        return seasonal_scores.get(month_of_year, 0.0)
    
    def _get_month_of_year_impact(self, month_of_year: int) -> str:
        """Отримати психологічний вплив місяця року"""
        impacts = {
            1: "New Year optimism",
            2: "Valentine's effect",
            3: "Spring optimism",
            4: "Tax season pressure",
            5: "Summer approach",
            6: "Summer start",
            7: "Summer lull",
            8: "Summer end anxiety",
            9: "Back to school",
            10: "Halloween effect",
            11: "Pre-holiday stress",
            12: "Holiday season"
        }
        return impacts.get(month_of_year, "Unknown")
    
    def _calculate_quarter_score(self, quarter: int) -> float:
        """Розрахувати скор для кварталу"""
        # Q1 - позитивний (New Year)
        # Q4 - позитивний (holiday season)
        # Q2, Q3 - нейтральні
        if quarter == 1:
            return 0.02
        elif quarter == 4:
            return 0.03
        else:
            return 0.0
    
    def _get_quarter_impact(self, quarter: int) -> str:
        """Отримати психологічний вплив кварталу"""
        impacts = {
            1: "Q1 - New Year optimism",
            2: "Q2 - Mid-year neutral",
            3: "Q3 - Summer lull",
            4: "Q4 - Holiday season"
        }
        return impacts.get(quarter, "Unknown")
    
    def _determine_technical_regime(self, technical_score: float) -> str:
        """Визначити технічний режим"""
        if technical_score > 0.15:
            return 'bullish'
        elif technical_score < -0.15:
            return 'bearish'
        else:
            return 'neutral'
    
    def _determine_technical_strength(self, indicators: Dict[str, Any]) -> str:
        """Визначити силу технічних сигналів"""
        strong_signals = sum(1 for ind in indicators.values() 
                           if ind.get('signal_strength', 0) > 0.7)
        
        if strong_signals >= 3:
            return 'strong'
        elif strong_signals >= 1:
            return 'medium'
        else:
            return 'weak'
    
    def _determine_psychological_bias(self, psychological_score: float) -> str:
        """Визначити психологічний bias"""
        if psychological_score > 0.1:
            return 'optimistic'
        elif psychological_score < -0.1:
            return 'pessimistic'
        else:
            return 'neutral'
    
    def _determine_seasonal_factor(self, month: int, quarter: int) -> str:
        """Визначити сезонний фактор"""
        if month in [12, 1, 2]:  # Winter
            return 'winter_effect'
        elif month in [3, 4, 5]:  # Spring
            return 'spring_effect'
        elif month in [6, 7, 8]:  # Summer
            return 'summer_effect'
        else:  # Fall
            return 'fall_effect'
    
    def _determine_dominant_regime(self, economic: str, technical: str, psychological: str) -> str:
        """Визначити домінуючий режим"""
        regimes = [economic, technical, psychological]
        
        # Підрахуємо голоси
        bullish_votes = sum(1 for r in regimes if r == 'bullish' or r == 'optimistic')
        bearish_votes = sum(1 for r in regimes if r == 'bearish' or r == 'pessimistic')
        
        if bullish_votes > bearish_votes:
            return 'bullish'
        elif bearish_votes > bullish_votes:
            return 'bearish'
        else:
            return 'neutral'
    
    def _determine_confidence_level(self, overall_score: float) -> str:
        """Визначити рівень впевненості"""
        abs_score = abs(overall_score)
        if abs_score > 0.2:
            return 'high'
        elif abs_score > 0.1:
            return 'medium'
        else:
            return 'low'
    
    def _identify_key_drivers(self, economic: Dict, technical: Dict, psychological: Dict) -> List[str]:
        """Ідентифікувати ключові драйвери"""
        drivers = []
        
        # Економічні драйвери
        economic_indicators = economic.get('economic_indicators', {})
        top_economic = sorted(economic_indicators.items(), 
                           key=lambda x: abs(x[1].get('score', 0)), 
                           reverse=True)[:2]
        
        for name, data in top_economic:
            drivers.append(f"Economic: {name} (score: {data.get('score', 0):.3f})")
        
        # Технічні драйвери
        technical_indicators = technical.get('technical_indicators', {})
        top_technical = sorted(technical_indicators.items(), 
                             key=lambda x: abs(x[1].get('score', 0)), 
                             reverse=True)[:2]
        
        for name, data in top_technical:
            drivers.append(f"Technical: {name} (score: {data.get('score', 0):.3f})")
        
        # Психологічні драйвери
        psych_indicators = psychological.get('psychological_indicators', {})
        top_psych = sorted(psych_indicators.items(), 
                          key=lambda x: abs(x[1].get('score', 0)), 
                          reverse=True)[:1]
        
        for name, data in top_psych:
            drivers.append(f"Psychological: {name} (score: {data.get('score', 0):.3f})")
        
        return drivers
    
    def _assess_overall_risk(self, context: Dict[str, Any]) -> str:
        """Оцінити загальний ризик"""
        risk_factors = 0
        
        # Економічний ризик
        if context['economic_context'].get('risk_level') == 'high':
            risk_factors += 1
        
        # Технічний ризик
        if context['technical_context'].get('technical_regime') == 'bearish':
            risk_factors += 1
        
        # Психологічний ризик
        if context['psychological_context'].get('psychological_bias') == 'pessimistic':
            risk_factors += 1
        
        # Загальний скор
        if abs(context['overall_score']) > 0.2:
            risk_factors += 1
        
        if risk_factors >= 3:
            return 'high'
        elif risk_factors >= 2:
            return 'medium'
        else:
            return 'low' 
    def get_economic_context(self, current_data: Dict[str, Any], historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Отримати економічний контекст
        
        Args:
            current_data: Поточні значення показників
            historical_data: Історичні дані для порівняння
            
        Returns:
            Dict: Економічний контекст
        """
        context_key = json.dumps(current_data, sort_keys=True)
        
        # Перевіряємо кеш
        if context_key in self.context_cache:
            cached_time, cached_context = self.context_cache[context_key]
            if datetime.now() - cached_time < self.cache_duration:
                return cached_context
        
        context = {
            'timestamp': datetime.now().isoformat(),
            'economic_indicators': {},
            'temporal_indicators': {},
            'overall_score': 0,
            'market_regime': 'neutral',
            'risk_level': 'medium'
        }
        
        total_score = 0
        total_weight = 0
        
        # Обробляємо економічні показники
        for indicator_name, indicator_config in self.economic_indicators.items():
            current_value = current_data.get(indicator_name)
            
            if current_value is None:
                continue
            
            # Отримуємо попереднє значення
            previous_value = self._get_previous_value(
                indicator_name, current_value, historical_data
            )
            
            # Порівнюємо значення
            comparison = self.compare_indicator_values(
                current_value, previous_value, indicator_config
            )
            
            # Розраховуємо скор
            indicator_score = comparison * indicator_config.weight
            
            context['economic_indicators'][indicator_name] = {
                'current_value': current_value,
                'previous_value': previous_value,
                'comparison': comparison,  # 1, -1, 0
                'score': indicator_score,
                'weight': indicator_config.weight,
                'direction': indicator_config.direction
            }
            
            total_score += indicator_score
            total_weight += indicator_config.weight
        
        # Обробляємо часові показники
        for temporal_name, temporal_config in self.temporal_indicators.items():
            value = current_data.get(temporal_name)
            
            if value is not None:
                # Нормалізуємо значення
                min_val, max_val = temporal_config['range']
                normalized_value = (value - min_val) / (max_val - min_val)
                
                # Розраховуємо скор
                temporal_score = (normalized_value - 0.5) * temporal_config['weight']
                
                context['temporal_indicators'][temporal_name] = {
                    'value': value,
                    'normalized_value': normalized_value,
                    'score': temporal_score,
                    'weight': temporal_config['weight']
                }
                
                total_score += temporal_score
                total_weight += temporal_config['weight']
        
        # Розраховуємо загальний скор
        if total_weight > 0:
            context['overall_score'] = total_score / total_weight
        
        # Визначаємо режим ринку
        context['market_regime'] = self._determine_market_regime(context['overall_score'])
        context['risk_level'] = self._determine_risk_level(context['overall_score'])
        
        # Кешуємо результат
        self.context_cache[context_key] = (datetime.now(), context)
        
        return context
    
    def _get_previous_value(self, indicator_name: str, current_value: float,
                          historical_data: Optional[pd.DataFrame]) -> Optional[float]:
        """Отримати попереднє значення показника"""
        if historical_data is None or indicator_name not in historical_data.columns:
            # Симуляція попереднього значення
            return current_value * np.random.uniform(0.95, 1.05)
        
        # Отримуємо останнє значення з історії
        indicator_data = historical_data[indicator_name].dropna()
        if len(indicator_data) > 1:
            return indicator_data.iloc[-2]
        
        return current_value
    
    def _determine_market_regime(self, score: float) -> str:
        """Визначити режим ринку на основі скору"""
        if score > 0.15:
            return 'bullish'
        elif score < -0.15:
            return 'bearish'
        else:
            return 'neutral'
    
    def _determine_risk_level(self, score: float) -> str:
        """Визначити рівень ризику"""
        abs_score = abs(score)
        if abs_score > 0.25:
            return 'high'
        elif abs_score > 0.1:
            return 'medium'
        else:
            return 'low'
    
    def calculate_universal_indicators(self, data: pd.DataFrame, indicator_type: str, 
                                   window: int = 20) -> Dict[str, pd.Series]:
        """
        Обчислює універсальні індикатори для всіх тікерів одночасно
        
        Args:
            data: DataFrame з даними (тікер-специфічні колонки)
            indicator_type: Тип індикатора ('sma', 'ema', 'rsi', 'macd', 'momentum', 'volatility')
            window: Вікно для розрахунку
            
        Returns:
            Dict: {ticker_name: pd.Series з індикатором}
        """
        indicators = {}
        
        # Знаходимо всі тікер-специфічні колонки для ціни
        price_columns = [col for col in data.columns if col.endswith('_close')]
        
        for price_col in price_columns:
            # Витягуємо назву тікера
            ticker = price_col.split('_')[0]
            
            # Отримуємо ціни для цього тікера
            prices = data[price_col].dropna()
            
            if len(prices) < window:
                continue
            
            # Обчислюємо індикатор залежно від типу
            if indicator_type == 'sma':
                indicator = prices.rolling(window=window, min_periods=1).mean()
            elif indicator_type == 'ema':
                indicator = prices.ewm(span=window, adjust=False, min_periods=1).mean()
            elif indicator_type == 'rsi':
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
                rs = gain / loss
                indicator = 100 - (100 / (1 + rs))
            elif indicator_type == 'momentum':
                indicator = prices.pct_change(periods=window)
            elif indicator_type == 'volatility':
                indicator = prices.rolling(window=window, min_periods=1).std()
            elif indicator_type == 'bollinger_upper':
                sma = prices.rolling(window=window, min_periods=1).mean()
                std = prices.rolling(window=window, min_periods=1).std()
                indicator = sma + (std * 2)
            elif indicator_type == 'bollinger_lower':
                sma = prices.rolling(window=window, min_periods=1).mean()
                std = prices.rolling(window=window, min_periods=1).std()
                indicator = sma - (std * 2)
            elif indicator_type == 'atr':
                if f'{ticker}_high' in data.columns and f'{ticker}_low' in data.columns:
                    high = data[f'{ticker}_high']
                    low = data[f'{ticker}_low']
                    close = prices
                    tr1 = high - low
                    tr2 = (high - close.shift()).abs()
                    tr3 = (low - close.shift()).abs()
                    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    indicator = true_range.rolling(window=window, min_periods=1).mean()
                else:
                    continue
            elif indicator_type == 'volume_sma':
                if f'{ticker}_volume' in data.columns:
                    volume = data[f'{ticker}_volume']
                    indicator = volume.rolling(window=window, min_periods=1).mean()
                else:
                    continue
            elif indicator_type == 'price_change':
                indicator = prices.pct_change()
            elif indicator_type == 'log_return':
                indicator = np.log(prices / prices.shift(1))
            elif indicator_type == 'cumulative_return':
                indicator = (prices / prices.iloc[0] - 1) * 100
            else:
                continue
            
            indicators[ticker] = indicator
        
        return indicators
    
    def get_universal_context(self, data: pd.DataFrame, 
                             indicators_config: List[Dict] = None) -> Dict[str, Any]:
        """
        Отримує універсальний контекст для всіх тікерів
        
        Args:
            data: DataFrame з даними
            indicators_config: Конфігурація індикаторів
            
        Returns:
            Dict: Універсальний контекст
        """
        if indicators_config is None:
            indicators_config = [
                {'type': 'sma', 'window': 20, 'weight': 0.1},
                {'type': 'ema', 'window': 12, 'weight': 0.1},
                {'type': 'rsi', 'window': 14, 'weight': 0.15},
                {'type': 'momentum', 'window': 5, 'weight': 0.1},
                {'type': 'volatility', 'window': 20, 'weight': 0.15},
                {'type': 'bollinger_upper', 'window': 20, 'weight': 0.05},
                {'type': 'bollinger_lower', 'window': 20, 'weight': 0.05},
                {'type': 'atr', 'window': 14, 'weight': 0.1},
                {'type': 'volume_sma', 'window': 20, 'weight': 0.05},
                {'type': 'price_change', 'window': 1, 'weight': 0.1},
                {'type': 'log_return', 'window': 1, 'weight': 0.05},
                {'type': 'cumulative_return', 'window': 0, 'weight': 0.15}
            ]
        
        universal_context = {
            'timestamp': datetime.now().isoformat(),
            'tickers': {},
            'market_overview': {},
            'top_performers': {},
            'bottom_performers': {},
            'market_signals': {}
        }
        
        # Обчислюємо всі індикатори для всіх тікерів
        all_indicators = {}
        for config in indicators_config:
            indicator_type = config['type']
            window = config['weight']
            indicators = self.calculate_universal_indicators(data, indicator_type, window)
            
            for ticker, indicator_series in indicators.items():
                if ticker not in all_indicators:
                    all_indicators[ticker] = {}
                
                all_indicators[ticker][indicator_type] = {
                    'series': indicator_series,
                    'current_value': indicator_series.iloc[-1] if not indicator_series.empty else None,
                    'previous_value': indicator_series.iloc[-2] if len(indicator_series) > 1 else None,
                    'weight': config['weight']
                }
        
        # Обробляємо кожен тікер
        ticker_scores = {}
        for ticker, indicators_data in all_indicators.items():
            ticker_context = {
                'indicators': {},
                'overall_score': 0,
                'signal': 'neutral',
                'performance': {}
            }
            
            total_score = 0
            total_weight = 0
            
            # Розраховуємо скор для кожного індикатора
            for indicator_type, indicator_data in indicators_data.items():
                current_value = indicator_data['current_value']
                previous_value = indicator_data['previous_value']
                weight = indicator_data['weight']
                
                if current_value is not None and previous_value is not None:
                    # Просте порівняння
                    if current_value > previous_value:
                        comparison = 1
                    elif current_value < previous_value:
                        comparison = -1
                    else:
                        comparison = 0
                    
                    # Специфічна логіка для різних індикаторів
                    if indicator_type in ['rsi', 'volatility', 'atr']:
                        # Для RSI, волатильності - екстремальні значення важливіші
                        if indicator_type == 'rsi':
                            if current_value > 70:  # Overbought
                                score = -1 * weight
                            elif current_value < 30:  # Oversold
                                score = 1 * weight
                            else:
                                score = comparison * weight * 0.5
                        elif indicator_type == 'volatility':
                            # Висока волатильність може бути добре або погано
                            score = comparison * weight * 0.7
                        else:  # ATR
                            score = comparison * weight * 0.6
                    elif indicator_type in ['momentum', 'price_change', 'log_return']:
                        # Для моментуму - позитивні зміни кращі
                        score = comparison * weight * 1.2
                    elif indicator_type in ['sma', 'ema']:
                        # Для ковзних середніх - порівнюємо з ціною
                        price_col = f'{ticker}_close'
                        if price_col in data.columns:
                            current_price = data[price_col].iloc[-1]
                            if current_value is not None:
                                price_vs_ma = (current_price - current_value) / current_value
                                score = np.sign(price_vs_ma) * weight
                            else:
                                score = 0
                        else:
                            score = 0
                    else:
                        # Для інших індикаторів
                        score = comparison * weight
                    
                    ticker_context['indicators'][indicator_type] = {
                        'current_value': current_value,
                        'previous_value': previous_value,
                        'comparison': comparison,
                        'score': score,
                        'weight': weight
                    }
                    
                    total_score += score
                    total_weight += weight
            
            # Розраховуємо загальний скор і сигнал
            if total_weight > 0:
                ticker_context['overall_score'] = total_score / total_weight
                
                if ticker_context['overall_score'] > 0.1:
                    ticker_context['signal'] = 'bullish'
                elif ticker_context['overall_score'] < -0.1:
                    ticker_context['signal'] = 'bearish'
                else:
                    ticker_context['signal'] = 'neutral'
            
            # Додаємо перформанс дані
            if f'{ticker}_close' in data.columns:
                prices = data[f'{ticker}_close'].dropna()
                if len(prices) > 1:
                    ticker_context['performance'] = {
                        'current_price': prices.iloc[-1],
                        'previous_price': prices.iloc[-2],
                        'price_change_pct': (prices.iloc[-1] / prices.iloc[-2] - 1) * 100,
                        'cumulative_return': (prices.iloc[-1] / prices.iloc[0] - 1) * 100 if len(prices) > 0 else 0
                    }
            
            universal_context['tickers'][ticker] = ticker_context
            ticker_scores[ticker] = ticker_context['overall_score']
        
        # Ранжуємо тікери
        sorted_tickers = sorted(ticker_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Топ перформери
        universal_context['top_performers'] = {
            'bullish': [ticker for ticker, score in sorted_tickers[:5] if score > 0.1],
            'bearish': [ticker for ticker, score in sorted_tickers[-5:] if score < -0.1],
            'neutral': [ticker for ticker, score in sorted_tickers if -0.1 <= score <= 0.1][:5]
        }
        
        # Загальний огляд ринку
        bullish_count = sum(1 for ticker, score in ticker_scores.items() if score > 0.1)
        bearish_count = sum(1 for ticker, score in ticker_scores.items() if score < -0.1)
        total_count = len(ticker_scores)
        
        universal_context['market_overview'] = {
            'total_tickers': total_count,
            'bullish_tickers': bullish_count,
            'bearish_tickers': bearish_count,
            'neutral_tickers': total_count - bullish_count - bearish_count,
            'market_sentiment': 'bullish' if bullish_count > bearish_count else 'bearish' if bearish_count > bullish_count else 'neutral',
            'average_score': sum(ticker_scores.values()) / total_count if total_count > 0 else 0
        }
        
        # Ринкові сигнали
        universal_context['market_signals'] = {
            'strong_bullish': [ticker for ticker, score in sorted_tickers[:3] if score > 0.2],
            'strong_bearish': [ticker for ticker, score in sorted_tickers[-3:] if score < -0.2],
            'reversal_candidates': [ticker for ticker, score in sorted_tickers if abs(score) > 0.15][:5]
        }
        
        return universal_context
    
    def get_ticker_universe_analysis(self, data: pd.DataFrame, 
                                    universe_tickers: List[str] = None) -> Dict[str, Any]:
        """
        Аналізує всесвіт тікерів
        
        Args:
            data: DataFrame з даними
            universe_tickers: Список тікерів для аналізу
            
        Returns:
            Dict: Аналіз всесвіту
        """
        if universe_tickers is None:
            # Автоматично знаходимо всі тікери в даних
            universe_tickers = list(set([col.split('_')[0] for col in data.columns if '_' in col]))
        
        # Фільтруємо дані тільки для тікерів зі всесвіту
        universe_data = data.copy()
        universe_columns = []
        
        for col in universe_data.columns:
            if '_' in col:
                ticker = col.split('_')[0]
                if ticker in universe_tickers:
                    universe_columns.append(col)
        
        universe_data = universe_data[universe_columns]
        
        # Отримуємо універсальний контекст
        universe_context = self.get_universal_context(universe_data)
        
        # Додаємо специфічний аналіз всесвіту
        universe_analysis = {
            **universe_context,
            'universe_size': len(universe_tickers),
            'universe_performance': {},
            'sector_analysis': {},
            'correlation_analysis': {},
            'risk_metrics': {}
        }
        
        # Аналіз перформансу всесвіту
        all_returns = []
        for ticker, context in universe_context['tickers'].items():
            if 'performance' in context:
                perf = context['performance']
                all_returns.append(perf.get('price_change_pct', 0))
        
        if all_returns:
            universe_analysis['universe_performance'] = {
                'average_return': np.mean(all_returns),
                'return_std': np.std(all_returns),
                'positive_returns': sum(1 for r in all_returns if r > 0),
                'negative_returns': sum(1 for r in all_returns if r < 0),
                'best_performer': max(all_returns),
                'worst_performer': min(all_returns)
            }
        
        return universe_analysis 
    def get_ticker_target_timeframe_selection(self, context: Dict[str, Any], ticker: str, target: str, timeframe: str) -> Dict[str, Any]:
        """
        Отримати вибір найважливіших показників для конкретного тікера, таргета, таймфрейму
        
        Args:
            context: Економічний контекст
            ticker: Тікер (наприклад, 'TSLA')
            target: Таргет (наприклад, 'momentum_5d')
            timeframe: Таймфрейм (наприклад, '15m')
            
        Returns:
            Dict: Вибір з важливими показниками для конкретної комбінації
        """
        # Отримуємо базовий вибір
        base_selection = self.get_performance_based_selection(context)
        
        # Отримуємо важливі показники
        important_indicators = self.get_important_indicators(context)
        
        # Отримуємо сигнали показників
        indicator_signals = self.get_indicator_signals(context)
        
        # Створюємо специфічний вибір для тікера/таргета/таймфрейму
        specific_selection = {
            'ticker': ticker,
            'target': target,
            'timeframe': timeframe,
            'market_regime': context['market_regime'],
            'risk_level': context['risk_level'],
            'overall_score': context['overall_score'],
            'important_indicators': important_indicators,
            'indicator_signals': indicator_signals,
            'selection_confidence': self._calculate_selection_confidence(context, ticker, target, timeframe),
            'recommended_actions': self._get_recommended_actions(context, ticker, target, timeframe),
            'risk_adjustments': self._get_risk_adjustments(context, ticker, target, timeframe),
            'performance_expectations': self._get_performance_expectations(context, ticker, target, timeframe)
        }
        
        return specific_selection
    
    def get_all_combinations_selection(self, context: Dict[str, any], 
                                       tickers: List[str], targets: List[str], 
                                       timeframes: List[str]) -> Dict[str, Any]:
        """
        Отримати вибір для всіх комбінацій тікер/таргет/таймфрейм
        
        Args:
            context: Економічний контекст
            tickers: Список тікерів
            targets: Список таргетів
            timeframes: Список таймфреймів
            
        Returns:
            Dict: Вибір для всіх комбінацій з рейтингами
        """
        all_selections = {}
        rankings = {}
        
        # Обробляємо кожну комбінацію
        for ticker in tickers:
            for target in targets:
                for timeframe in timeframes:
                    combination_key = f"{ticker}_{target}_{timeframe}"
                    
                    # Отримуємо специфічний вибір
                    selection = self.get_ticker_target_timeframe_selection(
                        context, ticker, target, timeframe
                    )
                    
                    all_selections[combination_key] = selection
                    
                    # Розраховуємо загальний рейтинг
                    rankings[combination_key] = self._calculate_combination_ranking(selection)
        
        # Сортуємо комбінації за рейтингом
        sorted_combinations = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        
        # Створюємо фінальний результат
        result = {
            'context': context,
            'all_selections': all_selections,
            'rankings': rankings,
            'top_combinations': dict(sorted_combinations[:10]),  # Топ-10
            'bottom_combinations': dict(sorted_combinations[-5:]),  # Найгірші 5
            'best_by_ticker': self._get_best_by_ticker(all_selections, tickers),
            'best_by_target': self._get_best_by_target(all_selections, targets),
            'best_by_timeframe': self._get_best_by_timeframe(all_selections, timeframes),
            'selection_summary': self._generate_selection_summary(all_selections, rankings)
        }
        
        return result
    
    def _calculate_selection_confidence(self, context: Dict[str, any], 
                                       ticker: str, target: str, timeframe: str) -> float:
        """
        Розрахувати впевненість вибору для конкретної комбінації
        
        Args:
            context: Економічний контекст
            ticker: Тікер
            target: Таргет
            timeframe: Таймфрейм
            
        Returns:
            float: Впевненість (0-1)
        """
        base_confidence = 0.5
        
        # Корекція на основі режиму ринку
        regime = context['market_regime']
        if regime == 'bullish':
            base_confidence += 0.2
        elif regime == 'bearish':
            base_confidence += 0.1
        
        # Корекція на основі рівня ризику
        risk_level = context['risk_level']
        if risk_level == 'low':
            base_confidence += 0.1
        elif risk_level == 'high':
            base_confidence -= 0.1
        
        # Корекція на основі загального скору
        overall_score = abs(context['overall_score'])
        base_confidence += min(overall_score, 0.2)
        
        # Корекція на основі таргета
        if 'momentum' in target.lower():
            if regime == 'bullish':
                base_confidence += 0.1
        elif 'volatility' in target.lower():
            if risk_level == 'high':
                base_confidence += 0.1
        
        # Корекція на основі таймфрейму
        if timeframe in ['15m', '5m']:
            if risk_level == 'high':
                base_confidence += 0.05  # Короткі терміни для високого ризику
        elif timeframe in ['1d', '4h']:
            if risk_level == 'low':
                base_confidence += 0.05  # Довгі терміни для низького ризику
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _get_recommended_actions(self, context: Dict[str, any], 
                               ticker: str, target: str, timeframe: str) -> List[str]:
        """
        Отримати рекомендовані дії для комбінації
        
        Args:
            context: Економічний контекст
            ticker: Тікер
            target: Таргет
            timeframe: Таймфрейм
            
        Returns:
            List[str]: Список рекомендацій
        """
        actions = []
        
        regime = context['market_regime']
        risk_level = context['risk_level']
        
        # Базові рекомендації по режиму
        if regime == 'bullish':
            actions.append("Consider long positions")
            actions.append("Monitor for breakout opportunities")
        elif regime == 'bearish':
            actions.append("Consider short positions or cash")
            actions.append("Set tighter stop losses")
        else:
            actions.append("Wait for clear directional signals")
        
        # Рекомендації по ризику
        if risk_level == 'high':
            actions.append("Reduce position size")
            actions.append("Use wider stop losses")
        elif risk_level == 'low':
            actions.append("Can increase position size")
            actions.append("Consider leverage if available")
        
        # Рекомендації по таргету
        if 'momentum' in target.lower():
            actions.append("Monitor momentum indicators")
            actions.append("Watch for trend reversals")
        elif 'volatility' in target.lower():
            actions.append("Use volatility-based position sizing")
            actions.append("Consider options strategies")
        
        # Рекомендації по таймфрейму
        if timeframe in ['15m', '5m']:
            actions.append("Monitor closely for intraday moves")
            actions.append("Use tight profit targets")
        elif timeframe in ['1d', '4h']:
            actions.append("Allow for larger price swings")
            actions.append("Use wider profit targets")
        
        return actions
    
    def _get_risk_adjustments(self, context: Dict[str, any], 
                             ticker: str, target: str, timeframe: str) -> Dict[str, float]:
        """
        Отримати корекції ризику для комбінації
        
        Args:
            context: Економічний контекст
            ticker: Тікер
            target: Таргет
            timeframe: Таймфрейм
            
        Returns:
            Dict: Корекції ризику
        """
        base_adjustments = {
            'position_size_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'risk_per_trade': 0.02
        }
        
        # Корекції по режиму ринку
        regime = context['market_regime']
        if regime == 'bullish':
            base_adjustments['position_size_multiplier'] *= 1.2
            base_adjustments['take_profit_multiplier'] *= 1.1
        elif regime == 'bearish':
            base_adjustments['position_size_multiplier'] *= 0.8
            base_adjustments['stop_loss_multiplier'] *= 0.9
        
        # Корекції по рівню ризику
        risk_level = context['risk_level']
        if risk_level == 'high':
            base_adjustments['position_size_multiplier'] *= 0.7
            base_adjustments['risk_per_trade'] *= 0.8
        elif risk_level == 'low':
            base_adjustments['position_size_multiplier'] *= 1.3
            base_adjustments['risk_per_trade'] *= 1.2
        
        # Корекції по таймфрейму
        if timeframe in ['15m', '5m']:
            base_adjustments['stop_loss_multiplier'] *= 0.8
            base_adjustments['take_profit_multiplier'] *= 0.7
        elif timeframe in ['1d', '4h']:
            base_adjustments['stop_loss_multiplier'] *= 1.2
            base_adjustments['take_profit_multiplier'] *= 1.3
        
        return base_adjustments
    
    def _get_performance_expectations(self, context: Dict[str, any], 
                                     ticker: str, target: str, timeframe: str) -> Dict[str, float]:
        """
        Отримати очікування продуктивності для комбінації
        
        Args:
            context: Економічний контекст
            ticker: Тікер
            target: Таргет
            timeframe: Таймфрейм
            
        Returns:
            Dict: Очікування продуктивності
        """
        base_expectations = {
            'expected_win_rate': 0.55,
            'expected_profit_factor': 1.5,
            'expected_sharpe_ratio': 1.0,
            'expected_max_drawdown': 0.15
        }
        
        # Корекції по режиму ринку
        regime = context['market_regime']
        if regime == 'bullish':
            base_expectations['expected_win_rate'] += 0.05
            base_expectations['expected_profit_factor'] += 0.3
        elif regime == 'bearish':
            base_expectations['expected_win_rate'] -= 0.05
            base_expectations['expected_profit_factor'] -= 0.2
        
        # Корекції по рівню ризику
        risk_level = context['risk_level']
        if risk_level == 'high':
            base_expectations['expected_win_rate'] -= 0.03
            base_expectations['expected_max_drawdown'] += 0.05
        elif risk_level == 'low':
            base_expectations['expected_win_rate'] += 0.03
            base_expectations['expected_max_drawdown'] -= 0.05
        
        # Корекції по таймфрейму
        if timeframe in ['15m', '5m']:
            base_expectations['expected_win_rate'] -= 0.02
            base_expectations['expected_profit_factor'] -= 0.2
        elif timeframe in ['1d', '4h']:
            base_expectations['expected_win_rate'] += 0.02
            base_expectations['expected_profit_factor'] += 0.2
        
        return base_expectations
    
    def _calculate_combination_ranking(self, selection: Dict[str, Any]) -> float:
        """
        Розрахувати загальний рейтинг комбінації
        
        Args:
            selection: Вибір для комбінації
            
        Returns:
            float: Рейтинг (0-1)
        """
        ranking = 0.0
        
        # Впевненість (40%)
        ranking += selection.get('selection_confidence', 0.5) * 0.4
        
        # Очікування продуктивності (30%)
        expectations = selection.get('performance_expectations', {})
        ranking += expectations.get('expected_win_rate', 0.5) * 0.15
        ranking += min(expectations.get('expected_profit_factor', 1.5) / 3.0, 1.0) * 0.15
        
        # Ризикові корекції (20%)
        risk_adj = selection.get('risk_adjustments', {})
        position_mult = risk_adj.get('position_size_multiplier', 1.0)
        ranking += min(position_mult / 1.5, 1.0) * 0.2
        
        # Загальний скор контексту (10%)
        ranking += min(abs(selection.get('overall_score', 0)), 1.0) * 0.1
        
        return min(max(ranking, 0.0), 1.0)
    
    def _get_best_by_ticker(self, selections: Dict[str, Any], tickers: List[str]) -> Dict[str, str]:
        """Отримати найкращі комбінації по тікерах"""
        best_by_ticker = {}
        for ticker in tickers:
            ticker_combinations = {k: v for k, v in selections.items() if k.startswith(ticker + '_')}
            if ticker_combinations:
                best_combination = max(ticker_combinations.items(), 
                                      key=lambda x: self._calculate_combination_ranking(x[1]))
                best_by_ticker[ticker] = best_combination[0]
        return best_by_ticker
    
    def _get_best_by_target(self, selections: Dict[str, Any], targets: List[str]) -> Dict[str, str]:
        """Отримати найкращі комбінації по таргетах"""
        best_by_target = {}
        for target in targets:
            target_combinations = {k: v for k, v in selections.items() if target in k}
            if target_combinations:
                best_combination = max(target_combinations.items(), 
                                      key=lambda x: self._calculate_combination_ranking(x[1]))
                best_by_target[target] = best_combination[0]
        return best_by_target
    
    def _get_best_by_timeframe(self, selections: Dict[str, Any], timeframes: List[str]) -> Dict[str, str]:
        """Отримати найкращі комбінації по таймфреймах"""
        best_by_timeframe = {}
        for timeframe in timeframes:
            tf_combinations = {k: v for k, v in selections.items() if timeframe in k}
            if tf_combinations:
                best_combination = max(tf_combinations.items(), 
                                      key=lambda x: self._calculate_combination_ranking(x[1]))
                best_by_timeframe[timeframe] = best_combination[0]
        return best_by_timeframe
    
    def _generate_selection_summary(self, selections: Dict[str, Any], rankings: Dict[str, float]) -> Dict[str, Any]:
        """Згенерувати підсумок вибору"""
        total_combinations = len(selections)
        avg_ranking = sum(rankings.values()) / len(rankings) if rankings else 0
        
        # Розподіл по рейтингах
        high_ranking = sum(1 for r in rankings.values() if r > 0.7)
        medium_ranking = sum(1 for r in rankings.values() if 0.4 <= r <= 0.7)
        low_ranking = sum(1 for r in rankings.values() if r < 0.4)
        
        return {
            'total_combinations': total_combinations,
            'average_ranking': avg_ranking,
            'high_ranking_combinations': high_ranking,
            'medium_ranking_combinations': medium_ranking,
            'low_ranking_combinations': low_ranking,
            'ranking_distribution': {
                'high': high_ranking / total_combinations if total_combinations > 0 else 0,
                'medium': medium_ranking / total_combinations if total_combinations > 0 else 0,
                'low': low_ranking / total_combinations if total_combinations > 0 else 0
            }
        }
        """
        Отримати важливі показники для поточного контексту
        
        Args:
            context: Економічний контекст
            
        Returns:
            Dict: Важливі показники з порівняннями
        """
        important_indicators = {}
        
        # Вибираємо топ-5 показників за абсолютним значенням score
        indicator_scores = []
        for indicator_name, indicator_data in context['economic_indicators'].items():
            score = abs(indicator_data['score'])
            indicator_scores.append((score, indicator_name, indicator_data))
        
        # Сортуємо за score
        indicator_scores.sort(reverse=True)
        
        # Беремо топ-5
        for score, indicator_name, indicator_data in indicator_scores[:5]:
            important_indicators[indicator_name] = {
                'current_value': indicator_data['current_value'],
                'previous_value': indicator_data['previous_value'],
                'comparison': indicator_data['comparison'],
                'score': indicator_data['score'],
                'weight': indicator_data['weight'],
                'direction': indicator_data['direction'],
                'threshold': self.economic_indicators[indicator_name].threshold,
                'noise_filter': self.economic_indicators[indicator_name].noise_filter,
                'importance_rank': len(important_indicators) + 1,
                'signal_strength': abs(indicator_data['score']) / indicator_data['weight']
            }
        
        return important_indicators
    
    def get_performance_based_selection(self, context: Dict[str, any]) -> Dict[str, any]:
        """
        Отримати вибір на основі продуктивності з урахуванням економічного контексту
        
        Args:
            context: Економічний контекст
            
        Returns:
            Dict: Розширений вибір з важливими показниками
        """
        # Отримуємо базовий вибір на продуктивності
        base_selection = self.get_optimal_model_selection(context)
        
        # Отримуємо важливі показники
        important_indicators = self.get_important_indicators(context)
        
        # Отримуємо сигнали показників
        indicator_signals = self.get_indicator_signals(context)
        
        # Отримуємо трейдинг рекомендації
        trading_recommendations = self.get_trading_recommendations(context)
        
        # Розширюємо вибір
        enhanced_selection = {
            **base_selection,
            'important_indicators': important_indicators,
            'indicator_signals': indicator_signals,
            'trading_recommendations': trading_recommendations,
            'context_enhanced': True,
            'selection_rationale': self._generate_selection_rationale(
                base_selection, important_indicators, context
            )
        }
        
        # Адаптивні корекції на основі важливих показників
        enhanced_selection = self._apply_indicator_based_adjustments(
            enhanced_selection, important_indicators, indicator_signals
        )
        
        return enhanced_selection
    
    def _generate_selection_rationale(self, base_selection: Dict, important_indicators: Dict, context: Dict) -> str:
        """
        Генерувати обґрунтування вибору
        
        Args:
            base_selection: Базовий вибір
            important_indicators: Важливі показники
            context: Економічний контекст
            
        Returns:
            str: Обґрунтування
        """
        rationale_parts = []
        
        # Режим ринку
        regime = context['market_regime']
        rationale_parts.append(f"Режим ринку: {regime}")
        
        # Рівень ризику
        risk_level = context['risk_level']
        rationale_parts.append(f"Рівень ризику: {risk_level}")
        
        # Важливі показники
        if important_indicators:
            top_indicators = list(important_indicators.keys())[:3]
            rationale_parts.append(f"Ключові показники: {', '.join(top_indicators)}")
        
        # Загальний скор
        overall_score = context['overall_score']
        rationale_parts.append(f"Загальний скор: {overall_score:.3f}")
        
        return " | ".join(rationale_parts)
    
    def _apply_indicator_based_adjustments(self, selection: Dict, indicators: Dict, signals: Dict) -> Dict:
        """
        Застосувати корекції на основі показників
        
        Args:
            selection: Поточний вибір
            indicators: Важливі показники
            signals: Сигнали показників
            
        Returns:
            Dict: Скорегований вибір
        """
        adjusted_selection = selection.copy()
        
        # Корекція розміру позиції на основі сигналів
        bull_count = len(signals.get('bullish_signals', []))
        bear_count = len(signals.get('bearish_signals', []))
        
        if bull_count > bear_count * 2:
            adjusted_selection['position_sizing']['max_position_size'] *= 1.2
            adjusted_selection['position_sizing']['risk_per_trade'] *= 1.1
        elif bear_count > bull_count * 2:
            adjusted_selection['position_sizing']['max_position_size'] *= 0.8
            adjusted_selection['position_sizing']['risk_per_trade'] *= 0.9
        
        # Корекція впевненості
        strong_signals = len(signals.get('strong_signals', []))
        if strong_signals > 3:
            adjusted_selection['confidence_score'] = min(0.95, adjusted_selection.get('confidence_score', 0.5) + 0.1)
        
        return adjusted_selection
    
    def get_indicator_signals(self, context: Dict[str, any]) -> Dict[str, Any]:
        """
        Отримати сигнали від показників для прийняття рішень
        
        Args:
            context: Економічний контекст
            
        Returns:
            Dict: Сигнали показників
        """
        signals = {
            'bullish_signals': [],
            'bearish_signals': [],
            'neutral_signals': [],
            'strong_signals': [],
            'weak_signals': []
        }
        
        for indicator_name, indicator_data in context['economic_indicators'].items():
            comparison = indicator_data['comparison']
            score = abs(indicator_data['score'])
            
            # Класифікація сигналів
            if comparison == 1:
                signals['bullish_signals'].append({
                    'indicator': indicator_name,
                    'score': score,
                    'current_value': indicator_data['current_value'],
                    'weight': indicator_data['weight']
                })
            elif comparison == -1:
                signals['bearish_signals'].append({
                    'indicator': indicator_name,
                    'score': score,
                    'current_value': indicator_data['current_value'],
                    'weight': indicator_data['weight']
                })
            else:
                signals['neutral_signals'].append({
                    'indicator': indicator_name,
                    'score': score,
                    'current_value': indicator_data['current_value'],
                    'weight': indicator_data['weight']
                })
            
            # Сильні/слабкі сигнали
            if score > 0.05:
                signals['strong_signals'].append({
                    'indicator': indicator_name,
                    'score': score,
                    'comparison': comparison,
                    'direction': 'bullish' if comparison == 1 else 'bearish' if comparison == -1 else 'neutral'
                })
            elif score < 0.01:
                signals['weak_signals'].append({
                    'indicator': indicator_name,
                    'score': score,
                    'comparison': comparison,
                    'direction': 'bullish' if comparison == 1 else 'bearish' if comparison == -1 else 'neutral'
                })
        
        return signals
    
    def get_trading_recommendations(self, context: Dict[str, any]) -> Dict[str, Any]:
        """
        Отримати рекомендації для трейдингу на основі контексту
        
        Args:
            context: Економічний контекст
            
        Returns:
            Dict: Рекомендації
        """
        signals = self.get_indicator_signals(context)
        regime = context['market_regime']
        risk_level = context['risk_level']
        
        recommendations = {
            'market_regime': regime,
            'risk_level': risk_level,
            'overall_score': context['overall_score'],
            'trading_bias': 'neutral',
            'position_sizing': 'normal',
            'stop_loss_adjustment': 'normal',
            'take_profit_adjustment': 'normal',
            'indicator_summary': signals,
            'key_signals': []
        }
        
        # Визначаємо трейдинг біас
        bull_count = len(signals['bullish_signals'])
        bear_count = len(signals['bearish_signals'])
        
        if bull_count > bear_count * 1.5:
            recommendations['trading_bias'] = 'bullish'
            recommendations['position_sizing'] = 'aggressive' if risk_level == 'high' else 'normal'
        elif bear_count > bull_count * 1.5:
            recommendations['trading_bias'] = 'bearish'
            recommendations['position_sizing'] = 'conservative'
        else:
            recommendations['trading_bias'] = 'neutral'
        
        # Визначаємо ключові сигнали
        top_bullish = sorted(signals['bullish_signals'], key=lambda x: x['score'], reverse=True)[:2]
        top_bearish = sorted(signals['bearish_signals'], key=lambda x: x['score'], reverse=True)[:2]
        
        recommendations['key_signals'] = {
            'top_bullish': top_bullish,
            'top_bearish': top_bearish,
            'strong_signals': signals['strong_signals'][:3],
            'weak_signals': signals['weak_signals'][:3]
        }
        
        return recommendations
    
    def add_context_to_training_data(self, training_data: pd.DataFrame) -> pd.DataFrame:
        """
        Додати економічний контекст до тренувальних data
        
        Args:
            training_data: Тренувальні дані
            
        Returns:
            pd.DataFrame: Дані з доданим контекстом
        """
        enhanced_data = training_data.copy()
        
        # Додаємо економічні показники як фічі
        for indicator_name in self.economic_indicators.keys():
            if indicator_name in enhanced_data.columns:
                # Створюємо лагові значення для порівняння
                enhanced_data[f'{indicator_name}_lag1'] = enhanced_data[indicator_name].shift(1)
                
                # Створюємо порівняльні фічі
                enhanced_data[f'{indicator_name}_comparison'] = enhanced_data.apply(
                    lambda row: self.compare_indicator_values(
                        row[indicator_name], 
                        row[f'{indicator_name}_lag1'],
                        self.economic_indicators[indicator_name]
                    ), axis=1
                )
                
                # Створюємо зважені фічі
                weight = self.economic_indicators[indicator_name].weight
                enhanced_data[f'{indicator_name}_weighted'] = (
                    enhanced_data[f'{indicator_name}_comparison'] * weight
                )
        
        # Додаємо часові показники
        if 'date' in enhanced_data.columns:
            enhanced_data['date'] = pd.to_datetime(enhanced_data['date'])
            enhanced_data['weekday'] = enhanced_data['date'].dt.weekday
            enhanced_data['hour_of_day'] = enhanced_data['date'].dt.hour
            enhanced_data['month'] = enhanced_data['date'].dt.month
            enhanced_data['quarter'] = enhanced_data['date'].dt.quarter
            enhanced_data['is_market_hours'] = (
                enhanced_data['hour_of_day'].between(9, 16) & 
                enhanced_data['weekday'].between(0, 4)
            ).astype(int)
        
        # Створюємо загальний контекстний скор
        context_columns = []
        for indicator_name in self.economic_indicators.keys():
            weighted_col = f'{indicator_name}_weighted'
            if weighted_col in enhanced_data.columns:
                context_columns.append(weighted_col)
        
        if context_columns:
            enhanced_data['economic_context_score'] = enhanced_data[context_columns].sum(axis=1)
        
        self.logger.info(f"Added economic context features to training data: {len(context_columns)} indicators")
        
        return enhanced_data
    
    def get_optimal_model_selection(self, context: Dict[str, any]) -> Dict[str, any]:
        """
        Отримати оптимальний вибір моделі на основі контексту
        
        Args:
            context: Економічний контекст
            
        Returns:
            Dict: Рекомендації по моделях
        """
        regime = context['market_regime']
        risk_level = context['risk_level']
        overall_score = context['overall_score']
        
        recommendations = {
            'regime': regime,
            'risk_level': risk_level,
            'model_preferences': {},
            'target_preferences': {},
            'timeframe_preferences': {},
            'position_sizing': {}
        }
        
        # Рекомендації по моделях
        if regime == 'bullish':
            recommendations['model_preferences'] = {
                'primary': ['LGBM', 'XGBoost', 'Ensemble'],
                'secondary': ['RandomForest', 'MLP'],
                'avoid': ['MeanReversion']
            }
            recommendations['target_preferences'] = {
                'primary': ['momentum_5d', 'trend_strength'],
                'secondary': ['volatility_20d', 'breakout_probability']
            }
        elif regime == 'bearish':
            recommendations['model_preferences'] = {
                'primary': ['LSTM', 'GRU', 'Transformer'],
                'secondary': ['Ensemble', 'LGBM'],
                'avoid': ['Momentum']
            }
            recommendations['target_preferences'] = {
                'primary': ['volatility_5d', 'max_drawdown'],
                'secondary': ['mean_reversion', 'support_resistance']
            }
        else:  # neutral
            recommendations['model_preferences'] = {
                'primary': ['Ensemble', 'LGBM', 'XGBoost'],
                'secondary': ['RandomForest', 'MLP'],
                'avoid': []
            }
            recommendations['target_preferences'] = {
                'primary': ['price_change_5d', 'direction_5d'],
                'secondary': ['volatility_ratio', 'trend_strength']
            }
        
        # Рекомендації по таймфреймах
        if risk_level == 'high':
            recommendations['timeframe_preferences'] = {
                'primary': ['15m', '1h'],
                'secondary': ['4h'],
                'avoid': ['1d']
            }
        elif risk_level == 'medium':
            recommendations['timeframe_preferences'] = {
                'primary': ['1h', '4h'],
                'secondary': ['15m', '1d'],
                'avoid': []
            }
        else:  # low
            recommendations['timeframe_preferences'] = {
                'primary': ['4h', '1d'],
                'secondary': ['1h'],
                'avoid': ['15m']
            }
        
        # Розміри позицій
        if risk_level == 'high':
            recommendations['position_sizing'] = {
                'max_position_size': 0.05,  # 5%
                'risk_per_trade': 0.01,    # 1%
                'max_positions': 3
            }
        elif risk_level == 'medium':
            recommendations['position_sizing'] = {
                'max_position_size': 0.08,  # 8%
                'risk_per_trade': 0.02,    # 2%
                'max_positions': 5
            }
        else:  # low
            recommendations['position_sizing'] = {
                'max_position_size': 0.12,  # 12%
                'risk_per_trade': 0.03,    # 3%
                'max_positions': 8
            }
        
        return recommendations


# Глобальний екземпляр
economic_context_mapper = EconomicContextMapper()


def get_economic_context(current_data: Dict[str, float], 
                       historical_data: Optional[pd.DataFrame] = None) -> Dict[str, any]:
    """Отримати економічний контекст"""
    return economic_context_mapper.get_economic_context(current_data, historical_data)


def add_economic_context_to_training(training_data: pd.DataFrame) -> pd.DataFrame:
    """Додати економічний контекст до тренувальних data"""
    return economic_context_mapper.add_context_to_training_data(training_data)


def get_optimal_model_selection(context: Dict[str, any]) -> Dict[str, any]:
    """Отримати оптимальний вибір моделі"""
    return economic_context_mapper.get_optimal_model_selection(context)


if __name__ == "__main__":
    # Приклад використання
    logging.basicConfig(level=logging.INFO)
    
    print("[SEARCH] Economic Context Mapper Test")
    print("="*50)
    
    # Тестові дані
    current_data = {
        'fedfunds': 5.25,
        't10y2y': 0.8,
        'vix': 18.5,
        'unrate': 3.8,
        'cpi': 298.4,
        'weekday': 2,
        'hour_of_day': 14,
        'month': 10,
        'quarter': 4
    }
    
    # Отримуємо контекст
    context = get_economic_context(current_data)
    
    print(f"[DATA] Economic Context:")
    print(f"   Overall Score: {context['overall_score']:.3f}")
    print(f"   Market Regime: {context['market_regime']}")
    print(f"   Risk Level: {context['risk_level']}")
    
    print(f"\n[TARGET] Model Recommendations:")
    model_rec = get_optimal_model_selection(context)
    print(f"   Primary Models: {', '.join(model_rec['model_preferences']['primary'])}")
    print(f"   Primary Targets: {', '.join(model_rec['target_preferences']['primary'])}")
    print(f"   Primary Timeframes: {', '.join(model_rec['timeframe_preferences']['primary'])}")
    print(f"   Max Position Size: {model_rec['position_sizing']['max_position_size']:.1%}")
    
    print(f"\n[OK] Economic Context Mapper working correctly!")
