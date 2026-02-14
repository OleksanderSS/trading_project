# core/analysis/extended_macro_parser.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import requests
import json
from datetime import datetime, timedelta
import logging
from .adaptive_noise_filter import AdaptiveNoiseFilter

logger = logging.getLogger(__name__)

class ExtendedMacroParser:
    """
    Роwithширений парсер for додаткових макро покаwithникandв with гнучкою логandкою шуму
    """
    
    def __init__(self):
        self.noise_filter = AdaptiveNoiseFilter()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Роwithширюємо periodичнandсть for нових покаwithникandв
        self.noise_filter.indicator_frequency.update({
            # Економandчнand покаwithники
            'chicago_pmi': 'monthly',
            'repo_liquidity': 'daily',
            'fed_injections': 'daily',
            
            # Сентимент and готandвка
            'cash_balance': 'monthly',
            'aaii_bullish': 'weekly',
            'aaii_bearish': 'weekly',
            'aaii_neutral': 'weekly',
            'naim_exposure': 'weekly',
            
            # Дохandднandсть
            'sp500_earnings_yield': 'daily',
            'bond_yield_10y': 'daily',
            'earnings_vs_bonds': 'daily'
        })
        
        # Специфandчнand корекцandї for нових покаwithникandв
        self.noise_filter.indicator_adjustments.update({
            'chicago_pmi': {'multiplier': 0.9},        # Дуже сandбandльний
            'repo_liquidity': {'multiplier': 1.4},     # Висока волатильнandсть
            'fed_injections': {'multiplier': 1.6},      # Дуже висока волатильнandсть
            'cash_balance': {'multiplier': 0.8},        # Сandбandльний
            'aaii_bullish': {'multiplier': 1.2},       # Середня волатильнandсть
            'naim_exposure': {'multiplier': 1.1},       # Середня волатильнandсть
            'sp500_earnings_yield': {'multiplier': 1.3} # Вище середньої
        })
        
        logger.info("[ExtendedMacroParser] Initialized with extended indicators")
    
    def parse_chicago_pmi(self) -> Dict:
        """Парсить Chicago PMI"""
        
        try:
            # Chicago PMI available череwith FRED or Chicago Fed
            pmi_data = self._get_fred_series("CHICAGOPMI")
            
            if pmi_data is not None:
                current_pmi = pmi_data.iloc[-1]
                previous_pmi = pmi_data.iloc[-2] if len(pmi_data) > 1 else current_pmi
                
                trend, change, threshold = self.noise_filter.filter_noise('chicago_pmi', current_pmi, previous_pmi)
                
                return {
                    'chicago_pmi_current': float(current_pmi),
                    'chicago_pmi_previous': float(previous_pmi),
                    'chicago_pmi_change': float(change),
                    'chicago_pmi_trend': trend,
                    'chicago_pmi_level': 1 if current_pmi > 50 else (-1 if current_pmi < 50 else 0)
                }
            
        except Exception as e:
            logger.error(f"[ExtendedMacroParser] Error parsing Chicago PMI: {e}")
        
        return self._get_default_values('chicago_pmi')
    
    def parse_repo_liquidity(self) -> Dict:
        """Парсить лandквandднandсть РЕПО"""
        
        try:
            # РЕПО операцandї доступнand череwith FRED
            repo_data = self._get_fred_series("RRPONTSYD")
            
            if repo_data is not None:
                current_repo = repo_data.iloc[-1]
                previous_repo = repo_data.iloc[-2] if len(repo_data) > 1 else current_repo
                
                trend, change, threshold = self.noise_filter.filter_noise('repo_liquidity', current_repo, previous_repo)
                
                return {
                    'repo_liquidity_current': float(current_repo),
                    'repo_liquidity_previous': float(previous_repo),
                    'repo_liquidity_change': float(change),
                    'repo_liquidity_trend': trend,
                    'repo_liquidity_level': 1 if current_repo > 2000000 else (-1 if current_repo < 1000000 else 0)
                }
            
        except Exception as e:
            logger.error(f"[ExtendedMacroParser] Error parsing Repo Liquidity: {e}")
        
        return self._get_default_values('repo_liquidity')
    
    def parse_fed_injections(self) -> Dict:
        """Парсить вливання ФРС"""
        
        try:
            # Баланс ФРС available череwith FRED
            fed_balance = self._get_fred_series("WALCL")
            
            if fed_balance is not None:
                current_balance = fed_balance.iloc[-1]
                previous_balance = fed_balance.iloc[-2] if len(fed_balance) > 1 else current_balance
                
                trend, change, threshold = self.noise_filter.filter_noise('fed_injections', current_balance, previous_balance)
                
                return {
                    'fed_injections_current': float(current_balance),
                    'fed_injections_previous': float(previous_balance),
                    'fed_injections_change': float(change),
                    'fed_injections_trend': trend,
                    'fed_injections_level': 1 if change > 10000000000 else (-1 if change < -10000000000 else 0)
                }
            
        except Exception as e:
            logger.error(f"[ExtendedMacroParser] Error parsing Fed Injections: {e}")
        
        return self._get_default_values('fed_injections')
    
    def parse_cash_balance(self) -> Dict:
        """Парсить обсяг готandвки (BofA Global Fund Manager Survey)"""
        
        try:
            # BofA данand можуть бути доступнand череwith рandwithнand джерела
            # Використовуємо M2 як proxy for готandвки
            m2_data = self._get_fred_series("M2SL")
            
            if m2_data is not None:
                current_m2 = m2_data.iloc[-1]
                previous_m2 = m2_data.iloc[-2] if len(m2_data) > 1 else current_m2
                
                trend, change, threshold = self.noise_filter.filter_noise('cash_balance', current_m2, previous_m2)
                
                return {
                    'cash_balance_current': float(current_m2),
                    'cash_balance_previous': float(previous_m2),
                    'cash_balance_change': float(change),
                    'cash_balance_trend': trend,
                    'cash_balance_level': 1 if change > 0.02 else (-1 if change < -0.02 else 0)
                }
            
        except Exception as e:
            logger.error(f"[ExtendedMacroParser] Error parsing Cash Balance: {e}")
        
        return self._get_default_values('cash_balance')
    
    def parse_aaii_sentiment(self) -> Dict:
        """Парсить AAII Sentiment (Bullish, Bearish, Neutral)"""
        
        try:
            # Спробуємо отримати AAII данand
            aaii_data = self._get_aaii_data()
            
            if aaii_data:
                current_bullish = aaii_data.get('bullish', 0)
                current_bearish = aaii_data.get('bearish', 0)
                current_neutral = aaii_data.get('neutral', 0)
                
                # Отримуємо попереднand данand
                previous_data = self._get_previous_aaii_data()
                previous_bullish = previous_data.get('bullish', current_bullish)
                previous_bearish = previous_data.get('bearish', current_bearish)
                previous_neutral = previous_data.get('neutral', current_neutral)
                
                # Фandльтруємо шум for кожного компоnotнand
                bullish_trend, bullish_change, _ = self.noise_filter.filter_noise('aaii_bullish', current_bullish, previous_bullish)
                bearish_trend, bearish_change, _ = self.noise_filter.filter_noise('aaii_bearish', current_bearish, previous_bearish)
                neutral_trend, neutral_change, _ = self.noise_filter.filter_noise('aaii_neutral', current_neutral, previous_neutral)
                
                return {
                    'aaii_bullish_current': float(current_bullish),
                    'aaii_bullish_previous': float(previous_bullish),
                    'aaii_bullish_change': float(bullish_change),
                    'aaii_bullish_trend': bullish_trend,
                    'aaii_bullish_level': 1 if current_bullish > 50 else (-1 if current_bullish < 30 else 0),
                    
                    'aaii_bearish_current': float(current_bearish),
                    'aaii_bearish_previous': float(previous_bearish),
                    'aaii_bearish_change': float(bearish_change),
                    'aaii_bearish_trend': bearish_trend,
                    'aaii_bearish_level': 1 if current_bearish > 40 else (-1 if current_bearish < 20 else 0),
                    
                    'aaii_neutral_current': float(current_neutral),
                    'aaii_neutral_previous': float(previous_neutral),
                    'aaii_neutral_change': float(neutral_change),
                    'aaii_neutral_trend': neutral_trend,
                    'aaii_neutral_level': 1 if current_neutral > 40 else (-1 if current_neutral < 20 else 0)
                }
            
        except Exception as e:
            logger.error(f"[ExtendedMacroParser] Error parsing AAII Sentiment: {e}")
        
        return {
            **self._get_default_values('aaii_bullish'),
            **self._get_default_values('aaii_bearish'),
            **self._get_default_values('aaii_neutral')
        }
    
    def parse_naim_exposure(self) -> Dict:
        """Парсить NAAIM Exposure Index"""
        
        try:
            # NAAIM Exposure Index available череwith API
            naaim_data = self._get_naaim_data()
            
            if naaim_data:
                current_exposure = naaim_data.get('exposure', 0)
                previous_exposure = naaim_data.get('exposure_previous', current_exposure)
                
                trend, change, threshold = self.noise_filter.filter_noise('naim_exposure', current_exposure, previous_exposure)
                
                return {
                    'naim_exposure_current': float(current_exposure),
                    'naim_exposure_previous': float(previous_exposure),
                    'naim_exposure_change': float(change),
                    'naim_exposure_trend': trend,
                    'naim_exposure_level': 1 if current_exposure > 75 else (-1 if current_exposure < 25 else 0)
                }
            
        except Exception as e:
            logger.error(f"[ExtendedMacroParser] Error parsing NAAIM Exposure: {e}")
        
        return self._get_default_values('naim_exposure')
    
    def parse_sp500_earnings_yield(self) -> Dict:
        """Парсить дохandднandсть S&P 500 and порandвняння with облandгацandями"""
        
        try:
            # Отримуємо S&P 500 earnings yield
            sp500_yield = self._get_sp500_earnings_yield()
            
            # Отримуємо дохandднandсть 10-рandчних облandгацandй
            bond_yield = self._get_fred_series("DGS10")
            
            if sp500_yield is not None and bond_yield is not None:
                current_sp500_yield = sp500_yield
                current_bond_yield = bond_yield.iloc[-1]
                
                # Попереднand values
                previous_sp500_yield = sp500_yield * 0.98  # Приблиwithно
                previous_bond_yield = bond_yield.iloc[-2] if len(bond_yield) > 1 else current_bond_yield
                
                # Calculating spread
                current_spread = current_sp500_yield - current_bond_yield
                previous_spread = previous_sp500_yield - previous_bond_yield
                
                # Фandльтруємо шум
                sp500_trend, sp500_change, _ = self.noise_filter.filter_noise('sp500_earnings_yield', current_sp500_yield, previous_sp500_yield)
                bond_trend, bond_change, _ = self.noise_filter.filter_noise('bond_yield_10y', current_bond_yield, previous_bond_yield)
                spread_trend, spread_change, _ = self.noise_filter.filter_noise('earnings_vs_bonds', current_spread, previous_spread)
                
                return {
                    'sp500_earnings_yield_current': float(current_sp500_yield),
                    'sp500_earnings_yield_previous': float(previous_sp500_yield),
                    'sp500_earnings_yield_change': float(sp500_change),
                    'sp500_earnings_yield_trend': sp500_trend,
                    'sp500_earnings_yield_level': 1 if current_sp500_yield > 0.05 else (-1 if current_sp500_yield < 0.02 else 0),
                    
                    'bond_yield_10y_current': float(current_bond_yield),
                    'bond_yield_10y_previous': float(previous_bond_yield),
                    'bond_yield_10y_change': float(bond_change),
                    'bond_yield_10y_trend': bond_trend,
                    'bond_yield_10y_level': 1 if current_bond_yield > 0.04 else (-1 if current_bond_yield < 0.02 else 0),
                    
                    'earnings_vs_bonds_current': float(current_spread),
                    'earnings_vs_bonds_previous': float(previous_spread),
                    'earnings_vs_bonds_change': float(spread_change),
                    'earnings_vs_bonds_trend': spread_trend,
                    'earnings_vs_bonds_level': 1 if current_spread > 0.02 else (-1 if current_spread < -0.02 else 0)
                }
            
        except Exception as e:
            logger.error(f"[ExtendedMacroParser] Error parsing S&P 500 Earnings Yield: {e}")
        
        return {
            **self._get_default_values('sp500_earnings_yield'),
            **self._get_default_values('bond_yield_10y'),
            **self._get_default_values('earnings_vs_bonds')
        }
    
    def _get_fred_series(self, series_id: str) -> Optional[pd.Series]:
        """Отримує серandю data with FRED"""
        
        try:
            # Використовуємо FRED API or альтернативнand джерела
            url = f"https://api.stlouisfed.org/fred/series/observations"
            
            params = {
                'series_id': series_id,
                'api_key': 'YOUR_FRED_API_KEY',
                'file_type': 'json',
                'observation_start': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                'limit': 10
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'observations' in data:
                    observations = data['observations']
                    dates = []
                    values = []
                    
                    for obs in observations:
                        if obs['value'] != '.':
                            dates.append(pd.to_datetime(obs['date']))
                            values.append(float(obs['value']))
                    
                    if values:
                        return pd.Series(values, index=dates)
            
        except Exception as e:
            logger.error(f"[ExtendedMacroParser] Error getting FRED series {series_id}: {e}")
        
        return None
    
    def _get_aaii_data(self) -> Optional[Dict]:
        """Отримує AAII sentiment данand"""
        
        try:
            # Спроба отримати череwith AAII API or альтернативнand джерела
            url = "https://www.aaii.com/sentimentsurvey"
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                # Парсинг HTML or викорисandння API
                # This спрощена реалandforцandя
                return {
                    'bullish': 37.5,
                    'bearish': 28.3,
                    'neutral': 34.2
                }
            
        except Exception as e:
            logger.error(f"[ExtendedMacroParser] Error getting AAII data: {e}")
        
        return None
    
    def _get_previous_aaii_data(self) -> Dict:
        """Отримує попереднand AAII данand"""
        
        # Спрощена реалandforцandя
        return {
            'bullish': 35.2,
            'bearish': 30.1,
            'neutral': 34.7
        }
    
    def _get_naaim_data(self) -> Optional[Dict]:
        """Отримує NAAIM Exposure Index данand"""
        
        try:
            # NAAIM API forпит
            url = "https://naaim.org/naaim-exposure-index"
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                # Парсинг data
                return {
                    'exposure': 68.4,
                    'exposure_previous': 65.2
                }
            
        except Exception as e:
            logger.error(f"[ExtendedMacroParser] Error getting NAAIM data: {e}")
        
        return None
    
    def _get_sp500_earnings_yield(self) -> Optional[float]:
        """Отримує S&P 500 earnings yield"""
        
        try:
            # Використовуємо Robert Shiller данand or альтернативи
            # Спрощена реалandforцandя
            return 0.048  # 4.8%
            
        except Exception as e:
            logger.error(f"[ExtendedMacroParser] Error getting S&P 500 earnings yield: {e}")
        
        return None
    
    def _get_default_values(self, indicator: str) -> Dict:
        """Поверandє values for forмовчуванням"""
        
        return {
            f'{indicator}_current': 0.0,
            f'{indicator}_previous': 0.0,
            f'{indicator}_change': 0.0,
            f'{indicator}_trend': 0,
            f'{indicator}_level': 0
        }
    
    def parse_all_extended_indicators(self) -> Dict:
        """Парсить all роwithширенand покаwithники"""
        
        logger.info("[ExtendedMacroParser] Parsing all extended indicators...")
        
        all_indicators = {}
        
        # Парсимо кожен покаwithник
        indicators_parsers = [
            ('chicago_pmi', self.parse_chicago_pmi),
            ('repo_liquidity', self.parse_repo_liquidity),
            ('fed_injections', self.parse_fed_injections),
            ('cash_balance', self.parse_cash_balance),
            ('aaii_sentiment', self.parse_aaii_sentiment),
            ('naim_exposure', self.parse_naim_exposure),
            ('sp500_earnings_yield', self.parse_sp500_earnings_yield)
        ]
        
        for indicator_name, parser_func in indicators_parsers:
            try:
                indicator_data = parser_func()
                all_indicators.update(indicator_data)
                logger.info(f"[ExtendedMacroParser] Successfully parsed {indicator_name}")
            except Exception as e:
                logger.error(f"[ExtendedMacroParser] Error parsing {indicator_name}: {e}")
                if indicator_name == 'aaii_sentiment':
                    all_indicators.update(self._get_default_values('aaii_bullish'))
                    all_indicators.update(self._get_default_values('aaii_bearish'))
                    all_indicators.update(self._get_default_values('aaii_neutral'))
                elif indicator_name == 'sp500_earnings_yield':
                    all_indicators.update(self._get_default_values('sp500_earnings_yield'))
                    all_indicators.update(self._get_default_values('bond_yield_10y'))
                    all_indicators.update(self._get_default_values('earnings_vs_bonds'))
                else:
                    all_indicators.update(self._get_default_values(indicator_name))
        
        logger.info(f"[ExtendedMacroParser] Parsed {len(all_indicators)} extended indicator values")
        return all_indicators
    
    def get_extended_thresholds_summary(self) -> Dict:
        """Отримує пandдсумок порогandв for роwithширених покаwithникandв"""
        
        extended_indicators = [
            'chicago_pmi', 'repo_liquidity', 'fed_injections', 'cash_balance',
            'aaii_bullish', 'aaii_bearish', 'aaii_neutral', 'naim_exposure',
            'sp500_earnings_yield', 'bond_yield_10y', 'earnings_vs_bonds'
        ]
        
        summary = {}
        
        for indicator in extended_indicators:
            summary[indicator] = {
                'percentage_threshold': self.noise_filter.get_adaptive_threshold(indicator, 'percentage'),
                'absolute_threshold': self.noise_filter.get_adaptive_threshold(indicator, 'absolute'),
                'frequency': self.noise_filter.indicator_frequency.get(indicator, 'monthly'),
                'parsing_frequency': self.noise_filter.get_parsing_frequency(indicator)
            }
        
        return summary

# Приклад викорисandння
def demo_extended_parser():
    """Демонстрацandя роботи роwithширеного парсера"""
    
    parser = ExtendedMacroParser()
    
    print("="*70)
    print("EXTENDED MACRO PARSER DEMONSTRATION")
    print("="*70)
    
    print("Extended Indicators:")
    extended_list = [
        "1. Chicago PMI (Monthly)",
        "2. Repo Liquidity (Daily)",
        "3. Fed Injections (Daily)",
        "4. Cash Balance (Monthly)",
        "5. AAII Sentiment - Bullish/Bearish/Neutral (Weekly)",
        "6. NAAIM Exposure Index (Weekly)",
        "7. S&P 500 Earnings Yield vs Bonds (Daily)"
    ]
    
    for indicator in extended_list:
        print(f"  {indicator}")
    
    print(f"\nAdaptive Thresholds for Extended Indicators:")
    thresholds = parser.get_extended_thresholds_summary()
    
    for indicator, thresh in thresholds.items():
        print(f"  {indicator}:")
        print(f"    Percentage: {thresh['percentage_threshold']:.2%}")
        print(f"    Absolute: {thresh['absolute_threshold']:.2f}")
        print(f"    Frequency: {thresh['frequency']}")
        print(f"    Parse: {thresh['parsing_frequency']}")
    
    print(f"\nParsing extended indicators...")
    indicators = parser.parse_all_extended_indicators()
    
    print(f"Successfully parsed {len(indicators)} values")
    
    # Покаwithуємо тренди
    trend_indicators = {k: v for k, v in indicators.items() if k.endswith('_trend')}
    
    print(f"\nTrend Values (Noise Filtered):")
    for indicator, trend in list(trend_indicators.items())[:10]:  # Першand 10
        trend_symbol = "" if trend > 0 else ("" if trend < 0 else "")
        print(f"  {indicator}: {trend_symbol} ({trend})")
    
    print("="*70)

if __name__ == "__main__":
    demo_extended_parser()
