# core/analysis/comprehensive_macro_parser.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import requests
import json
from datetime import datetime, timedelta
import logging
from .adaptive_noise_filter import AdaptiveNoiseFilter

logger = logging.getLogger(__name__)

class ComprehensiveMacroParser:
    """
    Комплексний парсер with усandма рекомендованими покаwithниками
    """
    
    def __init__(self):
        self.noise_filter = AdaptiveNoiseFilter()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Додаємо новand покаwithники до адаптивного фandльтра
        self.noise_filter.indicator_frequency.update({
            # Додатковand покаwithники
            'margin_debt': 'monthly',
            'michigan_consumer_sentiment': 'monthly',
            'cpi_vs_core_cpi': 'monthly',
            'investor_positioning_cash': 'weekly'
        })
        
        # Специфandчнand корекцandї for нових покаwithникandв
        self.noise_filter.indicator_adjustments.update({
            'margin_debt': {'multiplier': 1.3},              # Висока волатильнandсть
            'michigan_consumer_sentiment': {'multiplier': 1.2}, # Середня волатильнandсть
            'cpi_vs_core_cpi': {'multiplier': 0.8},           # Нижча волатильнandсть
            'investor_positioning_cash': {'multiplier': 1.1}    # Середня волатильнandсть
        })
        
        logger.info("[ComprehensiveMacroParser] Initialized with all indicators")
    
    def parse_margin_debt(self) -> Dict:
        """Парсить Маржинальний борг (Margin Debt)"""
        
        try:
            # FINRA/NYSE данand про маржинальний борг
            margin_data = self._get_margin_debt_data()
            
            if margin_data:
                current_margin = margin_data.get('margin_debt', 0)
                previous_margin = margin_data.get('margin_debt_previous', current_margin)
                
                # Calculating withмandну
                if previous_margin != 0:
                    margin_change = (current_margin - previous_margin) / abs(previous_margin)
                else:
                    margin_change = 0.0
                
                # Фandльтруємо шум
                trend, change, threshold = self.noise_filter.filter_noise(
                    'margin_debt', current_margin, previous_margin
                )
                
                return {
                    'margin_debt_current': float(current_margin),
                    'margin_debt_previous': float(previous_margin),
                    'margin_debt_change': float(margin_change),
                    'margin_debt_trend': trend,
                    'margin_debt_level': 1 if current_margin > 800000000000 else (-1 if current_margin < 500000000000 else 0)
                }
            
        except Exception as e:
            logger.error(f"[ComprehensiveMacroParser] Error parsing Margin Debt: {e}")
        
        return self._get_default_values('margin_debt')
    
    def parse_michigan_consumer_sentiment(self) -> Dict:
        """Парсить Michigan Consumer Sentiment (MCC)"""
        
        try:
            # University of Michigan данand
            michigan_data = self._get_michigan_data()
            
            if michigan_data:
                current_michigan = michigan_data.get('sentiment_index', 0)
                previous_michigan = michigan_data.get('sentiment_index_previous', current_michigan)
                
                # Фandльтруємо шум
                trend, change, threshold = self.noise_filter.filter_noise(
                    'michigan_consumer_sentiment', current_michigan, previous_michigan
                )
                
                return {
                    'michigan_consumer_sentiment_current': float(current_michigan),
                    'michigan_consumer_sentiment_previous': float(previous_michigan),
                    'michigan_consumer_sentiment_change': float(change),
                    'michigan_consumer_sentiment_trend': trend,
                    'michigan_consumer_sentiment_level': 1 if current_michigan > 85 else (-1 if current_michigan < 65 else 0)
                }
            
        except Exception as e:
            logger.error(f"[ComprehensiveMacroParser] Error parsing Michigan Consumer Sentiment: {e}")
        
        return self._get_default_values('michigan_consumer_sentiment')
    
    def parse_cpi_vs_core_cpi(self) -> Dict:
        """Парсить CPI vs Core CPI vs Truflation"""
        
        try:
            # Отримуємо all три покаwithники
            cpi_data = self._get_cpi_data()
            core_cpi_data = self._get_core_cpi_data()
            truflation_data = self._get_truflation_data()
            
            if cpi_data and core_cpi_data and truflation_data:
                current_cpi = cpi_data.get('cpi', 0)
                current_core_cpi = core_cpi_data.get('core_cpi', 0)
                current_truflation = truflation_data.get('truflation', 0)
                
                # Попереднand values
                previous_cpi = cpi_data.get('cpi_previous', current_cpi)
                previous_core_cpi = core_cpi_data.get('core_cpi_previous', current_core_cpi)
                previous_truflation = truflation_data.get('truflation_previous', current_truflation)
                
                # Calculating спреди
                cpi_vs_core = current_cpi - current_core_cpi
                cpi_vs_truflation = current_cpi - current_truflation
                core_vs_truflation = current_core_cpi - current_truflation
                
                # Попереднand спреди
                previous_cpi_vs_core = previous_cpi - previous_core_cpi
                previous_cpi_vs_truflation = previous_cpi - previous_truflation
                previous_core_vs_truflation = previous_core_cpi - previous_truflation
                
                # Фandльтруємо шум for спредandв
                cpi_core_trend, cpi_core_change, _ = self.noise_filter.filter_noise(
                    'cpi_vs_core_cpi', cpi_vs_core, previous_cpi_vs_core
                )
                cpi_truflation_trend, cpi_truflation_change, _ = self.noise_filter.filter_noise(
                    'cpi_vs_core_cpi', cpi_vs_truflation, previous_cpi_vs_truflation
                )
                core_truflation_trend, core_truflation_change, _ = self.noise_filter.filter_noise(
                    'cpi_vs_core_cpi', core_vs_truflation, previous_core_vs_truflation
                )
                
                return {
                    # CPI
                    'cpi_current': float(current_cpi),
                    'cpi_previous': float(previous_cpi),
                    'cpi_trend': 1 if current_cpi > previous_cpi else (-1 if current_cpi < previous_cpi else 0),
                    
                    # Core CPI
                    'core_cpi_current': float(current_core_cpi),
                    'core_cpi_previous': float(previous_core_cpi),
                    'core_cpi_trend': 1 if current_core_cpi > previous_core_cpi else (-1 if current_core_cpi < previous_core_cpi else 0),
                    
                    # Truflation
                    'truflation_current': float(current_truflation),
                    'truflation_previous': float(previous_truflation),
                    'truflation_trend': 1 if current_truflation > previous_truflation else (-1 if current_truflation < previous_truflation else 0),
                    
                    # Спреди
                    'cpi_vs_core_cpi_current': float(cpi_vs_core),
                    'cpi_vs_core_cpi_previous': float(previous_cpi_vs_core),
                    'cpi_vs_core_cpi_change': float(cpi_core_change),
                    'cpi_vs_core_cpi_trend': cpi_core_trend,
                    'cpi_vs_core_cpi_level': 1 if abs(cpi_vs_core) > 0.5 else 0,
                    
                    'cpi_vs_truflation_current': float(cpi_vs_truflation),
                    'cpi_vs_truflation_previous': float(previous_cpi_vs_truflation),
                    'cpi_vs_truflation_change': float(cpi_truflation_change),
                    'cpi_vs_truflation_trend': cpi_truflation_trend,
                    'cpi_vs_truflation_level': 1 if abs(cpi_vs_truflation) > 0.3 else 0,
                    
                    'core_vs_truflation_current': float(core_vs_truflation),
                    'core_vs_truflation_previous': float(previous_core_vs_truflation),
                    'core_vs_truflation_change': float(core_truflation_change),
                    'core_vs_truflation_trend': core_truflation_trend,
                    'core_vs_truflation_level': 1 if abs(core_vs_truflation) > 0.3 else 0
                }
            
        except Exception as e:
            logger.error(f"[ComprehensiveMacroParser] Error parsing CPI vs Core CPI vs Truflation: {e}")
        
        return {
            **self._get_default_values('cpi_vs_core_cpi'),
            **self._get_default_values('cpi_vs_truflation'),
            **self._get_default_values('core_vs_truflation')
        }
    
    def parse_investor_positioning_cash(self) -> Dict:
        """Парсить Поwithицandонування andнвесторandв (Cash Levels)"""
        
        try:
            # BofA Fund Manager Survey данand
            positioning_data = self._get_positioning_data()
            
            if positioning_data:
                current_cash_levels = positioning_data.get('cash_levels', 0)
                previous_cash_levels = positioning_data.get('cash_levels_previous', current_cash_levels)
                
                # Фandльтруємо шум
                trend, change, threshold = self.noise_filter.filter_noise(
                    'investor_positioning_cash', current_cash_levels, previous_cash_levels
                )
                
                return {
                    'investor_positioning_cash_current': float(current_cash_levels),
                    'investor_positioning_cash_previous': float(previous_cash_levels),
                    'investor_positioning_cash_change': float(change),
                    'investor_positioning_cash_trend': trend,
                    'investor_positioning_cash_level': 1 if current_cash_levels > 4.5 else (-1 if current_cash_levels < 3.5 else 0)
                }
            
        except Exception as e:
            logger.error(f"[ComprehensiveMacroParser] Error parsing Investor Positioning: {e}")
        
        return self._get_default_values('investor_positioning_cash')
    
    def _get_margin_debt_data(self) -> Optional[Dict]:
        """Отримує данand про маржинальний борг"""
        
        try:
            # FINRA/NYSE данand or симуляцandя
            return {
                'margin_debt': 850000000000,  # $850B
                'margin_debt_previous': 820000000000  # $820B
            }
            
        except Exception as e:
            logger.error(f"[ComprehensiveMacroParser] Error getting margin debt data: {e}")
        
        return None
    
    def _get_michigan_data(self) -> Optional[Dict]:
        """Отримує данand Michigan Consumer Sentiment"""
        
        try:
            # University of Michigan данand or симуляцandя
            return {
                'sentiment_index': 67.4,  # Поточний andнwhereкс
                'sentiment_index_previous': 65.2  # Попереднandй
            }
            
        except Exception as e:
            logger.error(f"[ComprehensiveMacroParser] Error getting Michigan data: {e}")
        
        return None
    
    def _get_cpi_data(self) -> Optional[Dict]:
        """Отримує данand CPI"""
        
        try:
            # BLS данand or симуляцandя
            return {
                'cpi': 298.4,
                'cpi_previous': 296.8
            }
            
        except Exception as e:
            logger.error(f"[ComprehensiveMacroParser] Error getting CPI data: {e}")
        
        return None
    
    def _get_core_cpi_data(self) -> Optional[Dict]:
        """Отримує данand Core CPI"""
        
        try:
            # BLS данand or симуляцandя
            return {
                'core_cpi': 306.1,
                'core_cpi_previous': 304.7
            }
            
        except Exception as e:
            logger.error(f"[ComprehensiveMacroParser] Error getting Core CPI data: {e}")
        
        return None
    
    def _get_truflation_data(self) -> Optional[Dict]:
        """Отримує данand Truflation"""
        
        try:
            # Truflation данand or симуляцandя
            return {
                'truflation': 3.2,
                'truflation_previous': 3.1
            }
            
        except Exception as e:
            logger.error(f"[ComprehensiveMacroParser] Error getting Truflation data: {e}")
        
        return None
    
    def _get_positioning_data(self) -> Optional[Dict]:
        """Отримує данand про поwithицandонування andнвесторandв"""
        
        try:
            # BofA Fund Manager Survey данand or симуляцandя
            return {
                'cash_levels': 4.2,  # 4.2%
                'cash_levels_previous': 4.0
            }
            
        except Exception as e:
            logger.error(f"[ComprehensiveMacroParser] Error getting positioning data: {e}")
        
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
    
    def parse_all_comprehensive_indicators(self) -> Dict:
        """Парсить all комплекснand покаwithники"""
        
        logger.info("[ComprehensiveMacroParser] Parsing all comprehensive indicators...")
        
        all_indicators = {}
        
        # Парсимо новand покаwithники
        indicators_parsers = [
            ('margin_debt', self.parse_margin_debt),
            ('michigan_consumer_sentiment', self.parse_michigan_consumer_sentiment),
            ('cpi_vs_core_cpi', self.parse_cpi_vs_core_cpi),
            ('investor_positioning_cash', self.parse_investor_positioning_cash)
        ]
        
        for indicator_name, parser_func in indicators_parsers:
            try:
                indicator_data = parser_func()
                all_indicators.update(indicator_data)
                logger.info(f"[ComprehensiveMacroParser] Successfully parsed {indicator_name}")
            except Exception as e:
                logger.error(f"[ComprehensiveMacroParser] Error parsing {indicator_name}: {e}")
                if indicator_name == 'cpi_vs_core_cpi':
                    all_indicators.update(self._get_default_values('cpi_vs_core_cpi'))
                    all_indicators.update(self._get_default_values('cpi_vs_truflation'))
                    all_indicators.update(self._get_default_values('core_vs_truflation'))
                else:
                    all_indicators.update(self._get_default_values(indicator_name))
        
        logger.info(f"[ComprehensiveMacroParser] Parsed {len(all_indicators)} comprehensive indicator values")
        return all_indicators
    
    def get_comprehensive_summary(self) -> Dict:
        """Отримує пandдсумок комплексних покаwithникandв"""
        
        indicators = [
            'margin_debt',
            'michigan_consumer_sentiment', 
            'cpi_vs_core_cpi',
            'investor_positioning_cash'
        ]
        
        summary = {}
        
        for indicator in indicators:
            summary[indicator] = {
                'percentage_threshold': self.noise_filter.get_adaptive_threshold(indicator, 'percentage'),
                'absolute_threshold': self.noise_filter.get_adaptive_threshold(indicator, 'absolute'),
                'frequency': self.noise_filter.indicator_frequency.get(indicator, 'monthly'),
                'parsing_frequency': self.noise_filter.get_parsing_frequency(indicator)
            }
        
        return summary

# Приклад викорисandння
def demo_comprehensive_parser():
    """Демонстрацandя комплексного парсера"""
    
    print("="*70)
    print("COMPREHENSIVE MACRO PARSER DEMONSTRATION")
    print("="*70)
    
    parser = ComprehensiveMacroParser()
    
    print("Comprehensive Indicators:")
    comprehensive_list = [
        "1. Margin Debt (Monthly)",
        "2. Michigan Consumer Sentiment (Monthly)",
        "3. CPI vs Core CPI vs Truflation (Monthly)",
        "4. Investor Positioning Cash (Weekly)"
    ]
    
    for indicator in comprehensive_list:
        print(f"  {indicator}")
    
    print(f"\nAdaptive Thresholds for Comprehensive Indicators:")
    thresholds = parser.get_comprehensive_summary()
    
    for indicator, thresh in thresholds.items():
        print(f"  {indicator}:")
        print(f"    Percentage: {thresh['percentage_threshold']:.2%}")
        print(f"    Absolute: {thresh['absolute_threshold']:.2f}")
        print(f"    Frequency: {thresh['frequency']}")
        print(f"    Parse: {thresh['parsing_frequency']}")
    
    print(f"\nParsing comprehensive indicators...")
    indicators = parser.parse_all_comprehensive_indicators()
    
    print(f"Successfully parsed {len(indicators)} values")
    
    # Покаwithуємо тренди
    trend_indicators = {k: v for k, v in indicators.items() if k.endswith('_trend')}
    
    print(f"\nTrend Values (Noise Filtered):")
    for indicator, trend in list(trend_indicators.items())[:10]:
        trend_symbol = "" if trend > 0 else ("" if trend < 0 else "")
        print(f"  {indicator}: {trend_symbol} ({trend})")
    
    print(f"\nCurrent Values:")
    current_indicators = {k: v for k, v in indicators.items() if k.endswith('_current')}
    for indicator, value in list(current_indicators.items())[:8]:
        print(f"  {indicator}: {value:.3f}")
    
    print(f"\nIntegration Benefits:")
    print("  - Margin Debt: Market leverage monitoring")
    print("  - Michigan Sentiment: Alternative sentiment measure")
    print("  - CPI vs Core CPI vs Truflation: Comprehensive inflation analysis")
    print("  - Investor Positioning: Investor behavior insights")
    print("  - All with adaptive noise filtering")
    print("  - Complementary to existing indicators")
    
    print("="*70)

if __name__ == "__main__":
    demo_comprehensive_parser()
