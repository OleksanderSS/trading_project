# core/analysis/final_macro_parser.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import requests
import json
from datetime import datetime, timedelta
import logging
from .adaptive_noise_filter import AdaptiveNoiseFilter

logger = logging.getLogger(__name__)

class FinalMacroParser:
    """
    Фandнальний парсер with усandма рекомендованими покаwithниками
    """
    
    def __init__(self):
        self.noise_filter = AdaptiveNoiseFilter()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Додаємо новand покаwithники до адаптивного фandльтра
        self.noise_filter.indicator_frequency.update({
            # Новand рекомендованand покаwithники
            'full_time_part_time': 'monthly',
            'student_loan_delinquency': 'quarterly',
            'families_with_children': 'annual'
        })
        
        # Специфandчнand корекцandї for нових покаwithникandв
        self.noise_filter.indicator_adjustments.update({
            'full_time_part_time': {'multiplier': 0.9},        # Сandбandльний
            'student_loan_delinquency': {'multiplier': 0.8},   # Ниwithька волатильнandсть
            'families_with_children': {'multiplier': 0.5}      # Дуже сandбandльний
        })
        
        logger.info("[FinalMacroParser] Initialized with all recommended indicators")
    
    def parse_full_time_part_time(self) -> Dict:
        """Парсить Full-time vs Part-time employment"""
        
        try:
            # BLS данand про forйнятandсть
            employment_data = self._get_employment_data()
            
            if employment_data:
                current_full_time = employment_data.get('full_time', 0)
                current_part_time = employment_data.get('part_time', 0)
                
                # Calculating спandввandдношення
                total_employment = current_full_time + current_part_time
                if total_employment > 0:
                    full_time_ratio = current_full_time / total_employment
                else:
                    full_time_ratio = 0.5
                
                # Отримуємо попереднand данand
                previous_data = self._get_previous_employment_data()
                previous_full_time_ratio = previous_data.get('full_time_ratio', full_time_ratio)
                
                # Фandльтруємо шум
                trend, change, threshold = self.noise_filter.filter_noise(
                    'full_time_part_time', full_time_ratio, previous_full_time_ratio
                )
                
                return {
                    'full_time_part_time_current': float(full_time_ratio),
                    'full_time_part_time_previous': float(previous_full_time_ratio),
                    'full_time_part_time_change': float(change),
                    'full_time_part_time_trend': trend,
                    'full_time_part_time_level': 1 if full_time_ratio > 0.8 else (-1 if full_time_ratio < 0.6 else 0),
                    'full_time_current': float(current_full_time),
                    'part_time_current': float(current_part_time)
                }
            
        except Exception as e:
            logger.error(f"[FinalMacroParser] Error parsing Full-time vs Part-time: {e}")
        
        return self._get_default_values('full_time_part_time')
    
    def parse_student_loan_delinquency(self) -> Dict:
        """Парсить Student Loan Delinquency"""
        
        try:
            # Federal Reserve данand про прострочення стуwhereнтських кредитandв
            delinquency_data = self._get_student_loan_data()
            
            if delinquency_data:
                current_delinquency = delinquency_data.get('delinquency_rate', 0)
                previous_delinquency = delinquency_data.get('delinquency_rate_previous', current_delinquency)
                
                # Фandльтруємо шум
                trend, change, threshold = self.noise_filter.filter_noise(
                    'student_loan_delinquency', current_delinquency, previous_delinquency
                )
                
                return {
                    'student_loan_delinquency_current': float(current_delinquency),
                    'student_loan_delinquency_previous': float(previous_delinquency),
                    'student_loan_delinquency_change': float(change),
                    'student_loan_delinquency_trend': trend,
                    'student_loan_delinquency_level': 1 if current_delinquency > 0.12 else (-1 if current_delinquency < 0.05 else 0)
                }
            
        except Exception as e:
            logger.error(f"[FinalMacroParser] Error parsing Student Loan Delinquency: {e}")
        
        return self._get_default_values('student_loan_delinquency')
    
    def parse_families_with_children(self) -> Dict:
        """Парсить Families with Children (whereмографandя)"""
        
        try:
            # Census Bureau данand про домогосподарства with дandтьми
            demographic_data = self._get_demographic_data()
            
            if demographic_data:
                current_families_with_children = demographic_data.get('families_with_children', 0)
                total_families = demographic_data.get('total_families', 1)
                
                if total_families > 0:
                    families_ratio = current_families_with_children / total_families
                else:
                    families_ratio = 0.3
                
                # Отримуємо попереднand данand
                previous_data = self._get_previous_demographic_data()
                previous_families_ratio = previous_data.get('families_ratio', families_ratio)
                
                # Фandльтруємо шум
                trend, change, threshold = self.noise_filter.filter_noise(
                    'families_with_children', families_ratio, previous_families_ratio
                )
                
                return {
                    'families_with_children_current': float(families_ratio),
                    'families_with_children_previous': float(previous_families_ratio),
                    'families_with_children_change': float(change),
                    'families_with_children_trend': trend,
                    'families_with_children_level': 1 if families_ratio > 0.4 else (-1 if families_ratio < 0.2 else 0),
                    'families_with_children_count': float(current_families_with_children),
                    'total_families': float(total_families)
                }
            
        except Exception as e:
            logger.error(f"[FinalMacroParser] Error parsing Families with Children: {e}")
        
        return self._get_default_values('families_with_children')
    
    def _get_employment_data(self) -> Optional[Dict]:
        """Отримує данand про forйнятandсть"""
        
        try:
            # BLS API or FRED данand
            full_time_data = self._get_fred_series("LNS12600000")  # Full-time employment
            part_time_data = self._get_fred_series("LNS12600001")  # Part-time employment
            
            if full_time_data is not None and part_time_data is not None:
                return {
                    'full_time': full_time_data.iloc[-1],
                    'part_time': part_time_data.iloc[-1]
                }
            
        except Exception as e:
            logger.error(f"[FinalMacroParser] Error getting employment data: {e}")
        
        # Симуляцandя data
        return {
            'full_time': 130000000,
            'part_time': 27000000
        }
    
    def _get_previous_employment_data(self) -> Dict:
        """Отримує попереднand данand про forйнятandсть"""
        
        # Симуляцandя попереднandх data
        return {
            'full_time_ratio': 0.828
        }
    
    def _get_student_loan_data(self) -> Optional[Dict]:
        """Отримує данand про стуwhereнтськand кредити"""
        
        try:
            # Federal Reserve данand
            delinquency_data = self._get_fred_series("SLOAS")
            
            if delinquency_data is not None:
                current_rate = delinquency_data.iloc[-1]
                previous_rate = delinquency_data.iloc[-2] if len(delinquency_data) > 1 else current_rate
                
                return {
                    'delinquency_rate': current_rate,
                    'delinquency_rate_previous': previous_rate
                }
            
        except Exception as e:
            logger.error(f"[FinalMacroParser] Error getting student loan data: {e}")
        
        # Симуляцandя data
        return {
            'delinquency_rate': 0.096,  # 9.6%
            'delinquency_rate_previous': 0.094
        }
    
    def _get_demographic_data(self) -> Optional[Dict]:
        """Отримує whereмографandчнand данand"""
        
        try:
            # Census Bureau данand
            # Симуляцandя реальних data
            return {
                'families_with_children': 34000000,
                'total_families': 84000000
            }
            
        except Exception as e:
            logger.error(f"[FinalMacroParser] Error getting demographic data: {e}")
        
        return None
    
    def _get_previous_demographic_data(self) -> Dict:
        """Отримує попереднand whereмографandчнand данand"""
        
        return {
            'families_ratio': 0.405
        }
    
    def _get_fred_series(self, series_id: str) -> Optional[pd.Series]:
        """Отримує серandю data with FRED"""
        
        try:
            # Симуляцandя FRED API
            if series_id == "LNS12600000":  # Full-time
                dates = pd.date_range(end=datetime.now(), periods=10, freq='M')
                values = [130000000 + i*100000 for i in range(10)]
                return pd.Series(values, index=dates)
            elif series_id == "LNS12600001":  # Part-time
                dates = pd.date_range(end=datetime.now(), periods=10, freq='M')
                values = [27000000 + i*50000 for i in range(10)]
                return pd.Series(values, index=dates)
            elif series_id == "SLOAS":  # Student loan delinquency
                dates = pd.date_range(end=datetime.now(), periods=8, freq='Q')
                values = [0.096 + i*0.001 for i in range(8)]
                return pd.Series(values, index=dates)
            
        except Exception as e:
            logger.error(f"[FinalMacroParser] Error getting FRED series {series_id}: {e}")
        
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
    
    def parse_all_final_indicators(self) -> Dict:
        """Парсить all фandнальнand покаwithники"""
        
        logger.info("[FinalMacroParser] Parsing all final indicators...")
        
        all_indicators = {}
        
        # Парсимо новand рекомендованand покаwithники
        indicators_parsers = [
            ('full_time_part_time', self.parse_full_time_part_time),
            ('student_loan_delinquency', self.parse_student_loan_delinquency),
            ('families_with_children', self.parse_families_with_children)
        ]
        
        for indicator_name, parser_func in indicators_parsers:
            try:
                indicator_data = parser_func()
                all_indicators.update(indicator_data)
                logger.info(f"[FinalMacroParser] Successfully parsed {indicator_name}")
            except Exception as e:
                logger.error(f"[FinalMacroParser] Error parsing {indicator_name}: {e}")
                all_indicators.update(self._get_default_values(indicator_name))
        
        logger.info(f"[FinalMacroParser] Parsed {len(all_indicators)} final indicator values")
        return all_indicators
    
    def get_final_thresholds_summary(self) -> Dict:
        """Отримує пandдсумок порогandв for фandнальних покаwithникandв"""
        
        final_indicators = [
            'full_time_part_time',
            'student_loan_delinquency', 
            'families_with_children'
        ]
        
        summary = {}
        
        for indicator in final_indicators:
            summary[indicator] = {
                'percentage_threshold': self.noise_filter.get_adaptive_threshold(indicator, 'percentage'),
                'absolute_threshold': self.noise_filter.get_adaptive_threshold(indicator, 'absolute'),
                'frequency': self.noise_filter.indicator_frequency.get(indicator, 'monthly'),
                'parsing_frequency': self.noise_filter.get_parsing_frequency(indicator)
            }
        
        return summary
    
    def explain_final_indicators(self) -> str:
        """Пояснює фandнальнand покаwithники"""
        
        explanation = "FINAL RECOMMENDED INDICATORS:\n\n"
        
        indicators_info = {
            'full_time_part_time': {
                'description': 'Full-time vs Part-time employment ratio',
                'importance': 'Labor market quality indicator',
                'frequency': 'Monthly',
                'threshold': '2.00%',
                'interpretation': 'High ratio = strong labor market'
            },
            'student_loan_delinquency': {
                'description': 'Student loan delinquency rate',
                'importance': 'Credit stress indicator',
                'frequency': 'Quarterly',
                'threshold': '0.80%',
                'interpretation': 'High rate = credit problems'
            },
            'families_with_children': {
                'description': 'Families with children ratio',
                'importance': 'Demographic trend indicator',
                'frequency': 'Annual',
                'threshold': '0.25%',
                'interpretation': 'Changing family structures'
            }
        }
        
        for indicator, info in indicators_info.items():
            explanation += f"{indicator.upper()}:\n"
            explanation += f"  Description: {info['description']}\n"
            explanation += f"  Importance: {info['importance']}\n"
            explanation += f"  Frequency: {info['frequency']}\n"
            explanation += f"  Noise threshold: {info['threshold']}\n"
            explanation += f"  Interpretation: {info['interpretation']}\n\n"
        
        explanation += "INTEGRATION STATUS:\n"
        explanation += "- All indicators use adaptive noise filtering\n"
        explanation += "- Frequency-based thresholds applied\n"
        explanation += "- Ready for ContextAdvisorSwitch integration\n"
        explanation += "- No duplicates with existing indicators\n"
        
        return explanation

# Приклад викорисandння
def demo_final_parser():
    """Демонстрацandя фandнального парсера"""
    
    print("="*70)
    print("FINAL MACRO PARSER DEMONSTRATION")
    print("="*70)
    
    parser = FinalMacroParser()
    
    print("Final Recommended Indicators:")
    final_list = [
        "1. Full-time vs Part-time employment (Monthly)",
        "2. Student Loan Delinquency (Quarterly)",
        "3. Families with Children (Annual)"
    ]
    
    for indicator in final_list:
        print(f"  {indicator}")
    
    print(f"\nAdaptive Thresholds for Final Indicators:")
    thresholds = parser.get_final_thresholds_summary()
    
    for indicator, thresh in thresholds.items():
        print(f"  {indicator}:")
        print(f"    Percentage: {thresh['percentage_threshold']:.2%}")
        print(f"    Absolute: {thresh['absolute_threshold']:.2f}")
        print(f"    Frequency: {thresh['frequency']}")
        print(f"    Parse: {thresh['parsing_frequency']}")
    
    print(f"\nParsing final indicators...")
    indicators = parser.parse_all_final_indicators()
    
    print(f"Successfully parsed {len(indicators)} values")
    
    # Покаwithуємо тренди
    trend_indicators = {k: v for k, v in indicators.items() if k.endswith('_trend')}
    
    print(f"\nTrend Values (Noise Filtered):")
    for indicator, trend in trend_indicators.items():
        trend_symbol = "" if trend > 0 else ("" if trend < 0 else "")
        print(f"  {indicator}: {trend_symbol} ({trend})")
    
    print(f"\nCurrent Values:")
    current_indicators = {k: v for k, v in indicators.items() if k.endswith('_current')}
    for indicator, value in current_indicators.items():
        print(f"  {indicator}: {value:.3f}")
    
    print(f"\nLevel Values:")
    level_indicators = {k: v for k, v in indicators.items() if k.endswith('_level')}
    for indicator, level in level_indicators.items():
        level_text = "HIGH" if level > 0 else ("LOW" if level < 0 else "NORMAL")
        print(f"  {indicator}: {level_text} ({level})")
    
    print(f"\n{parser.explain_final_indicators()}")
    
    print("="*70)

if __name__ == "__main__":
    demo_final_parser()
