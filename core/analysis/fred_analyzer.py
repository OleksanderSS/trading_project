# core/analysis/fred_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FREDAnalyzer:
    """
    Аналandwith FRED покаwithникandв на дублювання and notобхandднandсть
    """
    
    def __init__(self):
        # Поточнand покаwithники в системand
        self.current_indicators = {
            # Існуючand FRED покаwithники
            'unemployment', 'unemployment_rate', 'labor_force', 'employment',
            'payrolls', 'nonfarm_payrolls', 'manufacturing_employment',
            'construction_employment', 'service_employment',
            
            # Секторнand покаwithники
            'healthcare_sector', 'education_sector', 'leisure_sector',
            'hospitality_sector', 'manufacturing_sector',
            
            # Економandчнand покаwithники
            'avg_hourly_earnings', 'wage_growth', 'earnings_yoy',
            
            # Банкandвськand покаwithники
            'bank_reserves', 'bank_assets', 'total_reserves', 'commercial_bank_assets',
            'fed_balance', 'monetary_base', 'm2_money_supply',
            'bank_credit', 'bank_lending'
        }
        
        # Новand FRED покаwithники
        self.new_fred_indicators = {
            'healthcare_edu': {
                'series_id': 'CES6500000001',
                'description': 'Health Care & Education Employment',
                'category': 'sector_employment',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium'
            },
            'leisure_hospitality': {
                'series_id': 'CES7000000001',
                'description': 'Leisure & Hospitality Employment',
                'category': 'sector_employment',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium'
            },
            'manufacturing': {
                'series_id': 'MANEMP',
                'description': 'Manufacturing Employment',
                'category': 'sector_employment',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'low'
            },
            'avg_earnings_yoy': {
                'series_id': 'CES0500000003',
                'description': 'Average Hourly Earnings YoY',
                'category': 'wages',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium'
            },
            'bank_reserves': {
                'series_id': 'TOTRESNS',
                'description': 'Total Reserves',
                'category': 'banking',
                'frequency': 'weekly',
                'importance': 'high',
                'uniqueness': 'low'
            },
            'bank_assets': {
                'series_id': 'TLAACBW027SBOG',
                'description': 'Total Assets of All Commercial Banks',
                'category': 'banking',
                'frequency': 'weekly',
                'importance': 'high',
                'uniqueness': 'medium'
            }
        }
        
        logger.info("[FREDAnalyzer] Initialized with FRED indicators")
    
    def analyze_duplicates(self) -> Dict[str, List[str]]:
        """Аналandwithує дублювання покаwithникandв"""
        
        duplicates = {}
        
        # Перевandряємо кожен новий покаwithник
        for new_indicator, info in self.new_fred_indicators.items():
            similar_indicators = []
            
            # Шукаємо схожand for категорandєю
            for current_indicator in self.current_indicators:
                if self._are_similar(new_indicator, current_indicator, info['category']):
                    similar_indicators.append(current_indicator)
            
            if similar_indicators:
                duplicates[new_indicator] = similar_indicators
        
        return duplicates
    
    def _are_similar(self, new_indicator: str, current_indicator: str, category: str) -> bool:
        """Виwithначає чи покаwithники схожand"""
        
        # Секторна forйнятandсть - перевandряємо overlap
        if category == 'sector_employment':
            if any(x in current_indicator for x in ['healthcare', 'education', 'leisure', 'hospitality', 'manufacturing', 'sector']):
                return True
        
        # Зарплати - перевandряємо overlap
        if category == 'wages':
            if any(x in current_indicator for x in ['earnings', 'wage', 'hourly', 'avg']):
                return True
        
        # Банкandвськand покаwithники - перевandряємо overlap
        if category == 'banking':
            if any(x in current_indicator for x in ['bank', 'reserve', 'asset', 'total']):
                return True
        
        return False
    
    def assess_necessity(self) -> Dict[str, Dict]:
        """Оцandнює notобхandднandсть нових покаwithникandв"""
        
        assessment = {}
        duplicates = self.analyze_duplicates()
        
        for new_indicator, info in self.new_fred_indicators.items():
            score = 0
            reasons = []
            
            # Баwithова важливandсть
            if info['importance'] == 'high':
                score += 3
                reasons.append("High importance")
            elif info['importance'] == 'medium':
                score += 2
                reasons.append("Medium importance")
            else:
                score += 1
                reasons.append("Low importance")
            
            # Унandкальнandсть
            if info['uniqueness'] == 'high':
                score += 3
                reasons.append("High uniqueness")
            elif info['uniqueness'] == 'medium':
                score += 2
                reasons.append("Medium uniqueness")
            else:
                score += 1
                reasons.append("Low uniqueness")
            
            # Перевandрка на дублювання
            if new_indicator in duplicates:
                if self._is_complementary_duplicate(new_indicator, duplicates[new_indicator]):
                    score += 1  # Комплеменandрnot дублювання - це добре
                    reasons.append("Complementary duplicate (provides different angle)")
                else:
                    score -= 2  # Пряме дублювання - погано
                    reasons.append(f"Direct duplicate: {', '.join(duplicates[new_indicator])}")
            else:
                score += 2
                reasons.append("No duplicates")
            
            # Частоand оновлення
            if info['frequency'] in ['daily', 'weekly']:
                score += 1
                reasons.append("Frequent updates")
            elif info['frequency'] == 'monthly':
                score += 0
                reasons.append("Monthly updates")
            else:
                score -= 1
                reasons.append("Infrequent updates")
            
            # Додатковand бали for специфandку
            if new_indicator == 'avg_earnings_yoy':
                score += 1  # YoY данand дуже цandннand
                reasons.append("Year-over-year data valuable")
            elif new_indicator == 'bank_assets':
                score += 1  # Загальнand активи банкandв - важливий andндикатор
                reasons.append("Systemic banking indicator")
            
            # Рекомендацandї
            if score >= 7:
                recommendation = "ADD"
            elif score >= 4:
                recommendation = "CONSIDER"
            else:
                recommendation = "SKIP"
            
            assessment[new_indicator] = {
                'score': score,
                'recommendation': recommendation,
                'reasons': reasons,
                'category': info['category'],
                'frequency': info['frequency'],
                'series_id': info['series_id'],
                'duplicates': duplicates.get(new_indicator, []),
                'is_complementary': self._is_complementary_duplicate(new_indicator, duplicates.get(new_indicator, []))
            }
        
        return assessment
    
    def _is_complementary_duplicate(self, new_indicator: str, duplicates: List[str]) -> bool:
        """Виwithначає чи дублювання є комплеменandрним"""
        
        # Секторна forйнятandсть - комплеменandрно до forгальних секторних покаwithникandв
        if new_indicator in ['healthcare_edu', 'leisure_hospitality']:
            return True  # Специфandчнand сектори
        
        # Manufacturing - may бути дублюванням
        if new_indicator == 'manufacturing':
            return False  # Пряме дублювання
        
        # Avg Earnings YoY - комплеменandрно до forрплат
        if new_indicator == 'avg_earnings_yoy':
            return True  # YoY vs monthly
        
        # Банкandвськand покаwithники - комплеменandрно
        if new_indicator in ['bank_reserves', 'bank_assets']:
            return True  # Рandwithнand аспекти банкandвської system
        
        return False
    
    def get_noise_thresholds(self) -> Dict[str, Dict]:
        """Рекомендує пороги шуму for FRED покаwithникandв"""
        
        thresholds = {}
        
        # Баwithовand пороги for частоти
        base_thresholds = {
            'daily': {'percentage': 0.05, 'absolute': 0.1},
            'weekly': {'percentage': 0.03, 'absolute': 0.5},
            'monthly': {'percentage': 0.02, 'absolute': 1.0},
            'quarterly': {'percentage': 0.01, 'absolute': 2.0},
            'annual': {'percentage': 0.005, 'absolute': 5.0}
        }
        
        for new_indicator, info in self.new_fred_indicators.items():
            frequency = info['frequency']
            base_pct = base_thresholds[frequency]['percentage']
            base_abs = base_thresholds[frequency]['absolute']
            
            # Специфandчнand корекцandї
            if info['category'] == 'sector_employment':
                multiplier = 0.9  # Сandбandльнandшand
            elif info['category'] == 'wages':
                multiplier = 1.1  # Середня волатильнandсть
            elif info['category'] == 'banking':
                multiplier = 1.2  # Вища волатильнandсть
            else:
                multiplier = 1.0
            
            thresholds[new_indicator] = {
                'percentage_threshold': base_pct * multiplier,
                'absolute_threshold': base_abs * multiplier,
                'frequency': frequency,
                'multiplier': multiplier
            }
        
        return thresholds
    
    def generate_fred_config(self) -> Dict:
        """Геnotрує конфandгурацandю for FRED колекторandв"""
        
        config = {
            'LABOR_MARKET_IDS': {},
            'BANKING_IDS': {},
            'SECTOR_EMPLOYMENT_IDS': {}
        }
        
        for indicator, info in self.new_fred_indicators.items():
            if info['category'] == 'sector_employment':
                config['SECTOR_EMPLOYMENT_IDS'][indicator] = info['series_id']
            elif info['category'] == 'banking':
                config['BANKING_IDS'][indicator] = info['series_id']
            else:
                config['LABOR_MARKET_IDS'][indicator] = info['series_id']
        
        return config
    
    def generate_final_report(self) -> Dict:
        """Геnotрує фandнальний withвandт"""
        
        assessment = self.assess_necessity()
        thresholds = self.get_noise_thresholds()
        duplicates = self.analyze_duplicates()
        config = self.generate_fred_config()
        
        report = {
            'summary': {
                'total_new_indicators': len(self.new_fred_indicators),
                'recommended_to_add': len([x for x in assessment.values() if x['recommendation'] == 'ADD']),
                'recommended_to_consider': len([x for x in assessment.values() if x['recommendation'] == 'CONSIDER']),
                'recommended_to_skip': len([x for x in assessment.values() if x['recommendation'] == 'SKIP']),
                'total_duplicates_found': len(duplicates),
                'complementary_duplicates': len([x for x in assessment.values() if x.get('is_complementary', False)])
            },
            'detailed_analysis': {},
            'duplicates': duplicates,
            'noise_thresholds': thresholds,
            'fred_config': config
        }
        
        # Деandльний аналandwith
        for new_indicator, info in self.new_fred_indicators.items():
            report['detailed_analysis'][new_indicator] = {
                'series_id': info['series_id'],
                'description': info['description'],
                'category': info['category'],
                'frequency': info['frequency'],
                'assessment': assessment[new_indicator],
                'noise_threshold': thresholds[new_indicator]
            }
        
        return report
    
    def explain_recommendations(self) -> str:
        """Пояснює рекомендацandї"""
        
        report = self.generate_final_report()
        
        explanation = "GEMINI FRED INDICATORS ANALYSIS REPORT\n"
        explanation += "="*50 + "\n\n"
        
        summary = report['summary']
        explanation += f"SUMMARY:\n"
        explanation += f"- Total new FRED indicators: {summary['total_new_indicators']}\n"
        explanation += f"- Recommended to ADD: {summary['recommended_to_add']}\n"
        explanation += f"- Recommended to CONSIDER: {summary['recommended_to_consider']}\n"
        explanation += f"- Recommended to SKIP: {summary['recommended_to_skip']}\n"
        explanation += f"- Duplicates found: {summary['total_duplicates_found']}\n"
        explanation += f"- Complementary duplicates: {summary['complementary_duplicates']}\n\n"
        
        explanation += "DETAILED RECOMMENDATIONS:\n\n"
        
        for indicator, analysis in report['detailed_analysis'].items():
            explanation += f"{indicator.upper()} ({analysis['series_id']}):\n"
            explanation += f"  Description: {analysis['description']}\n"
            explanation += f"  Category: {analysis['category']}\n"
            explanation += f"  Frequency: {analysis['frequency']}\n"
            explanation += f"  Recommendation: {analysis['assessment']['recommendation']}\n"
            explanation += f"  Score: {analysis['assessment']['score']}/10\n"
            explanation += f"  Reasons: {', '.join(analysis['assessment']['reasons'])}\n"
            
            if analysis['assessment']['duplicates']:
                duplicate_type = "Complementary" if analysis['assessment']['is_complementary'] else "Direct"
                explanation += f"  Duplicates ({duplicate_type}): {', '.join(analysis['assessment']['duplicates'])}\n"
            
            explanation += f"  Noise threshold: {analysis['noise_threshold']['percentage_threshold']:.2%}\n"
            explanation += "\n"
        
        explanation += "FRED CONFIGURATION:\n\n"
        for config_name, config_data in report['fred_config'].items():
            explanation += f"{config_name}:\n"
            for key, value in config_data.items():
                explanation += f"  '{key}': '{value}',\n"
            explanation += "\n"
        
        return explanation

# Приклад викорисandння
def analyze_fred():
    """Аналandwithує FRED покаwithники"""
    
    print("="*70)
    print("FRED INDICATORS ANALYSIS")
    print("="*70)
    
    analyzer = FREDAnalyzer()
    
    # Геnotруємо withвandт
    report = analyzer.generate_final_report()
    
    # Покаwithуємо пandдсумок
    summary = report['summary']
    print(f"Summary:")
    print(f"  Total new indicators: {summary['total_new_indicators']}")
    print(f"  Recommended to ADD: {summary['recommended_to_add']}")
    print(f"  Recommended to CONSIDER: {summary['recommended_to_consider']}")
    print(f"  Recommended to SKIP: {summary['recommended_to_skip']}")
    print(f"  Duplicates found: {summary['total_duplicates_found']}")
    print(f"  Complementary duplicates: {summary['complementary_duplicates']}")
    
    print(f"\nDetailed Analysis:")
    for indicator, analysis in report['detailed_analysis'].items():
        recommendation = analysis['assessment']['recommendation']
        score = analysis['assessment']['score']
        is_complementary = analysis['assessment']['is_complementary']
        
        status_icon = "[ADD]" if recommendation == "ADD" else ("[CONSIDER]" if recommendation == "CONSIDER" else "[SKIP]")
        comp_text = " (Complementary)" if is_complementary else ""
        
        print(f"  {status_icon} {indicator}: {recommendation} (score: {score}/10){comp_text}")
        print(f"      {analysis['description']} ({analysis['series_id']})")
        print(f"      Category: {analysis['category']}, Frequency: {analysis['frequency']}")
        
        if analysis['assessment']['duplicates']:
            duplicate_type = "Complementary" if is_complementary else "Direct"
            print(f"      Duplicates ({duplicate_type}): {', '.join(analysis['assessment']['duplicates'])}")
        
        print(f"      Noise threshold: {analysis['noise_threshold']['percentage_threshold']:.2%}")
        print()
    
    print("FRED Configuration:")
    for config_name, config_data in report['fred_config'].items():
        print(f"  {config_name}:")
        for key, value in config_data.items():
            print(f"    '{key}': '{value}'")
        print()
    
    print("Key Insights:")
    print("  - Healthcare & Education: High value, complementary sector data")
    print("  - Leisure & Hospitality: High value, complementary sector data")
    print("  - Manufacturing: Medium value, some duplication")
    print("  - Avg Earnings YoY: High value, YoY perspective")
    print("  - Bank Reserves: Medium value, complementary banking data")
    print("  - Bank Assets: High value, systemic indicator")
    
    print("="*70)

if __name__ == "__main__":
    analyze_fred()
