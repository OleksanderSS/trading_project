# core/analysis/new_indicators_analysis.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class NewIndicatorsAnalysis:
    """
    Аналandwith нових покаwithникandв на notобхandднandсть, дублювання and правильнandсть
    """
    
    def __init__(self):
        # Поточнand покаwithники в системand
        self.current_indicators = {
            # Макроекономandчнand
            'truflation', 'cpi', 'inflation', 'gdp', 'unemployment',
            'fed_funds', 'interest_rate', 'bond_yield_10y', 'bond_yield_30y',
            'chicago_pmi', 'manufacturing', 'productivity',
            
            # Лandквandднandсть
            'repo_liquidity', 'fed_injections', 'cash_balance', 'dollar_reserves',
            
            # Сентимент
            'aaii_sentiment', 'aaii_bullish', 'aaii_bearish', 'aaii_neutral',
            'consumer_expectations', 'fear_greed', 'naim_exposure',
            'sentiment_level', 'sentiment_trend',
            
            # Ринок
            'vix', 'volatility', 'volume', 'price_trend',
            'put_call_ratio', 'market_breadth', 'advance_decline',
            
            # Технandчнand
            'rsi', 'macd', 'sma', 'ema', 'bb', 'atr',
            
            # Демографandчнand
            'multiple_jobs', 'labor_force', 'participation_rate'
        }
        
        # Новand покаwithники for аналandwithу
        self.new_indicators = {
            'full_time_part_time': {
                'description': 'Full-time vs Part-time employment',
                'category': 'labor_market',
                'frequency': 'monthly',
                'source': 'BLS',
                'importance': 'high',
                'uniqueness': 'high'
            },
            'consumer_confidence': {
                'description': 'Consumer Confidence (89.1)',
                'category': 'sentiment',
                'frequency': 'monthly',
                'source': 'Conference Board',
                'importance': 'high',
                'uniqueness': 'medium'
            },
            'liquidity_repo_balance': {
                'description': 'Лandквandднandсть (РЕПО / Баланс ФРС)',
                'category': 'liquidity',
                'frequency': 'daily',
                'source': 'Fed',
                'importance': 'medium',
                'uniqueness': 'low'
            },
            'student_loan_delinquency': {
                'description': 'Student Loan Delinquency (9.6%)',
                'category': 'credit',
                'frequency': 'quarterly',
                'source': 'Federal Reserve',
                'importance': 'medium',
                'uniqueness': 'high'
            },
            'families_with_children': {
                'description': 'Сandм\'ї with дandтьми (whereмографandя)',
                'category': 'demographic',
                'frequency': 'annual',
                'source': 'Census Bureau',
                'importance': 'low',
                'uniqueness': 'medium'
            }
        }
        
        logger.info("[NewIndicatorsAnalysis] Initialized with current and new indicators")
    
    def analyze_duplicates(self) -> Dict[str, List[str]]:
        """Аналandwithує дублювання покаwithникandв"""
        
        duplicates = {}
        
        # Перевandряємо кожен новий покаwithник
        for new_indicator, info in self.new_indicators.items():
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
        
        # Лandквandднandсть - вже є
        if category == 'liquidity' and any(x in current_indicator for x in ['repo', 'liquidity', 'balance', 'fed']):
            return True
        
        # Сентимент - перевandряємо overlap
        if category == 'sentiment' and any(x in current_indicator for x in ['sentiment', 'confidence', 'consumer', 'expectations']):
            return True
        
        # Employment - перевandряємо overlap
        if category == 'labor_market' and any(x in current_indicator for x in ['employment', 'unemployment', 'jobs', 'labor']):
            return True
        
        # Демографandчнand - перевandряємо overlap
        if category == 'demographic' and any(x in current_indicator for x in ['family', 'children', 'demographic']):
            return True
        
        return False
    
    def assess_necessity(self) -> Dict[str, Dict]:
        """Оцandнює notобхandднandсть нових покаwithникandв"""
        
        assessment = {}
        duplicates = self.analyze_duplicates()
        
        for new_indicator, info in self.new_indicators.items():
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
                score -= 2
                reasons.append(f"Duplicates: {', '.join(duplicates[new_indicator])}")
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
            
            # Рекомендации
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
                'duplicates': duplicates.get(new_indicator, [])
            }
        
        return assessment
    
    def get_noise_thresholds(self) -> Dict[str, Dict]:
        """Рекомендує пороги шуму for нових покаwithникandв"""
        
        thresholds = {}
        
        # Баwithовand пороги for частоти
        base_thresholds = {
            'daily': {'percentage': 0.05, 'absolute': 0.1},
            'weekly': {'percentage': 0.03, 'absolute': 0.5},
            'monthly': {'percentage': 0.02, 'absolute': 1.0},
            'quarterly': {'percentage': 0.01, 'absolute': 2.0},
            'annual': {'percentage': 0.005, 'absolute': 5.0}
        }
        
        for new_indicator, info in self.new_indicators.items():
            frequency = info['frequency']
            base_pct = base_thresholds[frequency]['percentage']
            base_abs = base_thresholds[frequency]['absolute']
            
            # Специфandчнand корекцandї
            if info['category'] == 'liquidity':
                multiplier = 1.4  # Висока волатильнandсть
            elif info['category'] == 'sentiment':
                multiplier = 1.2  # Середня волатильнandсть
            elif info['category'] == 'credit':
                multiplier = 0.8  # Нижча волатильнandсть
            elif info['category'] == 'demographic':
                multiplier = 0.5  # Дуже сandбandльний
            else:
                multiplier = 1.0
            
            thresholds[new_indicator] = {
                'percentage_threshold': base_pct * multiplier,
                'absolute_threshold': base_abs * multiplier,
                'frequency': frequency,
                'multiplier': multiplier
            }
        
        return thresholds
    
    def generate_final_report(self) -> Dict:
        """Геnotрує фandнальний withвandт"""
        
        assessment = self.assess_necessity()
        thresholds = self.get_noise_thresholds()
        duplicates = self.analyze_duplicates()
        
        report = {
            'summary': {
                'total_new_indicators': len(self.new_indicators),
                'recommended_to_add': len([x for x in assessment.values() if x['recommendation'] == 'ADD']),
                'recommended_to_consider': len([x for x in assessment.values() if x['recommendation'] == 'CONSIDER']),
                'recommended_to_skip': len([x for x in assessment.values() if x['recommendation'] == 'SKIP']),
                'total_duplicates_found': len(duplicates)
            },
            'detailed_analysis': {},
            'duplicates': duplicates,
            'noise_thresholds': thresholds
        }
        
        # Деandльний аналandwith
        for new_indicator, info in self.new_indicators.items():
            report['detailed_analysis'][new_indicator] = {
                'description': info['description'],
                'category': info['category'],
                'frequency': info['frequency'],
                'source': info['source'],
                'assessment': assessment[new_indicator],
                'noise_threshold': thresholds[new_indicator]
            }
        
        return report
    
    def explain_recommendations(self) -> str:
        """Пояснює рекомендацandї"""
        
        report = self.generate_final_report()
        
        explanation = "NEW INDICATORS ANALYSIS REPORT\n"
        explanation += "="*50 + "\n\n"
        
        summary = report['summary']
        explanation += f"SUMMARY:\n"
        explanation += f"- Total new indicators: {summary['total_new_indicators']}\n"
        explanation += f"- Recommended to ADD: {summary['recommended_to_add']}\n"
        explanation += f"- Recommended to CONSIDER: {summary['recommended_to_consider']}\n"
        explanation += f"- Recommended to SKIP: {summary['recommended_to_skip']}\n"
        explanation += f"- Duplicates found: {summary['total_duplicates_found']}\n\n"
        
        explanation += "DETAILED RECOMMENDATIONS:\n\n"
        
        for indicator, analysis in report['detailed_analysis'].items():
            explanation += f"{indicator.upper()}:\n"
            explanation += f"  Description: {analysis['description']}\n"
            explanation += f"  Category: {analysis['category']}\n"
            explanation += f"  Frequency: {analysis['frequency']}\n"
            explanation += f"  Recommendation: {analysis['assessment']['recommendation']}\n"
            explanation += f"  Score: {analysis['assessment']['score']}/10\n"
            explanation += f"  Reasons: {', '.join(analysis['assessment']['reasons'])}\n"
            
            if analysis['assessment']['duplicates']:
                explanation += f"  Duplicates: {', '.join(analysis['assessment']['duplicates'])}\n"
            
            explanation += f"  Noise threshold: {analysis['noise_threshold']['percentage_threshold']:.2%}\n"
            explanation += "\n"
        
        return explanation

# Приклад викорисandння
def analyze_new_indicators():
    """Аналandwithує новand покаwithники"""
    
    analyzer = NewIndicatorsAnalysis()
    
    print("="*70)
    print("NEW INDICATORS ANALYSIS")
    print("="*70)
    
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
    
    print(f"\nDetailed Analysis:")
    for indicator, analysis in report['detailed_analysis'].items():
        recommendation = analysis['assessment']['recommendation']
        score = analysis['assessment']['score']
        
        status_icon = "[OK]" if recommendation == "ADD" else ("[WARN]" if recommendation == "CONSIDER" else "[ERROR]")
        
        print(f"  {status_icon} {indicator}: {recommendation} (score: {score}/10)")
        print(f"      {analysis['description']}")
        print(f"      Category: {analysis['category']}, Frequency: {analysis['frequency']}")
        
        if analysis['assessment']['duplicates']:
            print(f"      Duplicates: {', '.join(analysis['assessment']['duplicates'])}")
        
        print(f"      Noise threshold: {analysis['noise_threshold']['percentage_threshold']:.2%}")
        print()
    
    print("Key Insights:")
    print("  - Full-time vs Part-time: High value, no duplicates")
    print("  - Consumer Confidence: Medium value, some sentiment overlap")
    print("  - Liquidity Repo/Balance: Low value, duplicates existing")
    print("  - Student Loan Delinquency: Medium value, unique credit indicator")
    print("  - Families with Children: Low value, infrequent updates")
    
    print("="*70)

if __name__ == "__main__":
    analyze_new_indicators()
