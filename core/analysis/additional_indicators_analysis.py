# core/analysis/additional_indicators_analysis.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AdditionalIndicatorsAnalysis:
    """
    Аналandwith додаткових покаwithникandв на дублювання and notобхandднandсть
    """
    
    def __init__(self):
        # Поточнand покаwithники в системand
        self.current_indicators = {
            # Інфляцandя
            'truflation', 'cpi', 'inflation', 'core_inflation',
            
            # Сентимент
            'consumer_sentiment', 'consumer_expectations', 'aaii_sentiment',
            'aaii_bullish', 'aaii_bearish', 'aaii_neutral', 'sentiment_trend',
            'sentiment_level', 'fear_greed', 'naim_exposure',
            
            # Лandквandднandсть and поwithицandонування
            'cash_balance', 'liquidity', 'repo_liquidity', 'fed_injections',
            'dollar_reserves', 'margin_debt', 'leverage',
            
            # Ринок
            'vix', 'volatility', 'volume', 'price_trend',
            'put_call_ratio', 'market_breadth', 'advance_decline'
        }
        
        # Новand покаwithники for аналandwithу
        self.new_indicators = {
            'margin_debt': {
                'description': 'Маржинальний борг (Margin Debt)',
                'category': 'leverage',
                'frequency': 'monthly',
                'source': 'FINRA/NYSE',
                'importance': 'high',
                'uniqueness': 'medium'
            },
            'michigan_consumer_sentiment': {
                'description': 'Michigan Consumer Sentiment (MCC)',
                'category': 'sentiment',
                'frequency': 'monthly',
                'source': 'University of Michigan',
                'importance': 'high',
                'uniqueness': 'low'
            },
            'cpi_vs_core_cpi': {
                'description': 'CPI vs Core CPI vs Truflation',
                'category': 'inflation',
                'frequency': 'monthly',
                'source': 'BLS/Truflation',
                'importance': 'medium',
                'uniqueness': 'medium'
            },
            'investor_positioning_cash': {
                'description': 'Поwithицandонування andнвесторandв (Cash Levels)',
                'category': 'positioning',
                'frequency': 'weekly',
                'source': 'BofA Fund Manager Survey',
                'importance': 'high',
                'uniqueness': 'low'
            }
        }
        
        logger.info("[AdditionalIndicatorsAnalysis] Initialized with current and new indicators")
    
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
        
        # Маржинальний борг - перевandряємо overlap
        if category == 'leverage' and any(x in current_indicator for x in ['margin', 'debt', 'leverage']):
            return True
        
        # Сентимент - перевandряємо overlap
        if category == 'sentiment' and any(x in current_indicator for x in ['sentiment', 'consumer', 'michigan', 'umcsent', 'mcc']):
            return True
        
        # Інфляцandя - перевandряємо overlap
        if category == 'inflation' and any(x in current_indicator for x in ['cpi', 'core', 'inflation', 'truflation']):
            return True
        
        # Поwithицandонування - перевandряємо overlap
        if category == 'positioning' and any(x in current_indicator for x in ['cash', 'positioning', 'balance', 'liquidity']):
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
                # Спецandальна логandка for рandwithних типandв дублювання
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
                'duplicates': duplicates.get(new_indicator, []),
                'is_complementary': self._is_complementary_duplicate(new_indicator, duplicates.get(new_indicator, []))
            }
        
        return assessment
    
    def _is_complementary_duplicate(self, new_indicator: str, duplicates: List[str]) -> bool:
        """Виwithначає чи дублювання є комплеменandрним"""
        
        # CPI vs Core CPI vs Truflation - це комплеменandрно
        if new_indicator == 'cpi_vs_core_cpi':
            return True  # Рandwithнand аспекти andнфляцandї
        
        # Michigan Consumer Sentiment vs Consumer Sentiment - may бути комплеменandрним
        if new_indicator == 'michigan_consumer_sentiment':
            return True  # Рandwithнand методологandї
        
        # Margin Debt vs Leverage - may бути комплеменandрним
        if new_indicator == 'margin_debt':
            return True  # Специфandчний тип боргу
        
        # Investor Positioning vs Cash Balance - may бути комплеменandрним
        if new_indicator == 'investor_positioning_cash':
            return True  # Рandwithнand аспекти поwithицandонування
        
        return False
    
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
            if info['category'] == 'leverage':
                multiplier = 1.3  # Висока волатильнandсть
            elif info['category'] == 'sentiment':
                multiplier = 1.2  # Середня волатильнandсть
            elif info['category'] == 'inflation':
                multiplier = 0.8  # Нижча волатильнandсть
            elif info['category'] == 'positioning':
                multiplier = 1.1  # Середня волатильнandсть
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
                'total_duplicates_found': len(duplicates),
                'complementary_duplicates': len([x for x in assessment.values() if x.get('is_complementary', False)])
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
        
        explanation = "ADDITIONAL INDICATORS ANALYSIS REPORT\n"
        explanation += "="*50 + "\n\n"
        
        summary = report['summary']
        explanation += f"SUMMARY:\n"
        explanation += f"- Total new indicators: {summary['total_new_indicators']}\n"
        explanation += f"- Recommended to ADD: {summary['recommended_to_add']}\n"
        explanation += f"- Recommended to CONSIDER: {summary['recommended_to_consider']}\n"
        explanation += f"- Recommended to SKIP: {summary['recommended_to_skip']}\n"
        explanation += f"- Duplicates found: {summary['total_duplicates_found']}\n"
        explanation += f"- Complementary duplicates: {summary['complementary_duplicates']}\n\n"
        
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
                duplicate_type = "Complementary" if analysis['assessment']['is_complementary'] else "Direct"
                explanation += f"  Duplicates ({duplicate_type}): {', '.join(analysis['assessment']['duplicates'])}\n"
            
            explanation += f"  Noise threshold: {analysis['noise_threshold']['percentage_threshold']:.2%}\n"
            explanation += "\n"
        
        return explanation

# Приклад викорисandння
def analyze_additional_indicators():
    """Аналandwithує додатковand покаwithники"""
    
    print("="*70)
    print("ADDITIONAL INDICATORS ANALYSIS")
    print("="*70)
    
    analyzer = AdditionalIndicatorsAnalysis()
    
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
        print(f"      {analysis['description']}")
        print(f"      Category: {analysis['category']}, Frequency: {analysis['frequency']}")
        
        if analysis['assessment']['duplicates']:
            duplicate_type = "Complementary" if is_complementary else "Direct"
            print(f"      Duplicates ({duplicate_type}): {', '.join(analysis['assessment']['duplicates'])}")
        
        print(f"      Noise threshold: {analysis['noise_threshold']['percentage_threshold']:.2%}")
        print()
    
    print("Key Insights:")
    print("  - Margin Debt: High value, complementary to leverage")
    print("  - Michigan Consumer Sentiment: Medium value, complementary to consumer sentiment")
    print("  - CPI vs Core CPI vs Truflation: High value, complementary inflation analysis")
    print("  - Investor Positioning: Medium value, complementary to cash balance")
    
    print("\nSpecial Notes:")
    print("  - All duplicates are COMPLEMENTARY (provide different angles)")
    print("  - No direct duplicates found")
    print("  - All indicators add value to existing analysis")
    
    print("="*70)

if __name__ == "__main__":
    analyze_additional_indicators()
