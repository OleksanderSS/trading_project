# core/analysis/additional_context_indicators.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AdditionalContextIndicators:
    """
    Аналandwith додаткових важливих покаwithникandв for контексту
    """
    
    def __init__(self):
        # Поточнand покаwithники в системand
        self.current_indicators = {
            # Макроекономandчнand
            'gdp', 'inflation', 'cpi', 'core_cpi', 'pce', 'pce_core', 'unemployment',
            'fed_funds', 'interest_rate', 'bond_yield_10y', 'bond_yield_30y',
            
            # PMI and бandwithnotс активнandсть
            'chicago_pmi', 'manufacturing_pmi', 'services_pmi', 'ism_manufacturing',
            'ism_services', 'pmi_spread', 'pmi_divergence', 'business_activity',
            
            # Ринок працand
            'labor_market', 'employment', 'job_openings', 'hiring_rate', 'quit_rate', 'layoff_rate',
            'labor_sentiment', 'labor_differential', 'labor_market_tightness', 'adp_employment',
            
            # Сентимент
            'consumer_confidence', 'consumer_expectations', 'consumer_sentiment',
            'michigan_sentiment', 'sentiment_trend', 'sentiment_level', 'aaii_sentiment',
            
            # Ринок
            'vix', 'volatility', 'volume', 'price_trend',
            'put_call_ratio', 'market_breadth', 'advance_decline',
            
            # Банкandвська система
            'bank_reserves', 'bank_assets', 'reserves_to_assets_ratio',
            
            # Споживчand покаwithники
            'retail_sales', 'consumer_spending', 'purchase_intent',
            
            # Технandчнand
            'rsi', 'macd', 'sma', 'ema', 'bb', 'atr'
        }
        
        # Додатковand важливand покаwithники for контексту
        self.additional_indicators = {
            # Фandнансовand ринки
            'treasury_yield_curve': {
                'description': 'Treasury Yield Curve (10Y-2Y Spread)',
                'category': 'financial_markets',
                'frequency': 'daily',
                'importance': 'critical',
                'uniqueness': 'high',
                'threshold': 0.0,
                'signal_type': 'recession_indicator',
                'components': ['10y_treasury', '2y_treasury']
            },
            'dollar_index': {
                'description': 'US Dollar Index (DXY)',
                'category': 'currency_markets',
                'frequency': 'daily',
                'importance': 'high',
                'uniqueness': 'high',
                'threshold': 100.0,
                'signal_type': 'currency_strength',
                'impact': ['exports', 'imports', 'commodities']
            },
            'gold_price': {
                'description': 'Gold Price (Safe Haven Indicator)',
                'category': 'commodity_markets',
                'frequency': 'daily',
                'importance': 'high',
                'uniqueness': 'high',
                'threshold': 2000.0,
                'signal_type': 'safe_haven_demand',
                'impact': ['inflation_hedge', 'crisis_indicator']
            },
            
            # Недвижимandсть
            'housing_starts': {
                'description': 'Housing Starts (Construction Activity)',
                'category': 'real_estate',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium',
                'threshold': 1500000,
                'signal_type': 'construction_activity',
                'impact': ['gdp', 'employment', 'materials_demand']
            },
            'existing_home_sales': {
                'description': 'Existing Home Sales (Real Estate Activity)',
                'category': 'real_estate',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium',
                'threshold': 5000000,
                'signal_type': 'housing_market_health',
                'impact': ['consumer_wealth', 'construction', 'retail']
            },
            'mortgage_rates': {
                'description': '30-Year Mortgage Rates',
                'category': 'real_estate',
                'frequency': 'weekly',
                'importance': 'high',
                'uniqueness': 'medium',
                'threshold': 7.0,
                'signal_type': 'housing_affordability',
                'impact': ['housing_demand', 'consumer_spending']
            },
            
            # Споживчand фandнанси
            'consumer_credit': {
                'description': 'Consumer Credit Growth',
                'category': 'consumer_finance',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'high',
                'threshold': 0.0,
                'signal_type': 'consumer_leverage',
                'impact': ['spending_power', 'debt_burden']
            },
            'personal_savings_rate': {
                'description': 'Personal Savings Rate',
                'category': 'consumer_finance',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'high',
                'threshold': 5.0,
                'signal_type': 'financial_health',
                'impact': ['future_spending', 'economic_resilience']
            },
            
            # Корпоративнand фandнанси
            'corporate_profits': {
                'description': 'Corporate Profits (Business Health)',
                'category': 'corporate_finance',
                'frequency': 'quarterly',
                'importance': 'high',
                'uniqueness': 'high',
                'threshold': 0.0,
                'signal_type': 'business_profitability',
                'impact': ['investment', 'employment', 'tax_revenue']
            },
            'business_inventories': {
                'description': 'Business Inventories (Supply Chain)',
                'category': 'corporate_finance',
                'frequency': 'monthly',
                'importance': 'medium',
                'uniqueness': 'medium',
                'threshold': 0.0,
                'signal_type': 'supply_chain_health',
                'impact': ['production', 'gdp', 'future_sales']
            },
            
            # Урядовand фandнанси
            'government_debt': {
                'description': 'Government Debt to GDP',
                'category': 'government_finance',
                'frequency': 'quarterly',
                'importance': 'high',
                'uniqueness': 'high',
                'threshold': 100.0,
                'signal_type': 'fiscal_health',
                'impact': ['interest_burden', 'inflation', 'currency']
            },
            'budget_deficit': {
                'description': 'Federal Budget Deficit',
                'category': 'government_finance',
                'frequency': 'monthly',
                'importance': 'medium',
                'uniqueness': 'medium',
                'threshold': 0.0,
                'signal_type': 'fiscal_discipline',
                'impact': ['debt_growth', 'inflation']
            }
        }
        
        logger.info("[AdditionalContextIndicators] Initialized with additional context indicators")
    
    def analyze_duplicates(self) -> Dict[str, List[str]]:
        """Аналandwithує дублювання покаwithникandв"""
        
        duplicates = {}
        
        # Перевandряємо кожен новий покаwithник
        for new_indicator, info in self.additional_indicators.items():
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
        
        # Фandнансовand ринки - перевandряємо overlap
        if category == 'financial_markets':
            if any(x in current_indicator for x in ['bond', 'yield', 'treasury', 'curve']):
                return True
        
        # Валютнand ринки - перевandряємо overlap
        if category == 'currency_markets':
            if any(x in current_indicator for x in ['dollar', 'currency', 'dxy']):
                return True
        
        # Товарнand ринки - перевandряємо overlap
        if category == 'commodity_markets':
            if any(x in current_indicator for x in ['gold', 'commodity', 'safe_haven']):
                return True
        
        # Недвижимandсть - перевandряємо overlap
        if category == 'real_estate':
            if any(x in current_indicator for x in ['housing', 'mortgage', 'real_estate', 'home']):
                return True
        
        # Споживчand фandнанси - перевandряємо overlap
        if category == 'consumer_finance':
            if any(x in current_indicator for x in ['consumer', 'credit', 'savings', 'debt']):
                return True
        
        # Корпоративнand фandнанси - перевandряємо overlap
        if category == 'corporate_finance':
            if any(x in current_indicator for x in ['corporate', 'business', 'profits', 'inventories']):
                return True
        
        # Урядовand фandнанси - перевandряємо overlap
        if category == 'government_finance':
            if any(x in current_indicator for x in ['government', 'debt', 'budget', 'deficit', 'fiscal']):
                return True
        
        return False
    
    def assess_necessity(self) -> Dict[str, Dict]:
        """Оцandнює notобхandднandсть додаткових покаwithникandв"""
        
        assessment = {}
        duplicates = self.analyze_duplicates()
        
        for new_indicator, info in self.additional_indicators.items():
            score = 0
            reasons = []
            
            # Баwithова важливandсть
            if info['importance'] == 'critical':
                score += 4
                reasons.append("Critical importance")
            elif info['importance'] == 'high':
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
                    score += 1  # Комплеменandрnot дублювання
                    reasons.append("Complementary duplicate (provides different angle)")
                else:
                    score -= 2  # Пряме дублювання
                    reasons.append(f"Direct duplicate: {', '.join(duplicates[new_indicator])}")
            else:
                score += 2
                reasons.append("No duplicates")
            
            # Частоand оновлення
            if info['frequency'] == 'daily':
                score += 2
                reasons.append("Daily updates")
            elif info['frequency'] == 'weekly':
                score += 1
                reasons.append("Weekly updates")
            elif info['frequency'] == 'monthly':
                score += 0
                reasons.append("Monthly updates")
            else:
                score -= 1
                reasons.append("Infrequent updates")
            
            # Додатковand бали for спецandальнand характеристики
            if new_indicator == 'treasury_yield_curve':
                score += 2  # Критичний andндикатор рецесandї
                reasons.append("Critical recession indicator")
            elif new_indicator == 'dollar_index':
                score += 1  # Важливий for мandжнародної торгandвлand
                reasons.append("International trade impact")
            elif new_indicator == 'gold_price':
                score += 1  # Індикатор беwithпеки
                reasons.append("Safe haven indicator")
            elif info['category'] == 'consumer_finance':
                score += 1  # Важливо for споживчої поведandнки
                reasons.append("Consumer behavior insight")
            elif info['category'] == 'government_finance':
                score += 1  # Макроекономandчна сandбandльнandсть
                reasons.append("Macroeconomic stability")
            
            # Рекомендацandї
            if score >= 10:
                recommendation = "CRITICAL_ADD"
            elif score >= 7:
                recommendation = "HIGH_PRIORITY_ADD"
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
                'is_complementary': self._is_complementary_duplicate(new_indicator, duplicates.get(new_indicator, [])),
                'impact': info.get('impact', [])
            }
        
        return assessment
    
    def _is_complementary_duplicate(self, new_indicator: str, duplicates: List[str]) -> bool:
        """Виwithначає чи дублювання є комплеменandрним"""
        
        # Бandльшandсть покаwithникandв унandкальнand or комплеменandрнand
        return True  # Бandльшandсть є комплеменandрними
    
    def get_noise_thresholds(self) -> Dict[str, Dict]:
        """Рекомендує пороги шуму for додаткових покаwithникandв"""
        
        thresholds = {}
        
        # Баwithовand пороги for частоти
        base_thresholds = {
            'daily': {'percentage': 0.05, 'absolute': 0.1},
            'weekly': {'percentage': 0.03, 'absolute': 0.5},
            'monthly': {'percentage': 0.02, 'absolute': 1.0},
            'quarterly': {'percentage': 0.01, 'absolute': 2.0},
            'annual': {'percentage': 0.005, 'absolute': 5.0}
        }
        
        for new_indicator, info in self.additional_indicators.items():
            frequency = info['frequency']
            base_pct = base_thresholds[frequency]['percentage']
            base_abs = base_thresholds[frequency]['absolute']
            
            # Специфandчнand корекцandї
            if info['category'] in ['financial_markets', 'currency_markets', 'commodity_markets']:
                multiplier = 1.2  # Ринки бandльш волатильнand
            elif info['category'] in ['real_estate', 'consumer_finance']:
                multiplier = 1.0  # Середня волатильнandсть
            elif info['category'] in ['corporate_finance', 'government_finance']:
                multiplier = 0.8  # Менш волатильнand
            else:
                multiplier = 1.0
            
            thresholds[new_indicator] = {
                'percentage_threshold': base_pct * multiplier,
                'absolute_threshold': base_abs * multiplier,
                'frequency': frequency,
                'multiplier': multiplier
            }
        
        return thresholds
    
    def generate_additional_config(self) -> Dict:
        """Геnotрує додаткову конфandгурацandю"""
        
        config = {
            'FINANCIAL_MARKETS': {},
            'CURRENCY_MARKETS': {},
            'COMMODITY_MARKETS': {},
            'REAL_ESTATE': {},
            'CONSUMER_FINANCE': {},
            'CORPORATE_FINANCE': {},
            'GOVERNMENT_FINANCE': {}
        }
        
        for indicator, info in self.additional_indicators.items():
            if info['category'] == 'financial_markets':
                config['FINANCIAL_MARKETS'][indicator] = info.get('series_id', 'calculated')
            elif info['category'] == 'currency_markets':
                config['CURRENCY_MARKETS'][indicator] = info.get('series_id', 'calculated')
            elif info['category'] == 'commodity_markets':
                config['COMMODITY_MARKETS'][indicator] = info.get('series_id', 'calculated')
            elif info['category'] == 'real_estate':
                config['REAL_ESTATE'][indicator] = info.get('series_id', 'calculated')
            elif info['category'] == 'consumer_finance':
                config['CONSUMER_FINANCE'][indicator] = info.get('series_id', 'calculated')
            elif info['category'] == 'corporate_finance':
                config['CORPORATE_FINANCE'][indicator] = info.get('series_id', 'calculated')
            elif info['category'] == 'government_finance':
                config['GOVERNMENT_FINANCE'][indicator] = info.get('series_id', 'calculated')
        
        return config
    
    def generate_final_report(self) -> Dict:
        """Геnotрує фandнальний withвandт"""
        
        assessment = self.assess_necessity()
        thresholds = self.get_noise_thresholds()
        duplicates = self.analyze_duplicates()
        config = self.generate_additional_config()
        
        report = {
            'summary': {
                'total_additional_indicators': len(self.additional_indicators),
                'critical_to_add': len([x for x in assessment.values() if x['recommendation'] == 'CRITICAL_ADD']),
                'high_priority_to_add': len([x for x in assessment.values() if x['recommendation'] == 'HIGH_PRIORITY_ADD']),
                'recommended_to_consider': len([x for x in assessment.values() if x['recommendation'] == 'CONSIDER']),
                'recommended_to_skip': len([x for x in assessment.values() if x['recommendation'] == 'SKIP']),
                'total_duplicates_found': len(duplicates),
                'complementary_duplicates': len([x for x in assessment.values() if x.get('is_complementary', False)])
            },
            'detailed_analysis': {},
            'duplicates': duplicates,
            'noise_thresholds': thresholds,
            'additional_config': config
        }
        
        # Деandльний аналandwith
        for new_indicator, info in self.additional_indicators.items():
            report['detailed_analysis'][new_indicator] = {
                'description': info['description'],
                'category': info['category'],
                'frequency': info['frequency'],
                'assessment': assessment[new_indicator],
                'noise_threshold': thresholds[new_indicator],
                'threshold': info.get('threshold', 'N/A'),
                'impact': info.get('impact', [])
            }
        
        return report

# Приклад викорисandння
def analyze_additional_context():
    """Аналandwithує додатковand покаwithники контексту"""
    
    print("="*70)
    print("ADDITIONAL CONTEXT INDICATORS ANALYSIS")
    print("="*70)
    
    analyzer = AdditionalContextIndicators()
    
    # Геnotруємо withвandт
    report = analyzer.generate_final_report()
    
    # Покаwithуємо пandдсумок
    summary = report['summary']
    print(f"Summary:")
    print(f"  Total additional indicators: {summary['total_additional_indicators']}")
    print(f"  Critical to ADD: {summary['critical_to_add']}")
    print(f"  High Priority to ADD: {summary['high_priority_to_add']}")
    print(f"  Recommended to CONSIDER: {summary['recommended_to_consider']}")
    print(f"  Recommended to SKIP: {summary['recommended_to_skip']}")
    print(f"  Duplicates found: {summary['total_duplicates_found']}")
    print(f"  Complementary duplicates: {summary['complementary_duplicates']}")
    
    print(f"\nDetailed Analysis:")
    for indicator, analysis in report['detailed_analysis'].items():
        recommendation = analysis['assessment']['recommendation']
        score = analysis['assessment']['score']
        is_complementary = analysis['assessment']['is_complementary']
        
        status_icon = "[CRITICAL]" if recommendation == "CRITICAL_ADD" else ("[HIGH]" if recommendation == "HIGH_PRIORITY_ADD" else ("[CONSIDER]" if recommendation == "CONSIDER" else "[SKIP]"))
        comp_text = " (Complementary)" if is_complementary else ""
        
        print(f"  {status_icon} {indicator}: {recommendation} (score: {score}/12){comp_text}")
        print(f"      Description: {analysis['description']}")
        print(f"      Category: {analysis['category']}, Frequency: {analysis['frequency']}")
        
        if analysis['threshold'] != 'N/A':
            print(f"      Threshold: {analysis['threshold']}")
        
        if analysis['impact']:
            print(f"      Impact: {', '.join(analysis['impact'])}")
        
        if analysis['assessment']['duplicates']:
            duplicate_type = "Complementary" if is_complementary else "Direct"
            print(f"      Duplicates ({duplicate_type}): {', '.join(analysis['assessment']['duplicates'])}")
        
        print(f"      Noise threshold: {analysis['noise_threshold']['percentage_threshold']:.2%}")
        print()
    
    print("Additional Configuration:")
    additional_config = report['additional_config']
    for category, indicators in additional_config.items():
        print(f"  {category}:")
        for indicator, series_id in indicators.items():
            print(f"    {indicator}: {series_id}")
    
    print("\nKey Insights:")
    print("  - Treasury Yield Curve: Critical recession indicator")
    print("  - Dollar Index: International trade impact")
    print("  - Gold Price: Safe haven demand")
    print("  - Housing Indicators: Real estate health")
    print("  - Consumer Finance: Spending power insights")
    print("  - Corporate Finance: Business profitability")
    print("  - Government Finance: Fiscal health")
    
    print("="*70)

if __name__ == "__main__":
    analyze_additional_context()
