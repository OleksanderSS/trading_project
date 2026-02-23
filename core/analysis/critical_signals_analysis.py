# core/analysis/critical_signals_analysis.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CriticalSignalsAnalysis:
    """
    Аналandwith критичних сигналandв and problems data
    """
    
    def __init__(self):
        # Поточнand покаwithники в системand
        self.current_indicators = {
            # PMI покаwithники
            'chicago_pmi', 'manufacturing_pmi', 'services_pmi', 'ism_manufacturing',
            'ism_services', 'pmi_spread', 'pmi_divergence', 'business_activity',
            
            # Ринок працand
            'labor_market', 'labor_fragmentation', 'employment_quality',
            'job_openings', 'hiring_rate', 'quit_rate', 'layoff_rate',
            'labor_sentiment', 'labor_differential', 'labor_market_tightness',
            
            # Сентимент
            'consumer_confidence', 'consumer_expectations', 'consumer_sentiment',
            'michigan_sentiment', 'sentiment_trend', 'sentiment_level',
            
            # Інфляцandя
            'ppi', 'producer_price_index', 'inflation', 'core_inflation',
            'cpi', 'core_cpi', 'pce_core', 'pce_total'
        }
        
        # Новand критичнand сигнали
        self.new_critical_signals = {
            'chicago_pmi_black_swan': {
                'description': 'Chicago PMI "Black Swan" Signal',
                'category': 'critical_pmi',
                'frequency': 'monthly',
                'importance': 'critical',
                'uniqueness': 'high',
                'signal_type': 'black_swan',
                'threshold': 45.0,  # Black Swan рandвень
                'impact': 'market_crash_warning'
            },
            'labor_market_fragmentation': {
                'description': 'Labor Market Fragmentation (Роwithшарування)',
                'category': 'critical_labor',
                'frequency': 'monthly',
                'importance': 'critical',
                'uniqueness': 'high',
                'signal_type': 'fragmentation',
                'components': ['skill_gap', 'wage_polarization', 'sector_mismatch'],
                'impact': 'structural_employment_issues'
            },
            'consumer_confidence_breakdown': {
                'description': 'Consumer Confidence (88.7) Psychological Breakdown',
                'category': 'critical_sentiment',
                'frequency': 'monthly',
                'importance': 'critical',
                'uniqueness': 'high',
                'signal_type': 'psychological_breakdown',
                'threshold': 85.0,  # Психологandчний withлам
                'impact': 'consumer_behavior_shift'
            },
            'shutdown_data_lag': {
                'description': 'Shutdown & Data Lag Problem',
                'category': 'data_quality',
                'frequency': 'irregular',
                'importance': 'critical',
                'uniqueness': 'high',
                'signal_type': 'data_lag',
                'problem': 'september_data_in_january',
                'impact': 'forecasting_vs_learning'
            },
            'ppi_revision_trend': {
                'description': 'PPI Revision Trend (andнфляцandя виробникandв)',
                'category': 'inflation_validation',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium',
                'signal_type': 'trend_validation',
                'components': ['ppi_initial', 'ppi_revised', 'revision_direction'],
                'impact': 'stagflation_confirmation'
            }
        }
        
        logger.info("[CriticalSignalsAnalysis] Initialized with critical signals")
    
    def analyze_duplicates(self) -> Dict[str, List[str]]:
        """Аналandwithує дублювання покаwithникandв"""
        
        duplicates = {}
        
        # Перевandряємо кожен новий покаwithник
        for new_signal, info in self.new_critical_signals.items():
            similar_indicators = []
            
            # Шукаємо схожand for категорandєю
            for current_indicator in self.current_indicators:
                if self._are_similar(new_signal, current_indicator, info['category']):
                    similar_indicators.append(current_indicator)
            
            if similar_indicators:
                duplicates[new_signal] = similar_indicators
        
        return duplicates
    
    def _are_similar(self, new_signal: str, current_indicator: str, category: str) -> bool:
        """Виwithначає чи покаwithники схожand"""
        
        # PMI покаwithники - перевandряємо overlap
        if category == 'critical_pmi':
            if any(x in current_indicator for x in ['pmi', 'chicago', 'manufacturing', 'services', 'ism']):
                return True
        
        # Ринок працand - перевandряємо overlap
        if category == 'critical_labor':
            if any(x in current_indicator for x in ['labor', 'employment', 'job', 'hiring', 'fragmentation']):
                return True
        
        # Сентимент - перевandряємо overlap
        if category == 'critical_sentiment':
            if any(x in current_indicator for x in ['confidence', 'sentiment', 'consumer', 'expectation']):
                return True
        
        # Якandсть data - унandкальна категорandя
        if category == 'data_quality':
            return False  # Унandкальний покаwithник
        
        # Інфляцandя - перевandряємо overlap
        if category == 'inflation_validation':
            if any(x in current_indicator for x in ['ppi', 'inflation', 'producer', 'price']):
                return True
        
        return False
    
    def assess_criticality(self) -> Dict[str, Dict]:
        """Оцandнює критичнandсть сигналandв"""
        
        assessment = {}
        duplicates = self.analyze_duplicates()
        
        for new_signal, info in self.new_critical_signals.items():
            score = 0
            reasons = []
            
            # Критична важливandсть
            if info['importance'] == 'critical':
                score += 5  # Максимальний бал for критичних
                reasons.append("Critical importance")
            elif info['importance'] == 'high':
                score += 3
                reasons.append("High importance")
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
            if new_signal in duplicates:
                if self._is_complementary_duplicate(new_signal, duplicates[new_signal]):
                    score += 1  # Комплеменandрnot дублювання
                    reasons.append("Complementary duplicate (provides different angle)")
                else:
                    score -= 1  # Пряме дублювання
                    reasons.append(f"Direct duplicate: {', '.join(duplicates[new_signal])}")
            else:
                score += 2
                reasons.append("No duplicates")
            
            # Тип сигналу
            signal_type = info.get('signal_type', 'unknown')
            if signal_type == 'black_swan':
                score += 3
                reasons.append("Black Swan signal - extremely valuable")
            elif signal_type == 'psychological_breakdown':
                score += 3
                reasons.append("Psychological breakdown - critical")
            elif signal_type == 'fragmentation':
                score += 2
                reasons.append("Fragmentation signal - structural")
            elif signal_type == 'data_lag':
                score += 2
                reasons.append("Data lag problem - systemic")
            elif signal_type == 'trend_validation':
                score += 1
                reasons.append("Trend validation - useful")
            
            # Вплив
            impact = info.get('impact', 'unknown')
            if impact in ['market_crash_warning', 'structural_employment_issues', 'consumer_behavior_shift']:
                score += 2
                reasons.append(f"High impact: {impact}")
            elif impact in ['forecasting_vs_learning', 'stagflation_confirmation']:
                score += 1
                reasons.append(f"Medium impact: {impact}")
            
            # Рекомендацandї
            if score >= 10:
                recommendation = "CRITICAL_ADD"
            elif score >= 7:
                recommendation = "HIGH_PRIORITY_ADD"
            elif score >= 4:
                recommendation = "CONSIDER"
            else:
                recommendation = "SKIP"
            
            assessment[new_signal] = {
                'score': score,
                'recommendation': recommendation,
                'reasons': reasons,
                'category': info['category'],
                'frequency': info['frequency'],
                'duplicates': duplicates.get(new_signal, []),
                'is_complementary': self._is_complementary_duplicate(new_signal, duplicates.get(new_signal, [])),
                'signal_type': signal_type,
                'impact': impact
            }
        
        return assessment
    
    def _is_complementary_duplicate(self, new_signal: str, duplicates: List[str]) -> bool:
        """Виwithначає чи дублювання є комплеменandрним"""
        
        # Chicago PMI Black Swan - унandкальний сигнал
        if new_signal == 'chicago_pmi_black_swan':
            return True  # Унandкальний Black Swan аспект
        
        # Labor fragmentation - унandкальний покаwithник
        if new_signal == 'labor_market_fragmentation':
            return True  # Унandкальний структурний покаwithник
        
        # Consumer confidence breakdown - комплеменandрно
        if new_signal == 'consumer_confidence_breakdown':
            return True  # Психологandчний аспект
        
        # Data lag - унandкальна problemsа
        if new_signal == 'shutdown_data_lag':
            return False  # Унandкальний покаwithник
        
        # PPI revision - комплеменandрно
        if new_signal == 'ppi_revision_trend':
            return True  # Валandдацandя тренду
        
        return False
    
    def get_noise_thresholds(self) -> Dict[str, Dict]:
        """Рекомендує пороги шуму for критичних сигналandв"""
        
        thresholds = {}
        
        # Баwithовand пороги for частоти
        base_thresholds = {
            'daily': {'percentage': 0.05, 'absolute': 0.1},
            'weekly': {'percentage': 0.03, 'absolute': 0.5},
            'monthly': {'percentage': 0.02, 'absolute': 1.0},
            'quarterly': {'percentage': 0.01, 'absolute': 2.0},
            'irregular': {'percentage': 0.01, 'absolute': 5.0}
        }
        
        for new_signal, info in self.new_critical_signals.items():
            frequency = info['frequency']
            base_pct = base_thresholds[frequency]['percentage']
            base_abs = base_thresholds[frequency]['absolute']
            
            # Специфandчнand корекцandї for критичних сигналandв
            if info['category'] == 'critical_pmi':
                multiplier = 0.7  # Нижчий шум for критичних PMI
            elif info['category'] == 'critical_labor':
                multiplier = 0.8  # Нижчий шум for структурних
            elif info['category'] == 'critical_sentiment':
                multiplier = 0.9  # Середнandй шум for сентименту
            elif info['category'] == 'data_quality':
                multiplier = 0.5  # Дуже ниwithький шум for якостand data
            elif info['category'] == 'inflation_validation':
                multiplier = 0.8  # Нижчий шум for валandдацandї
            else:
                multiplier = 1.0
            
            thresholds[new_signal] = {
                'percentage_threshold': base_pct * multiplier,
                'absolute_threshold': base_abs * multiplier,
                'frequency': frequency,
                'multiplier': multiplier
            }
        
        return thresholds
    
    def generate_data_lag_solution(self) -> Dict:
        """Геnotрує рandшення for problemsи лагу data"""
        
        solution = {
            'problem': 'September data released in January',
            'impact': {
                'forecasting': 'September data is "garbage" for tomorrow prediction',
                'learning': 'September data is "gold" for model training',
                'conflict': 'Same data has different value for different purposes'
            },
            'solution': {
                'separate_pipelines': {
                    'operational_pipeline': {
                        'purpose': 'Real-time trading signals',
                        'data_sources': ['real_time_data', 'nowcasts', 'high_frequency'],
                        'exclude': ['delayed_revisions', 'seasonal_adjustments_late']
                    },
                    'learning_pipeline': {
                        'purpose': 'Model training and validation',
                        'data_sources': ['historical_revisions', 'complete_data', 'seasonal_patterns'],
                        'include': ['delayed_revisions', 'final_revised_data']
                    }
                },
                'data_classification': {
                    'real_time': 'For operational decisions',
                    'revised': 'For model learning',
                    'final': 'For backtesting'
                },
                'validation_logic': {
                    'trend_validation': 'Use revised data to confirm trends',
                    'example': 'PPI up + PMI down = Stagflation confirmation',
                    'operational_use': 'Real-time PMI for trading',
                    'learning_use': 'Revised PPI for model training'
                }
            },
            'implementation': {
                'data_tagging': 'Tag all data with release_date and revision_status',
                'pipeline_routing': 'Route data to appropriate pipeline',
                'model_integration': 'Use different data for different model components'
            }
        }
        
        return solution
    
    def generate_critical_config(self) -> Dict:
        """Геnotрує конфandгурацandю for критичних сигналandв"""
        
        config = {
            'CRITICAL_PMI': {},
            'CRITICAL_LABOR': {},
            'CRITICAL_SENTIMENT': {},
            'DATA_QUALITY': {},
            'INFLATION_VALIDATION': {}
        }
        
        for signal, info in self.new_critical_signals.items():
            if info['category'] == 'critical_pmi':
                config['CRITICAL_PMI'][signal] = info.get('threshold', 'calculated')
            elif info['category'] == 'critical_labor':
                config['CRITICAL_LABOR'][signal] = 'calculated'
            elif info['category'] == 'critical_sentiment':
                config['CRITICAL_SENTIMENT'][signal] = info.get('threshold', 'calculated')
            elif info['category'] == 'data_quality':
                config['DATA_QUALITY'][signal] = 'monitored'
            elif info['category'] == 'inflation_validation':
                config['INFLATION_VALIDATION'][signal] = 'calculated'
        
        return config
    
    def generate_final_report(self) -> Dict:
        """Геnotрує фandнальний withвandт"""
        
        assessment = self.assess_criticality()
        thresholds = self.get_noise_thresholds()
        duplicates = self.analyze_duplicates()
        config = self.generate_critical_config()
        data_lag_solution = self.generate_data_lag_solution()
        
        report = {
            'summary': {
                'total_new_signals': len(self.new_critical_signals),
                'critical_add': len([x for x in assessment.values() if x['recommendation'] == 'CRITICAL_ADD']),
                'high_priority_add': len([x for x in assessment.values() if x['recommendation'] == 'HIGH_PRIORITY_ADD']),
                'recommended_to_consider': len([x for x in assessment.values() if x['recommendation'] == 'CONSIDER']),
                'recommended_to_skip': len([x for x in assessment.values() if x['recommendation'] == 'SKIP']),
                'total_duplicates_found': len(duplicates),
                'complementary_duplicates': len([x for x in assessment.values() if x.get('is_complementary', False)])
            },
            'detailed_analysis': {},
            'duplicates': duplicates,
            'noise_thresholds': thresholds,
            'critical_config': config,
            'data_lag_solution': data_lag_solution
        }
        
        # Деandльний аналandwith
        for new_signal, info in self.new_critical_signals.items():
            report['detailed_analysis'][new_signal] = {
                'description': info['description'],
                'category': info['category'],
                'frequency': info['frequency'],
                'assessment': assessment[new_signal],
                'noise_threshold': thresholds[new_signal],
                'signal_type': info['signal_type'],
                'impact': info['impact'],
                'threshold': info.get('threshold', 'N/A'),
                'components': info.get('components', [])
            }
        
        return report

# Приклад викорисandння
def analyze_critical_signals():
    """Аналandwithує критичнand сигнали"""
    
    print("="*70)
    print("CRITICAL SIGNALS ANALYSIS")
    print("="*70)
    
    analyzer = CriticalSignalsAnalysis()
    
    # Геnotруємо withвandт
    report = analyzer.generate_final_report()
    
    # Покаwithуємо пandдсумок
    summary = report['summary']
    print(f"Summary:")
    print(f"  Total new critical signals: {summary['total_new_signals']}")
    print(f"  Critical to ADD: {summary['critical_add']}")
    print(f"  High Priority to ADD: {summary['high_priority_add']}")
    print(f"  Recommended to CONSIDER: {summary['recommended_to_consider']}")
    print(f"  Recommended to SKIP: {summary['recommended_to_skip']}")
    print(f"  Duplicates found: {summary['total_duplicates_found']}")
    print(f"  Complementary duplicates: {summary['complementary_duplicates']}")
    
    print(f"\nDetailed Analysis:")
    for signal, analysis in report['detailed_analysis'].items():
        recommendation = analysis['assessment']['recommendation']
        score = analysis['assessment']['score']
        is_complementary = analysis['assessment']['is_complementary']
        
        status_icon = "[CRITICAL]" if recommendation == "CRITICAL_ADD" else ("[HIGH]" if recommendation == "HIGH_PRIORITY_ADD" else ("[CONSIDER]" if recommendation == "CONSIDER" else "[SKIP]"))
        comp_text = " (Complementary)" if is_complementary else ""
        
        print(f"  {status_icon} {signal}: {recommendation} (score: {score}/15){comp_text}")
        print(f"      Description: {analysis['description']}")
        print(f"      Category: {analysis['category']}, Frequency: {analysis['frequency']}")
        print(f"      Signal Type: {analysis['signal_type']}, Impact: {analysis['impact']}")
        
        if analysis['threshold'] != 'N/A':
            print(f"      Threshold: {analysis['threshold']}")
        
        if analysis['components']:
            print(f"      Components: {', '.join(analysis['components'])}")
        
        if analysis['assessment']['duplicates']:
            duplicate_type = "Complementary" if is_complementary else "Direct"
            print(f"      Duplicates ({duplicate_type}): {', '.join(analysis['assessment']['duplicates'])}")
        
        print(f"      Noise threshold: {analysis['noise_threshold']['percentage_threshold']:.2%}")
        print()
    
    # Покаwithуємо рandшення for problemsи лагу data
    print("Data Lag Solution:")
    solution = report['data_lag_solution']
    print(f"  Problem: {solution['problem']}")
    print(f"  Impact:")
    for impact_type, impact_desc in solution['impact'].items():
        print(f"    {impact_type}: {impact_desc}")
    print(f"  Solution:")
    print(f"    Separate pipelines for operational vs learning")
    print(f"    Data classification: real_time, revised, final")
    print(f"    Validation logic: Use revised data for trend confirmation")
    
    print("\nKey Insights:")
    print("  - Chicago PMI Black Swan: Critical market crash warning")
    print("  - Labor Fragmentation: Structural employment issues")
    print("  - Consumer Confidence Breakdown: Psychological shift")
    print("  - Data Lag: Need separate operational vs learning pipelines")
    print("  - PPI Revision: Trend validation for stagflation")
    
    print("="*70)

if __name__ == "__main__":
    analyze_critical_signals()
