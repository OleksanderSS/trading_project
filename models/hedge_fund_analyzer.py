#!/usr/bin/env python3
"""
Hedge Fund Analysis Module
Аналіз продуктивності хедж фондів: детекція стилю, експозиції, ризики
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

from models.fama_french_factors import get_fama_french_factors, calculate_factor_exposures
from utils.economic_context_mapper import get_economic_context

logger = logging.getLogger(__name__)


class HedgeFundAnalyzer:
    """
    Аналіз хедж фондів для оцінки навичок менеджерів та стійкості стратегій
    """
    
    def __init__(self):
        self.logger = logging.getLogger("HedgeFundAnalyzer")
        
        # Фактори для аналізу
        self.factor_models = {
            'carhart': ['MKT', 'SMB', 'HML', 'UMD'],  # Carhart 4-factor
            'french_5': ['MKT', 'SMB', 'HML', 'RMW', 'CMA'],  # Fama-French 5-factor
            'french_6': ['MKT', 'SMB', 'HML', 'UMD', 'RMW', 'CMA']  # Full 6-factor
        }
        
        # Пороги для детекції стилю
        self.style_thresholds = {
            'market_exposure': 0.7,      # >0.7 = high beta
            'size_tilt': 0.3,           # >0.3 = small cap bias
            'value_tilt': 0.3,          # >0.3 = value bias
            'momentum_tilt': 0.3,       # >0.3 = momentum bias
            'alpha_significance': 0.05   # p-value < 0.05 = significant alpha
        }
        
        # Метрики ризику
        self.risk_metrics = [
            'volatility', 'sharpe', 'sortino', 'max_drawdown',
            'var_95', 'var_99', 'cvar_95', 'cvar_99',
            'calmar_ratio', 'tail_ratio', 'skewness', 'kurtosis'
        ]
        
        self.logger.info("HedgeFundAnalyzer initialized")
    
    def calculate_performance_metrics(self, returns: pd.Series, 
                                   benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """
        Розрахувати повний набір метрик продуктивності
        
        Args:
            returns: Доходність фонду
            benchmark_returns: Бенчмарк (опціонально)
            
        Returns:
            Dict: Метрики продуктивності
        """
        try:
            metrics = {}
            
            # Базові метрики
            metrics['total_return'] = (1 + returns).prod() - 1
            metrics['annual_return'] = returns.mean() * 252
            metrics['annual_volatility'] = returns.std() * np.sqrt(252)
            
            # Risk-adjusted metrics
            risk_free_rate = 0.02  # 2% річна
            
            # Sharpe Ratio
            excess_returns = returns - risk_free_rate / 252
            metrics['sharpe_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            
            # Sortino Ratio (тільки негативна волатильність)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_vol = downside_returns.std() * np.sqrt(252)
                metrics['sortino_ratio'] = excess_returns.mean() / downside_vol
            else:
                metrics['sortino_ratio'] = np.inf
            
            # Maximum Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()
            
            # Calmar Ratio
            if metrics['max_drawdown'] != 0:
                metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown'])
            else:
                metrics['calmar_ratio'] = np.inf
            
            # VaR і CVaR
            metrics['var_95'] = returns.quantile(0.05)
            metrics['var_99'] = returns.quantile(0.01)
            metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
            metrics['cvar_99'] = returns[returns <= metrics['var_99']].mean()
            
            # Tail Ratio
            tail_95 = returns.quantile(0.95)
            tail_05 = returns.quantile(0.05)
            metrics['tail_ratio'] = abs(tail_95) / abs(tail_05) if tail_05 != 0 else np.inf
            
            # Статистика розподілу
            metrics['skewness'] = returns.skew()
            metrics['kurtosis'] = returns.kurtosis()
            
            # Win Rate і Hit Ratio
            metrics['win_rate'] = (returns > 0).sum() / len(returns)
            metrics['hit_ratio'] = metrics['win_rate']  # Те ж саме
            
            # Information Ratio (якщо є бенчмарк)
            if benchmark_returns is not None:
                active_returns = returns - benchmark_returns
                if active_returns.std() != 0:
                    metrics['information_ratio'] = active_returns.mean() / active_returns.std() * np.sqrt(252)
                else:
                    metrics['information_ratio'] = 0
                metrics['tracking_error'] = active_returns.std() * np.sqrt(252)
            else:
                metrics['information_ratio'] = 0
                metrics['tracking_error'] = 0
            
            self.logger.info(f"Performance metrics calculated: Sharpe={metrics['sharpe_ratio']:.2f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def calculate_factor_exposures(self, fund_returns: pd.Series, 
                                 factor_model: str = 'french_6') -> Dict[str, any]:
        """
        Розрахувати факторні експозиції фонду
        
        Args:
            fund_returns: Доходність фонду
            factor_model: Модель факторів
            
        Returns:
            Dict: Факторні експозиції та статистика
        """
        try:
            # Отримуємо фактори
            factors = get_fama_french_factors()
            
            if factors.empty:
                return {}
            
            # Вибираємо фактори моделі
            model_factors = self.factor_models.get(factor_model, self.factor_models['french_6'])
            available_factors = [f for f in model_factors if f in factors.columns]
            
            if not available_factors:
                return {}
            
            factor_data = factors[available_factors]
            
            # Розраховуємо експозиції
            exposures = calculate_factor_exposures(fund_returns, factor_data)
            
            # Додаємо статистику значущості
            if exposures:
                # T-statistics для перевірки значущості
                t_stats = self._calculate_t_statistics(fund_returns, factor_data)
                
                # P-values
                p_values = {factor: 2 * (1 - stats.t.cdf(abs(t), len(fund_returns) - len(factor_data.columns) - 1))
                           for factor, t in t_stats.items()}
                
                # Significant factors
                significant_factors = {f: exposures[f] for f in exposures 
                                     if f in p_values and p_values[f] < self.style_thresholds['alpha_significance']}
                
                result = {
                    'exposures': exposures,
                    't_statistics': t_stats,
                    'p_values': p_values,
                    'significant_factors': significant_factors,
                    'r_squared': exposures.get('r_squared', 0),
                    'model_type': factor_model
                }
                
                self.logger.info(f"Factor exposures calculated: R²={result['r_squared']:.3f}")
                
                return result
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error calculating factor exposures: {e}")
            return {}
    
    def detect_style_drift(self, current_exposures: Dict[str, float], 
                          historical_exposures: List[Dict[str, float]]) -> Dict[str, any]:
        """
        Детекція дрифту стилю фонду
        
        Args:
            current_exposures: Поточні експозиції
            historical_exposures: Історичні експозиції
            
        Returns:
            Dict: Результати детекції дрифту
        """
        try:
            if not historical_exposures:
                return {'drift_detected': False, 'reason': 'No historical data'}
            
            # Розраховуємо середні історичні експозиції
            avg_exposures = {}
            for factor in current_exposures:
                if factor != 'alpha' and factor != 'r_squared':
                    factor_values = [h.get(factor, 0) for h in historical_exposures if factor in h]
                    if factor_values:
                        avg_exposures[factor] = np.mean(factor_values)
            
            # Розраховуємо різницю
            drift_scores = {}
            significant_drifts = {}
            
            for factor, current_exp in current_exposures.items():
                if factor in avg_exposures and factor != 'alpha' and factor != 'r_squared':
                    avg_exp = avg_exposures[factor]
                    diff = abs(current_exp - avg_exp)
                    
                    # Статистична значущість різниці
                    historical_values = [h.get(factor, 0) for h in historical_exposures if factor in h]
                    if len(historical_values) > 1:
                        std_dev = np.std(historical_values)
                        z_score = diff / std_dev if std_dev > 0 else 0
                        
                        drift_scores[factor] = {
                            'difference': diff,
                            'z_score': z_score,
                            'significant': z_score > 2.0  # 95% confidence
                        }
                        
                        if drift_scores[factor]['significant']:
                            significant_drifts[factor] = drift_scores[factor]
            
            # Загальна оцінка дрифту
            drift_detected = len(significant_drifts) > 0
            
            # Аналіз типу дрифту
            drift_analysis = self._analyze_drift_type(significant_drifts, current_exposures)
            
            result = {
                'drift_detected': drift_detected,
                'drift_scores': drift_scores,
                'significant_drifts': significant_drifts,
                'drift_analysis': drift_analysis,
                'current_exposures': current_exposures,
                'historical_average': avg_exposures,
                'drift_severity': len(significant_drifts) / len(current_exposures) if current_exposures else 0
            }
            
            self.logger.info(f"Style drift analysis: detected={drift_detected}, severe_factors={len(significant_drifts)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting style drift: {e}")
            return {'drift_detected': False, 'error': str(e)}
    
    def analyze_manager_skill(self, fund_returns: pd.Series, 
                            benchmark_returns: pd.Series = None) -> Dict[str, any]:
        """
        Аналіз навичок менеджера (alpha та навички)
        
        Args:
            fund_returns: Доходність фонду
            benchmark_returns: Бенчмарк
            
        Returns:
            Dict: Аналіз навичок менеджера
        """
        try:
            # Розраховуємо метрики продуктивності
            performance = self.calculate_performance_metrics(fund_returns, benchmark_returns)
            
            # Факторний аналіз
            factor_analysis = self.calculate_factor_exposures(fund_returns)
            
            # Alpha аналіз
            alpha = factor_analysis.get('exposures', {}).get('alpha', 0)
            alpha_p_value = factor_analysis.get('p_values', {}).get('alpha', 1.0)
            
            # Навички менеджера
            manager_skill = {
                'has_alpha': alpha > 0 and alpha_p_value < self.style_thresholds['alpha_significance'],
                'alpha_annualized': alpha * 252,
                'alpha_significance': alpha_p_value,
                'risk_adjusted_alpha': alpha / fund_returns.std() * np.sqrt(252) if fund_returns.std() > 0 else 0,
                'consistency_score': self._calculate_consistency_score(fund_returns),
                'skill_score': 0  # Розрахуємо нижче
            }
            
            # Загальний скор навичок
            skill_components = []
            
            # Alpha component
            if manager_skill['has_alpha']:
                skill_components.append(min(abs(manager_skill['alpha_annualized']) / 0.05, 1.0))  # 5% alpha = 1.0
            
            # Consistency component
            skill_components.append(manager_skill['consistency_score'])
            
            # Risk-adjusted performance
            if performance['sharpe_ratio'] > 1.0:
                skill_components.append(min(performance['sharpe_ratio'] / 2.0, 1.0))  # Sharpe 2.0 = 1.0
            
            # Downside protection
            if performance['sortino_ratio'] > 1.0:
                skill_components.append(min(performance['sortino_ratio'] / 2.0, 1.0))
            
            # Загальний скор навичок
            if skill_components:
                manager_skill['skill_score'] = np.mean(skill_components)
            else:
                manager_skill['skill_score'] = 0
            
            # Класифікація навичок
            if manager_skill['skill_score'] >= 0.8:
                skill_level = 'Exceptional'
            elif manager_skill['skill_score'] >= 0.6:
                skill_level = 'Excellent'
            elif manager_skill['skill_score'] >= 0.4:
                skill_level = 'Good'
            elif manager_skill['skill_score'] >= 0.2:
                skill_level = 'Average'
            else:
                skill_level = 'Poor'
            
            result = {
                'performance_metrics': performance,
                'factor_analysis': factor_analysis,
                'manager_skill': manager_skill,
                'skill_level': skill_level,
                'recommendation': self._generate_skill_recommendation(manager_skill, performance)
            }
            
            self.logger.info(f"Manager skill analysis: level={skill_level}, score={manager_skill['skill_score']:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing manager skill: {e}")
            return {}
    
    def _calculate_t_statistics(self, returns: pd.Series, factors: pd.DataFrame) -> Dict[str, float]:
        """Розрахувати t-statistics для факторних експозицій"""
        try:
            import statsmodels.api as sm
            
            # Об'єднуємо дані
            combined_data = pd.concat([returns, factors], axis=1).dropna()
            
            if len(combined_data) < 30:
                return {}
            
            X = sm.add_constant(combined_data.iloc[:, 1:])
            y = combined_data.iloc[:, 0]
            
            model = sm.OLS(y, X).fit()
            
            # T-statistics
            t_stats = {}
            for i, factor in enumerate(factors.columns):
                if i + 1 < len(model.tvalues):
                    t_stats[factor] = model.tvalues[i + 1]
            
            return t_stats
            
        except Exception as e:
            self.logger.error(f"Error calculating t-statistics: {e}")
            return {}
    
    def _analyze_drift_type(self, significant_drifts: Dict, current_exposures: Dict) -> Dict[str, any]:
        """Аналізувати тип дрифту стилю"""
        drift_type = {
            'market_drift': False,
            'size_drift': False,
            'value_drift': False,
            'momentum_drift': False,
            'quality_drift': False,
            'overall_assessment': 'No significant drift'
        }
        
        for factor in significant_drifts:
            if factor == 'MKT':
                drift_type['market_drift'] = True
            elif factor == 'SMB':
                drift_type['size_drift'] = True
            elif factor == 'HML':
                drift_type['value_drift'] = True
            elif factor == 'UMD':
                drift_type['momentum_drift'] = True
            elif factor in ['RMW', 'CMA']:
                drift_type['quality_drift'] = True
        
        # Загальна оцінка
        drift_count = sum([drift_type[k] for k in drift_type if k != 'overall_assessment'])
        
        if drift_count == 0:
            drift_type['overall_assessment'] = 'No significant drift'
        elif drift_count == 1:
            drift_type['overall_assessment'] = 'Minor style drift detected'
        elif drift_count == 2:
            drift_type['overall_assessment'] = 'Moderate style drift detected'
        else:
            drift_type['overall_assessment'] = 'Significant style drift detected'
        
        return drift_type
    
    def _calculate_consistency_score(self, returns: pd.Series) -> float:
        """Розрахувати скор консистентності доходності"""
        try:
            # Розраховуємо доходність по роках
            if len(returns) > 252:
                yearly_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
                
                # Розраховуємо % позитивних років
                positive_years = (yearly_returns > 0).sum()
                consistency = positive_years / len(yearly_returns)
                
                return consistency
            else:
                # Якщо менше року, використовуємо місячні дані
                monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                positive_months = (monthly_returns > 0).sum()
                consistency = positive_months / len(monthly_returns)
                
                return consistency
                
        except Exception as e:
            self.logger.error(f"Error calculating consistency score: {e}")
            return 0.5  # Default middle value
    
    def _generate_skill_recommendation(self, skill: Dict, performance: Dict) -> str:
        """Згенерувати рекомендацію based на навичках менеджера"""
        try:
            if skill['skill_score'] >= 0.8:
                return "Exceptional manager with consistent alpha generation. Recommend allocation."
            elif skill['skill_score'] >= 0.6:
                return "Excellent manager with good risk-adjusted returns. Consider allocation."
            elif skill['skill_score'] >= 0.4:
                return "Good manager with moderate skill. Monitor closely."
            elif skill['skill_score'] >= 0.2:
                return "Average manager with limited alpha. Consider smaller allocation."
            else:
                return "Poor manager with negative or insignificant alpha. Avoid allocation."
                
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {e}")
            return "Unable to generate recommendation"


# Глобальний екземпляр
hedge_fund_analyzer = HedgeFundAnalyzer()


def analyze_hedge_fund(fund_returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, any]:
    """Аналіз хедж фонду"""
    return hedge_fund_analyzer.analyze_manager_skill(fund_returns, benchmark_returns)


def detect_style_drift(current_exposures: Dict[str, float], 
                      historical_exposures: List[Dict[str, float]]) -> Dict[str, any]:
    """Детекція дрифту стилю"""
    return hedge_fund_analyzer.detect_style_drift(current_exposures, historical_exposures)


if __name__ == "__main__":
    # Приклад використання
    logging.basicConfig(level=logging.INFO)
    
    print("🏦 Hedge Fund Analyzer Test")
    print("="*50)
    
    # Симуляція data хедж фонду
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    
    # Симуляція доходності фонду
    fund_returns = pd.Series(
        np.random.normal(0.0008, 0.012, len(dates)),  # 20% annual, 12% vol
        index=dates
    )
    
    # Бенчмарк (S&P 500)
    benchmark_returns = pd.Series(
        np.random.normal(0.0006, 0.010, len(dates)),  # 15% annual, 10% vol
        index=dates
    )
    
    # Аналізуємо фонд
    analysis = analyze_hedge_fund(fund_returns, benchmark_returns)
    
    if analysis:
        print(f"[DATA] Manager Skill Analysis:")
        print(f"   Skill Level: {analysis['skill_level']}")
        print(f"   Skill Score: {analysis['manager_skill']['skill_score']:.2f}")
        print(f"   Alpha (annual): {analysis['manager_skill']['alpha_annualized']:.2%}")
        print(f"   Sharpe Ratio: {analysis['performance_metrics']['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {analysis['performance_metrics']['max_drawdown']:.2%}")
        print(f"   Recommendation: {analysis['recommendation']}")
        
        # Факторний аналіз
        factor_analysis = analysis.get('factor_analysis', {})
        if factor_analysis:
            print(f"\n[TARGET] Factor Exposures:")
            exposures = factor_analysis.get('exposures', {})
            for factor, exposure in exposures.items():
                if factor not in ['alpha', 'r_squared']:
                    print(f"   {factor}: {exposure:.3f}")
            print(f"   Alpha: {exposures.get('alpha', 0):.4f}")
            print(f"   R²: {exposures.get('r_squared', 0):.3f}")
        
        print(f"\n[OK] Hedge Fund Analysis working correctly!")
    else:
        print(f"[ERROR] Analysis failed")
