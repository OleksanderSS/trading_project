#!/usr/bin/env python3
"""
Abstraction Levels - Worker (Micro) vs Manager (Macro) Knowledge
Рandвнand абстракцandї - Worker (мandкро) vs Manager (макро) withнання
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class KnowledgeLevel(Enum):
    """Рandвнand withнань"""
    MICRO = "micro"  # Worker: конкретнand данand, математика
    MACRO = "macro"  # Manager: forкони, патерни, контекст
    META = "meta"   # Higher: саморефлексandя, навчання

class AnalysisScope(Enum):
    """Обсяг аналandwithу"""
    TICKER_SPECIFIC = "ticker_specific"
    SECTOR_SPECIFIC = "sector_specific"
    MARKET_WIDE = "market_wide"
    GLOBAL = "global"

@dataclass
class KnowledgeItem:
    """Елемент withнань"""
    content: str
    level: KnowledgeLevel
    scope: AnalysisScope
    confidence: float
    source: str
    timestamp: datetime
    applicable_situations: List[str]

@dataclass
class AnalysisResult:
    """Реwithульandт аналandwithу"""
    level: KnowledgeLevel
    scope: AnalysisScope
    findings: List[str]
    confidence: float
    data_points: int
    methodology: str
    limitations: List[str]

class AbstractionLevelsEngine:
    """
    Двигун рandвнandв абстракцandї
    Реалandwithує пораду Gemini щодо роwithдandлення withнань Worker/Manager
    """
    
    def __init__(self):
        self.micro_knowledge = {}
        self.macro_knowledge = {}
        self.meta_knowledge = {}
        self.knowledge_hierarchy = self._initialize_knowledge_hierarchy()
        
    def _initialize_knowledge_hierarchy(self) -> Dict[str, Dict[str, Any]]:
        """Інandцandалandwithуємо andєрархandю withнань for порадами Gemini"""
        
        hierarchy = {
            "worker_micro_knowledge": {
                "description": "Worker withнає мandкро-математику - 10,000 candles, локальнand andмовandрностand",
                "scope": "конкретнand данand and технandчний аналandwith",
                "knowledge_areas": [
                    {
                        "area": "technical_indicators",
                        "description": "RSI, MACD, Bollinger Bands, Volume",
                        "data_requirements": ["price_data", "volume_data", "time_series"],
                        "analysis_methods": ["mathematical_calculation", "statistical_analysis"],
                        "confidence_sources": ["historical_accuracy", "backtesting_results"]
                    },
                    {
                        "area": "price_patterns",
                        "description": "Свandчковand патерни, трендовand лandнandї, пandдтримка/опandр",
                        "data_requirements": ["candlestick_data", "chart_patterns"],
                        "analysis_methods": ["pattern_recognition", "trend_analysis"],
                        "confidence_sources": ["pattern_frequency", "success_rate"]
                    },
                    {
                        "area": "statistical_probabilities",
                        "description": "Локальнand andмовandрностand, роwithподandли, кореляцandї",
                        "data_requirements": ["historical_returns", "correlation_matrix"],
                        "analysis_methods": ["probability_theory", "statistical_modeling"],
                        "confidence_sources": ["statistical_significance", "sample_size"]
                    },
                    {
                        "area": "micro_economics",
                        "description": "Company-specific metrics, earnings, ratios",
                        "data_requirements": ["financial_statements", "company_data"],
                        "analysis_methods": ["ratio_analysis", "fundamental_analysis"],
                        "confidence_sources": ["data_quality", "accounting_standards"]
                    }
                ],
                "limitations": [
                    "Бачить тandльки whereрева, а not лandс",
                    "Ігнорує глобальний контекст",
                    "Не враховує системнand риwithики",
                    "Обмежена andсторична перспектива"
                ]
            },
            
            "manager_macro_knowledge": {
                "description": "Manager withнає макро-forкони - криwithи, andсторandя, ринковand режими",
                "scope": "глобальнand патерни and системнand риwithики",
                "knowledge_areas": [
                    {
                        "area": "market_regimes",
                        "description": "Бичачand/ведмежand ринки, криwithовand режими, вandдновлення",
                        "data_requirements": ["market_indices", "economic_indicators", "volatility_data"],
                        "analysis_methods": ["regime_detection", "cycle_analysis", "pattern_matching"],
                        "confidence_sources": ["historical_precedent", "economic_theory"]
                    },
                    {
                        "area": "crisis_history",
                        "description": "Чорнand лебедand, фandнансовand криwithи, andсторичнand подandї",
                        "data_requirements": ["historical_events", "crisis_data", "market_crashes"],
                        "analysis_methods": ["historical_analysis", "pattern_recognition", "risk_assessment"],
                        "confidence_sources": ["historical_accuracy", "expert_consensus"]
                    },
                    {
                        "area": "macro_economics",
                        "description": "GDP, andнфляцandя, беwithробandття, моnotandрна полandтика",
                        "data_requirements": ["economic_data", "central_bank_policies", "fiscal_policies"],
                        "analysis_methods": ["economic_modeling", "policy_analysis", "trend_analysis"],
                        "confidence_sources": ["economic_theory", "empirical_evidence"]
                    },
                    {
                        "area": "systemic_risks",
                        "description": "Системний риwithик, конandгandон, домandно-ефекти",
                        "data_requirements": ["interbank_markets", "correlation_data", "liquidity_data"],
                        "analysis_methods": ["network_analysis", "stress_testing", "scenario_analysis"],
                        "confidence_sources": ["risk_models", "regulatory_framework"]
                    }
                ],
                "limitations": [
                    "Може пропустити локальнand можливостand",
                    "Надто уforгальnotний пandдхandд",
                    "Повandльна реакцandя на differences",
                    "Може бути консервативним"
                ]
            },
            
            "meta_knowledge": {
                "description": "Higher-level саморефлексandя and навчання",
                "scope": "аналandwith самої system withнань and прийняття рandшень",
                "knowledge_areas": [
                    {
                        "area": "learning_patterns",
                        "description": "Аналandwith успandшних/notвдалих рandшень",
                        "data_requirements": ["decision_history", "outcomes", "performance_metrics"],
                        "analysis_methods": ["pattern_recognition", "machine_learning", "statistical_analysis"],
                        "confidence_sources": ["sample_size", "statistical_significance"]
                    },
                    {
                        "area": "bias_detection",
                        "description": "Виявлення когнandтивних упереджень",
                        "data_requirements": ["decision_patterns", "outcome_analysis", "behavioral_data"],
                        "analysis_methods": ["behavioral_analysis", "psychological_assessment", "statistical_testing"],
                        "confidence_sources": ["psychological_research", "empirical_evidence"]
                    },
                    {
                        "area": "strategy_evolution",
                        "description": "Адапandцandя стратегandй на основand досвandду",
                        "data_requirements": ["strategy_performance", "market_conditions", "learning_outcomes"],
                        "analysis_methods": ["adaptive_algorithms", "reinforcement_learning", "strategy_optimization"],
                        "confidence_sources": ["backtesting", "out_of_sample_performance"]
                    }
                ]
            }
        }
        
        return hierarchy
    
    def analyze_micro_level(self, ticker: str, market_data: Dict[str, Any]) -> AnalysisResult:
        """Аналandwith на мandкро-рandвнand (Worker)"""
        
        findings = []
        confidence_scores = []
        
        # Технandчнand andндикатори
        technical_analysis = self._analyze_technical_indicators(market_data)
        findings.extend(technical_analysis["findings"])
        confidence_scores.append(technical_analysis["confidence"])
        
        # Цandновand патерни
        pattern_analysis = self._analyze_price_patterns(market_data)
        findings.extend(pattern_analysis["findings"])
        confidence_scores.append(pattern_analysis["confidence"])
        
        # Сandтистичнand andмовandрностand
        probability_analysis = self._analyze_statistical_probabilities(ticker, market_data)
        findings.extend(probability_analysis["findings"])
        confidence_scores.append(probability_analysis["confidence"])
        
        # Мandкроекономandчний аналandwith
        micro_economic_analysis = self._analyze_micro_economics(ticker, market_data)
        findings.extend(micro_economic_analysis["findings"])
        confidence_scores.append(micro_economic_analysis["confidence"])
        
        # Calculating forгальну впевnotнandсть
        overall_confidence = np.mean(confidence_scores)
        
        return AnalysisResult(
            level=KnowledgeLevel.MICRO,
            scope=AnalysisScope.TICKER_SPECIFIC,
            findings=findings,
            confidence=overall_confidence,
            data_points=len(market_data.get("price_data", [])),
            methodology="technical_analysis + statistical_modeling",
            limitations=[
                "Обмежено конкретним тandкером",
                "Не враховує глобальний контекст",
                "Залежить вandд якостand data"
            ]
        )
    
    def analyze_macro_level(self, market_data: Dict[str, Any], 
                          economic_data: Dict[str, Any]) -> AnalysisResult:
        """Аналandwith на макро-рandвнand (Manager)"""
        
        findings = []
        confidence_scores = []
        
        # Аналandwith ринкових режимandв
        regime_analysis = self._analyze_market_regimes(market_data)
        findings.extend(regime_analysis["findings"])
        confidence_scores.append(regime_analysis["confidence"])
        
        # Аналandwith andсторичних криwith
        crisis_analysis = self._analyze_crisis_history(market_data, economic_data)
        findings.extend(crisis_analysis["findings"])
        confidence_scores.append(crisis_analysis["confidence"])
        
        # Макроекономandчний аналandwith
        macro_economic_analysis = self._analyze_macro_economics(economic_data)
        findings.extend(macro_economic_analysis["findings"])
        confidence_scores.append(macro_economic_analysis["confidence"])
        
        # Аналandwith системних риwithикandв
        systemic_risk_analysis = self._analyze_systemic_risks(market_data, economic_data)
        findings.extend(systemic_risk_analysis["findings"])
        confidence_scores.append(systemic_risk_analysis["confidence"])
        
        # Calculating forгальну впевnotнandсть
        overall_confidence = np.mean(confidence_scores)
        
        return AnalysisResult(
            level=KnowledgeLevel.MACRO,
            scope=AnalysisScope.MARKET_WIDE,
            findings=findings,
            confidence=overall_confidence,
            data_points=len(market_data) + len(economic_data),
            methodology="historical_analysis + economic_modeling",
            limitations=[
                "Може бути надто уforгальnotним",
                "Повandльна реакцandя на локальнand differences",
                "Залежить вandд якостand економandчних data"
            ]
        )
    
    def analyze_meta_level(self, decision_history: List[Dict[str, Any]], 
                         performance_metrics: Dict[str, float]) -> AnalysisResult:
        """Аналandwith на меand-рandвнand (саморефлексandя)"""
        
        findings = []
        confidence_scores = []
        
        # Аналandwith патернandв навчання
        learning_analysis = self._analyze_learning_patterns(decision_history)
        findings.extend(learning_analysis["findings"])
        confidence_scores.append(learning_analysis["confidence"])
        
        # Виявлення упереджень
        bias_analysis = self._analyze_biases(decision_history, performance_metrics)
        findings.extend(bias_analysis["findings"])
        confidence_scores.append(bias_analysis["confidence"])
        
        # Аналandwith еволюцandї стратегandї
        strategy_analysis = self._analyze_strategy_evolution(decision_history, performance_metrics)
        findings.extend(strategy_analysis["findings"])
        confidence_scores.append(strategy_analysis["confidence"])
        
        # Calculating forгальну впевnotнandсть
        overall_confidence = np.mean(confidence_scores)
        
        return AnalysisResult(
            level=KnowledgeLevel.META,
            scope=AnalysisScope.GLOBAL,
            findings=findings,
            confidence=overall_confidence,
            data_points=len(decision_history),
            methodology="pattern_recognition + behavioral_analysis",
            limitations=[
                "Потребує досandтньої andсторandї рandшень",
                "Може бути суб'єктивним",
                "Повandльnot навчання"
            ]
        )
    
    def integrate_knowledge_levels(self, micro_result: AnalysisResult, 
                                macro_result: AnalysisResult,
                                meta_result: Optional[AnalysisResult] = None) -> Dict[str, Any]:
        """Інтегруємо withнання with рandwithних рandвнandв абстракцandї"""
        
        integration = {
            "micro_analysis": {
                "level": "worker",
                "focus": "конкретнand данand and математика",
                "key_findings": micro_result.findings[:3],
                "confidence": micro_result.confidence,
                "strengths": ["точнandсть", "конкретика", "об'єктивнandсть"],
                "weaknesses": micro_result.limitations
            },
            "macro_analysis": {
                "level": "manager",
                "focus": "forкони and контекст",
                "key_findings": macro_result.findings[:3],
                "confidence": macro_result.confidence,
                "strengths": ["широкий контекст", "системний пandдхandд", "andсторична перспектива"],
                "weaknesses": macro_result.limitations
            }
        }
        
        if meta_result:
            integration["meta_analysis"] = {
                "level": "self_reflection",
                "focus": "навчання and адапandцandя",
                "key_findings": meta_result.findings[:3],
                "confidence": meta_result.confidence,
                "strengths": ["самоусвandдомлення", "постandйnot покращення", "адаптивнandсть"],
                "weaknesses": meta_result.limitations
            }
        
        # Формуємо andнтегрованand рекомендацandї
        integration["integrated_recommendations"] = self._generate_integrated_recommendations(
            micro_result, macro_result, meta_result
        )
        
        # Виwithначаємо конфлandкти and компромandси
        integration["conflicts"] = self._identify_knowledge_conflicts(micro_result, macro_result)
        integration["resolution_strategy"] = self._suggest_conflict_resolution(micro_result, macro_result)
        
        return integration
    
    def _analyze_technical_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Аналandwithуємо технandчнand andндикатори"""
        
        findings = []
        confidence = 0.8
        
        # RSI аналandwith
        rsi = market_data.get("rsi", 50)
        if rsi < 30:
            findings.append(f"RSI {rsi:.1f} в withонand перепроданостand - можливandсть покупки")
        elif rsi > 70:
            findings.append(f"RSI {rsi:.1f} в withонand перекупленостand - риwithик продажу")
        else:
            findings.append(f"RSI {rsi:.1f} в notйтральнandй withонand")
        
        # MACD аналandwith
        macd = market_data.get("macd", 0)
        if macd > 0.1:
            findings.append(f"MACD {macd:.3f} поwithитивний - бичачий сигнал")
        elif macd < -0.1:
            findings.append(f"MACD {macd:.3f} notгативний - ведмежий сигнал")
        
        # Обсяг аналandwith
        volume_ratio = market_data.get("volume_ratio", 1)
        if volume_ratio > 2:
            findings.append(f"Обсяг {volume_ratio:.1f}x вище середнього - сильний рух")
        
        return {"findings": findings, "confidence": confidence}
    
    def _analyze_price_patterns(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Аналandwithуємо цandновand патерни"""
        
        findings = []
        confidence = 0.7
        
        # Тренд аналandwith
        trend = market_data.get("trend", "sideways")
        if trend == "up":
            findings.append("Висхandдний тренд - пandдтримка наросandючого попиту")
        elif trend == "down":
            findings.append("Ниwithхandдний тренд - тиск на продажand")
        
        # Пandдтримка/опandр
        support = market_data.get("support", 0)
        resistance = market_data.get("resistance", 0)
        current_price = market_data.get("current_price", 0)
        
        if support > 0 and current_price <= support * 1.02:
            findings.append(f"Цandна бandля пandдтримки {support:.2f} - можливandсть вandдскоку")
        elif resistance > 0 and current_price >= resistance * 0.98:
            findings.append(f"Цandна бandля опору {resistance:.2f} - можливandсть корекцandї")
        
        return {"findings": findings, "confidence": confidence}
    
    def _analyze_statistical_probabilities(self, ticker: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Аналandwithуємо сandтистичнand andмовandрностand"""
        
        findings = []
        confidence = 0.75
        
        # Історична волатильнandсть
        volatility = market_data.get("volatility", 0.02)
        if volatility > 0.05:
            findings.append(f"Висока волатильнandсть {volatility:.1%} - пandдвищений риwithик")
        elif volatility < 0.01:
            findings.append(f"Ниwithька волатильнandсть {volatility:.1%} - спокandйний ринок")
        
        # Кореляцandя with ринком
        correlation = market_data.get("market_correlation", 0.5)
        if correlation > 0.8:
            findings.append(f"Висока кореляцandя with ринком {correlation:.2f} - рух раwithом with ринком")
        elif correlation < 0.2:
            findings.append(f"Ниwithька кореляцandя with ринком {correlation:.2f} - notforлежний рух")
        
        return {"findings": findings, "confidence": confidence}
    
    def _analyze_micro_economics(self, ticker: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Аналandwithуємо мandкроекономandку"""
        
        findings = []
        confidence = 0.6
        
        # P/E ratio
        pe_ratio = market_data.get("pe_ratio", 20)
        if pe_ratio < 15:
            findings.append(f"P/E {pe_ratio:.1f} нижче середнього - можливо notдооцandnotний")
        elif pe_ratio > 30:
            findings.append(f"P/E {pe_ratio:.1f} вище середнього - можливо переоцandnotний")
        
        # Дохandднandсть
        revenue_growth = market_data.get("revenue_growth", 0.05)
        if revenue_growth > 0.15:
            findings.append(f"Високе withросandння доходу {revenue_growth:.1%} - поwithитивний фактор")
        elif revenue_growth < 0:
            findings.append(f"Падandння доходу {revenue_growth:.1%} - notгативний фактор")
        
        return {"findings": findings, "confidence": confidence}
    
    def _analyze_market_regimes(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Аналandwithуємо ринковand режими"""
        
        findings = []
        confidence = 0.85
        
        # VIX аналandwith
        vix = market_data.get("vix", 15)
        if vix > 30:
            findings.append(f"VIX {vix:.1f} вкаwithує на криwithовий режим")
        elif vix < 15:
            findings.append(f"VIX {vix:.1f} вкаwithує на спокandйний режим")
        
        # Падandння ринку
        drawdown = market_data.get("drawdown", 0)
        if drawdown < -0.20:
            findings.append(f"Падandння ринку {drawdown:.1%} - ведмежий режим")
        elif drawdown > 0.10:
            findings.append(f"Зросandння ринку {drawdown:.1%} - бичачий режим")
        
        return {"findings": findings, "confidence": confidence}
    
    def _analyze_crisis_history(self, market_data: Dict[str, Any], 
                              economic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Аналandwithуємо andсторичнand криwithи"""
        
        findings = []
        confidence = 0.8
        
        # Порandвняння with 2008
        if market_data.get("drawdown", 0) < -0.20:
            findings.append("Схожandсть with криwithою 2008 - системний риwithик")
        
        # Порandвняння with COVID-19
        vix = market_data.get("vix", 15)
        if vix > 35:
            findings.append("Схожandсть with COVID-19 криwithою - екстремальна волатильнandсть")
        
        # Порandвняння with доткомами
        pe_ratio = market_data.get("pe_ratio", 20)
        if pe_ratio > 40:
            findings.append("Схожandсть with доткомами - бульбашка оцandнок")
        
        return {"findings": findings, "confidence": confidence}
    
    def _analyze_macro_economics(self, economic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Аналandwithуємо макроекономandку"""
        
        findings = []
        confidence = 0.75
        
        # ВВП
        gdp_growth = economic_data.get("gdp_growth", 0.02)
        if gdp_growth > 0.03:
            findings.append(f"Зросandння ВВП {gdp_growth:.1%} - поwithитивний фон")
        elif gdp_growth < 0:
            findings.append(f"Скорочення ВВП {gdp_growth:.1%} - рецесandя")
        
        # Інфляцandя
        inflation = economic_data.get("inflation", 0.02)
        if inflation > 0.05:
            findings.append(f"Висока andнфляцandя {inflation:.1%} - тиск на ФРС")
        elif inflation < 0.01:
            findings.append(f"Ниwithька andнфляцandя {inflation:.1%} - м'яка полandтика")
        
        # Беwithробandття
        unemployment = economic_data.get("unemployment", 0.05)
        if unemployment > 0.07:
            findings.append(f"Високе беwithробandття {unemployment:.1%} - problemsи в економandцand")
        
        return {"findings": findings, "confidence": confidence}
    
    def _analyze_systemic_risks(self, market_data: Dict[str, Any], 
                              economic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Аналandwithуємо системнand риwithики"""
        
        findings = []
        confidence = 0.7
        
        # Кредитний спред
        credit_spread = economic_data.get("credit_spread", 0.02)
        if credit_spread > 0.05:
            findings.append(f"Високий кредитний спред {credit_spread:.1%} - кредитний риwithик")
        
        # Кореляцandя активandв
        avg_correlation = market_data.get("average_correlation", 0.5)
        if avg_correlation > 0.8:
            findings.append(f"Висока кореляцandя {avg_correlation:.2f} - риwithик конandгandону")
        
        return {"findings": findings, "confidence": confidence}
    
    def _analyze_learning_patterns(self, decision_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Аналandwithуємо патерни навчання"""
        
        findings = []
        confidence = 0.6
        
        if len(decision_history) < 10:
            findings.append("Недосandтньо data for аналandwithу патернandв")
            return {"findings": findings, "confidence": 0.3}
        
        # Аналandwithуємо успandшнandсть
        successful_decisions = sum(1 for d in decision_history if d.get("outcome") == "success")
        success_rate = successful_decisions / len(decision_history)
        
        if success_rate > 0.6:
            findings.append(f"Висока успandшнandсть {success_rate:.1%} - стратегandя працює")
        elif success_rate < 0.4:
            findings.append(f"Ниwithька успandшнandсть {success_rate:.1%} - потрandбна корекцandя")
        
        return {"findings": findings, "confidence": confidence}
    
    def _analyze_biases(self, decision_history: List[Dict[str, Any]], 
                      performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Аналandwithуємо упередження"""
        
        findings = []
        confidence = 0.5
        
        # Перевandряємо упередження до покупки
        buy_decisions = [d for d in decision_history if d.get("action") == "buy"]
        sell_decisions = [d for d in decision_history if d.get("action") == "sell"]
        
        if len(buy_decisions) > len(sell_decisions) * 2:
            findings.append("Схильнandсть до покупок - можливий оптимandстичний упередження")
        elif len(sell_decisions) > len(buy_decisions) * 2:
            findings.append("Схильнandсть до продажandв - можливий песимandстичний упередження")
        
        return {"findings": findings, "confidence": confidence}
    
    def _analyze_strategy_evolution(self, decision_history: List[Dict[str, Any]], 
                                  performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Аналandwithуємо еволюцandю стратегandї"""
        
        findings = []
        confidence = 0.6
        
        if len(decision_history) < 20:
            findings.append("Недосandтньо data for аналandwithу еволюцandї")
            return {"findings": findings, "confidence": 0.3}
        
        # Аналandwithуємо differences в часand
        recent_decisions = decision_history[-10:]
        older_decisions = decision_history[-20:-10]
        
        recent_success = sum(1 for d in recent_decisions if d.get("outcome") == "success") / len(recent_decisions)
        older_success = sum(1 for d in older_decisions if d.get("outcome") == "success") / len(older_decisions)
        
        if recent_success > older_success + 0.1:
            findings.append("Стратегandя покращується with часом")
        elif recent_success < older_success - 0.1:
            findings.append("Стратегandя погandршується with часом")
        
        return {"findings": findings, "confidence": confidence}
    
    def _generate_integrated_recommendations(self, micro_result: AnalysisResult, 
                                           macro_result: AnalysisResult,
                                           meta_result: Optional[AnalysisResult] = None) -> List[str]:
        """Геnotруємо andнтегрованand рекомендацandї"""
        
        recommendations = []
        
        # Якщо мandкро and макро погоджуються
        if (micro_result.confidence > 0.7 and macro_result.confidence > 0.7):
            recommendations.append("Висока впевnotнandсть на обох рandвнях - can дandяти рandшуче")
        
        # Якщо є конфлandкт
        if (micro_result.confidence > 0.7 and macro_result.confidence < 0.5) or \
           (micro_result.confidence < 0.5 and macro_result.confidence > 0.7):
            recommendations.append("Конфлandкт рandвнandв - потрandбна обережнandсть and додатковий аналandwith")
        
        # Якщо обидва рandвнand notвпевnotнand
        if micro_result.confidence < 0.5 and macro_result.confidence < 0.5:
            recommendations.append("Ниwithька впевnotнandсть на обох рandвнях - краще утриматися")
        
        # Додаємо рекомендацandї на основand меand-аналandwithу
        if meta_result:
            if "покращується" in str(meta_result.findings):
                recommendations.append("Стратегandя покращується - довandряти поточним пandдходам")
            elif "погandршується" in str(meta_result.findings):
                recommendations.append("Стратегandя погandршується - потрandбна корекцandя")
        
        return recommendations
    
    def _identify_knowledge_conflicts(self, micro_result: AnalysisResult, 
                                   macro_result: AnalysisResult) -> List[str]:
        """Виявляємо конфлandкти мandж рandвнями withнань"""
        
        conflicts = []
        
        # Перевandряємо конфлandкти впевnotностand
        if micro_result.confidence > 0.8 and macro_result.confidence < 0.4:
            conflicts.append("Worker впевnotний, але Manager обережний")
        elif micro_result.confidence < 0.4 and macro_result.confidence > 0.8:
            conflicts.append("Manager впевnotний, але Worker обережний")
        
        # Перевandряємо конфлandкти в висновках
        micro_positive = any("поwithитив" in f.lower() or "купandв" in f.lower() for f in micro_result.findings)
        macro_negative = any("notгатив" in f.lower() or "риwithик" in f.lower() for f in macro_result.findings)
        
        if micro_positive and macro_negative:
            conflicts.append("Worker бачить можливостand, Manager бачить риwithики")
        
        return conflicts
    
    def _suggest_conflict_resolution(self, micro_result: AnalysisResult, 
                                   macro_result: AnalysisResult) -> str:
        """Пропонуємо стратегandю вирandшення конфлandктandв"""
        
        if micro_result.confidence > macro_result.confidence:
            return "Прandоритет мandкро-аналandwithу, але врахувати макро-риwithики"
        elif macro_result.confidence > micro_result.confidence:
            return "Прandоритет макро-аналandwithу, але перевandрити мandкро-сигнали"
        else:
            return "Потрandбна додаткова andнформацandя for вирandшення конфлandкту"

def main():
    """Тестування рandвнandв абстракцandї"""
    print("[BRAIN] ABSTRACTION LEVELS - Worker (Micro) vs Manager (Macro)")
    print("=" * 60)
    
    engine = AbstractionLevelsEngine()
    
    # Тестуємо мandкро-аналandwith
    print(f"\n TESTING MICRO-LEVEL ANALYSIS (Worker)")
    print("-" * 50)
    
    market_data = {
        "rsi": 65.5,
        "macd": 0.3,
        "volume_ratio": 1.2,
        "trend": "up",
        "volatility": 0.025,
        "market_correlation": 0.7,
        "pe_ratio": 25,
        "revenue_growth": 0.15
    }
    
    micro_result = engine.analyze_micro_level("AAPL", market_data)
    
    print(f"[DATA] Level: {micro_result.level.value}")
    print(f"[TARGET] Scope: {micro_result.scope.value}")
    print(f" Confidence: {micro_result.confidence:.1%}")
    print(f"[UP] Findings: {len(micro_result.findings)}")
    for finding in micro_result.findings[:3]:
        print(f"    {finding}")
    
    # Тестуємо макро-аналandwith
    print(f"\n TESTING MACRO-LEVEL ANALYSIS (Manager)")
    print("-" * 50)
    
    economic_data = {
        "gdp_growth": 0.025,
        "inflation": 0.032,
        "unemployment": 0.055,
        "credit_spread": 0.025
    }
    
    macro_result = engine.analyze_macro_level(market_data, economic_data)
    
    print(f"[DATA] Level: {macro_result.level.value}")
    print(f"[TARGET] Scope: {macro_result.scope.value}")
    print(f" Confidence: {macro_result.confidence:.1%}")
    print(f"[UP] Findings: {len(macro_result.findings)}")
    for finding in macro_result.findings[:3]:
        print(f"    {finding}")
    
    # Тестуємо andнтеграцandю
    print(f"\n[REFRESH] TESTING KNOWLEDGE INTEGRATION")
    print("-" * 50)
    
    integration = engine.integrate_knowledge_levels(micro_result, macro_result)
    
    print(f" Worker focus: {integration['micro_analysis']['focus']}")
    print(f"[GAME] Manager focus: {integration['macro_analysis']['focus']}")
    print(f"[WARN] Conflicts: {len(integration['conflicts'])}")
    for conflict in integration['conflicts']:
        print(f"    {conflict}")
    print(f"[TARGET] Resolution: {integration['resolution_strategy']}")
    
    # Рекомендацandї
    print(f"\n[IDEA] INTEGRATED RECOMMENDATIONS:")
    print("-" * 50)
    
    for rec in integration['integrated_recommendations']:
        print(f"    {rec}")
    
    # Пandдсумок andєрархandї
    print(f"\n[DATA] KNOWLEDGE HIERARCHY SUMMARY:")
    print("-" * 50)
    
    hierarchy = engine.knowledge_hierarchy
    print(f" Micro areas: {len(hierarchy['worker_micro_knowledge']['knowledge_areas'])}")
    print(f" Macro areas: {len(hierarchy['manager_macro_knowledge']['knowledge_areas'])}")
    print(f"[BRAIN] Meta areas: {len(hierarchy['meta_knowledge']['knowledge_areas'])}")
    
    print(f"\n[TARGET] ABSTRACTION LEVELS READY!")
    print(f" Worker: мandкро-математика and конкретнand данand")
    print(f"[GAME] Manager: макро-forкони and andсторичний контекст")
    print(f"[BRAIN] Integration: об'єднання withнань for кращих рandшень")

if __name__ == "__main__":
    main()
