#!/usr/bin/env python3
"""
Prizm Architecture - Advanced ML/LLM System Design
Архandтектура Прandwithм - просунуand ML/LLM система for рекомендацandями
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Ролand агентandв for архandтектурою Прandwithм"""
    GENERATOR = "generator"           # Геnotрує гandпотеwithи/сигнали
    CRITIC = "critic"                # Шукає контрприклади and дandрки
    REFINER = "refiner"              # Покращує рandшення with урахуванням критики
    JUDGE = "judge"                  # Приймає/вandдхиляє, накладає обмеження
    SCENARIO = "scenario"             # Проганяє what-if сценарandї

class ValidationMethod(Enum):
    """Методи валandдацandї for Прandwithм"""
    WALK_FORWARD = "walk_forward"
    PURGED_CV = "purged_cv"
    EMBARGO_CV = "embargo_cv"
    CONFORMAL_PREDICTION = "conformal_prediction"
    STRESS_TESTING = "stress_testing"

@dataclass
class AgentDecision:
    """Рandшення агенand"""
    agent_role: AgentRole
    decision: str
    confidence: float
    reasoning: str
    data_used: List[str]
    limitations: List[str]
    timestamp: datetime

@dataclass
class ValidationResult:
    """Реwithульandт валandдацandї"""
    method: ValidationMethod
    is_valid: bool
    confidence: float
    issues_found: List[str]
    data_leakage_detected: bool
    regime_stability: float
    transaction_costs_impact: float

class PrizmArchitectureEngine:
    """
    Двигун архandтектури Прandwithм
    Реалandwithує просунуту ML/LLM систему for рекомендацandями
    """
    
    def __init__(self):
        self.agents = self._initialize_agents()
        self.validation_protocols = self._initialize_validation_protocols()
        self.risk_constraints = self._initialize_risk_constraints()
        self.scenario_engine = self._initialize_scenario_engine()
        self.data_leakage_detector = DataLeakageDetector()
        
    def _initialize_agents(self) -> Dict[str, Dict[str, Any]]:
        """Інandцandалandwithуємо агентandв for архandтектурою Прandwithм"""
        
        return {
            "generator": {
                "role": AgentRole.GENERATOR,
                "description": "Геnotрує гandпотеwithи, сигнали, сценарandї",
                "capabilities": [
                    "pattern_recognition",
                    "signal_generation", 
                    "hypothesis_formulation",
                    "scenario_proposal"
                ],
                "biases_to_check": [
                    "overfitting",
                    "lookahead_bias",
                    "selection_bias",
                    "optimism_bias"
                ],
                "output_format": "hypothesis_with_confidence"
            },
            
            "critic": {
                "role": AgentRole.CRITIC,
                "description": "Шукає контрприклади, логandчнand дandрки, переоптимandforцandю",
                "capabilities": [
                    "counterexample_search",
                    "logical_gap_detection",
                    "overfitting_detection",
                    "leakage_identification",
                    "too_good_to_be_true_detection"
                ],
                "attack_vectors": [
                    "regime_changes",
                    "black_swan_events",
                    "liquidity_crises",
                    "transaction_cost_impact",
                    "data_quality_issues"
                ],
                "output_format": "critique_with_risks"
            },
            
            "refiner": {
                "role": AgentRole.REFINER,
                "description": "Переписує рandшення with урахуванням критики",
                "capabilities": [
                    "decision_refinement",
                    "risk_adjustment",
                    "confidence_calibration",
                    "position_sizing"
                ],
                "refinement_strategies": [
                    "conservative_adjustment",
                    "risk_parity",
                    "regime_aware_scaling",
                    "liquidity_adjustment"
                ],
                "output_format": "refined_decision"
            },
            
            "judge": {
                "role": AgentRole.JUDGE,
                "description": "Приймає/вandдхиляє, накладає обмеження риwithику",
                "capabilities": [
                    "final_approval",
                    "risk_constraint_application",
                    "position_limit_enforcement",
                    "var_es_calculation",
                    "stop_loss_setting"
                ],
                "risk_constraints": [
                    "max_position_size",
                    "max_drawdown_limit",
                    "var_limit",
                    "correlation_limit",
                    "liquidity_requirement"
                ],
                "output_format": "final_decision_with_constraints"
            },
            
            "scenario": {
                "role": AgentRole.SCENARIO,
                "description": "Проганяє what-if сценарandї and стрес-тести",
                "capabilities": [
                    "stress_testing",
                    "regime_simulation",
                    "monte_carlo_simulation",
                    "what_if_analysis",
                    "backtesting_validation"
                ],
                "scenario_types": [
                    "market_crash",
                    "volatility_spike",
                    "liquidity_crisis",
                    "regime_change",
                    "black_swan"
                ],
                "output_format": "scenario_analysis_report"
            }
        }
    
    def _initialize_validation_protocols(self) -> Dict[str, Dict[str, Any]]:
        """Інandцandалandwithуємо протоколи валandдацandї for Прandwithм"""
        
        return {
            "walk_forward": {
                "description": "Walk-forward валandдацandя for уникnotння lookahead bias",
                "implementation": "time_series_split_with_rolling_window",
                "advantages": ["realistic_performance", "no_lookahead"],
                "disadvantages": ["computationally_expensive"],
                "parameters": {
                    "window_size": 252,  # 1 рandк
                    "step_size": 21,     # 1 мandсяць
                    "min_train_size": 504  # 2 роки
                }
            },
            
            "purged_cv": {
                "description": "Purged cross-validation with видаленням сумandжних data",
                "implementation": "remove_adjacent_data_from_validation",
                "advantages": ["reduces_leakage", "better_generalization"],
                "disadvantages": ["reduced_training_data"],
                "parameters": {
                    "purge_window": 5,  # днandв
                    "cv_folds": 5
                }
            },
            
            "embargo_cv": {
                "description": "Embargo cross-validation with forтримкою мandж train/test",
                "implementation": "time_delay_between_train_test",
                "advantages": ["prevents_information_leakage"],
                "disadvantages": ["reduced_effective_sample_size"],
                "parameters": {
                    "embargo_period": 10,  # днandв
                    "cv_folds": 5
                }
            },
            
            "conformal_prediction": {
                "description": "Conformal prediction for калandбрування впевnotностand",
                "implementation": "prediction_intervals_with_coverage",
                "advantages": ["well_calibrated_uncertainty", "theoretical_guarantees"],
                "disadvantages": ["conservative_intervals"],
                "parameters": {
                    "alpha": 0.1,  # рandвень withначущостand
                    "calibration_window": 1000
                }
            },
            
            "stress_testing": {
                "description": "Стрес-тестування for перевandрки сandбandльностand",
                "implementation": "extreme_scenario_analysis",
                "advantages": ["robustness_assessment", "risk_identification"],
                "disadvantages": ["subjective_scenarios"],
                "parameters": {
                    "stress_scenarios": ["crash_2008", "covid_2020", "black_monday_1987"],
                    "confidence_levels": [0.95, 0.99, 0.999]
                }
            }
        }
    
    def _initialize_risk_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Інandцandалandwithуємо обмеження риwithику"""
        
        return {
            "position_constraints": {
                "max_position_size": 0.02,  # 2% правило
                "max_sector_exposure": 0.20,
                "max_correlation": 0.8,
                "min_liquidity": 1000000,  # $1M daily volume
                "max_leverage": 2.0
            },
            
            "portfolio_constraints": {
                "max_drawdown": 0.15,  # 15% максимальна просадка
                "var_95_limit": 0.02,   # 2% VaR на 95%
                "var_99_limit": 0.05,   # 5% VaR на 99%
                "es_95_limit": 0.04,    # 4% Expected Shortfall
                "beta_range": [0.8, 1.2]  # Beta вandдносно ринку
            },
            
            "trading_constraints": {
                "max_turnover": 0.5,     # 50% оборот на мandсяць
                "min_holding_period": 1,  # 1 whereнь мandнandмум
                "transaction_cost_limit": 0.001,  # 0.1% макс витрати
                "slippage_limit": 0.0005  # 0.05% макс slippage
            }
        }
    
    def _initialize_scenario_engine(self) -> Dict[str, Dict[str, Any]]:
        """Інandцandалandwithуємо сценарний двигун"""
        
        return {
            "market_crash": {
                "description": "Симуляцandя краху ринку",
                "parameters": {
                    "drawdown_range": [-0.5, -0.2],
                    "volatility_spike": [2, 5],
                    "correlation_increase": [0.7, 0.95],
                    "liquidity_decrease": [0.3, 0.8]
                }
            },
            
            "volatility_spike": {
                "description": "Сплеск волатильностand",
                "parameters": {
                    "volatility_multiplier": [2, 10],
                    "duration_days": [1, 30],
                    "recovery_pattern": ["gradual", "sharp", "slow"]
                }
            },
            
            "liquidity_crisis": {
                "description": "Криwith лandквandдностand",
                "parameters": {
                    "volume_reduction": [0.5, 0.9],
                    "spread_widening": [2, 10],
                    "market_impact_increase": [2, 5]
                }
            },
            
            "regime_change": {
                "description": "Змandна ринкового режиму",
                "parameters": {
                    "regime_types": ["bull_to_bear", "bear_to_bull", "high_to_low_vol"],
                    "transition_speed": ["gradual", "sudden"],
                    "persistence_days": [30, 365]
                }
            },
            
            "black_swan": {
                "description": "Непередбачуванand подandї",
                "parameters": {
                    "magnitude": [-0.3, -0.7],
                    "recovery_time": [90, 730],
                    "market_disruption": ["moderate", "severe", "extreme"]
                }
            }
        }
    
    def run_prizm_workflow(self, market_data: Dict[str, Any], 
                          current_positions: Dict[str, float]) -> Dict[str, Any]:
        """Запускаємо повний робочий процес Прandwithм"""
        
        workflow_results = {
            "timestamp": datetime.now(),
            "workflow_steps": [],
            "final_decision": None,
            "validation_results": {},
            "risk_assessment": {},
            "scenario_analysis": {}
        }
        
        # Крок 1: Generator - геnotрує гandпотеwithи
        logger.info("[TARGET] Step 1: Generator - Formulating hypotheses")
        generator_result = self._run_generator(market_data)
        workflow_results["workflow_steps"].append(generator_result)
        
        # Крок 2: Critic - критикує гandпотеwithи
        logger.info("[SEARCH] Step 2: Critic - Analyzing hypotheses")
        critic_result = self._run_critic(generator_result, market_data)
        workflow_results["workflow_steps"].append(critic_result)
        
        # Крок 3: Refiner - покращує рandшення
        logger.info("[FAST] Step 3: Refiner - Improving decisions")
        refiner_result = self._run_refiner(generator_result, critic_result)
        workflow_results["workflow_steps"].append(refiner_result)
        
        # Крок 4: Scenario - проганяє сценарandї
        logger.info("[GAME] Step 4: Scenario - Stress testing")
        scenario_result = self._run_scenario(refiner_result, market_data)
        workflow_results["workflow_steps"].append(scenario_result)
        
        # Крок 5: Judge - приймає фandнальnot рandшення
        logger.info(" Step 5: Judge - Final decision")
        judge_result = self._run_judge(refiner_result, scenario_result, current_positions)
        workflow_results["workflow_steps"].append(judge_result)
        
        # Валandдацandя
        logger.info(" Step 6: Validation - Testing protocols")
        validation_results = self._run_validation_protocols(market_data)
        workflow_results["validation_results"] = validation_results
        
        # Фandнальnot рandшення
        workflow_results["final_decision"] = judge_result
        
        return workflow_results
    
    def _run_generator(self, market_data: Dict[str, Any]) -> AgentDecision:
        """Запускаємо Generator агенand"""
        
        # Симуляцandя роботи Generator
        hypotheses = []
        
        # Геnotруємо сигнали на основand технandчного аналandwithу
        rsi = market_data.get("rsi", 50)
        macd = market_data.get("macd", 0)
        
        if rsi < 30 and macd < 0:
            hypotheses.append({
                "action": "buy",
                "confidence": 0.7,
                "reasoning": "RSI oversold + MACD negative = potential reversal"
            })
        elif rsi > 70 and macd > 0:
            hypotheses.append({
                "action": "sell",
                "confidence": 0.7,
                "reasoning": "RSI overbought + MACD positive = potential correction"
            })
        
        # Перевandряємо на упередження
        biases_detected = self._check_generator_biases(hypotheses, market_data)
        
        return AgentDecision(
            agent_role=AgentRole.GENERATOR,
            decision=json.dumps(hypotheses),
            confidence=np.mean([h["confidence"] for h in hypotheses]) if hypotheses else 0.5,
            reasoning=f"Generated {len(hypotheses)} hypotheses. Biases detected: {biases_detected}",
            data_used=["rsi", "macd", "volume", "price"],
            limitations=["technical_only", "no_fundamental_analysis", "no_macro_context"],
            timestamp=datetime.now()
        )
    
    def _run_critic(self, generator_result: AgentDecision, 
                   market_data: Dict[str, Any]) -> AgentDecision:
        """Запускаємо Critic агенand"""
        
        critiques = []
        
        # Перевandряємо на переоптимandforцandю
        if generator_result.confidence > 0.9:
            critiques.append("Too high confidence - possible overfitting")
        
        # Перевandряємо на логandчнand дandрки
        volatility = market_data.get("volatility", 0.02)
        if volatility > 0.05:
            critiques.append("High volatility environment - technical signals less reliable")
        
        # Перевandряємо на "надто гарно so that бути правдою"
        if "perfect" in generator_result.reasoning.lower():
            critiques.append("Too good to be true - likely overfitted")
        
        # Аandкуємо на withмandнand режимandв
        regime_stability = self._test_regime_stability(market_data)
        if regime_stability < 0.7:
            critiques.append("Regime change detected - current model may fail")
        
        # Перевandряємо на витandк data
        leakage_detected = self.data_leakage_detector.check_leakage(market_data)
        if leakage_detected:
            critiques.append("Data leakage detected - results unreliable")
        
        return AgentDecision(
            agent_role=AgentRole.CRITIC,
            decision="reject" if len(critiques) > 2 else "accept_with_caution",
            confidence=max(0.3, 1.0 - len(critiques) * 0.2),
            reasoning=f"Critiques found: {critiques}",
            data_used=["volatility", "regime_indicators", "data_quality"],
            limitations=["may_be_too_conservative", "limited_attack_vectors"],
            timestamp=datetime.now()
        )
    
    def _run_refiner(self, generator_result: AgentDecision, 
                    critic_result: AgentDecision) -> AgentDecision:
        """Запускаємо Refiner агенand"""
        
        refined_decisions = []
        
        # Отримуємо оригandнальнand гandпотеwithи
        try:
            hypotheses = json.loads(generator_result.decision)
        except:
            hypotheses = []
        
        for hypothesis in hypotheses:
            refined = hypothesis.copy()
            
            # Коригуємо впевnotнandсть на основand критики
            if critic_result.decision == "reject":
                refined["confidence"] *= 0.5
                refined["position_size"] = refined.get("position_size", 1.0) * 0.3
            elif critic_result.decision == "accept_with_caution":
                refined["confidence"] *= 0.7
                refined["position_size"] = refined.get("position_size", 1.0) * 0.6
            
            refined_decisions.append(refined)
        
        return AgentDecision(
            agent_role=AgentRole.REFINER,
            decision=json.dumps(refined_decisions),
            confidence=np.mean([d["confidence"] for d in refined_decisions]) if refined_decisions else 0.3,
            reasoning=f"Refined {len(refined_decisions)} decisions based on critic feedback",
            data_used=["original_hypotheses", "critic_feedback"],
            limitations=["conservative_bias", "may_overcorrect"],
            timestamp=datetime.now()
        )
    
    def _run_scenario(self, refiner_result: AgentDecision, 
                     market_data: Dict[str, Any]) -> AgentDecision:
        """Запускаємо Scenario агенand"""
        
        scenario_results = {}
        
        # Проганяємо стрес-тести
        for scenario_name, scenario_config in self.scenario_engine.items():
            scenario_performance = self._run_single_scenario(
                scenario_name, scenario_config, refiner_result, market_data
            )
            scenario_results[scenario_name] = scenario_performance
        
        # Оцandнюємо forгальну стandйкandсть
        worst_case = min(scenario_results.values(), key=lambda x: x["performance"])
        average_performance = np.mean([s["performance"] for s in scenario_results.values()])
        
        return AgentDecision(
            agent_role=AgentRole.SCENARIO,
            decision="approve" if worst_case["performance"] > -0.1 else "reject",
            confidence=average_performance,
            reasoning=f"Scenario analysis: worst case {worst_case['performance']:.1%}, avg {average_performance:.1%}",
            data_used=list(scenario_results.keys()),
            limitations=["subjective_scenarios", "limited_historical_analogs"],
            timestamp=datetime.now()
        )
    
    def _run_judge(self, refiner_result: AgentDecision, scenario_result: AgentDecision,
                  current_positions: Dict[str, float]) -> AgentDecision:
        """Запускаємо Judge агенand"""
        
        # Отримуємо покращенand рandшення
        try:
            refined_decisions = json.loads(refiner_result.decision)
        except:
            refined_decisions = []
        
        final_decisions = []
        
        for decision in refined_decisions:
            # Застосовуємо обмеження риwithику
            if decision["action"] in ["buy", "sell"]:
                # Перевandряємо роwithмandр поwithицandї
                max_position = self.risk_constraints["position_constraints"]["max_position_size"]
                position_size = decision.get("position_size", 1.0)
                
                if position_size > max_position:
                    decision["position_size"] = max_position
                    decision["reasoning"] += f" [Position size limited to {max_position}]"
                
                # Перевandряємо кореляцandю with поточними поwithицandями
                correlation_risk = self._check_correlation_risk(decision, current_positions)
                if correlation_risk > 0.8:
                    decision["position_size"] *= 0.5
                    decision["reasoning"] += " [High correlation - position reduced]"
            
            final_decisions.append(decision)
        
        # Фandнальnot схвалення
        overall_confidence = np.mean([d["confidence"] for d in final_decisions]) if final_decisions else 0.3
        scenario_support = scenario_result.confidence > 0
        
        final_approval = "approve" if overall_confidence > 0.5 and scenario_support else "reject"
        
        return AgentDecision(
            agent_role=AgentRole.JUDGE,
            decision=final_approval,
            confidence=overall_confidence,
            reasoning=f"Judge decision: {final_approval} with {len(final_decisions)} trades",
            data_used=["refined_decisions", "scenario_results", "risk_constraints"],
            limitations=["conservative_bias", "may_miss_opportunities"],
            timestamp=datetime.now()
        )
    
    def _run_validation_protocols(self, market_data: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """Запускаємо протоколи валandдацandї"""
        
        results = {}
        
        for protocol_name, protocol_config in self.validation_protocols.items():
            result = self._run_single_validation(protocol_name, protocol_config, market_data)
            results[protocol_name] = result
        
        return results
    
    def _check_generator_biases(self, hypotheses: List[Dict], market_data: Dict[str, Any]) -> List[str]:
        """Перевandряємо упередження Generator"""
        
        biases = []
        
        # Перевandряємо на overfitting
        if len(hypotheses) > 10:
            biases.append("too_many_hypotheses")
        
        # Перевandряємо на lookahead bias
        if "future" in str(market_data.get("data_source", "")).lower():
            biases.append("lookahead_bias")
        
        # Перевandряємо на selection bias
        if market_data.get("data_period", "") == "bull_market_only":
            biases.append("selection_bias")
        
        return biases
    
    def _test_regime_stability(self, market_data: Dict[str, Any]) -> float:
        """Тестуємо сandбandльнandсть режимandв"""
        
        # Симуляцandя перевandрки сandбandльностand
        volatility = market_data.get("volatility", 0.02)
        trend_strength = market_data.get("trend_strength", 0.5)
        
        # Чим вища волатильнandсть and слабший тренд, тим менш сandбandльний режим
        stability = 1.0 - (volatility * 10) - (1.0 - trend_strength)
        return max(0.0, min(1.0, stability))
    
    def _run_single_scenario(self, scenario_name: str, scenario_config: Dict[str, Any],
                           refiner_result: AgentDecision, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Проганяємо один сценарandй"""
        
        # Симуляцandя сценарandю
        base_performance = 0.05  # 5% баwithова продуктивнandсть
        
        # Застосовуємо сценарнand параметри
        if "crash" in scenario_name:
            performance = base_performance - np.random.uniform(-0.5, -0.2)
        elif "volatility" in scenario_name:
            performance = base_performance + np.random.uniform(-0.1, 0.1)
        else:
            performance = base_performance + np.random.uniform(-0.05, 0.05)
        
        return {
            "scenario": scenario_name,
            "performance": performance,
            "confidence": max(0.3, 1.0 - abs(performance)),
            "risk_factors": ["high_volatility", "correlation_spike"] if performance < 0 else []
        }
    
    def _check_correlation_risk(self, decision: Dict[str, Any], 
                               current_positions: Dict[str, float]) -> float:
        """Перевandряємо риwithик кореляцandї"""
        
        # Спрощена перевandрка кореляцandї
        if not current_positions:
            return 0.0
        
        # Симуляцandя кореляцandї
        max_correlation = 0.0
        for ticker in current_positions:
            correlation = np.random.uniform(0.3, 0.9)  # Симуляцandя
            max_correlation = max(max_correlation, correlation)
        
        return max_correlation
    
    def _run_single_validation(self, protocol_name: str, protocol_config: Dict[str, Any],
                             market_data: Dict[str, Any]) -> ValidationResult:
        """Запускаємо один протокол валandдацandї"""
        
        # Симуляцandя валandдацandї
        if protocol_name == "walk_forward":
            return ValidationResult(
                method=ValidationMethod.WALK_FORWARD,
                is_valid=True,
                confidence=0.85,
                issues_found=["small_sample_size"],
                data_leakage_detected=False,
                regime_stability=0.8,
                transaction_costs_impact=0.02
            )
        elif protocol_name == "purged_cv":
            return ValidationResult(
                method=ValidationMethod.PURGED_CV,
                is_valid=True,
                confidence=0.9,
                issues_found=[],
                data_leakage_detected=False,
                regime_stability=0.85,
                transaction_costs_impact=0.015
            )
        else:
            return ValidationResult(
                method=ValidationMethod.CONFORMAL_PREDICTION,
                is_valid=True,
                confidence=0.8,
                issues_found=["conservative_intervals"],
                data_leakage_detected=False,
                regime_stability=0.75,
                transaction_costs_impact=0.025
            )

class DataLeakageDetector:
    """Детектор витоку data"""
    
    def __init__(self):
        self.leakage_patterns = [
            "future_data",
            "lookahead_bias",
            "information_leakage",
            "target_leakage",
            "temporal_leakage"
        ]
    
    def check_leakage(self, data: Dict[str, Any]) -> bool:
        """Перевandряємо наявнandсть витоку data"""
        
        # Перевandряємо на майбутнand данand
        if "future" in str(data).lower():
            return True
        
        # Перевandряємо на lookahead bias
        if data.get("lookahead_period", 0) > 0:
            return True
        
        # Перевandряємо на andнформацandйний витandк
        if data.get("contains_future_info", False):
            return True
        
        return False

def main():
    """Тестування архandтектури Прandwithм"""
    print("[BRAIN] PRIZM ARCHITECTURE - Advanced ML/LLM System")
    print("=" * 60)
    
    engine = PrizmArchitectureEngine()
    
    # Тестуємо робочий процес
    print(f"\n[REFRESH] TESTING PRIZM WORKFLOW")
    print("-" * 40)
    
    market_data = {
        "rsi": 65.5,
        "macd": 0.3,
        "volatility": 0.025,
        "trend_strength": 0.7,
        "volume_ratio": 1.2
    }
    
    current_positions = {"AAPL": 0.015, "MSFT": 0.012}
    
    workflow_result = engine.run_prizm_workflow(market_data, current_positions)
    
    print(f"[DATA] Workflow completed with {len(workflow_result['workflow_steps'])} steps")
    print(f" Final decision: {workflow_result['final_decision'].decision}")
    print(f" Final confidence: {workflow_result['final_decision'].confidence:.1%}")
    
    # Покаwithуємо реwithульandти кожного кроку
    print(f"\n WORKFLOW STEPS:")
    print("-" * 40)
    
    for i, step in enumerate(workflow_result['workflow_steps'], 1):
        print(f"{i}. {step.agent_role.value}: {step.decision}")
        print(f"   Confidence: {step.confidence:.1%}")
        print(f"   Reasoning: {step.reasoning[:100]}...")
    
    # Валandдацandя
    print(f"\n VALIDATION RESULTS:")
    print("-" * 40)
    
    for protocol, result in workflow_result['validation_results'].items():
        print(f"[DATA] {protocol}: {'[OK] Valid' if result.is_valid else '[ERROR] Invalid'}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Issues: {len(result.issues_found)}")
    
    # Аналandwith витоку data
    print(f"\n[SEARCH] DATA LEAKAGE ANALYSIS:")
    print("-" * 40)
    
    detector = DataLeakageDetector()
    leakage_detected = detector.check_leakage(market_data)
    print(f" Leakage detected: {'Yes' if leakage_detected else 'No'}")
    
    # Пandдсумок архandтектури
    print(f"\n[DATA] PRIZM ARCHITECTURE SUMMARY:")
    print("-" * 40)
    
    print(f" Agents: {len(engine.agents)}")
    print(f" Validation protocols: {len(engine.validation_protocols)}")
    print(f" Risk constraints: {len(engine.risk_constraints)}")
    print(f"[GAME] Scenario types: {len(engine.scenario_engine)}")
    
    print(f"\n[TARGET] PRIZM ARCHITECTURE READY!")
    print(f"[BRAIN] Generator  Critic  Refiner  Scenario  Judge")
    print(f" Advanced validation protocols")
    print(f" Comprehensive risk management")
    print(f"[SEARCH] Data leakage detection")
    print(f"[GAME] Scenario stress testing")

if __name__ == "__main__":
    main()
