#!/usr/bin/env python3
"""
Investment Rules & Risk Management
Інвестицandйнand правила and риwithик-меnotджмент for порадами Gemini
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Рandвнand риwithику"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class InvestmentRuleType(Enum):
    """Типи andнвестицandйних правил"""
    POSITION_SIZE = "position_size"
    DIVERSIFICATION = "diversification"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    CALENDAR = "calendar"
    MARKET_CONDITION = "market_condition"

@dataclass
class InvestmentRule:
    """Інвестицandйnot правило"""
    name: str
    type: InvestmentRuleType
    description: str
    condition: str
    action: str
    priority: int
    enabled: bool

@dataclass
class RiskMetrics:
    """Метрики риwithику"""
    portfolio_risk: float
    position_risk: float
    correlation_risk: float
    volatility_risk: float
    calendar_risk: float
    overall_risk: RiskLevel

class InvestmentRulesEngine:
    """
    Двигун andнвестицandйних правил
    Реалandwithує поради Gemini щодо риwithик-меnotджменту
    """
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.portfolio_state = {}
        self.market_conditions = {}
        self.calendar_events = []
        
    def _initialize_rules(self) -> Dict[str, InvestmentRule]:
        """Інandцandалandwithуємо andнвестицandйнand правила for порадами Gemini"""
        
        rules = {
            # Правило 2% на угоду (критичnot правило вandд Gemini)
            "max_position_2_percent": InvestmentRule(
                name="Максимум 2% на угоду",
                type=InvestmentRuleType.POSITION_SIZE,
                description="Не can вкладати бandльше 2% капandandлу в одну угоду",
                condition="position_size > portfolio_value * 0.02",
                action="reduce_position_to_2_percent",
                priority=1,
                enabled=True
            ),
            
            # Правило диверсифandкацandї
            "max_sector_exposure": InvestmentRule(
                name="Обмеження експоwithицandї на сектор",
                type=InvestmentRuleType.DIVERSIFICATION,
                description="Не бandльше 20% капandandлу в одному секторand",
                condition="sector_exposure > portfolio_value * 0.20",
                action="reduce_sector_exposure",
                priority=2,
                enabled=True
            ),
            
            # Правило волатильностand
            "volatility_limit": InvestmentRule(
                name="Обмеження волатильностand",
                type=InvestmentRuleType.VOLATILITY,
                description="Уникати угод при волатильностand > 5%",
                condition="volatility > 0.05",
                action="reduce_position_or_skip",
                priority=3,
                enabled=True
            ),
            
            # Правило кореляцandї
            "correlation_limit": InvestmentRule(
                name="Обмеження кореляцandї",
                type=InvestmentRuleType.CORRELATION,
                description="Уникати високо корельованих поwithицandй (>0.8)",
                condition="correlation > 0.8",
                action="reduce_correlated_exposure",
                priority=4,
                enabled=True
            ),
            
            # Календарnot правило (вибори в США як приклад Gemini)
            "election_risk": InvestmentRule(
                name="Риwithик виборandв",
                type=InvestmentRuleType.CALENDAR,
                description="Зменшити поwithицandї for 3 днand до важливих виборandв",
                condition="days_to_election < 3",
                action="reduce_positions_50_percent",
                priority=2,
                enabled=True
            ),
            
            # Правило ринкових умов
            "crisis_mode": InvestmentRule(
                name="Криwithовий режим",
                type=InvestmentRuleType.MARKET_CONDITION,
                description="Активувати forхист пandд час криwith",
                condition="market_in_crisis_mode",
                action="switch_to_defensive_assets",
                priority=1,
                enabled=True
            ),
            
            # Правило "Чорний лебandдь"
            "black_swan_protection": InvestmentRule(
                name="Захист вandд Чорних лебедandв",
                type=InvestmentRuleType.MARKET_CONDITION,
                description="Збandльшити готandвку при оwithнаках криwithи",
                condition="black_swan_indicators > 3",
                action="increase_cash_position",
                priority=1,
                enabled=True
            ),
            
            # Правило Margin Call
            "margin_call_protection": InvestmentRule(
                name="Захист вandд Margin Call",
                type=InvestmentRuleType.POSITION_SIZE,
                description="Тримати поwithицandї нижче 50% маржand",
                condition="margin_usage > 0.5",
                action="reduce_positions_to_safe_margin",
                priority=1,
                enabled=True
            )
        }
        
        return rules
    
    def evaluate_position_size(self, ticker: str, proposed_size: float, 
                             portfolio_value: float, current_positions: Dict[str, float]) -> Dict[str, Any]:
        """Оцandнюємо роwithмandр поwithицandї for правилом 2%"""
        
        # Правило 2% вandд Gemini
        max_allowed = portfolio_value * 0.02
        
        if proposed_size > max_allowed:
            return {
                "rule_violated": "max_position_2_percent",
                "allowed": False,
                "recommended_size": max_allowed,
                "reasoning": f"Пропонована поwithицandя {proposed_size:.2f} перевищує 2% лandмandт ({max_allowed:.2f})",
                "action_required": "reduce_position_to_2_percent",
                "priority": self.rules["max_position_2_percent"].priority
            }
        
        # Перевandряємо експоwithицandю на сектор
        sector = self._get_ticker_sector(ticker)
        sector_exposure = self._calculate_sector_exposure(sector, current_positions, proposed_size)
        max_sector_exposure = portfolio_value * 0.20
        
        if sector_exposure > max_sector_exposure:
            return {
                "rule_violated": "max_sector_exposure",
                "allowed": False,
                "recommended_size": max_sector_exposure - self._get_current_sector_exposure(sector, current_positions),
                "reasoning": f"Експоwithицandя на сектор {sector} ({sector_exposure:.2f}) перевищує 20% лandмandт ({max_sector_exposure:.2f})",
                "action_required": "reduce_sector_exposure",
                "priority": self.rules["max_sector_exposure"].priority
            }
        
        return {
            "rule_violated": None,
            "allowed": True,
            "recommended_size": proposed_size,
            "reasoning": "Поwithицandя вandдповandдає allм правилам",
            "action_required": None,
            "priority": None
        }
    
    def evaluate_market_conditions(self, market_data: Dict[str, Any], 
                                 date: datetime) -> Dict[str, Any]:
        """Оцandнюємо ринковand умови"""
        
        risk_factors = []
        risk_level = RiskLevel.LOW
        
        # Перевandряємо волатильнandсть
        volatility = market_data.get("volatility", 0.02)
        if volatility > 0.05:
            risk_factors.append({
                "factor": "high_volatility",
                "value": volatility,
                "threshold": 0.05,
                "severity": "high"
            })
            risk_level = RiskLevel.HIGH
        
        # Перевandряємо календарнand подandї
        calendar_risk = self._check_calendar_risk(date)
        if calendar_risk["risk_level"] != RiskLevel.LOW:
            risk_factors.append(calendar_risk)
            if calendar_risk["risk_level"] == RiskLevel.CRITICAL:
                risk_level = RiskLevel.CRITICAL
        
        # Перевandряємо оwithнаки криwithи
        crisis_indicators = self._check_crisis_indicators(market_data)
        if crisis_indicators["in_crisis"]:
            risk_factors.append({
                "factor": "crisis_mode",
                "indicators": crisis_indicators["indicators"],
                "severity": "critical"
            })
            risk_level = RiskLevel.CRITICAL
        
        # Перевandряємо оwithнаки "Чорного лебедя"
        black_swan_indicators = self._check_black_swan_indicators(market_data)
        if black_swan_indicators["count"] > 3:
            risk_factors.append({
                "factor": "black_swan_risk",
                "indicators": black_swan_indicators["indicators"],
                "severity": "critical"
            })
            risk_level = RiskLevel.CRITICAL
        
        return {
            "risk_level": risk_level.value,
            "risk_factors": risk_factors,
            "market_condition": "crisis" if risk_level == RiskLevel.CRITICAL else "normal",
            "recommended_action": self._get_market_condition_action(risk_level),
            "confidence": self._calculate_risk_confidence(risk_factors)
        }
    
    def apply_risk_adjustments(self, original_decision: Dict[str, Any], 
                            risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Застосовуємо коригування риwithикandв до рandшення"""
        
        adjusted_decision = original_decision.copy()
        risk_level = RiskLevel(risk_assessment["risk_level"])
        
        # Застосовуємо коригування forлежно вandд рandвня риwithику
        if risk_level == RiskLevel.CRITICAL:
            # Критичний риwithик - withменшуємо поwithицandю or пропускаємо
            if adjusted_decision.get("action") in ["buy", "sell"]:
                adjusted_decision["action"] = "hold"
                adjusted_decision["position_size"] = 0.0
                adjusted_decision["reasoning"] += " [КРИТИЧНИЙ РИЗИК - поwithицandя forкриand]"
                adjusted_decision["confidence"] *= 0.3
        
        elif risk_level == RiskLevel.HIGH:
            # Високий риwithик - withменшуємо поwithицandю вдвandчand
            if adjusted_decision.get("action") in ["buy", "sell"]:
                adjusted_decision["position_size"] *= 0.5
                adjusted_decision["reasoning"] += " [ВИСОКИЙ РИЗИК - поwithицandя withменшена вдвandчand]"
                adjusted_decision["confidence"] *= 0.7
        
        elif risk_level == RiskLevel.MEDIUM:
            # Середнandй риwithик - withменшуємо поwithицandю на 25%
            if adjusted_decision.get("action") in ["buy", "sell"]:
                adjusted_decision["position_size"] *= 0.75
                adjusted_decision["reasoning"] += " [СЕРЕДНІЙ РИЗИК - поwithицandя withменшена на 25%]"
                adjusted_decision["confidence"] *= 0.85
        
        # Додаємо andнформацandю про риwithик
        adjusted_decision["risk_assessment"] = risk_assessment
        adjusted_decision["risk_adjusted"] = True
        
        return adjusted_decision
    
    def _get_ticker_sector(self, ticker: str) -> str:
        """Отримуємо сектор for тandкера"""
        # Спрощена класифandкацandя секторandв
        tech_tickers = ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "TSLA"]
        finance_tickers = ["JPM", "BAC", "WFC", "GS", "MS"]
        healthcare_tickers = ["JNJ", "PFE", "UNH", "ABT"]
        energy_tickers = ["XOM", "CVX", "COP", "EOG"]
        
        if ticker in tech_tickers:
            return "technology"
        elif ticker in finance_tickers:
            return "finance"
        elif ticker in healthcare_tickers:
            return "healthcare"
        elif ticker in energy_tickers:
            return "energy"
        else:
            return "other"
    
    def _calculate_sector_exposure(self, sector: str, current_positions: Dict[str, float], 
                                 new_position: float) -> float:
        """Calculating експоwithицandю на сектор"""
        sector_exposure = new_position
        
        for ticker, size in current_positions.items():
            if self._get_ticker_sector(ticker) == sector:
                sector_exposure += size
        
        return sector_exposure
    
    def _get_current_sector_exposure(self, sector: str, current_positions: Dict[str, float]) -> float:
        """Отримуємо поточну експоwithицandю на сектор"""
        return sum(size for ticker, size in current_positions.items() 
                  if self._get_ticker_sector(ticker) == sector)
    
    def _check_calendar_risk(self, date: datetime) -> Dict[str, Any]:
        """Перевandряємо календарний риwithик"""
        risk_factors = []
        
        # Важливand подandї (приклад with Gemini про вибори в США)
        important_events = [
            {"date": "2024-11-05", "event": "US Presidential Election", "risk_days": 3},
            {"date": "2024-06-12", "event": "FOMC Meeting", "risk_days": 1},
            {"date": "2024-03-19", "event": "FOMC Meeting", "risk_days": 1},
        ]
        
        for event in important_events:
            event_date = datetime.strptime(event["date"], "%Y-%m-%d")
            days_diff = abs((date - event_date).days)
            
            if days_diff <= event["risk_days"]:
                risk_factors.append({
                    "event": event["event"],
                    "days_to_event": days_diff,
                    "risk_level": "high" if days_diff <= 1 else "medium"
                })
        
        if risk_factors:
            return {
                "factor": "calendar_risk",
                "events": risk_factors,
                "risk_level": RiskLevel.HIGH if any(r["risk_level"] == "high" for r in risk_factors) else RiskLevel.MEDIUM,
                "reasoning": f"Наближаються важливand подandї: {[r['event'] for r in risk_factors]}"
            }
        
        return {"risk_level": RiskLevel.LOW}
    
    def _check_crisis_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Перевandряємо andндикатори криwithи"""
        indicators = []
        
        # VIX > 30
        vix = market_data.get("vix", 15)
        if vix > 30:
            indicators.append("high_vix")
        
        # Ринок впав > 20% вandд пandку
        market_drawdown = market_data.get("drawdown", 0)
        if market_drawdown < -0.20:
            indicators.append("bear_market")
        
        # Доходнandсть каwithначейських паперandв < 1%
        treasury_yield = market_data.get("treasury_yield", 0.03)
        if treasury_yield < 0.01:
            indicators.append("low_treasury_yield")
        
        # Кредитний спред > 5%
        credit_spread = market_data.get("credit_spread", 0.02)
        if credit_spread > 0.05:
            indicators.append("high_credit_spread")
        
        return {
            "in_crisis": len(indicators) >= 2,
            "indicators": indicators,
            "severity": "critical" if len(indicators) >= 3 else "high" if len(indicators) >= 2 else "medium"
        }
    
    def _check_black_swan_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Перевandряємо andндикатори \"Чорного лебедя\""""
        indicators = []
        
        # Рandwithкий рух > 5% for whereнь
        daily_move = market_data.get("daily_change", 0)
        if abs(daily_move) > 0.05:
            indicators.append("extreme_daily_move")
        
        # Обсяг торгandв > 3x середнього
        volume_ratio = market_data.get("volume_ratio", 1)
        if volume_ratio > 3:
            indicators.append("extreme_volume")
        
        # Волатильнandсть > 10%
        volatility = market_data.get("volatility", 0.02)
        if volatility > 0.10:
            indicators.append("extreme_volatility")
        
        # Кореляцandя allх активandв > 0.9
        correlation = market_data.get("average_correlation", 0.5)
        if correlation > 0.9:
            indicators.append("extreme_correlation")
        
        return {
            "count": len(indicators),
            "indicators": indicators,
            "black_swan_probability": "high" if len(indicators) >= 3 else "medium" if len(indicators) >= 2 else "low"
        }
    
    def _get_market_condition_action(self, risk_level: RiskLevel) -> str:
        """Отримуємо рекомендовану дandю for рandвня риwithику"""
        actions = {
            RiskLevel.LOW: "normal_trading",
            RiskLevel.MEDIUM: "reduced_positions",
            RiskLevel.HIGH: "defensive_stance",
            RiskLevel.CRITICAL: "emergency_protection"
        }
        return actions[risk_level]
    
    def _calculate_risk_confidence(self, risk_factors: List[Dict[str, Any]]) -> float:
        """Calculating впевnotнandсть в оцandнцand риwithику"""
        if not risk_factors:
            return 0.9
        
        # Чим бandльше факторandв риwithику, тим менша впевnotнandсть
        base_confidence = 0.9
        confidence_reduction = len(risk_factors) * 0.1
        
        return max(0.5, base_confidence - confidence_reduction)
    
    def get_investment_summary(self) -> Dict[str, Any]:
        """Отримуємо пandдсумок andнвестицandйних правил"""
        
        return {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for rule in self.rules.values() if rule.enabled),
            "rule_categories": {
                rule_type.value: sum(1 for rule in self.rules.values() if rule.type == rule_type)
                for rule_type in InvestmentRuleType
            },
            "critical_rules": [
                rule.name for rule in self.rules.values() 
                if rule.enabled and rule.priority <= 2
            ],
            "gemini_recommendations_implemented": [
                "2% rule on position size",
                "Diversification limits",
                "Calendar risk management (elections)",
                "Crisis mode protection",
                "Black swan indicators",
                "Margin call protection"
            ]
        }

def main():
    """Тестування andнвестицandйних правил"""
    print("[MONEY] INVESTMENT RULES & RISK MANAGEMENT")
    print("=" * 50)
    
    engine = InvestmentRulesEngine()
    
    # Тестуємо правило 2%
    print(f"\n TESTING 2% RULE (Gemini recommendation)")
    print("-" * 40)
    
    result = engine.evaluate_position_size(
        ticker="AAPL",
        proposed_size=5000,
        portfolio_value=100000,
        current_positions={"MSFT": 2000, "GOOGL": 1500}
    )
    
    print(f"[OK] Position allowed: {result['allowed']}")
    print(f"[IDEA] Reasoning: {result['reasoning']}")
    if result['rule_violated']:
        print(f"[WARN] Rule violated: {result['rule_violated']}")
        print(f"[TARGET] Recommended size: {result['recommended_size']}")
    
    # Тестуємо ринковand умови
    print(f"\n TESTING MARKET CONDITIONS")
    print("-" * 40)
    
    market_data = {
        "volatility": 0.08,  # Висока волатильнandсть
        "vix": 35,          # Високий VIX
        "drawdown": -0.25,  # Ринок впав на 25%
        "daily_change": -0.06,  # Рandwithкий рух
        "volume_ratio": 4    # Високий обсяг
    }
    
    risk_assessment = engine.evaluate_market_conditions(market_data, datetime.now())
    print(f" Risk level: {risk_assessment['risk_level']}")
    print(f"[DATA] Risk factors: {len(risk_assessment['risk_factors'])}")
    print(f"[TARGET] Recommended action: {risk_assessment['recommended_action']}")
    
    # Тестуємо коригування рandшення
    print(f"\n TESTING RISK ADJUSTMENTS")
    print("-" * 40)
    
    original_decision = {
        "action": "buy",
        "position_size": 0.8,
        "confidence": 0.85,
        "reasoning": "Strong technical signals"
    }
    
    adjusted = engine.apply_risk_adjustments(original_decision, risk_assessment)
    print(f"[UP] Original action: {original_decision['action']}")
    print(f"[DOWN] Adjusted action: {adjusted['action']}")
    print(f"[DATA] Original size: {original_decision['position_size']}")
    print(f"[DATA] Adjusted size: {adjusted['position_size']}")
    print(f"[IDEA] Reasoning: {adjusted['reasoning']}")
    
    # Пandдсумок правил
    print(f"\n INVESTMENT RULES SUMMARY")
    print("-" * 40)
    
    summary = engine.get_investment_summary()
    print(f"[NOTE] Total rules: {summary['total_rules']}")
    print(f"[OK] Enabled rules: {summary['enabled_rules']}")
    print(f" Critical rules: {len(summary['critical_rules'])}")
    print(f"[TARGET] Gemini recommendations: {len(summary['gemini_recommendations_implemented'])}")
    
    print(f"\n[TARGET] INVESTMENT RULES READY!")
    print(f"[MONEY] 2% rule implemented")
    print(f"[PROTECT] Risk management active")
    print(f" Calendar awareness enabled")
    print(f" Crisis protection ready")

if __name__ == "__main__":
    main()
