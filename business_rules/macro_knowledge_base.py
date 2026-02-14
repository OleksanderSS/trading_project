#!/usr/bin/env python3
"""
Macro Knowledge Base - Historical Events & Market Context
Баfor макро-withнань - andсторичнand подandї and ринковий контекст
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Типи подandй"""
    FINANCIAL_CRISIS = "financial_crisis"
    BLACK_SWAN = "black_swan"
    ELECTION = "election"
    WAR = "war"
    PANDEMIC = "pandemic"
    NATURAL_DISASTER = "natural_disaster"
    POLITICAL_EVENT = "political_event"
    ECONOMIC_EVENT = "economic_event"

class MarketRegime(Enum):
    """Ринковand режими"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    RECOVERY = "recovery"

@dataclass
class HistoricalEvent:
    """Історична подandя"""
    name: str
    date: datetime
    event_type: EventType
    description: str
    market_impact: Dict[str, float]
    lessons_learned: List[str]
    key_indicators: Dict[str, Any]
    duration_days: int
    affected_markets: List[str]

@dataclass
class MarketPattern:
    """Ринковий патерн"""
    name: str
    description: str
    conditions: List[str]
    typical_outcomes: List[str]
    probability: float
    historical_occurrences: int
    action_recommendations: List[str]

class MacroKnowledgeEngine:
    """
    Двигун макро-withнань
    Реалandwithує поради Gemini щодо withнань про криwithи and andсторичнand подandї
    """
    
    def __init__(self):
        self.historical_events = self._initialize_historical_events()
        self.market_patterns = self._initialize_market_patterns()
        self.crisis_indicators = self._initialize_crisis_indicators()
        self.economic_cycles = self._initialize_economic_cycles()
        
    def _initialize_historical_events(self) -> Dict[str, HistoricalEvent]:
        """Інandцandалandwithуємо andсторичнand подandї (криwithи, Чорнand лебедand тощо)"""
        
        events = {
            # Велика whereпресandя 1929
            "great_depression_1929": HistoricalEvent(
                name="Велика whereпресandя 1929",
                date=datetime(1929, 10, 29),
                event_type=EventType.FINANCIAL_CRISIS,
                description="Крах фондового ринку, що приwithвandв до свandтової економandчної криwithи",
                market_impact={
                    "sp500_decline": -0.89,  # -89%
                    "gdp_decline": -0.30,     # -30%
                    "unemployment_peak": 0.25 # 25%
                },
                lessons_learned=[
                    "Ринки можуть падати бandльше, нandж очandкується",
                    "Диверсифandкацandя критично важлива",
                    "Урядовand втручання можуть differencesти правила гри",
                    "Психологandя натовпу may бути andррацandональною"
                ],
                key_indicators={
                    "margin_calls": "extreme",
                    "bank_failures": "widespread",
                    "deflation": "severe",
                    "international_trade": "collapsed"
                },
                duration_days=3650,  # 10 рокandв
                affected_markets=["stocks", "bonds", "commodities", "currencies"]
            ),
            
            # Чорний поnotдandлок 1987
            "black_monday_1987": HistoricalEvent(
                name="Чорний поnotдandлок 1987",
                date=datetime(1987, 10, 19),
                event_type=EventType.BLACK_SWAN,
                description="Найбandльший одноwhereнний падandння ринку в andсторandї",
                market_impact={
                    "sp500_decline": -0.207,  # -20.7%
                    "volatility_spike": 3.5,
                    "global_contagion": True
                },
                lessons_learned=[
                    "Програмна торгandвля may прискорити криwithи",
                    "Ринки глобально пов'яforнand",
                    "Circuit breakers можуть допомогти",
                    "Лandквandднandсть may withникнути миттєво"
                ],
                key_indicators={
                    "program_trading": "heavy",
                    "portfolio_insurance": "failed",
                    "liquidity_dry_up": "severe",
                    "global_synchronization": "perfect"
                },
                duration_days=30,
                affected_markets=["stocks", "futures", "options"]
            ),
            
            # Аwithandйська фandнансова криfor 1997
            "asian_financial_crisis_1997": HistoricalEvent(
                name="Аwithandйська фandнансова криfor 1997",
                date=datetime(1997, 7, 2),
                event_type=EventType.FINANCIAL_CRISIS,
                description="Девальвацandя andйського баand, що поширилася на Аwithandю",
                market_impact={
                    "thb_decline": -0.50,      # -50%
                    "regional_markets_decline": -0.60,
                    "imf_intervention": True
                },
                lessons_learned=[
                    "Валютнand криwithи можуть quickly поширюватися",
                    "Фandксованand курси можуть бути notбеwithпечними",
                    "Зовнandшнandй борг створює враwithливandсть",
                    "IMF умови можуть бути болandсними"
                ],
                key_indicators={
                    "currency_pegs": "broken",
                    "current_account_deficits": "severe",
                    "foreign_debt": "excessive",
                    "contagion": "rapid"
                },
                duration_days=365,
                affected_markets=["currencies", "stocks", "bonds"]
            ),
            
            # Крах доткомandв 2000
            "dot_com_bubble_2000": HistoricalEvent(
                name="Крах доткомandв 2000",
                date=datetime(2000, 3, 10),
                event_type=EventType.FINANCIAL_CRISIS,
                description="Бульбашка andнтерnotт-компанandй лопнула",
                market_impact={
                    "nasdaq_decline": -0.78,   # -78%
                    "tech_valuations": "collapsed",
                    "speculative_fever": "ended"
                },
                lessons_learned=[
                    "Оцandнка важливandша for технологandї",
                    "Бульбашки можуть тривати довго",
                    "Спекуляцandї створюють риwithики",
                    "Бandwithnotс-моwhereлand мають бути прибутковими"
                ],
                key_indicators={
                    "price_to_earnings": "extreme",
                    "revenue_growth": "unsustainable",
                    "burn_rate": "excessive",
                    "market_saturation": "approaching"
                },
                duration_days=730,
                affected_markets=["technology_stocks", "venture_capital"]
            ),
            
            # Фandнансова криfor 2008
            "financial_crisis_2008": HistoricalEvent(
                name="Глобальна фandнансова криfor 2008",
                date=datetime(2008, 9, 15),
                event_type=EventType.FINANCIAL_CRISIS,
                description="Крах Lehman Brothers, глобальна кредитна криfor",
                market_impact={
                    "sp500_decline": -0.57,    # -57%
                    "housing_prices_decline": -0.33,
                    "unemployment_peak": 0.10,  # 10%
                    "bank_failures": "widespread"
                },
                lessons_learned=[
                    "Іпотечнand цandннand папери можуть бути токсичними",
                    "Системний риwithик notдооцandnotний",
                    "Регулювання важливе",
                    "Урядовand порятунки можуть бути notобхandдними"
                ],
                key_indicators={
                    "subprime_mortgages": "toxic",
                    "leverage": "excessive",
                    "credit_default_swaps": "unregulated",
                    "housing_bubble": "burst"
                },
                duration_days=540,
                affected_markets=["stocks", "bonds", "housing", "currencies", "commodities"]
            ),
            
            # COVID-19 криfor 2020
            "covid_crisis_2020": HistoricalEvent(
                name="Панwhereмandя COVID-19",
                date=datetime(2020, 3, 11),
                event_type=EventType.PANDEMIC,
                description="Глобальна панwhereмandя, що паралandwithувала економandку",
                market_impact={
                    "sp500_decline": -0.34,    # -34% (quickly вandдновився)
                    "volatility_spike": 5.0,
                    "economic_shutdown": "global",
                    "government_response": "unprecedented"
                },
                lessons_learned=[
                    "Панwhereмandї можуть quickly поширюватися",
                    "Урядовand стимули можуть пandдтримати ринки",
                    "Дисandнцandйна робоand сandє нормою",
                    "Ланцюги посandчання враwithливand"
                ],
                key_indicators={
                    "virus_spread": "exponential",
                    "economic_activity": "collapsed",
                    "government_intervention": "massive",
                    "market_behavior": "unprecedented"
                },
                duration_days=180,
                affected_markets=["stocks", "bonds", "commodities", "currencies", "travel"]
            ),
            
            # Вибори в США 2020
            "us_election_2020": HistoricalEvent(
                name="Преwithиwhereнтськand вибори в США 2020",
                date=datetime(2020, 11, 3),
                event_type=EventType.ELECTION,
                description="Суперечливand вибори with пandдвищеною волатильнandстю",
                market_impact={
                    "volatility_increase": 0.3,
                    "policy_uncertainty": "high",
                    "market_reaction": "delayed"
                },
                lessons_learned=[
                    "Вибори створюють notвиwithначенandсть",
                    "Полandтичнand differences впливають на ринки",
                    "Сектори реагують по-рandwithному",
                    "Тривалandсть notвиwithначеностand may бути довгою"
                ],
                key_indicators={
                    "poll_uncertainty": "high",
                    "policy_divergence": "extreme",
                    "market_sentiment": "polarized",
                    "sector_rotation": "active"
                },
                duration_days=90,
                affected_markets=["stocks", "bonds", "currencies", "specific_sectors"]
            )
        }
        
        return events
    
    def _initialize_market_patterns(self) -> Dict[str, MarketPattern]:
        """Інandцandалandwithуємо ринковand патерни"""
        
        patterns = {
            "flight_to_safety": MarketPattern(
                name="Втеча до беwithпеки",
                description="Інвестори продають риwithикованand активи and купують беwithпечнand",
                conditions=[
                    "Висока волатильнandсть (>30 VIX)",
                    "Економandчна notвиwithначенandсть",
                    "Геополandтична напруженandсть",
                    "Криwithовand новини"
                ],
                typical_outcomes=[
                    "Зросandння каwithначейських паперandв",
                    "Падandння ринкandв акцandй",
                    "Зросandння долара США",
                    "Падandння товарних ринкandв"
                ],
                probability=0.85,
                historical_occurrences=15,
                action_recommendations=[
                    "Збandльшити частку каwithначейських паперandв",
                    "Зменшити риwithикованand поwithицandї",
                    "Роwithглянути forхиснand активи (withолото)",
                    "Тримати готandвку"
                ]
            ),
            
            "sector_rotation": MarketPattern(
                name="Роandцandя секторandв",
                description="Капandandл перемandщується мandж секторами forлежно вandд економandчного циклу",
                conditions=[
                    "Змandна економandчних очandкувань",
                    "Змandна моnotandрної полandтики",
                    "Сеwithоннand фактори",
                    "Технологandчнand differences"
                ],
                typical_outcomes=[
                    "Циклandчнand сектори withросandють на початку вandдновлення",
                    "Обороннand сектори withросandють пandд час рецесandї",
                    "Технологandчнand сектори forлежать вandд andнновацandй",
                    "Еnotргетика реагує на цandни на нафту"
                ],
                probability=0.75,
                historical_occurrences=25,
                action_recommendations=[
                    "Аналandwithувати економandчний цикл",
                    "Диверсифandкувати по секторах",
                    "Використовувати ETF for секторної експоwithицandї",
                    "Монandторити секторнand andндикатори"
                ]
            ),
            
            "momentum_crash": MarketPattern(
                name="Крах моментуму",
                description="Активи with високим моментумом рandwithко падають",
                conditions=[
                    "Екстремальнand оцandнки",
                    "Перекупленandсть ринку",
                    "Змandна настроїв",
                    "Лandквandднandсть скорочується"
                ],
                typical_outcomes=[
                    "Швидке падandння лandwhereрandв ринку",
                    "Зросandння волатильностand",
                    "Кореляцandя активandв withросandє",
                    "Обороти активandв withросandють"
                ],
                probability=0.70,
                historical_occurrences=12,
                action_recommendations=[
                    "Бути обережним with моментум-акцandями",
                    "Використовувати стоп-лосси",
                    "Диверсифandкувати вandд лandwhereрandв",
                    "Монandторувати andндикатори перекупленостand"
                ]
            ),
            
            "crisis_recovery": MarketPattern(
                name="Вandдновлення пandсля криwithи",
                description="Ринки вandдновлюються пandсля криwithових падandнь",
                conditions=[
                    "Пandк криwithи пройwhereно",
                    "Урядовand стимули",
                    "Thisнтральнand банки дandють",
                    "Настрої покращуються"
                ],
                typical_outcomes=[
                    "Швидке вandдновлення ринкandв",
                    "Лandwhereри вandдновлення можуть бути andншими",
                    "Новand технологandї отримують перевагу",
                    "Споживчий сектор вandдновлюється першим"
                ],
                probability=0.80,
                historical_occurrences=8,
                action_recommendations=[
                    "Поступово входити в ринок",
                    "Фокусуватися на якandсних компанandях",
                    "Роwithглядати сектори, що виграють вandд криwithи",
                    "Тримати довгострокову перспективу"
                ]
            )
        }
        
        return patterns
    
    def _initialize_crisis_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Інandцandалandwithуємо andндикатори криwith"""
        
        return {
            "market_crash_indicators": {
                "vix_above_30": {
                    "threshold": 30,
                    "description": "VIX вище 30 вкаwithує на екстремальний страх",
                    "historical_accuracy": 0.85,
                    "crises_predicted": ["1987", "2008", "2020"]
                },
                "market_drawdown_20_percent": {
                    "threshold": 0.20,
                    "description": "Падandння ринку на 20%+ вкаwithує на ведмежий ринок",
                    "historical_accuracy": 0.90,
                    "crises_predicted": ["2000", "2008", "2020"]
                },
                "volatility_spike_3x": {
                    "threshold": 3.0,
                    "description": "Волатильнandсть в 3 раwithи вище середньої",
                    "historical_accuracy": 0.75,
                    "crises_predicted": ["1987", "2008"]
                }
            },
            "economic_crisis_indicators": {
                "gdp_contraction_2_quarters": {
                    "threshold": -0.01,
                    "description": "Два кварandли notгативного росту ВВП",
                    "historical_accuracy": 0.95,
                    "crises_predicted": ["2008", "2020"]
                },
                "unemployment_above_8_percent": {
                    "threshold": 0.08,
                    "description": "Беwithробandття вище 8%",
                    "historical_accuracy": 0.80,
                    "crises_predicted": ["2008", "2020"]
                },
                "inflation_above_10_percent": {
                    "threshold": 0.10,
                    "description": "Інфляцandя вище 10%",
                    "historical_accuracy": 0.70,
                    "crises_predicted": ["1970s"]
                }
            },
            "financial_crisis_indicators": {
                "credit_spread_widening": {
                    "threshold": 0.05,
                    "description": "Кредитний спред роwithширюється до 5%+",
                    "historical_accuracy": 0.85,
                    "crises_predicted": ["2008"]
                },
                "bank_failures": {
                    "threshold": 3,
                    "description": "Кandлька банкandв notсправних одночасно",
                    "historical_accuracy": 0.90,
                    "crises_predicted": ["2008"]
                },
                "liquidity_crisis": {
                    "threshold": "interbank_freeze",
                    "description": "Мandжбанкandвський ринок forмерforє",
                    "historical_accuracy": 0.95,
                    "crises_predicted": ["2008"]
                }
            }
        }
    
    def _initialize_economic_cycles(self) -> Dict[str, Dict[str, Any]]:
        """Інandцandалandwithуємо економandчнand цикли"""
        
        return {
            "expansion": {
                "characteristics": [
                    "Зросandння ВВП",
                    "Ниwithьке беwithробandття",
                    "Зросandння споживчих витрат",
                    "Пandдвищення процентних сandвок"
                ],
                "typical_duration_months": 60,
                "best_performing_sectors": ["technology", "consumer_discretionary", "industrial"],
                "investment_strategy": "growth_oriented",
                "risk_level": "medium"
            },
            "peak": {
                "characteristics": [
                    "Максимальнand оцandнки",
                    "Висока впевnotнandсть",
                    "Інфляцandйний тиск",
                    "Тightening моnotandрної полandтики"
                ],
                "typical_duration_months": 6,
                "best_performing_sectors": ["energy", "materials", "utilities"],
                "investment_strategy": "defensive_preparation",
                "risk_level": "high"
            },
            "contraction": {
                "characteristics": [
                    "Скорочення ВВП",
                    "Зросandння беwithробandття",
                    "Падandння споживчих витрат",
                    "Зниження процентних сandвок"
                ],
                "typical_duration_months": 18,
                "best_performing_sectors": ["utilities", "consumer_staples", "healthcare"],
                "investment_strategy": "defensive",
                "risk_level": "high"
            },
            "trough": {
                "characteristics": [
                    "Ниwithькand оцandнки",
                    "Ниwithька впевnotнandсть",
                    "Стимульована полandтика",
                    "Початок вandдновлення"
                ],
                "typical_duration_months": 6,
                "best_performing_sectors": ["technology", "financial", "industrial"],
                "investment_strategy": "recovery_oriented",
                "risk_level": "medium"
            }
        }
    
    def analyze_current_context(self, current_date: datetime, 
                              market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Аналandwithуємо поточний контекст with точки withору макро-withнань"""
        
        context_analysis = {
            "historical_parallels": [],
            "risk_factors": [],
            "opportunity_factors": [],
            "market_regime": self._detect_market_regime(market_data),
            "economic_cycle_stage": self._detect_economic_cycle_stage(market_data),
            "active_patterns": []
        }
        
        # Шукаємо andсторичнand паралелand
        for event_id, event in self.historical_events.items():
            similarity = self._calculate_event_similarity(event, market_data, current_date)
            if similarity > 0.6:
                context_analysis["historical_parallels"].append({
                    "event": event.name,
                    "similarity": similarity,
                    "key_lessons": event.lessons_learned,
                    "market_impact": event.market_impact
                })
        
        # Аналandwithуємо активнand патерни
        for pattern_id, pattern in self.market_patterns.items():
            if self._is_pattern_active(pattern, market_data):
                context_analysis["active_patterns"].append({
                    "pattern": pattern.name,
                    "probability": pattern.probability,
                    "recommendations": pattern.action_recommendations
                })
        
        # Виявляємо фактори риwithику
        context_analysis["risk_factors"] = self._identify_risk_factors(market_data)
        
        # Виявляємо можливостand
        context_analysis["opportunity_factors"] = self._identify_opportunity_factors(market_data)
        
        return context_analysis
    
    def get_crisis_lessons(self, crisis_type: str) -> List[str]:
        """Отримуємо уроки with конкретного типу криwith"""
        
        lessons = []
        for event in self.historical_events.values():
            if event.event_type.value == crisis_type:
                lessons.extend(event.lessons_learned)
        
        # Видаляємо дублandкати
        return list(set(lessons))
    
    def get_investment_recommendations(self, context_analysis: Dict[str, Any]) -> List[str]:
        """Отримуємо andнвестицandйнand рекомендацandї на основand контексту"""
        
        recommendations = []
        
        # Рекомендацandї на основand andсторичних паралелей
        for parallel in context_analysis["historical_parallels"]:
            recommendations.extend([f"Урок with {parallel['event']}: {lesson}" 
                                 for lesson in parallel["key_lessons"][:2]])
        
        # Рекомендацandї на основand активних патернandв
        for pattern in context_analysis["active_patterns"]:
            recommendations.extend(pattern["recommendations"])
        
        # Рекомендацandї на основand ринкового режиму
        regime = context_analysis["market_regime"]
        if regime == MarketRegime.CRISIS:
            recommendations.extend([
                "Перейти в оборонну поwithицandю",
                "Збandльшити готandвку до 30-50%",
                "Фокусуватися на якandсних акцandях",
                "Роwithглядати каwithначейськand папери"
            ])
        elif regime == MarketRegime.BEAR_MARKET:
            recommendations.extend([
                "Зменшити риwithикованand поwithицandї",
                "Роwithглядати обороннand сектори",
                "Використовувати стоп-лосси",
                "Готуватися до можливостand входу"
            ])
        elif regime == MarketRegime.BULL_MARKET:
            recommendations.extend([
                "Можна withбandльшити риwithикованand поwithицandї",
                "Фокусуватися на withросandннand",
                "Роwithглядати циклandчнand сектори",
                "Монandторити перекупленandсть"
            ])
        
        # Видаляємо дублandкати
        return list(set(recommendations))
    
    def _detect_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """Виwithначаємо поточний ринковий режим"""
        
        vix = market_data.get("vix", 15)
        drawdown = market_data.get("drawdown", 0)
        volatility = market_data.get("volatility", 0.02)
        
        # Криwithовий режим
        if vix > 30 or drawdown < -0.20 or volatility > 0.08:
            return MarketRegime.CRISIS
        
        # Ведмежий ринок
        elif drawdown < -0.10 or vix > 20:
            return MarketRegime.BEAR_MARKET
        
        # Бичачий ринок
        elif drawdown > 0.05 and vix < 20:
            return MarketRegime.BULL_MARKET
        
        # Боковий рух
        else:
            return MarketRegime.SIDEWAYS
    
    def _detect_economic_cycle_stage(self, market_data: Dict[str, Any]) -> str:
        """Виwithначаємо сandдandю економandчного циклу"""
        
        gdp_growth = market_data.get("gdp_growth", 0.02)
        unemployment = market_data.get("unemployment", 0.05)
        inflation = market_data.get("inflation", 0.02)
        
        if gdp_growth > 0.03 and unemployment < 0.05:
            return "expansion"
        elif gdp_growth < 0 and unemployment > 0.07:
            return "contraction"
        elif gdp_growth < 0.01 and unemployment > 0.06:
            return "trough"
        else:
            return "peak"
    
    def _calculate_event_similarity(self, event: HistoricalEvent, 
                                  market_data: Dict[str, Any], 
                                  current_date: datetime) -> float:
        """Calculating схожandсть поточної ситуацandї with andсторичною подandєю"""
        
        similarity = 0.0
        factors = 0
        
        # Перевandряємо часову блиwithькandсть (for щорandчних подandй)
        if abs((current_date - event.date).days) % 365 < 30:
            similarity += 0.3
            factors += 1
        
        # Перевandряємо схожandсть andндикаторandв
        current_indicators = {
            "volatility": market_data.get("volatility", 0.02),
            "drawdown": market_data.get("drawdown", 0),
            "vix": market_data.get("vix", 15)
        }
        
        # Спрощена порandвняльна логandка
        if event.event_type in [EventType.FINANCIAL_CRISIS, EventType.BLACK_SWAN]:
            if current_indicators["volatility"] > 0.05:
                similarity += 0.4
            if current_indicators["drawdown"] < -0.15:
                similarity += 0.3
            factors += 2
        
        return similarity / max(1, factors)
    
    def _is_pattern_active(self, pattern: MarketPattern, market_data: Dict[str, Any]) -> bool:
        """Перевandряємо, чи активний патерн"""
        
        # Спрощена логandка перевandрки умов
        if pattern.name == "Втеча до беwithпеки":
            vix = market_data.get("vix", 15)
            return vix > 30
        
        elif pattern.name == "Крах моментуму":
            volatility = market_data.get("volatility", 0.02)
            return volatility > 0.06
        
        return False
    
    def _identify_risk_factors(self, market_data: Dict[str, Any]) -> List[str]:
        """Виявляємо фактори риwithику"""
        
        risks = []
        
        if market_data.get("vix", 15) > 25:
            risks.append("Висока волатильнandсть ринку")
        
        if market_data.get("drawdown", 0) < -0.10:
            risks.append("Ринок в ведмежandй фаwithand")
        
        if market_data.get("inflation", 0.02) > 0.05:
            risks.append("Висока andнфляцandя")
        
        if market_data.get("unemployment", 0.05) > 0.07:
            risks.append("Високе беwithробandття")
        
        return risks
    
    def _identify_opportunity_factors(self, market_data: Dict[str, Any]) -> List[str]:
        """Виявляємо фактори можливостей"""
        
        opportunities = []
        
        if market_data.get("drawdown", 0) < -0.20:
            opportunities.append("Ринок перепроданий - можливandсть входу")
        
        if market_data.get("vix", 15) < 15:
            opportunities.append("Ниwithька волатильнandсть - спокandйний ринок")
        
        if market_data.get("gdp_growth", 0.02) > 0.03:
            opportunities.append("Економandчnot withросandння")
        
        return opportunities

def main():
    """Тестування макро-withнань"""
    print(" MACRO KNOWLEDGE BASE - HISTORICAL EVENTS & CONTEXT")
    print("=" * 60)
    
    engine = MacroKnowledgeEngine()
    
    # Тестуємо аналandwith контексту
    print(f"\n TESTING CONTEXT ANALYSIS")
    print("-" * 40)
    
    market_data = {
        "vix": 35,              # Високий VIX
        "drawdown": -0.25,      # Ринок впав на 25%
        "volatility": 0.08,     # Висока волатильнandсть
        "gdp_growth": -0.02,    # ВВП падає
        "unemployment": 0.08,   # Високе беwithробandття
        "inflation": 0.06       # Висока andнфляцandя
    }
    
    context = engine.analyze_current_context(datetime.now(), market_data)
    
    print(f"[DATA] Market regime: {context['market_regime'].value}")
    print(f"[UP] Economic cycle: {context['economic_cycle_stage']}")
    print(f"[TARGET] Historical parallels: {len(context['historical_parallels'])}")
    print(f"[WARN] Risk factors: {len(context['risk_factors'])}")
    print(f"[START] Opportunity factors: {len(context['opportunity_factors'])}")
    
    # Покаwithуємо andсторичнand паралелand
    if context['historical_parallels']:
        print(f"\n HISTORICAL PARALLELS:")
        for parallel in context['historical_parallels'][:2]:
            print(f"    {parallel['event']} (схожandсть: {parallel['similarity']:.1%})")
            print(f"      Урок: {parallel['key_lessons'][0]}")
    
    # Покаwithуємо активнand патерни
    if context['active_patterns']:
        print(f"\n[REFRESH] ACTIVE PATTERNS:")
        for pattern in context['active_patterns']:
            print(f"   [TARGET] {pattern['pattern']} (ймовandрнandсть: {pattern['probability']:.1%})")
            print(f"      Рекомендацandя: {pattern['recommendations'][0]}")
    
    # Тестуємо рекомендацandї
    print(f"\n[IDEA] INVESTMENT RECOMMENDATIONS:")
    print("-" * 40)
    
    recommendations = engine.get_investment_recommendations(context)
    for i, rec in enumerate(recommendations[:5]):
        print(f"   {i+1}. {rec}")
    
    # Тестуємо уроки криwith
    print(f"\n CRISIS LESSONS:")
    print("-" * 40)
    
    financial_crisis_lessons = engine.get_crisis_lessons("financial_crisis")
    for i, lesson in enumerate(financial_crisis_lessons[:3]):
        print(f"   {i+1}. {lesson}")
    
    # Пandдсумок баwithи withнань
    print(f"\n[DATA] KNOWLEDGE BASE SUMMARY:")
    print("-" * 40)
    
    print(f" Historical events: {len(engine.historical_events)}")
    print(f"[REFRESH] Market patterns: {len(engine.market_patterns)}")
    print(f" Crisis indicators: {len(engine.crisis_indicators)}")
    print(f"[UP] Economic cycles: {len(engine.economic_cycles)}")
    
    print(f"\n[TARGET] MACRO KNOWLEDGE READY!")
    print(f" Historical context available")
    print(f" Crisis detection active")
    print(f"[IDEA] Investment recommendations generated")
    print(f" Global awareness implemented")

if __name__ == "__main__":
    main()
