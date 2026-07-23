import asyncio
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from src.agents.archive.veto_system import AgenticVetoSystem
from src.meta_learning.memory.diary_engine import DiaryEngine, DecisionRecord, DecisionType, DecisionOutcome
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ShadowArena")

import asyncio
import logging
from typing import Dict, Any, List

from src.agents.archive.veto_system import AgenticVetoSystem
from src.meta_learning.memory.diary_engine import DiaryEngine, DecisionRecord, DecisionType, DecisionOutcome
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ShadowArena")

class HistoricalTypologyArena:
    """
    Shadow Arena V2: Historical Typologies
    Тестування "чуйки" Агента-Керуючого на різних макроекономічних, 
    геополітичних та кліматичних архетипах.
    """
    def __init__(self):
        self.veto_system = AgenticVetoSystem()
        self.diary = DiaryEngine()

    async def _run_scenario(self, scenario_name: str, signals: List[Dict[str, Any]], context: Dict[str, Any], expected_vetoes: List[str], lesson: str):
        logger.info(f"\n{'='*50}\n=== RUNNING SCENARIO: {scenario_name} ===\n{'='*50}")
        context_string = str(context)
        
        logger.info(f"Market Context: {context['news_headlines'][0]}")
        logger.info(f"Math Models Output: {signals}")

        decisions = await self.veto_system.review_recommendations(signals, latest_news=context_string)
        
        actual_vetoes = []
        for decision in decisions:
            ticker = decision.get("ticker")
            if decision.get("vetoed"):
                actual_vetoes.append(ticker)
                logger.warning(f"🎯 AGENT VETOED {ticker}: {decision.get('veto_reason')}")
            else:
                logger.info(f"✅ AGENT APPROVED {ticker}: {decision.get('veto_reason')}")
                
            if "causal_graph" in decision:
                graph = decision.get("causal_graph")
                logger.info(f"   📊 CAUSAL GRAPH: {graph}")


        # Verification Logic
        success = set(actual_vetoes) == set(expected_vetoes)
        if success:
            logger.info(f"🏆 SCENARIO PASSED! Agent intuition matched expectations.")
        else:
            logger.error(f"❌ SCENARIO FAILED. Expected vetoes: {expected_vetoes}, Actual: {actual_vetoes}")

        # Replace old experience_journal usage with DiaryEngine
        record = DecisionRecord(
            agent_id="ShadowArena",
            ticker="PORTFOLIO",
            decision_type=DecisionType.METADATA,
            reasoning=lesson,
            market_context={
                "scenario": scenario_name,
                "context": context_string[:200],
                "vetoed": actual_vetoes
            },
            context_fingerprint="shadow_arena_scenario",
            outcome=DecisionOutcome.NEUTRAL
        )
        self.diary.record_decision(record)

    async def scenario_war_inflation(self):
        signals = [{"ticker": "SPY", "model": "MeanReversion", "action": "BUY", "confidence": 0.88}]
        context = {
            "date": "1973-10-15",
            "news_headlines": ["Oil embargo triggers massive energy crisis.", "Inflation spikes to double digits amid prolonged geopolitical conflict."],
            "fred_data_simulation": {"CPI": 12.0, "OIL_PRICE": 150.0}
        }
        await self._run_scenario("War & Inflation (1970s)", signals, context, expected_vetoes=["SPY"], lesson="During stagflation, avoid broad equity index buys.")

    async def scenario_swift_victory(self):
        signals = [{"ticker": "SPY", "model": "PanicSell_Model", "action": "SELL", "confidence": 0.95}]
        context = {
            "date": "1991-01-17",
            "news_headlines": ["Coalition forces launch Operation Desert Storm.", "Swift, decisive victories reported; oil supply secure."],
            "fred_data_simulation": {"VIX": 15.0} # Low fear despite war
        }
        await self._run_scenario("Swift Geopolitical Victory (1991)", signals, context, expected_vetoes=["SPY"], lesson="Do not panic sell if institutional response is overwhelming and decisive.")

    async def scenario_credit_bubble(self):
        signals = [{"ticker": "XLF", "model": "Value_PE", "action": "BUY", "confidence": 0.99}]
        context = {
            "date": "2007-08-09",
            "news_headlines": ["BNP Paribas freezes funds over subprime exposure.", "Interbank lending seizes up as toxic MBS defaults mount."],
            "fred_data_simulation": {"TED_SPREAD": 2.5}
        }
        await self._run_scenario("Credit Bubble Collapse (2007)", signals, context, expected_vetoes=["XLF"], lesson="Ignore low P/E ratios when the underlying assets (MBS) are toxic.")

    async def scenario_productivity_boom(self):
        signals = [{"ticker": "QQQ", "model": "Value_PE", "action": "SELL", "confidence": 0.85}] # Math says it's too expensive
        context = {
            "date": "1995-08-09",
            "news_headlines": ["Netscape IPO signals beginning of Internet era.", "Global connectivity fundamentally altering corporate productivity."],
            "fred_data_simulation": {"GDP_GROWTH": 4.5}
        }
        await self._run_scenario("Technological Paradigm Shift", signals, context, expected_vetoes=["QQQ"], lesson="Let profits run during paradigm shifts. Overvalued does not mean peak.")

    async def scenario_climate_shock(self):
        signals = [
            {"ticker": "SPY", "model": "Momentum", "action": "BUY", "confidence": 0.90},
            {"ticker": "XLE", "model": "MeanReversion", "action": "SELL", "confidence": 0.85} # Math wants to short energy
        ]
        context = {
            "date": "2005-08-29",
            "news_headlines": ["Category 5 Hurricane destroys massive energy infrastructure in Gulf of Mexico.", "Refineries offline, major supply chain disruption."],
            "fred_data_simulation": {"VIX": 25.0}
        }
        # Expected: Veto buying the broad market (SPY) due to shock, and Veto shorting Energy (XLE) because energy prices will spike.
        await self._run_scenario("Exogenous Climate Shock", signals, context, expected_vetoes=["SPY", "XLE"], lesson="Physical infrastructure destruction overrides standard momentum. Energy supply shocks make energy shorts deadly.")

    async def run_all(self):
        logger.info("Starting Shadow Arena V2: Full Typology Test...")
        await self.scenario_war_inflation()
        await self.scenario_swift_victory()
        await self.scenario_credit_bubble()
        await self.scenario_productivity_boom()
        await self.scenario_climate_shock()
        logger.info("=== ALL SCENARIOS COMPLETED ===")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    
    arena = HistoricalTypologyArena()
    asyncio.run(arena.run_all())
