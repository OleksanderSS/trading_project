import asyncio
import json
import logging
import sys
from pathlib import Path

# Додаємо корінь проєкту до PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.logging.logger import ProjectLogger
logging.basicConfig(level=logging.INFO)
logger = ProjectLogger.get_logger("TestModularPipeline")

from src.features.entity_linker import EntityLinker
from src.agents.modular_pipeline.orchestrator import ModularAnalystOrchestrator
from src.agents.modular_pipeline.lenses.technology_lens import TechnologySectorLens
from src.agents.modular_pipeline.lenses.macro_regime_lens import MacroRegimeLens
from src.agents.modular_pipeline.lenses.healthcare_biotech_lens import HealthcareBiotechLens
from src.agents.modular_pipeline.lenses.cross_platform_innovation_lens import CrossPlatformInnovationLens
from src.agents.modular_pipeline.lenses.geopolitical_risk_lens import GeopoliticalRiskLens
from src.agents.modular_pipeline.lenses.energy_commodity_lens import EnergyCommodityLens
from src.agents.modular_pipeline.lenses.financial_services_lens import FinancialServicesLens
from src.agents.modular_pipeline.lenses.industrial_logistics_lens import IndustrialLogisticsLens
from src.agents.modular_pipeline.lenses.metals_mining_lens import MetalsMiningLens
from src.agents.modular_pipeline.lenses.consumer_retail_lens import ConsumerRetailLens
from src.agents.modular_pipeline.lenses.real_estate_construction_lens import RealEstateConstructionLens
from src.agents.modular_pipeline.lenses.agriculture_food_lens import AgricultureFoodLens

# Мок класифікатора впливу новин для тестування (в реальності використовує yaml)
class MockNewsImpactClassifier:
    def classify(self, text: str):
        text_lower = text.lower()
        if "fda" in text_lower or "cancer" in text_lower:
            return ["healthcare", "biotech"]
        if "fed" in text_lower or "rate cut" in text_lower or "inflation" in text_lower:
            return ["macro_economic", "market_wide"]
        if "chip" in text_lower or "ban" in text_lower or "export" in text_lower:
            return ["geopolitics", "semiconductors"]
        return ["market_wide"]

async def main():
    logger.info("Initializing Modular Pipeline...")
    
    # 1. Ініціалізація компонентів
    entity_linker = EntityLinker()
    impact_classifier = MockNewsImpactClassifier()
    orchestrator = ModularAnalystOrchestrator()
    
    # 2. Реєстрація лінз
    orchestrator.register_lens(TechnologySectorLens())
    orchestrator.register_lens(MacroRegimeLens())
    orchestrator.register_lens(HealthcareBiotechLens())
    orchestrator.register_lens(CrossPlatformInnovationLens())
    orchestrator.register_lens(GeopoliticalRiskLens())
    orchestrator.register_lens(EnergyCommodityLens())
    orchestrator.register_lens(FinancialServicesLens())
    orchestrator.register_lens(IndustrialLogisticsLens())
    orchestrator.register_lens(MetalsMiningLens())
    orchestrator.register_lens(ConsumerRetailLens())
    orchestrator.register_lens(RealEstateConstructionLens())
    orchestrator.register_lens(AgricultureFoodLens())
    
    # 3. Тестові сценарії
    scenarios = [
        {
            "name": "Nvidia Chip Ban (Geopolitics + Tech)",
            "text": "US imposes new export ban on advanced AI chips to China, affecting Nvidia's H100 sales and TSMC production quotas."
        },
        {
            "name": "Macro Shock (Fed Decision)",
            "text": "Federal Reserve announces unexpected 50bps rate cut amid rising unemployment fears."
        },
        {
            "name": "Cross-Platform AI Biotech (Healthcare + AI)",
            "text": "Nvidia partners with major pharmaceutical company to use AI infrastructure for discovering novel cancer treatments, accelerating Phase 1 trials."
        }
    ]
    
    for idx, scenario in enumerate(scenarios, 1):
        logger.info(f"\n{'='*50}\nExecuting Scenario {idx}: {scenario['name']}\n{'='*50}")
        text = scenario["text"]
        
        # Крок А: Витягування макро-тегів (від класифікатора подій)
        event_tags = impact_classifier.classify(text)
        
        # Крок Б: Витягування специфічних тегів компаній (від EntityLinker)
        entity_tags = entity_linker.extract_tags(text)
        
        # Об'єднання тегів
        all_tags = list(set(event_tags + entity_tags))
        logger.info(f"Unified Affected Tags: {all_tags}")
        
        # Крок В: Запуск Оркестратора
        analysis_packet = await orchestrator.analyze(text, all_tags)
        
        # Вивід результату
        logger.info("FINAL ANALYSIS PACKET:")
        print(json.dumps(analysis_packet, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
