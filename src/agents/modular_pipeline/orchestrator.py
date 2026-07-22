import logging
from typing import Dict, Any, List

from src.core.logging.logger import ProjectLogger
from src.agents.modular_pipeline.base_lens import BaseLens

logger = ProjectLogger.get_logger("ModularAnalystOrchestrator")

class ModularAnalystOrchestrator:
    """
    Оркестратор Модульного Пайплайну.
    Приймає новину, визначає її теги (через класифікатор) та проганяє через
    всі відповідні Лінзи, збираючи фінальний AnalysisPacket.
    """
    def __init__(self):
        self.lenses: List[BaseLens] = []
        
    def register_lens(self, lens: BaseLens):
        """Реєстрація нової лінзи в пайплайні"""
        self.lenses.append(lens)
        logger.info(f"Registered Lens: {lens.lens_name} (Supports: {lens.supported_tags})")
        
    def _get_relevant_lenses(self, affected_tags: List[str]) -> List[BaseLens]:
        """Повертає тільки ті лінзи, які перетинаються з тегами новини"""
        relevant = []
        for lens in self.lenses:
            # Якщо є хоча б один спільний тег або лінза глобальна ('*')
            if '*' in lens.supported_tags or any(tag in lens.supported_tags for tag in affected_tags):
                relevant.append(lens)
        return relevant

    async def analyze(self, source_text: str, affected_tags: List[str]) -> Dict[str, Any]:
        """
        Проганяє новину через усі релевантні лінзи.
        """
        logger.info(f"Starting Modular Analysis for tags: {affected_tags}")
        
        # Початковий стан пакету
        analysis_packet = {
            "source_text": source_text,
            "affected_tags": affected_tags,
            "insights": {},
            "evidence_gaps": [],
            "scenario_nodes": [],
            "lens_status": {},
            "authority": {
                "proposal_only": True,
                "is_evidence": False,
                "may_confirm_hypothesis": False,
                "may_trade": False,
            },
        }
        
        relevant_lenses = self._get_relevant_lenses(affected_tags)
        if not relevant_lenses:
            logger.warning("No relevant lenses found for these tags.")
            return analysis_packet
            
        for lens in relevant_lenses:
            logger.debug(f"Applying Lens: {lens.lens_name}")
            try:
                delta = await lens.analyze(source_text, analysis_packet)
                self._merge_delta(analysis_packet, delta, lens.lens_name)
            except Exception as e:
                logger.error("Lens %s failed (%s)", lens.lens_name, type(e).__name__)
                analysis_packet["lens_status"][lens.lens_name] = {
                    "analysis_status": "failed_closed",
                    "error_type": type(e).__name__,
                    "authority": {
                        "proposal_only": True,
                        "is_evidence": False,
                        "may_confirm_hypothesis": False,
                        "may_trade": False,
                    },
                }
                
        return analysis_packet
        
    def _merge_delta(self, current_state: Dict[str, Any], delta: Dict[str, Any], lens_name: str):
        """
        Зливає дельту (результат лінзи) із загальним пакетом.
        """
        if delta.get("insights"):
            current_state["insights"][lens_name] = delta["insights"]
        if "evidence_gaps" in delta:
            current_state["evidence_gaps"].extend(delta["evidence_gaps"])
        if "scenario_nodes" in delta:
            current_state["scenario_nodes"].extend(delta["scenario_nodes"])
        current_state["lens_status"][lens_name] = {
            "analysis_status": delta.get("analysis_status", "invalid_delta"),
            "authority": delta.get("authority", {}),
        }

def get_default_orchestrator(llm_client=None) -> ModularAnalystOrchestrator:
    """
    Фабричний метод, що повертає готовий оркестратор з усіма 12 лінзами.
    """
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

    orchestrator = ModularAnalystOrchestrator()
    orchestrator.register_lens(TechnologySectorLens(llm_client))
    orchestrator.register_lens(MacroRegimeLens(llm_client))
    orchestrator.register_lens(HealthcareBiotechLens(llm_client))
    orchestrator.register_lens(CrossPlatformInnovationLens(llm_client))
    orchestrator.register_lens(GeopoliticalRiskLens(llm_client))
    orchestrator.register_lens(EnergyCommodityLens(llm_client))
    orchestrator.register_lens(FinancialServicesLens(llm_client))
    orchestrator.register_lens(IndustrialLogisticsLens(llm_client))
    orchestrator.register_lens(MetalsMiningLens(llm_client))
    orchestrator.register_lens(ConsumerRetailLens(llm_client))
    orchestrator.register_lens(RealEstateConstructionLens(llm_client))
    orchestrator.register_lens(AgricultureFoodLens(llm_client))
    
    return orchestrator
