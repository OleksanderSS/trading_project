from src.agents.modular_pipeline.lenses._prompted_lens import PromptedLens


class IndustrialLogisticsLens(PromptedLens):
    LENS_NAME = "IndustrialLogisticsLens"
    SUPPORTED_TAGS = ("logistics", "supply_chain", "transportation")
    PERSPECTIVE = "industrial capacity, logistics, freight, and supply chains"
