from src.agents.modular_pipeline.lenses._prompted_lens import PromptedLens


class GeopoliticalRiskLens(PromptedLens):
    LENS_NAME = "GeopoliticalRiskLens"
    SUPPORTED_TAGS = ("geopolitics", "global_trade", "sanctions", "tariffs")
    PERSPECTIVE = "geopolitical risk, trade restrictions, sanctions, and retaliation"
