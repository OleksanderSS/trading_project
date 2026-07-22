from src.agents.modular_pipeline.lenses._prompted_lens import PromptedLens


class MacroRegimeLens(PromptedLens):
    LENS_NAME = "MacroRegimeLens"
    SUPPORTED_TAGS = ("market_wide", "macro_economic", "finance", "banking")
    PERSPECTIVE = "rates, inflation, growth, liquidity, and macro regimes"
