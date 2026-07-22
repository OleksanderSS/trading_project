from src.agents.modular_pipeline.lenses._prompted_lens import PromptedLens


class FinancialServicesLens(PromptedLens):
    LENS_NAME = "FinancialServicesLens"
    SUPPORTED_TAGS = ("banking", "fintech", "insurance")
    PERSPECTIVE = "banking liquidity, credit, insurance, and financial stability"
