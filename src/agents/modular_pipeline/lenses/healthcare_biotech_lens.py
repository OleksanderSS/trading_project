from src.agents.modular_pipeline.lenses._prompted_lens import PromptedLens


class HealthcareBiotechLens(PromptedLens):
    LENS_NAME = "HealthcareBiotechLens"
    SUPPORTED_TAGS = ("healthcare", "biotech", "medical_devices")
    PERSPECTIVE = "clinical evidence, healthcare regulation, and biotechnology"
