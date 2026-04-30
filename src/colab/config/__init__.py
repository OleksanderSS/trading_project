"""Configuration module for Colab training"""
from .training_config import TrainingConfig
from .config_loader import RuntimeConfigLoader

__all__ = ['TrainingConfig', 'RuntimeConfigLoader']
