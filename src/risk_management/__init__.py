"""
Risk Management Module - Управління фінансовими ризиками

Цей модуль надає комплексні інструменти для:
- Розрахунку Value at Risk (VaR) різними методами
- Conditional VaR (CVaR) та Expected Shortfall
- Stress Testing для сценаріїв кризи
- Оцінки ризику ліквідності
- Управління лімітами ризику
- Комплексної оцінки ризиків портфеля
"""

from .framework import (
    RiskManagementFramework,
    VaRCalculator,
    StressTestingFramework,
    LiquidityRiskAssessor,
    RiskLimitsManager,
    RiskManagementError
)

__all__ = [
    'RiskManagementFramework',
    'VaRCalculator',
    'StressTestingFramework',
    'LiquidityRiskAssessor',
    'RiskLimitsManager',
    'RiskManagementError'
]