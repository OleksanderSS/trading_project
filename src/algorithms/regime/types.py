from enum import Enum


class MarketRegime(Enum):
    """Режими ринку з додатковими станами"""
    TRENDING_UP = 'TRENDING_UP'
    TRENDING_DOWN = 'TRENDING_DOWN'
    RANGING = 'RANGING'
    VOLATILE = 'VOLATILE'
    CRISIS = 'CRISIS'
    MEAN_REVERSION = 'MEAN_REVERSION'
    MOMENTUM = 'MOMENTUM'
    BREAKOUT = 'BREAKOUT'
    NORMAL = 'NORMAL'
