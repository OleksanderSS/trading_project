
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TradingStyle(Enum):
    SCALPING = "scalping"
    DAY_TRADING = "day_trading"
    SWING_TRADING = "swing_trading"
    POSITION_TRADING = "position_trading"
    INVESTING = "investing"

class MarketFocus(Enum):
    TECH = "tech"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    ENERGY = "energy"
    CRYPTO = "crypto"
    COMMODITIES = "commodities"
    ALL = "all"

@dataclass
class TradingConfig:
    style: TradingStyle
    market_focus: MarketFocus
    timeframes: List[str]
    tickers: List[str]
    max_tickers: int = 25
    risk_level: str = "medium"

@dataclass
class SectorConfig:
    name: str
    tickers: List[str]
    volatility_score: float
    profit_potential: float
    risk_level: str
    correlation_with_market: float
    recommended_position_size: float
    optimal_timeframes: List[str]

class AssetUniverseManager:
    def __init__(self, config: Dict):
        self.config = config.get('asset_universe', {})
        self.sectors = self._create_enhanced_sectors()
        self.volatility_ranking = self._create_volatility_ranking()
        self.profitability_ranking = self._create_profitability_ranking()
        self.presets = self._create_preset_configs()

    def _create_enhanced_sectors(self) -> Dict[str, SectorConfig]:
        sectors_data = self.config.get('sectors', {})
        return {
            sector_name: SectorConfig(**data)
            for sector_name, data in sectors_data.items()
        }

    def _create_volatility_ranking(self) -> List[str]:
        return [s[0] for s in sorted(self.sectors.items(), key=lambda x: x[1].volatility_score, reverse=True)]

    def _create_profitability_ranking(self) -> List[str]:
        return [s[0] for s in sorted(self.sectors.items(), key=lambda x: x[1].profit_potential, reverse=True)]

    def _map_market_focus_to_sectors(self, market_focus: MarketFocus) -> List[str]:
        mapping = self.config.get('market_focus_mapping', {})
        return mapping.get(market_focus.value, [])

    def get_tickers_by_market_focus(self, market_focus: MarketFocus, max_tickers: Optional[int] = None) -> List[str]:
        sector_names = self._map_market_focus_to_sectors(market_focus)
        all_tickers = []
        for sector_name in sector_names:
            if sector_name in self.sectors:
                all_tickers.extend(self.sectors[sector_name].tickers)

        if market_focus == MarketFocus.ALL:
             etf_sector = self.config.get('etf_strategic_sector_name', 'etf_strategic')
             if etf_sector in self.sectors:
                all_tickers.extend(self.sectors[etf_sector].tickers)

        unique_tickers = sorted(list(set(all_tickers)))
        return unique_tickers[:max_tickers] if max_tickers else unique_tickers

    def _create_preset_configs(self) -> Dict[str, TradingConfig]:
        presets_data = self.config.get('presets', {})
        return {
            preset_name: self.create_custom_config(
                style=TradingStyle(preset_data['style']),
                market_focus=MarketFocus(preset_data['market_focus']),
                max_tickers=preset_data.get('max_tickers'),
                risk_level=preset_data.get('risk_level')
            )
            for preset_name, preset_data in presets_data.items()
        }

    def get_timeframes_for_style(self, style: TradingStyle) -> List[str]:
        available_timeframes = self.config.get('available_timeframes', {})
        return [
            tf for tf, config in available_timeframes.items()
            if style.value in config.get('style', []) and config.get('recommended')
        ]

    def create_custom_config(self, style: TradingStyle, market_focus: MarketFocus, custom_tickers: Optional[List[str]] = None, custom_timeframes: Optional[List[str]] = None, max_tickers: int = 25, risk_level: str = "medium") -> TradingConfig:
        timeframes = custom_timeframes if custom_timeframes else self.get_timeframes_for_style(style)
        tickers = custom_tickers if custom_tickers else self.get_tickers_by_market_focus(market_focus, max_tickers)

        return TradingConfig(
            style=style,
            market_focus=market_focus,
            timeframes=timeframes,
            tickers=sorted(list(set(tickers))),
            max_tickers=max_tickers,
            risk_level=risk_level
        )

    def get_preset(self, preset_name: str) -> Optional[TradingConfig]:
        return self.presets.get(preset_name)

    def list_presets(self) -> Dict[str, str]:
        return self.config.get('preset_descriptions', {})

    def get_tickers_by_strategy(self, strategy: str, limit: Optional[int] = None) -> List[str]:
        strategy_map = self.config.get('strategy_map', {})
        sector_names = strategy_map.get(strategy, [])
        tickers = []
        for name in sector_names:
            if name in self.sectors:
                tickers.extend(self.sectors[name].tickers)
        return sorted(list(set(tickers)))[:limit] if limit else sorted(list(set(tickers)))

    def get_sector_analysis(self) -> pd.DataFrame:
        data = [
            {
                'sector': name, 'name': s.name, 'tickers_count': len(s.tickers),
                'volatility_score': s.volatility_score, 'profit_potential': s.profit_potential,
                'risk_level': s.risk_level, 'correlation': s.correlation_with_market,
                'position_size': s.recommended_position_size, 'optimal_timeframes': ', '.join(s.optimal_timeframes)
            }
            for name, s in self.sectors.items()
        ]
        return pd.DataFrame(data).sort_values('volatility_score', ascending=False).set_index('sector')
