from typing import Any


class TransactionCostModel:
    """Моделювання транзакційних витрат"""

    def __init__(self, config: (dict[str, Any] | None)=None):
        self.config = config or {}
        self.commission_pct = self.config.get('commission_pct', 0.001)
        self.spread_bps = self.config.get('spread_bps', 5)
        self.market_impact_coefficient = self.config.get(
            'market_impact_coefficient', 0.1)
        self.slippage_pct = self.config.get('slippage_pct', 0.001)

    def calculate_execution_costs(self, trade_value: float, daily_volume:
        float=1000000.0) ->float:
        """
        Розраховує повну вартість виконання: комісії + спред + вплив на ринок + прослизання.
        """
        commission = abs(trade_value) * self.commission_pct
        spread_cost = abs(trade_value) * (self.spread_bps / 10000.0) / 2.0
        participation_rate = abs(trade_value) / max(daily_volume, 1.0)
        market_impact = abs(trade_value
            ) * participation_rate * self.market_impact_coefficient
        slippage = abs(trade_value) * self.slippage_pct
        return commission + spread_cost + market_impact + slippage
