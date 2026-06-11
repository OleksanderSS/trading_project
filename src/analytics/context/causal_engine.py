import logging
from datetime import timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

class CausalEngine:
    """
    🎯 PATTERN-ADJUSTED CAUSAL VECTORS:
    Проектує наслідки подій на майбутнє з урахуванням ринкового режиму.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.causal_library = self.config.get('causal_library', {
            "monetary_tightening": [
                {"feature": "liquidity_crunch", "delay": 5, "impact": -0.4},
                {"feature": "credit_spread_widening", "delay": 15, "impact": -0.6},
                {"feature": "equity_valuation_compression", "delay": 30, "impact": -0.8}
            ],
            "monetary_expansion": [
                {"feature": "liquidity_flood", "delay": 3, "impact": 0.5},
                {"feature": "asset_inflation_momentum", "delay": 10, "impact": 0.7},
                {"feature": "risk_on_sentiment", "delay": 20, "impact": 0.8}
            ]
        })
        logger.info("CausalEngine initialized. Pattern-aware ripple effects enabled.")

    def generate_causal_vectors(self,
                                 events_df: pd.DataFrame,
                                 target_index: pd.DatetimeIndex,
                                 context_velocity: float = 0.0) -> pd.DataFrame:
        """
        Генерує вектори причинності з урахуванням швидкості хаосу.

        Args:
            context_velocity: 0.0 to 1.0. Високе значення прискорює та посилює ефекти.
        """
        implied_features = self._initialize_implied_features(target_index)

        if not events_df.empty:
            for _, row in events_df.iterrows():
                event_data = self._extract_event_data(row)

                # ✅ ELITE: Адаптація до хаосу
                # У хаотичному ринку ефекти наступають швидше (delay стискається)
                speed_factor = 1.0 - (context_velocity * 0.5) # До 50% швидше
                # І ефекти сильніші (impact посилюється)
                power_factor = 1.0 + (context_velocity * 0.8) # До 80% сильніше

                self._apply_event_causal_effects(event_data, implied_features, target_index, speed_factor, power_factor)

        return implied_features

    def _initialize_implied_features(self, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(0.0, index=target_index, columns=self._get_all_causal_features())

    def _apply_event_causal_effects(self, event_data: dict[str, Any], implied_features: pd.DataFrame,
                                   target_index: pd.DatetimeIndex, speed_factor: float, power_factor: float):
        trigger = event_data['trigger']
        if trigger not in self.causal_library:
            return

        for effect in self.causal_library[trigger]:
            feature_name = f"implied_{effect['feature']}"

            # Адаптована затримка та вплив
            adjusted_delay = max(1, int(effect['delay'] * speed_factor))
            adjusted_impact = effect['impact'] * event_data['magnitude'] * power_factor

            impact_time = event_data['event_time'] + timedelta(days=adjusted_delay)
            idx_pos = target_index.get_indexer([impact_time], method='ffill')

            if idx_pos.size > 0 and idx_pos[0] != -1:
                self._apply_decaying_impact(implied_features, feature_name, idx_pos[0], adjusted_impact)

    def _apply_decaying_impact(self, df: pd.DataFrame, column: str, start_idx: int, impact: float):
        current_impact = impact
        # Half-life 20 днів, але у хаосі згасання теж швидше
        decay_factor = 0.95

        for i in range(start_idx, len(df)):
            if column in df.columns:
                df.iloc[i, df.columns.get_loc(column)] += current_impact
                current_impact *= decay_factor
                if abs(current_impact) < 0.01:
                    break

    def _get_all_causal_features(self) -> list[str]:
        features = set()
        for chain in self.causal_library.values():
            for step in chain:
                features.add(f"implied_{step['feature']}")
        return list(features)

    def _extract_event_data(self, row: pd.Series) -> dict[str, Any]:
        return {
            'trigger': row.get('event_type'),
            'magnitude': row.get('magnitude', 1.0),
            'event_time': pd.to_datetime(row.get('timestamp'))
        }
