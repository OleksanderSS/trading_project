import logging
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CausalEngine:
    """
    Implements 'Causal Vectors' as described in DEAN's conceptual blueprints.
    Translates a single 'trigger event' into a sequence of expected future
    market ripple effects (implied features).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initializes the CausalEngine with a predefined library of causal chains.
        """
        self.config = config or {}
        # Causal chains: Trigger -> List of (Consequence, Delay_Days, Impact_Weight)
        self.causal_library = self.config.get('causal_library', {
            "monetary_tightening": [
                {"feature": "liquidity_crunch", "delay": 5, "impact": -0.4},
                {"feature": "credit_spread_widening", "delay": 15, "impact": -0.6},
                {"feature": "equity_valuation_compression", "delay": 30, "impact": -0.8},
                {"feature": "recession_probability_spike", "delay": 90, "impact": -0.5}
            ],
            "monetary_expansion": [
                {"feature": "liquidity_flood", "delay": 3, "impact": 0.5},
                {"feature": "asset_inflation_momentum", "delay": 10, "impact": 0.7},
                {"feature": "risk_on_sentiment", "delay": 20, "impact": 0.8},
                {"feature": "speculative_bubble_formation", "delay": 60, "impact": 0.4}
            ]
        })
        logger.info(f"CausalEngine initialized with {len(self.causal_library)} chains.")

    def generate_causal_vectors(self,
                                 events_df: pd.DataFrame,
                                 target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Processes a DataFrame of detected events and projects their causal ripple
        effects onto a target timeline.
        """
        implied_features = self._initialize_implied_features(target_index)

        if not events_df.empty:
            self._process_events(events_df, implied_features, target_index)

        return implied_features

    def _initialize_implied_features(self, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Initialize implied features DataFrame."""
        return pd.DataFrame(0.0, index=target_index, columns=self._get_all_causal_features())

    def _process_events(self, events_df: pd.DataFrame, implied_features: pd.DataFrame, target_index: pd.DatetimeIndex):
        """Process all events and apply their causal effects."""
        for _, row in events_df.iterrows():
            event_data = self._extract_event_data(row)
            self._apply_event_causal_effects(event_data, implied_features, target_index)

    def _extract_event_data(self, row: pd.Series) -> dict[str, Any]:
        """Extract event data from row."""
        return {
            'trigger': row.get('event_type'),
            'magnitude': row.get('magnitude', 1.0),
            'event_time': pd.to_datetime(row.get('timestamp'))
        }

    def _apply_event_causal_effects(self, event_data: dict[str, Any], implied_features: pd.DataFrame, target_index: pd.DatetimeIndex):
        """Apply causal effects for a single event."""
        trigger = event_data['trigger']

        if trigger not in self.causal_library:
            return

        chain = self.causal_library[trigger]
        for effect in chain:
            self._apply_single_effect(effect, event_data, implied_features, target_index)

    def _apply_single_effect(self, effect: dict[str, Any], event_data: dict[str, Any], implied_features: pd.DataFrame, target_index: pd.DatetimeIndex):
        """Apply a single causal effect."""
        feature_name = f"implied_{effect['feature']}"
        impact_time = self._calculate_impact_time(event_data['event_time'], effect['delay'])

        idx_pos = target_index.get_indexer([impact_time], method='ffill')

        if self._is_valid_index_position(idx_pos):
            impact_params = self._create_impact_params(effect['impact'], event_data['magnitude'])
            self._apply_decaying_impact(implied_features, feature_name, idx_pos[0], impact_params)

    def _calculate_impact_time(self, event_time, delay_days: int):
        """Calculate the time when the impact should be applied."""
        return event_time + timedelta(days=delay_days)

    def _create_impact_params(self, impact: float, magnitude: float) -> dict[str, Any]:
        """Create impact parameters for decaying effect."""
        return {
            'base_impact': impact * magnitude,
            'half_life': 20
        }

    def _is_valid_index_position(self, idx_pos: np.ndarray) -> bool:
        """Check if indexer returned a valid position."""
        return idx_pos.size > 0 and idx_pos[0] != -1

    def _apply_decaying_impact(self,
                               df: pd.DataFrame,
                               column: str,
                               start_idx: int,
                               impact_params: dict[str, Any]):
        """
        Applies a base impact at start_idx and lets it decay exponentially.
        """
        decay_config = self._create_decay_config(impact_params)

        self._apply_impact_decay_loop(df, column, start_idx, decay_config)

    def _create_decay_config(self, impact_params: dict[str, Any]) -> dict[str, Any]:
        """Create decay configuration from impact parameters."""
        base_impact = impact_params['base_impact']
        half_life = impact_params.get('half_life', 20)

        return {
            'current_impact': base_impact,
            'decay_factor': np.exp(-np.log(2) / half_life),
            'min_impact': 0.01
        }

    def _apply_impact_decay_loop(self, df: pd.DataFrame, column: str, start_idx: int, decay_config: dict[str, Any]):
        """Apply decaying impact in a loop."""
        current_impact = decay_config['current_impact']
        decay_factor = decay_config['decay_factor']
        min_impact = decay_config['min_impact']

        for i in range(start_idx, len(df)):
            if self._should_apply_impact(df, column, current_impact):
                df.iloc[i, df.columns.get_loc(column)] += current_impact
                current_impact *= decay_factor

                if abs(current_impact) < min_impact:
                    break

    def _should_apply_impact(self, df: pd.DataFrame, column: str, current_impact: float) -> bool:
        """Check if impact should be applied."""
        return column in df.columns and abs(current_impact) >= 0.01

    def _get_all_causal_features(self) -> list[str]:
        """Returns a unique list of all possible implied feature names."""
        features = set()
        for chain in self.causal_library.values():
            for step in chain:
                features.add(f"implied_{step['feature']}")
        return list(features) if features else []

    def register_custom_chain(self, trigger: str, steps: list[dict[str, Any]]):
        """Allows dynamic registration of new causal relationships."""
        if self._validate_and_register_chain(trigger, steps):
            logger.info(f"Registered new causal chain for: {trigger}")

    def _validate_and_register_chain(self, trigger: str, steps: list[dict[str, Any]]) -> bool:
        """Validate and register custom chain."""
        if not self._validate_custom_chain_steps(steps):
            self._log_invalid_chain_error(trigger)
            return False

        self.causal_library[trigger] = steps
        return True

    def _validate_custom_chain_steps(self, steps: list[dict[str, Any]]) -> bool:
        """Validate that all steps have required fields."""
        required_fields = ['feature', 'delay', 'impact']
        return all(self._has_required_fields(step, required_fields) for step in steps)

    def _has_required_fields(self, step: dict[str, Any], required_fields: list[str]) -> bool:
        """Check if step has all required fields."""
        return all(field in step for field in required_fields)

    def _log_invalid_chain_error(self, trigger: str):
        """Log error for invalid custom chain."""
        logger.error(f"Invalid format for custom chain '{trigger}'. Each step must have 'feature', 'delay', and 'impact'.")

    def get_explanation(self, trigger_event: str) -> str:
        """Returns a human-readable explanation of the causal chain for XAI."""
        if trigger_event not in self.causal_library:
            return f"No causal chain defined for '{trigger_event}'."

        chain = self.causal_library[trigger_event]
        explanation = f"Causal Chain for '{trigger_event}':\n"
        for step in chain:
            explanation += f" -> T+{step['delay']}d: {step['feature']} (Impact: {step['impact']})\n"
        return explanation
