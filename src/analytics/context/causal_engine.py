import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class CausalEngine:
    """
    Implements 'Causal Vectors' as described in DEAN's conceptual blueprints.
    Translates a single 'trigger event' into a sequence of expected future 
    market ripple effects (implied features).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
        implied_features = pd.DataFrame(0.0, index=target_index, columns=self._get_all_causal_features())
        if events_df.empty:
            return implied_features

        for _, row in events_df.iterrows():
            trigger = row.get('event_type')
            magnitude = row.get('magnitude', 1.0)
            event_time = pd.to_datetime(row.get('timestamp'))

            if trigger in self.causal_library:
                chain = self.causal_library[trigger]
                for effect in chain:
                    feature_name = f"implied_{effect['feature']}"
                    impact_time = event_time + timedelta(days=effect['delay'])
                    
                    idx_pos = target_index.get_indexer([impact_time], method='ffill')
                    
                    # Check if the indexer returned a valid position
                    if idx_pos.size > 0 and idx_pos[0] != -1:
                        self._apply_decaying_impact(
                            implied_features, 
                            feature_name, 
                            idx_pos[0], 
                            effect['impact'] * magnitude
                        )
        
        return implied_features

    def _apply_decaying_impact(self, 
                               df: pd.DataFrame, 
                               column: str, 
                               start_idx: int, 
                               base_impact: float, 
                               half_life: int = 20):
        """
        Applies a base impact at start_idx and lets it decay exponentially.
        """
        decay_factor = np.exp(-np.log(2) / half_life)
        current_impact = base_impact
        
        for i in range(start_idx, len(df)):
            # Ensure the column exists before trying to access it
            if column in df.columns:
                df.iloc[i, df.columns.get_loc(column)] += current_impact
                current_impact *= decay_factor
                if abs(current_impact) < 0.01: # Stop when impact is negligible
                    break

    def _get_all_causal_features(self) -> List[str]:
        """Returns a unique list of all possible implied feature names."""
        features = set()
        for chain in self.causal_library.values():
            for step in chain:
                features.add(f"implied_{step['feature']}")
        return list(features) if features else []

    def register_custom_chain(self, trigger: str, steps: List[Dict[str, Any]]):
        """Allows dynamic registration of new causal relationships."""
        if not all('feature' in s and 'delay' in s and 'impact' in s for s in steps):
            logger.error(f"Invalid format for custom chain '{trigger}'. Each step must have 'feature', 'delay', and 'impact'.")
            return
        self.causal_library[trigger] = steps
        logger.info(f"Registered new causal chain for: {trigger}")

    def get_explanation(self, trigger_event: str) -> str:
        """Returns a human-readable explanation of the causal chain for XAI."""
        if trigger_event not in self.causal_library:
            return f"No causal chain defined for '{trigger_event}'."
        
        chain = self.causal_library[trigger_event]
        explanation = f"Causal Chain for '{trigger_event}':\n"
        for step in chain:
            explanation += f" -> T+{step['delay']}d: {step['feature']} (Impact: {step['impact']})\n"
        return explanation
