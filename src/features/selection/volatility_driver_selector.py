
import pandas as pd
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class VolatilityDriverSelector:
    """
    Selects features that are top drivers of target volatility.
    This is based on the logic from the deprecated context_selector.py.
    """

    def __init__(self, top_n: int = 25):
        self.top_n = top_n
        self.model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
        self.selected_features: List[str] = []

    def select(self, df: pd.DataFrame, auxiliary_pool: List[str], target_col: str) -> List[str]:
        """
        Ranks and selects features from the auxiliary pool based on their
        ability to explain the target's absolute returns (volatility).
        """
        if df.empty or len(df) < 30:
            logger.warning("Insufficient data for volatility driver discovery.")
            return []

        # 1. Target: Realized Volatility (Proxy for regime shifts)
        y_vol = df[target_col].pct_change().abs().fillna(0)

        # 2. Prepare Auxiliary Pool
        valid_aux = [c for c in auxiliary_pool if c in df.columns]
        X_sub = df[valid_aux].ffill().fillna(0)
        
        # Remove low-variance/constant features
        selector_mask = X_sub.std() > 1e-6
        X_sub = X_sub.loc[:, selector_mask]
        
        if X_sub.empty:
            logger.error("Auxiliary pool contains no valid non-constant features.")
            return []

        try:
            # 3. Driver Discovery via Feature Importance
            self.model.fit(X_sub, y_vol)
            importances = pd.Series(self.model.feature_importances_, index=X_sub.columns).sort_values(ascending=False)
            self.selected_features = importances.head(self.top_n).index.tolist()
            
            logger.info(f"VolatilityDriverSelector selected {len(self.selected_features)} features: {self.selected_features}")
            return self.selected_features

        except Exception as e:
            logger.error(f"Volatility driver discovery failed: {e}", exc_info=True)
            return []

