
import pandas as pd
import numpy as np
from typing import List
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
        self.model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=7, 
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42, 
            n_jobs=-1
        )
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
        y_vol = (
            df[target_col]
            .pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            .abs()
            .rename("_target_volatility")
        )

        # 2. Prepare Auxiliary Pool
        valid_aux = [c for c in auxiliary_pool if c in df.columns]
        x_sub = df[valid_aux].ffill().replace([np.inf, -np.inf], np.nan)
        training_data = pd.concat([y_vol, x_sub], axis=1).dropna(how="any")
        if len(training_data) < 30:
            logger.warning("Insufficient complete data for volatility driver discovery.")
            return []

        y_vol = training_data["_target_volatility"]
        x_sub = training_data[valid_aux]
        
        # Remove low-variance/constant features
        selector_mask = x_sub.std() > 1e-6
        x_sub = x_sub.loc[:, selector_mask]
        
        if x_sub.empty:
            logger.error("Auxiliary pool contains no valid non-constant features.")
            return []

        try:
            # 3. Driver Discovery via Feature Importance
            self.model.fit(x_sub, y_vol)
            importances = pd.Series(self.model.feature_importances_, index=x_sub.columns).sort_values(ascending=False)
            self.selected_features = importances.head(self.top_n).index.tolist()
            
            logger.info(f"VolatilityDriverSelector selected {len(self.selected_features)} features: {self.selected_features}")
            return self.selected_features

        except Exception as e:
            logger.error(f"Volatility driver discovery failed: {e}", exc_info=True)
            raise RuntimeError("Volatility driver discovery failed") from e

