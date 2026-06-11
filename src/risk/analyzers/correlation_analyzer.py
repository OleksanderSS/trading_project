"""
Specialized analyzer for asset correlations within a portfolio.
Uses advanced metrics to detect hidden dependencies and systemic risk.
"""
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("CorrelationAnalyzer")

class CorrelationAnalyzer:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def analyze(self, market_data: pd.DataFrame, symbols: list[str]) -> dict[str, Any]:
        """Analyzes correlation matrix and identifies high-risk groups."""
        if len(symbols) < 2:
            return {"status": "insufficient_assets", "groups": []}

        # Get returns for all symbols
        returns_df = pd.DataFrame()
        for symbol in symbols:
            if symbol in market_data.columns:
                returns_df[symbol] = market_data[symbol].pct_change()
            elif 'close' in market_data and symbol in market_data['close'].columns:
                returns_df[symbol] = market_data['close'][symbol].pct_change()

        if returns_df.empty:
            return {"status": "no_data", "groups": []}

        corr_matrix = returns_df.corr().abs()
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.threshold:
                    high_corr_pairs.append({
                        'pair': (corr_matrix.columns[i], corr_matrix.columns[j]),
                        'correlation': float(corr_matrix.iloc[i, j])
                    })

        return {
            'status': 'success',
            'average_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),
            'high_correlation_pairs': high_corr_pairs,
            'matrix': corr_matrix.to_dict()
        }
