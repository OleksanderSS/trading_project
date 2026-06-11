"""
Max Exposure Monitor - Orchestrator for risk exposure monitoring.
Delegates to specialized analyzers and integrates EliteRiskMetrics.
"""
from datetime import datetime
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.risk.analyzers.concentration_analyzer import ConcentrationAnalyzer
from src.risk.analyzers.correlation_analyzer import CorrelationAnalyzer
from src.risk.elite_risk_metrics import EliteRiskMetrics

logger = ProjectLogger.get_logger("MaxExposureMonitor")

class MaxExposureMonitor:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logger

        # Specialized Components
        self.correlation_analyzer = CorrelationAnalyzer(
            threshold=self.config.get('correlation_threshold', 0.7)
        )
        self.concentration_analyzer = ConcentrationAnalyzer(
            max_asset_weight=self.config.get('max_asset_weight', 0.25)
        )
        self.elite_metrics = EliteRiskMetrics(config_manager=self.config)

        self.logger.info("✅ MaxExposureMonitor initialized with specialized analyzers.")

    async def monitor_exposure(self, portfolio_data: dict[str, Any], market_data: pd.DataFrame) -> dict[str, Any]:
        """
        Monitors portfolio exposure using multiple specialized layers.
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'ok',
            'analysis': {}
        }

        try:
            # 1. Concentration Analysis
            results['analysis']['concentration'] = self.concentration_analyzer.analyze(portfolio_data)

            # 2. Correlation Analysis
            symbols = list(portfolio_data.keys())
            results['analysis']['correlation'] = self.correlation_analyzer.analyze(market_data, symbols)

            # 3. Elite Risk Metrics (VaR/CVaR)
            # Update returns history for elite metrics
            for symbol in symbols:
                if symbol in market_data.columns:
                    self.elite_metrics.update_returns(symbol, market_data[symbol].pct_change().dropna())

            results['analysis']['elite_risk'] = self.elite_metrics.get_risk_report(
                positions={s: p.get('quantity', 0) for s, p in portfolio_data.items()},
                prices={s: p.get('current_price', 0) for s, p in portfolio_data.items()},
                portfolio_value=sum(p.get('current_value', 0) for p in portfolio_data.values())
            )

            # 4. Global Breach Check
            if results['analysis']['concentration'].get('breaches') or \
               results['analysis']['elite_risk'].get('risk_status') == 'high':
                results['status'] = 'warning'

            return results

        except Exception as e:
            self.logger.error(f"❌ Exposure monitoring failed: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}
