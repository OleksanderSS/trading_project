"""
Elite VaR & Risk Metrics Engine
- Historical Simulation VaR
- GARCH dynamic volatility
- Cornish-Fisher ES (Expected Shortfall)
- Tail risk metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from scipy import stats

class EliteRiskMetrics:
    """Advanced risk computation"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.returns_history = {}  # {ticker: pd.Series}
        self.volatility_estimates = {}
    
    def compute_historical_simulation_var(self,
                                         ticker: str,
                                         confidence_level: float = 0.95,
                                         lookback_days: int = 252) -> float:
        """
        Historical Simulation VaR
        
        The simplest (but effective) - take historical returns,
        sort, take the percentile
        
        Args:
            confidence_level: 0.95 = 95% VaR = 5% tail loss
            lookback_days: How far back to look
        
        Returns:
            VaR (as percentage, e.g., 0.03 = 3% loss)
        """
        if ticker not in self.returns_history:
            return 0.05  # Default 5%
        
        returns = self.returns_history[ticker]
        if len(returns) < 30:
            return 0.05
        
        # Take last N days
        recent_returns = returns[-lookback_days:] if len(returns) > lookback_days else returns
        
        # VaR = negative of the percentile (loss is negative return)
        var_percentile = (1 - confidence_level) * 100
        var = -np.percentile(recent_returns, var_percentile)
        
        return max(0.001, var)
    
    def compute_cornish_fisher_var(self,
                                  ticker: str,
                                  confidence_level: float = 0.95,
                                  lookback_days: int = 252) -> Tuple[float, float]:
        """
        Cornish-Fisher VaR (accounts for skewness and kurtosis)
        
        More accurate than normal VaR, especially for tails
        
        Returns:
            (VaR, CVaR)  - Value at Risk & Conditional VaR (Expected Shortfall)
        """
        if ticker not in self.returns_history:
            return 0.05, 0.08
        
        returns = self.returns_history[ticker]
        if len(returns) < 30:
            return 0.05, 0.08
        
        # Take recent
        recent_returns = returns[-lookback_days:] if len(returns) > lookback_days else returns
        
        mean = recent_returns.mean()
        std = recent_returns.std()
        skewness = stats.skew(recent_returns)
        kurtosis = stats.kurtosis(recent_returns)
        
        # Z-score for confidence level
        z = stats.norm.ppf(1 - confidence_level)
        
        # Cornish-Fisher modification
        z_cf = z + (z**2 - 1) * skewness / 6 + (z**3 - 3*z) * kurtosis / 24 - (2*z**3 - 5*z) * skewness**2 / 36
        
        # VaR as loss (positive number)
        var_cf = -(mean + z_cf * std)
        
        # CVaR = expected loss beyond VaR
        # Approximation: CVaR ≈ VaR * (1 + kurtosis/4)
        cvar_cf = var_cf * (1 + abs(kurtosis) / 4)
        
        return max(0.001, var_cf), max(0.001, cvar_cf)
    
    def compute_garch_var(self,
                         ticker: str,
                         confidence_level: float = 0.95) -> float:
        """
        GARCH(1,1) dynamic volatility VaR
        
        GARCH captures volatility clustering
        σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
        
        Returns:
            Predicted VaR for tomorrow
        """
        if ticker not in self.returns_history:
            return 0.05
        
        returns = self.returns_history[ticker]
        if len(returns) < 50:
            return 0.05
        
        try:
            from arch import arch_model
        except ImportError:
            self.logger.warning("ARCH package not installed. Falling back to simple VaR.")
            return self.compute_historical_simulation_var(ticker, confidence_level)
        
        try:
            # Fit GARCH(1,1)
            model = arch_model(returns[-252:] * 100, vol='Garch', p=1, q=1)
            res = model.fit(disp='off')
            
            # Get conditional volatility forecast for next day
            forecast = res.forecast(horizon=1)
            tomorrow_vol = float(forecast.variance.values[-1, 0] ** 0.5) / 100
            
            # Convert to VaR
            z = stats.norm.ppf(1 - confidence_level)
            var = -tomorrow_vol * z
            
            self.logger.debug(f"GARCH VaR for {ticker}: {var:.3%}")
            return max(0.001, var)
        
        except Exception as e:
            self.logger.warning(f"GARCH failed for {ticker}: {e}. Using historical VaR.")
            return self.compute_historical_simulation_var(ticker, confidence_level)
    
    def compute_comprehensive_risk_metrics(self,
                                          ticker: str,
                                          position_size: int,
                                          entry_price: float,
                                          portfolio_value: float) -> Dict:
        """
        Compute comprehensive set of risk metrics
        
        Returns:
            Dict with VaR, CVaR, portfolio impact, etc.
        """
        position_value = position_size * entry_price
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Compute different VaR approaches
        hs_var = self.compute_historical_simulation_var(ticker, 0.95)
        cf_var, cf_cvar = self.compute_cornish_fisher_var(ticker, 0.95)
        garch_var = self.compute_garch_var(ticker, 0.95)
        
        # Average ensemble
        ensemble_var = np.mean([hs_var, cf_var, garch_var])
        
        # Dollar terms
        var_dollars = position_value * ensemble_var
        cvar_dollars = position_value * cf_cvar
        
        # Portfolio impact
        portfolio_var = portfolio_value * ensemble_var
        
        return {
            'ticker': ticker,
            'position_size': position_size,
            'position_value': position_value,
            'position_pct': position_pct,
            'var_95_pct': {
                'historical_simulation': hs_var,
                'cornish_fisher': cf_var,
                'garch': garch_var,
                'ensemble': ensemble_var
            },
            'cvar_95_pct': cf_cvar,
            'var_95_dollars': var_dollars,
            'cvar_95_dollars': cvar_dollars,
            'portfolio_var_impact': portfolio_var,
            'status': 'ok' if var_dollars < portfolio_value * 0.05 else 'high_risk'
        }
    
    def update_returns(self, ticker: str, returns: pd.Series):
        """Update historical data"""
        self.returns_history[ticker] = returns
    
    def get_risk_report(self, positions: Dict[str, int], prices: Dict[str, float], portfolio_value: float) -> Dict:
        """
        Generate risk report for the portfolio
        """
        total_position_var = 0.0
        total_position_cvar = 0.0
        position_risks = []
        
        for ticker, qty in positions.items():
            if qty == 0 or ticker not in prices:
                continue
            
            risk_metrics = self.compute_comprehensive_risk_metrics(
                ticker, qty, prices[ticker], portfolio_value
            )
            
            total_position_var += risk_metrics['var_95_dollars']
            total_position_cvar += risk_metrics['cvar_95_dollars']
            position_risks.append(risk_metrics)
        
        # Portfolio level
        portfolio_var_pct = total_position_var / portfolio_value if portfolio_value > 0 else 0
        portfolio_cvar_pct = total_position_cvar / portfolio_value if portfolio_value > 0 else 0
        
        return {
            'portfolio_var_95_dollars': total_position_var,
            'portfolio_var_95_pct': portfolio_var_pct,
            'portfolio_cvar_95_dollars': total_position_cvar,
            'portfolio_cvar_95_pct': portfolio_cvar_pct,
            'position_risks': position_risks,
            'timestamp': datetime.now().isoformat(),
            'risk_status': self._determine_risk_status(portfolio_var_pct)
        }


def _determine_risk_status(self, portfolio_var_pct: float) -> str:
        """Determine risk status based on portfolio variance percentage."""
        if portfolio_var_pct < 0.05:
            return 'ok'
        if portfolio_var_pct < 0.10:
            return 'elevated'
        return 'high'

# Import datetime for timestamp
from datetime import datetime
