"""
Risk Manager
Comprehensive risk management with kill-switch, exposure limits, and volatility scaling.
"""
import logging

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("RiskManager")


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Position:
    """Position data class."""
    ticker: str
    value: float  # Position value in $
    quantity: float
    entry_price: float
    current_price: float
    sector: str = "unknown"
    
    @property
    def pnl(self) -> float:
        """Calculate P&L."""
        return (self.current_price - self.entry_price) * self.quantity
    
    @property
    def pnl_pct(self) -> float:
        """Calculate P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price


class RiskManager:
    """
    Comprehensive risk management system.
    
    Audit Points:
    - Kill-switch logic
    - Max exposure limits
    - Sector concentration limits
    - Volatility scaling
    """
    
    def __init__(
        self,
        portfolio_value: float,
        max_position_exposure: float = 0.25,  # 25% per position
        max_sector_exposure: float = 0.40,    # 40% per sector
        max_total_exposure: float = 1.0,      # 100% total
        kill_switch_drawdown: float = 0.10,   # 10% drawdown triggers kill-switch
        target_volatility: float = 0.02,      # 2% target volatility
        sector_map: Optional[Dict[str, str]] = None
    ):
        """
        Initialize risk manager.
        
        Args:
            portfolio_value: Total portfolio value
            max_position_exposure: Max exposure per position (0-1)
            max_sector_exposure: Max exposure per sector (0-1)
            max_total_exposure: Max total exposure (0-1)
            kill_switch_drawdown: Drawdown threshold for kill-switch (0-1)
            target_volatility: Target volatility for position sizing
            sector_map: Mapping of ticker -> sector
        """
        self.portfolio_value = portfolio_value
        self.max_position_exposure = max_position_exposure
        self.max_sector_exposure = max_sector_exposure
        self.max_total_exposure = max_total_exposure
        self.kill_switch_drawdown = kill_switch_drawdown
        self.target_volatility = target_volatility
        self.sector_map = sector_map or {}
        
        # State
        self.kill_switch_triggered = False
        self.kill_switch_reason = None
        self.positions: List[Position] = []
        self.peak_portfolio_value = portfolio_value
        
        # Metrics
        self.metrics = {
            'checks_performed': 0,
            'violations_detected': 0,
            'kill_switch_triggers': 0,
            'exposure_warnings': 0
        }
        
        logger.info(f"RiskManager initialized:")
        logger.info(f"  Portfolio value: ${portfolio_value:,.2f}")
        logger.info(f"  Max position exposure: {max_position_exposure:.1%}")
        logger.info(f"  Max sector exposure: {max_sector_exposure:.1%}")
        logger.info(f"  Kill-switch drawdown: {kill_switch_drawdown:.1%}")
    
    def check_kill_switch(self, current_portfolio_value: float) -> bool:
        """
        Check if kill-switch should be triggered.
        
        Args:
            current_portfolio_value: Current portfolio value
            
        Returns:
            True if kill-switch triggered
        """
        self.metrics['checks_performed'] += 1
        
        # Update peak
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value
        
        # Calculate drawdown
        drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
        
        # Check threshold
        if drawdown >= self.kill_switch_drawdown:
            if not self.kill_switch_triggered:
                self.kill_switch_triggered = True
                self.kill_switch_reason = f"Drawdown {drawdown:.1%} exceeded threshold {self.kill_switch_drawdown:.1%}"
                self.metrics['kill_switch_triggers'] += 1
                
                logger.error(f"🚨 KILL-SWITCH TRIGGERED: {self.kill_switch_reason}")
                logger.error(f"   Peak: ${self.peak_portfolio_value:,.2f}")
                logger.error(f"   Current: ${current_portfolio_value:,.2f}")
                logger.error(f"   Drawdown: {drawdown:.1%}")
                
                return True
        
        return self.kill_switch_triggered
    
    def check_position_exposure(self, position: Position) -> Tuple[bool, str]:
        """
        Check if position exposure is within limits.
        
        Args:
            position: Position to check
            
        Returns:
            (is_valid, message)
        """
        exposure = abs(position.value) / self.portfolio_value
        
        if exposure > self.max_position_exposure:
            self.metrics['violations_detected'] += 1
            self.metrics['exposure_warnings'] += 1
            
            message = (
                f"Position exposure {exposure:.1%} exceeds limit {self.max_position_exposure:.1%} "
                f"for {position.ticker}"
            )
            logger.error(f"❌ {message}")
            return False, message
        
        return True, "OK"
    
    def check_sector_exposure(self, positions: List[Position]) -> Tuple[bool, Dict[str, float]]:
        """
        Check sector exposure limits.
        
        Args:
            positions: List of positions
            
        Returns:
            (is_valid, sector_exposures)
        """
        sector_exposure = {}
        
        for pos in positions:
            sector = pos.sector
            if sector not in sector_exposure:
                sector_exposure[sector] = 0.0
            sector_exposure[sector] += abs(pos.value)
        
        # Check limits
        violations = []
        for sector, exposure_value in sector_exposure.items():
            exposure_pct = exposure_value / self.portfolio_value
            
            if exposure_pct > self.max_sector_exposure:
                self.metrics['violations_detected'] += 1
                self.metrics['exposure_warnings'] += 1
                
                violations.append(
                    f"Sector '{sector}' exposure {exposure_pct:.1%} exceeds limit {self.max_sector_exposure:.1%}"
                )
                logger.error(f"❌ {violations[-1]}")
        
        is_valid = len(violations) == 0
        
        if is_valid:
            logger.info(f"✅ Sector exposure check passed")
            for sector, exposure_value in sector_exposure.items():
                exposure_pct = exposure_value / self.portfolio_value
                logger.info(f"   {sector}: {exposure_pct:.1%}")
        
        return is_valid, sector_exposure
    
    def check_total_exposure(self, positions: List[Position]) -> Tuple[bool, float]:
        """
        Check total exposure limit.
        
        Args:
            positions: List of positions
            
        Returns:
            (is_valid, total_exposure)
        """
        total_exposure_value = sum(abs(pos.value) for pos in positions)
        total_exposure_pct = total_exposure_value / self.portfolio_value
        
        if total_exposure_pct > self.max_total_exposure:
            self.metrics['violations_detected'] += 1
            self.metrics['exposure_warnings'] += 1
            
            logger.error(
                f"❌ Total exposure {total_exposure_pct:.1%} exceeds limit {self.max_total_exposure:.1%}"
            )
            return False, total_exposure_pct
        
        logger.info(f"✅ Total exposure: {total_exposure_pct:.1%}")
        return True, total_exposure_pct
    
    def calculate_position_size(
        self,
        base_size: float,
        volatility: float,
        confidence: float = 1.0
    ) -> float:
        """
        Calculate position size with volatility scaling.
        
        Args:
            base_size: Base position size ($)
            volatility: Asset volatility (std dev of returns)
            confidence: Confidence level (0-1)
            
        Returns:
            Adjusted position size ($)
        """
        if volatility <= 0:
            logger.warning(f"⚠️ Invalid volatility {volatility}, using base size")
            return base_size
        
        # Volatility adjustment
        vol_adjustment = self.target_volatility / volatility
        
        # Confidence adjustment
        confidence_adjustment = confidence
        
        # Combined adjustment
        adjusted_size = base_size * vol_adjustment * confidence_adjustment
        
        # Cap at max position exposure
        max_size = self.portfolio_value * self.max_position_exposure
        adjusted_size = min(adjusted_size, max_size)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Position sizing: base=${base_size:,.0f}, vol={volatility:.3f}, "
                f"conf={confidence:.2f} → ${adjusted_size:,.0f}"
            )
        
        return adjusted_size
    
    def validate_trade(
        self,
        ticker: str,
        size: float,
        sector: str = "unknown"
    ) -> Tuple[bool, str]:
        """
        Validate if trade is allowed.
        
        Args:
            ticker: Ticker symbol
            size: Trade size ($)
            sector: Sector
            
        Returns:
            (is_valid, message)
        """
        # Check kill-switch
        if self.kill_switch_triggered:
            return False, f"Kill-switch triggered: {self.kill_switch_reason}"
        
        # Create hypothetical position
        hypothetical_pos = Position(
            ticker=ticker,
            value=size,
            quantity=0,
            entry_price=0,
            current_price=0,
            sector=sector
        )
        
        # Check position exposure
        is_valid, message = self.check_position_exposure(hypothetical_pos)
        if not is_valid:
            return False, message
        
        # Check sector exposure with new position
        hypothetical_positions = self.positions + [hypothetical_pos]
        is_valid, _ = self.check_sector_exposure(hypothetical_positions)
        if not is_valid:
            return False, "Sector exposure limit exceeded"
        
        # Check total exposure
        is_valid, _ = self.check_total_exposure(hypothetical_positions)
        if not is_valid:
            return False, "Total exposure limit exceeded"
        
        return True, "Trade validated"
    
    def add_position(self, position: Position):
        """Add position to portfolio."""
        self.positions.append(position)
        logger.info(f"Position added: {position.ticker} ${position.value:,.2f}")
    
    def remove_position(self, ticker: str):
        """Remove position from portfolio."""
        self.positions = [p for p in self.positions if p.ticker != ticker]
        logger.info(f"Position removed: {ticker}")
    
    def clear_all_positions(self):
        """Clear all positions (emergency exit)."""
        logger.warning("🚨 Clearing all positions (emergency exit)")
        self.positions = []
    
    def reset_kill_switch(self):
        """Reset kill-switch (manual override)."""
        logger.warning("⚠️ Kill-switch manually reset")
        self.kill_switch_triggered = False
        self.kill_switch_reason = None
    
    def get_risk_level(self) -> RiskLevel:
        """Get current risk level."""
        if self.kill_switch_triggered:
            return RiskLevel.CRITICAL
        
        # Calculate total exposure
        total_exposure = sum(abs(p.value) for p in self.positions) / self.portfolio_value
        
        if total_exposure > 0.8:
            return RiskLevel.HIGH
        elif total_exposure > 0.5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def get_metrics(self) -> Dict:
        """Get risk manager metrics."""
        return {
            **self.metrics,
            'kill_switch_triggered': self.kill_switch_triggered,
            'kill_switch_reason': self.kill_switch_reason,
            'positions_count': len(self.positions),
            'risk_level': self.get_risk_level().value
        }
    
    def generate_report(self) -> Dict:
        """Generate risk report."""
        total_exposure = sum(abs(p.value) for p in self.positions)
        total_pnl = sum(p.pnl for p in self.positions)
        
        sector_exposure = {}
        for pos in self.positions:
            if pos.sector not in sector_exposure:
                sector_exposure[pos.sector] = 0.0
            sector_exposure[pos.sector] += abs(pos.value)
        
        return {
            'portfolio_value': self.portfolio_value,
            'peak_portfolio_value': self.peak_portfolio_value,
            'total_exposure': total_exposure,
            'total_exposure_pct': total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl / self.portfolio_value if self.portfolio_value > 0 else 0,
            'positions_count': len(self.positions),
            'sector_exposure': sector_exposure,
            'risk_level': self.get_risk_level().value,
            'kill_switch_triggered': self.kill_switch_triggered,
            'metrics': self.get_metrics()
        }
