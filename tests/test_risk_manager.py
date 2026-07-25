"""
Tests for RiskManager - Kill-Switch and Exposure Limits
"""

import pytest
import numpy as np
from src.archive.risk.risk_manager import RiskManager, Position, RiskLevel


class TestKillSwitch:
    """Test kill-switch logic."""
    
    def test_kill_switch_not_triggered_on_small_drawdown(self):
        """Kill-switch should NOT trigger on small drawdown."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            kill_switch_drawdown=0.10  # 10% threshold
        )
        
        # 5% drawdown - should NOT trigger
        current_value = 95000
        triggered = risk_manager.check_kill_switch(current_value)
        
        assert not triggered
        assert not risk_manager.kill_switch_triggered
        assert risk_manager.kill_switch_reason is None
    
    def test_kill_switch_triggered_on_large_drawdown(self):
        """Kill-switch SHOULD trigger on large drawdown."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            kill_switch_drawdown=0.10  # 10% threshold
        )
        
        # 15% drawdown - SHOULD trigger
        current_value = 85000
        triggered = risk_manager.check_kill_switch(current_value)
        
        assert triggered
        assert risk_manager.kill_switch_triggered
        assert "Drawdown" in risk_manager.kill_switch_reason
        assert risk_manager.metrics['kill_switch_triggers'] == 1
    
    def test_kill_switch_synthetic_crash(self):
        """Test kill-switch on synthetic crash (-20%)."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            kill_switch_drawdown=0.10
        )
        
        # Simulate -20% crash
        crash_value = 80000
        triggered = risk_manager.check_kill_switch(crash_value)
        
        assert triggered
        assert risk_manager.kill_switch_triggered
        
        # Verify positions would be cleared
        risk_manager.clear_all_positions()
        assert len(risk_manager.positions) == 0
    
    def test_kill_switch_stays_triggered(self):
        """Kill-switch should stay triggered after recovery."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            kill_switch_drawdown=0.10
        )
        
        # Trigger kill-switch
        risk_manager.check_kill_switch(85000)
        assert risk_manager.kill_switch_triggered
        
        # Even if portfolio recovers, kill-switch stays triggered
        risk_manager.check_kill_switch(95000)
        assert risk_manager.kill_switch_triggered
    
    def test_kill_switch_manual_reset(self):
        """Test manual kill-switch reset."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            kill_switch_drawdown=0.10
        )
        
        # Trigger kill-switch
        risk_manager.check_kill_switch(85000)
        assert risk_manager.kill_switch_triggered
        
        # Manual reset
        risk_manager.reset_kill_switch()
        assert not risk_manager.kill_switch_triggered
        assert risk_manager.kill_switch_reason is None


class TestExposureLimits:
    """Test exposure limits."""
    
    def test_position_exposure_within_limit(self):
        """Position within exposure limit should pass."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            max_position_exposure=0.25  # 25%
        )
        
        position = Position(
            ticker="AAPL",
            value=20000,  # 20% of portfolio
            quantity=100,
            entry_price=200,
            current_price=200,
            sector="tech"
        )
        
        is_valid, message = risk_manager.check_position_exposure(position)
        assert is_valid
        assert message == "OK"
    
    def test_position_exposure_exceeds_limit(self):
        """Position exceeding exposure limit should fail."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            max_position_exposure=0.25  # 25%
        )
        
        position = Position(
            ticker="AAPL",
            value=30000,  # 30% of portfolio - EXCEEDS LIMIT
            quantity=150,
            entry_price=200,
            current_price=200,
            sector="tech"
        )
        
        is_valid, message = risk_manager.check_position_exposure(position)
        assert not is_valid
        assert "exceeds limit" in message
        assert risk_manager.metrics['violations_detected'] > 0
    
    def test_sector_exposure_within_limit(self):
        """Sector exposure within limit should pass."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            max_sector_exposure=0.40  # 40%
        )
        
        positions = [
            Position("AAPL", 15000, 75, 200, 200, "tech"),
            Position("MSFT", 15000, 75, 200, 200, "tech"),
            # Total tech: 30% - within 40% limit
        ]
        
        is_valid, sector_exp = risk_manager.check_sector_exposure(positions)
        assert is_valid
        assert sector_exp["tech"] == 30000
    
    def test_sector_exposure_exceeds_limit(self):
        """Sector exposure exceeding limit should fail."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            max_sector_exposure=0.40  # 40%
        )
        
        positions = [
            Position("AAPL", 20000, 100, 200, 200, "tech"),
            Position("MSFT", 20000, 100, 200, 200, "tech"),
            Position("GOOGL", 15000, 75, 200, 200, "tech"),
            # Total tech: 55% - EXCEEDS 40% limit
        ]
        
        is_valid, sector_exp = risk_manager.check_sector_exposure(positions)
        assert not is_valid
        assert sector_exp["tech"] == 55000
        assert risk_manager.metrics['violations_detected'] > 0
    
    def test_correlated_positions_semiconductor_example(self):
        """Test correlated positions (NVDA + AMD + INTC)."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            max_sector_exposure=0.40  # 40%
        )
        
        positions = [
            Position("NVDA", 20000, 100, 200, 200, "semiconductor"),
            Position("AMD", 15000, 75, 200, 200, "semiconductor"),
            Position("INTC", 10000, 50, 200, 200, "semiconductor"),
            # Total semiconductor: 45% - EXCEEDS 40%
        ]
        
        is_valid, sector_exp = risk_manager.check_sector_exposure(positions)
        assert not is_valid
        assert sector_exp["semiconductor"] == 45000
    
    def test_total_exposure_within_limit(self):
        """Total exposure within limit should pass."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            max_total_exposure=1.0  # 100%
        )
        
        positions = [
            Position("AAPL", 20000, 100, 200, 200, "tech"),
            Position("JPM", 20000, 100, 200, 200, "finance"),
            Position("XOM", 20000, 100, 200, 200, "energy"),
            # Total: 60%
        ]
        
        is_valid, total_exp = risk_manager.check_total_exposure(positions)
        assert is_valid
        assert total_exp == 0.60
    
    def test_total_exposure_exceeds_limit(self):
        """Total exposure exceeding limit should fail."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            max_total_exposure=0.80  # 80%
        )
        
        positions = [
            Position("AAPL", 30000, 150, 200, 200, "tech"),
            Position("JPM", 30000, 150, 200, 200, "finance"),
            Position("XOM", 30000, 150, 200, 200, "energy"),
            # Total: 90% - EXCEEDS 80%
        ]
        
        is_valid, total_exp = risk_manager.check_total_exposure(positions)
        assert not is_valid
        assert total_exp == 0.90


class TestVolatilityScaling:
    """Test volatility-based position sizing."""
    
    def test_position_sizing_low_volatility(self):
        """Low volatility should increase position size."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            target_volatility=0.02  # 2% target
        )
        
        base_size = 10000
        low_volatility = 0.01  # 1% volatility
        
        adjusted_size = risk_manager.calculate_position_size(base_size, low_volatility)
        
        # Should be larger than base (2x)
        assert adjusted_size > base_size
        assert adjusted_size == pytest.approx(20000, rel=0.01)
    
    def test_position_sizing_high_volatility(self):
        """High volatility should decrease position size."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            target_volatility=0.02  # 2% target
        )
        
        base_size = 10000
        high_volatility = 0.04  # 4% volatility
        
        adjusted_size = risk_manager.calculate_position_size(base_size, high_volatility)
        
        # Should be smaller than base (0.5x)
        assert adjusted_size < base_size
        assert adjusted_size == pytest.approx(5000, rel=0.01)
    
    def test_position_sizing_with_confidence(self):
        """Confidence should scale position size."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            target_volatility=0.02
        )
        
        base_size = 10000
        volatility = 0.02
        low_confidence = 0.5
        
        adjusted_size = risk_manager.calculate_position_size(
            base_size, volatility, confidence=low_confidence
        )
        
        # Should be 50% of base due to low confidence
        assert adjusted_size == pytest.approx(5000, rel=0.01)
    
    def test_position_sizing_capped_at_max_exposure(self):
        """Position size should be capped at max exposure."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            max_position_exposure=0.25,  # 25% max
            target_volatility=0.02
        )
        
        base_size = 50000  # 50% of portfolio
        low_volatility = 0.01  # Would double to 100%
        
        adjusted_size = risk_manager.calculate_position_size(base_size, low_volatility)
        
        # Should be capped at 25% = $25,000
        assert adjusted_size <= 25000


class TestTradeValidation:
    """Test trade validation."""
    
    def test_validate_trade_when_kill_switch_triggered(self):
        """Trade should be rejected when kill-switch is triggered."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            kill_switch_drawdown=0.10
        )
        
        # Trigger kill-switch
        risk_manager.check_kill_switch(85000)
        
        # Try to validate trade
        is_valid, message = risk_manager.validate_trade("AAPL", 10000, "tech")
        
        assert not is_valid
        assert "Kill-switch triggered" in message
    
    def test_validate_trade_within_all_limits(self):
        """Trade within all limits should be validated."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            max_position_exposure=0.25,
            max_sector_exposure=0.40
        )
        
        is_valid, message = risk_manager.validate_trade("AAPL", 20000, "tech")
        
        assert is_valid
        assert message == "Trade validated"
    
    def test_validate_trade_exceeding_position_limit(self):
        """Trade exceeding position limit should be rejected."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            max_position_exposure=0.25
        )
        
        is_valid, message = risk_manager.validate_trade("AAPL", 30000, "tech")
        
        assert not is_valid
        assert "exceeds limit" in message


class TestRiskLevel:
    """Test risk level calculation."""
    
    def test_risk_level_low(self):
        """Low exposure should result in LOW risk level."""
        risk_manager = RiskManager(portfolio_value=100000)
        
        risk_manager.positions = [
            Position("AAPL", 20000, 100, 200, 200, "tech"),
            # 20% exposure
        ]
        
        assert risk_manager.get_risk_level() == RiskLevel.LOW
    
    def test_risk_level_medium(self):
        """Medium exposure should result in MEDIUM risk level."""
        risk_manager = RiskManager(portfolio_value=100000)
        
        risk_manager.positions = [
            Position("AAPL", 30000, 150, 200, 200, "tech"),
            Position("JPM", 30000, 150, 200, 200, "finance"),
            # 60% exposure
        ]
        
        assert risk_manager.get_risk_level() == RiskLevel.MEDIUM
    
    def test_risk_level_high(self):
        """High exposure should result in HIGH risk level."""
        risk_manager = RiskManager(portfolio_value=100000)
        
        risk_manager.positions = [
            Position("AAPL", 30000, 150, 200, 200, "tech"),
            Position("JPM", 30000, 150, 200, 200, "finance"),
            Position("XOM", 30000, 150, 200, 200, "energy"),
            # 90% exposure
        ]
        
        assert risk_manager.get_risk_level() == RiskLevel.HIGH
    
    def test_risk_level_critical_when_kill_switch(self):
        """Kill-switch should result in CRITICAL risk level."""
        risk_manager = RiskManager(
            portfolio_value=100000,
            kill_switch_drawdown=0.10
        )
        
        # Trigger kill-switch
        risk_manager.check_kill_switch(85000)
        
        assert risk_manager.get_risk_level() == RiskLevel.CRITICAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
