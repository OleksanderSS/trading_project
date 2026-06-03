from src.trading.portfolio_manager import PortfolioManager
from src.trading.elite_risk_sizer import EliteRiskSizer


class FakePortfolio:
    positions = {}
    current_balance = 10_000.0

    def get_total_value(self, current_prices):
        return self.current_balance


class FailingAdaptiveSizer:
    def calculate_position_size(self, params):
        raise RuntimeError("adaptive sizing unavailable")


class EliteSizer:
    def compute_optimal_position_size(self, **kwargs):
        return 0.1, {"stages": {"kelly_size": 0.1}}


class PriceAwareEliteSizer:
    def __init__(self):
        self.kwargs = None

    def compute_optimal_position_size(self, **kwargs):
        self.kwargs = kwargs
        return 0.1, {"stages": {"kelly_size": 0.1}}


def test_position_size_falls_back_to_elite_sizer_when_adaptive_fails():
    manager = PortfolioManager(FakePortfolio(), elite_risk_sizer=EliteSizer())
    manager.position_sizer = FailingAdaptiveSizer()

    shares = manager._calculate_position_size("AAPL", price=100.0, confidence=0.8)

    assert shares == 10


def test_elite_fallback_receives_current_price():
    elite = PriceAwareEliteSizer()
    manager = PortfolioManager(FakePortfolio(), elite_risk_sizer=elite)
    manager.position_sizer = FailingAdaptiveSizer()

    manager._calculate_position_size("AAPL", price=125.0, confidence=0.8)

    assert elite.kwargs["current_price"] == 125.0


def test_elite_risk_sizer_respects_existing_position_limit():
    sizer = EliteRiskSizer(kelly_fraction=1.0)

    shares = sizer.calculate_optimal_position_size(
        ticker="AAPL",
        entry_price=100.0,
        win_rate=0.9,
        avg_win_loss_ratio=3.0,
        current_positions={"AAPL": {"quantity": 15}},
        total_equity=10_000.0,
        position_value_limit=0.15,
        portfolio_volatility=0.2,
        cash_available=10_000.0,
    )

    assert shares == 0
