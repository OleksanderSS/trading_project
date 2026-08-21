"""A limit called "daily" has to mean a day, in both of its halves.

`PortfolioManager.is_trading_allowed` latches a kill switch when
`VirtualPortfolio.get_daily_drawdown` breaches the configured daily limit.
Two things kept that from working:

  * `update_performance` stamped every valuation with `datetime.now()`, so
    inside a single backtest every record shared today's date, no record was
    ever "from a previous day", and the anchor fell through to the run's
    opening equity. The figure was the loss since the run began.
  * the switch, once tripped, never released -- so a slow 5% drift below
    starting equity ended the run permanently.
"""

from datetime import datetime, timedelta

import pytest

from src.trading.portfolio_manager import PortfolioManager
from src.trading.virtual_portfolio import VirtualPortfolio


@pytest.fixture
def portfolio(tmp_path):
    instance = VirtualPortfolio(initial_balance=100_000.0, portfolio_name='dd_test')
    instance.portfolio_file = tmp_path / 'dd_test.json'
    return instance


def _mark(portfolio, value, when):
    """Record a valuation of `value` at time `when`, with no open positions."""
    portfolio.current_balance = value
    portfolio.update_performance({}, as_of=when)


def test_a_multi_day_slide_is_measured_against_yesterday_not_inception(portfolio):
    """The case that used to end runs: down 20% overall, down 5% today."""
    day = datetime(2026, 3, 2, 16, 0)
    for offset, value in enumerate([100_000.0, 92_000.0, 84_000.0, 80_000.0]):
        _mark(portfolio, value, day + timedelta(days=offset))

    portfolio.current_balance = 76_000.0
    today = day + timedelta(days=4)

    drawdown = portfolio.get_daily_drawdown({}, as_of=today)
    assert drawdown == pytest.approx(-0.05)          # against yesterday's 80,000
    assert drawdown != pytest.approx(-0.24)          # not against inception


def test_the_anchor_is_the_previous_days_close(portfolio):
    day = datetime(2026, 3, 2, 16, 0)
    _mark(portfolio, 100_000.0, day)
    _mark(portfolio, 90_000.0, day + timedelta(days=1))
    _mark(portfolio, 99_000.0, day + timedelta(days=1, hours=1))

    portfolio.current_balance = 95_040.0
    drawdown = portfolio.get_daily_drawdown({}, as_of=day + timedelta(days=2))
    assert drawdown == pytest.approx(-0.04)          # 95,040 against 99,000


def test_live_callers_that_pass_no_time_still_use_the_clock(portfolio):
    _mark(portfolio, 100_000.0, datetime.now() - timedelta(days=1))
    portfolio.current_balance = 97_000.0
    assert portfolio.get_daily_drawdown({}) == pytest.approx(-0.03)


class _Portfolio:
    """Reports a fixed drawdown, so the manager's own logic is what is tested."""

    def __init__(self, drawdown):
        self.drawdown = drawdown

    def get_daily_drawdown(self, _prices, as_of=None):
        return self.drawdown


def _manager(drawdown):
    return PortfolioManager(
        virtual_portfolio=_Portfolio(drawdown),
        config={'max_daily_loss_pct': 0.05},
    )


def test_the_switch_trips_on_a_breach():
    manager = _manager(-0.08)
    monday = datetime(2026, 3, 2, 15, 0)
    assert manager.is_trading_allowed({}, as_of=monday) is False
    assert manager.kill_switch_active is True


def test_the_switch_holds_for_the_rest_of_that_day():
    manager = _manager(-0.08)
    monday = datetime(2026, 3, 2, 15, 0)
    manager.is_trading_allowed({}, as_of=monday)
    manager.portfolio.drawdown = 0.0                 # recovered, same day
    assert manager.is_trading_allowed({}, as_of=monday + timedelta(hours=1)) is False


def test_the_switch_releases_on_the_next_day():
    """A daily limit that never releases is not a daily limit."""
    manager = _manager(-0.08)
    monday = datetime(2026, 3, 2, 15, 0)
    manager.is_trading_allowed({}, as_of=monday)
    manager.portfolio.drawdown = -0.01
    assert manager.is_trading_allowed({}, as_of=monday + timedelta(days=1)) is True
    assert manager.kill_switch_active is False


def test_a_new_day_that_is_still_breaching_trips_again():
    manager = _manager(-0.08)
    monday = datetime(2026, 3, 2, 15, 0)
    manager.is_trading_allowed({}, as_of=monday)
    assert manager.is_trading_allowed({}, as_of=monday + timedelta(days=1)) is False
    assert manager.kill_switch_active is True
