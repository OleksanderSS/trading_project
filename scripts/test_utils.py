import sys
from datetime import date
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.trading_calendar import TradingCalendar


def test_calendar():
    calendar = TradingCalendar(start_year=2024, end_year=2024)
    
    # Test US Holiday (Jan 1, 2024 - New Year's Day)
    new_year = date(2024, 1, 1)
    print(f"Is {new_year} a trading day? {calendar.is_trading_day(new_year)}")
    
    # Test Business Day (Jan 2, 2024)
    jan_2 = date(2024, 1, 2)
    print(f"Is {jan_2} a trading day? {calendar.is_trading_day(jan_2)}")
    
    # Test Weekend (Jan 6, 2024)
    sat = date(2024, 1, 6)
    print(f"Is {sat} a trading day? {calendar.is_trading_day(sat)}")
    
    # Test Next Trading Day
    friday = date(2024, 1, 12)
    print(f"Next trading day after {friday}: {calendar.get_next_trading_day(friday)}")
    
    # Test Previous Trading Days
    monday = date(2024, 1, 15) # MLK Day
    print(f"Is {monday} (MLK Day) a trading day? {calendar.is_trading_day(monday)}")
    prev_days = calendar.get_previous_trading_days(monday, 3)
    print(f"3 trading days before {monday}: {prev_days}")

if __name__ == "__main__":
    test_calendar()
