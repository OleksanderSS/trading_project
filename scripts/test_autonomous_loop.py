from src.analytics.arena.arena_battle import get_trading_arena
from src.core.logging.logger import ProjectLogger
from src.trading.portfolio_manager import PortfolioManager
from src.trading.virtual_portfolio import VirtualPortfolio

logger = ProjectLogger.get_logger("AutonomousLoopTest")

def test_autonomous_feedback_loop():
    logger.info("Starting autonomous feedback loop test...")
    
    # Setup
    arena = get_trading_arena()
    portfolio = VirtualPortfolio(initial_balance=100000)
    pm = PortfolioManager(virtual_portfolio=portfolio)
    
    # 1. Register a test model
    model_name = "test_model_1"
    arena.register_model(model_name, None, model_type="traditional")
    
    # 2. Simulate Stop-Loss events
    logger.info("Simulating 3 stop-loss events...")
    prices = {'AAPL': 100}
    position = {'quantity': 10, 'stop_loss': 105, 'model_name': model_name}
    portfolio.positions = {'AAPL': position}
    
    for i in range(3):
        pm.check_risk_exits(prices)
        logger.info(f"Event {i+1}: Failure recorded.")
        
    # 3. Verify COOLDOWN
    status = arena.models[model_name].get('status')
    logger.info(f"Model status: {status}")
    assert status == 'COOLDOWN', "Model should be in COOLDOWN!"
    
    # 4. Try to open new trade (should be blocked)
    class MockReport:
        def __init__(self, name):
            self.model_name = name
            self.market_regime = 'NORMAL'
        
    order = pm._create_buy_order('AAPL', 95, 0.8, MockReport(model_name))
    if order is None:
        logger.info("SUCCESS: Trading blocked for cooled down model.")
    else:
        logger.error("FAILURE: Trade was not blocked!")

if __name__ == "__main__":
    test_autonomous_feedback_loop()
