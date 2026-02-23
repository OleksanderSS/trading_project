# auto_trader.py - АВТОМАТИЧНИЙ ТРЕЙДЕР

import time
# import schedule
from datetime import datetime
from core.stages.stage_5_pipeline import run_full_pipeline
from paper_trader import paper_trader
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("AutoTrader")

class AutoTrader:
    """Автоматичний трейдер з розкладом"""
    
    def __init__(self):
        self.is_running = False
        self.last_signals = {}
        
    def run_daily_pipeline(self):
        """Запускає щоденний pipeline"""
        try:
            logger.info("🚀 Starting daily pipeline...")
            
            # Запускаємо pipeline
            results = run_full_pipeline(
                models=["lgbm", "ensemble"],
                tickers=["SPY", "QQQ"],
                time_frames=["1d"],
                debug_no_network=False
            )
            
            # Обробляємо результати
            self.process_pipeline_results(results)
            
            # Показуємо стан рахунку
            account = paper_trader.get_account_info()
            logger.info(f"💰 Account equity: ${account['equity']:,.2f} ({account['total_return']:.2f}%)")
            
        except Exception as e:
            logger.error(f"❌ Daily pipeline error: {e}")
    
    def process_pipeline_results(self, results):
        """Обробляє результати pipeline та виконує торгівлю"""
        if not results:
            return
            
        for model_name, model_results in results.items():
            for combination, result in model_results.items():
                if result.get("status") != "success":
                    continue
                    
                ticker, interval = combination.split("_")
                signal = result.get("final_signal", "HOLD")
                
                # Зберігаємо сигнал
                self.last_signals[f"{ticker}_{model_name}"] = {
                    "signal": signal,
                    "timestamp": datetime.now(),
                    "model": model_name,
                    "ticker": ticker
                }
                
                # Виконуємо торгівлю тільки для основних тікерів
                if ticker in ["SPY", "QQQ"] and signal in ["BUY", "SELL"]:
                    self.execute_trade(ticker, signal, model_name)
    
    def execute_trade(self, ticker, signal, model_name):
        """Виконує торгівлю на основі сигналу"""
        try:
            shares = 10 if ticker == "SPY" else 5
            
            if signal == "BUY":
                result = paper_trader.place_order(ticker, "BUY", shares)
                if result["status"] == "success":
                    logger.info(f"✅ AUTO BUY: {shares} {ticker} @ ${result['price']:.2f} (Model: {model_name})")
                else:
                    logger.warning(f"❌ BUY failed for {ticker}: {result['message']}")
                    
            elif signal == "SELL":
                # Перевіряємо позицію
                account = paper_trader.get_account_info()
                position = next((p for p in account['positions'] if p['ticker'] == ticker), None)
                
                if position and position['shares'] >= shares:
                    result = paper_trader.place_order(ticker, "SELL", shares)
                    if result["status"] == "success":
                        logger.info(f"✅ AUTO SELL: {shares} {ticker} @ ${result['price']:.2f} (Model: {model_name})")
                    else:
                        logger.warning(f"❌ SELL failed for {ticker}: {result['message']}")
                else:
                    logger.info(f"ℹ️ No position to sell for {ticker}")
                    
        except Exception as e:
            logger.error(f"❌ Trade execution error for {ticker}: {e}")
    
    def check_positions(self):
        """Перевіряє позиції та виконує risk management"""
        try:
            account = paper_trader.get_account_info()
            
            for position in account['positions']:
                ticker = position['ticker']
                pnl_pct = (position['unrealized_pnl'] / (position['shares'] * position['avg_price'])) * 100
                
                # Stop-loss на -5%
                if pnl_pct <= -5:
                    result = paper_trader.place_order(ticker, "SELL", position['shares'])
                    if result["status"] == "success":
                        logger.warning(f"🛑 STOP-LOSS: Sold all {ticker} @ ${result['price']:.2f} (Loss: {pnl_pct:.1f}%)")
                
                # Take-profit на +10%
                elif pnl_pct >= 10:
                    # Продаємо половину позиції
                    sell_shares = position['shares'] // 2
                    if sell_shares > 0:
                        result = paper_trader.place_order(ticker, "SELL", sell_shares)
                        if result["status"] == "success":
                            logger.info(f"💰 TAKE-PROFIT: Sold {sell_shares} {ticker} @ ${result['price']:.2f} (Profit: {pnl_pct:.1f}%)")
                            
        except Exception as e:
            logger.error(f"❌ Position check error: {e}")
    
    def print_daily_summary(self):
        """Друкує щоденний підсумок"""
        try:
            account = paper_trader.get_account_info()
            
            print("\n" + "="*60)
            print(f"📊 DAILY SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            print("="*60)
            print(f"💰 Equity: ${account['equity']:,.2f}")
            print(f"📈 Total Return: {account['total_return']:.2f}%")
            print(f"🎯 Positions: {len(account['positions'])}")
            
            if self.last_signals:
                print(f"\n🤖 RECENT SIGNALS:")
                for key, signal_data in list(self.last_signals.items())[-5:]:
                    print(f"  {signal_data['ticker']} ({signal_data['model']}): {signal_data['signal']}")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"❌ Summary error: {e}")
    
    def start_auto_trading(self):
        """Запускає автоматичну торгівлю"""
        print("🤖 AUTO TRADING IS CURRENTLY DISABLED DUE TO MISSING DEPENDENCIES.")
        # print("🤖 STARTING AUTO TRADER")
        # print("=" * 50)
        # print("Schedule:")
        # print("  📊 Pipeline: Daily at 15:00 (before US market)")
        # print("  🔍 Position check: Every 30 minutes")
        # print("  📋 Summary: Every 4 hours")
        # print("=" * 50)
        
        # # Розклад
        # schedule.every().day.at("15:00").do(self.run_daily_pipeline)  # Перед відкриттям US ринку
        # schedule.every(30).minutes.do(self.check_positions)  # Кожні 30 хвилин
        # schedule.every(4).hours.do(self.print_daily_summary)  # Кожні 4 години
        
        # self.is_running = True
        
        # try:
        #     while self.is_running:
        #         schedule.run_pending()
        #         time.sleep(60)  # Перевірка кожну хвилину
                
        # except KeyboardInterrupt:
        #     print("\n🛑 Auto trader stopped by user")
        #     self.is_running = False
        # except Exception as e:
        #     logger.error(f"❌ Auto trader error: {e}")
        #     self.is_running = False

def main():
    """Головна функція"""
    trader = AutoTrader()
    
    # Показуємо поточний стан
    trader.print_daily_summary()
    
    # Запитуємо користувача
    print("\n🤖 AUTO TRADER OPTIONS:")
    print("1. Start auto trading (scheduled) - DISABLED")
    print("2. Run pipeline once")
    print("3. Check positions")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        trader.start_auto_trading()
    elif choice == "2":
        trader.run_daily_pipeline()
        trader.print_daily_summary()
    elif choice == "3":
        trader.check_positions()
        trader.print_daily_summary()
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()