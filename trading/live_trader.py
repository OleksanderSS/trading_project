# live_trader.py - РЕАЛЬНИЙ ТРЕЙДИНГ

import time
import yfinance as yf
from datetime import datetime
from core.stages.stage_4_modeling import run_stage_4_modeling
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("LiveTrader")

class LiveTrader:
    def __init__(self, tickers=["SPY", "QQQ"], models=["lgbm", "ensemble"]):
        self.tickers = tickers
        self.models = models
        self.positions = {}
        
    def get_live_data(self, ticker):
        """Отримати поточнand данand"""
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d", interval="1m")
        return hist.tail(100)  # Осandннand 100 хвилин
        
    def generate_signal(self, ticker):
        """Згеnotрувати торговий сигнал"""
        try:
            data = self.get_live_data(ticker)
            if len(data) < 50:
                return "HOLD"
                
            # Швидкand andндикатори
            data['RSI'] = self.calculate_rsi(data['Close'])
            data['SMA_20'] = data['Close'].rolling(20).mean()
            
            current_price = data['Close'].iloc[-1]
            rsi = data['RSI'].iloc[-1]
            sma = data['SMA_20'].iloc[-1]
            
            # Простand правила for quicklyго прибутку
            if rsi < 30 and current_price > sma:
                return "BUY"
            elif rsi > 70 and current_price < sma:
                return "SELL"
            else:
                return "HOLD"
                
        except Exception as e:
            logger.error(f"Signal error for {ticker}: {e}")
            return "HOLD"
            
    def calculate_rsi(self, prices, period=14):
        """Швидкий RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def run_live_trading(self):
        """Основний цикл живого трейдингу"""
        print("[START] LIVE TRADING STARTED")
        print("=" * 50)
        
        while True:
            try:
                for ticker in self.tickers:
                    signal = self.generate_signal(ticker)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    
                    if signal != "HOLD":
                        print(f"{current_time} | {ticker} | {signal} | [MONEY]")
                        
                        # ТУТ ДОДАТИ РЕАЛЬНІ ОРДЕРИ (Interactive Brokers, Alpaca, тощо)
                        # self.place_order(ticker, signal)
                        
                    else:
                        print(f"{current_time} | {ticker} | {signal}")
                        
                time.sleep(60)  # Перевandрка кожну хвилину
                
            except KeyboardInterrupt:
                print("\n Trading stopped by user")
                break
            except Exception as e:
                logger.error(f"Trading error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    trader = LiveTrader()
    trader.run_live_trading()