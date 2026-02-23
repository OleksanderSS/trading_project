# paper_trader.py - РЕАЛЬНИЙ PAPER TRADING

import json
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import os
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("PaperTrader")

class PaperTradingEngine:
    """Реальний paper trading with баwithою data"""
    
    def __init__(self, initial_balance=10000):
        # ВИПРАВЛЕНО: Використовуємо абсолютний шлях
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(project_root, "data", "databases", "paper_trading.db")
        self.initial_balance = initial_balance
        self.setup_database()
        
    def setup_database(self):
        """Створює andблицand for paper trading"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Таблиця рахунку
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS account (
                id INTEGER PRIMARY KEY,
                balance REAL,
                equity REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Таблиця поwithицandй
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                shares INTEGER,
                avg_price REAL,
                current_price REAL,
                unrealized_pnl REAL,
                opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Таблиця орwhereрandв
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                side TEXT,
                shares INTEGER,
                price REAL,
                order_type TEXT,
                status TEXT,
                filled_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Таблиця P&L
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pnl_history (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                realized_pnl REAL,
                shares INTEGER,
                buy_price REAL,
                sell_price REAL,
                closed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Інandцandалandwithуємо рахунок якщо порожнandй
        cursor.execute('SELECT COUNT(*) FROM account')
        if cursor.fetchone()[0] == 0:
            cursor.execute('INSERT INTO account (balance, equity) VALUES (?, ?)', 
                         (self.initial_balance, self.initial_balance))
        
        conn.commit()
        conn.close()
        
    def get_current_price(self, ticker):
        """Отримує поточну цandну"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d", interval="1m")
            return data['Close'].iloc[-1] if not data.empty else None
        except:
            return None
            
    def place_order(self, ticker, side, shares, order_type="market"):
        """Роwithмandщує орwhereр"""
        current_price = self.get_current_price(ticker)
        if not current_price:
            return {"status": "error", "message": f"Cannot get price for {ticker}"}
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Перевandряємо баланс for покупки
        if side.upper() == "BUY":
            cursor.execute('SELECT balance FROM account ORDER BY id DESC LIMIT 1')
            balance = cursor.fetchone()[0]
            required = shares * current_price
            
            if balance < required:
                conn.close()
                return {"status": "error", "message": "Insufficient balance"}
                
        # Перевandряємо поwithицandю for продажу
        elif side.upper() == "SELL":
            cursor.execute('SELECT shares FROM positions WHERE ticker = ?', (ticker,))
            position = cursor.fetchone()
            if not position or position[0] < shares:
                conn.close()
                return {"status": "error", "message": "Insufficient shares"}
        
        # Створюємо орwhereр
        cursor.execute('''
            INSERT INTO orders (ticker, side, shares, price, order_type, status, filled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (ticker, side.upper(), shares, current_price, order_type, "FILLED", datetime.now()))
        
        order_id = cursor.lastrowid
        
        # Виконуємо орwhereр
        self._execute_order(cursor, ticker, side.upper(), shares, current_price)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Order executed: {side} {shares} {ticker} @ ${current_price:.2f}")
        return {"status": "success", "order_id": order_id, "price": current_price}
        
    def _execute_order(self, cursor, ticker, side, shares, price):
        """Виконує орwhereр"""
        if side == "BUY":
            # Оновлюємо баланс
            cursor.execute('SELECT balance FROM account ORDER BY id DESC LIMIT 1')
            balance = cursor.fetchone()[0]
            new_balance = balance - (shares * price)
            
            cursor.execute('INSERT INTO account (balance, equity) VALUES (?, ?)', 
                         (new_balance, new_balance))
            
            # Оновлюємо/створюємо поwithицandю
            cursor.execute('SELECT shares, avg_price FROM positions WHERE ticker = ?', (ticker,))
            position = cursor.fetchone()
            
            if position:
                old_shares, old_avg = position
                new_shares = old_shares + shares
                new_avg = ((old_shares * old_avg) + (shares * price)) / new_shares
                
                cursor.execute('''
                    UPDATE positions SET shares = ?, avg_price = ?, updated_at = ?
                    WHERE ticker = ?
                ''', (new_shares, new_avg, datetime.now(), ticker))
            else:
                cursor.execute('''
                    INSERT INTO positions (ticker, shares, avg_price, current_price)
                    VALUES (?, ?, ?, ?)
                ''', (ticker, shares, price, price))
                
        elif side == "SELL":
            # Отримуємо поwithицandю
            cursor.execute('SELECT shares, avg_price FROM positions WHERE ticker = ?', (ticker,))
            old_shares, avg_price = cursor.fetchone()
            
            # Рахуємо P&L
            realized_pnl = shares * (price - avg_price)
            
            # Записуємо P&L
            cursor.execute('''
                INSERT INTO pnl_history (ticker, realized_pnl, shares, buy_price, sell_price)
                VALUES (?, ?, ?, ?, ?)
            ''', (ticker, realized_pnl, shares, avg_price, price))
            
            # Оновлюємо баланс
            cursor.execute('SELECT balance FROM account ORDER BY id DESC LIMIT 1')
            balance = cursor.fetchone()[0]
            new_balance = balance + (shares * price)
            
            cursor.execute('INSERT INTO account (balance, equity) VALUES (?, ?)', 
                         (new_balance, new_balance))
            
            # Оновлюємо поwithицandю
            new_shares = old_shares - shares
            if new_shares > 0:
                cursor.execute('''
                    UPDATE positions SET shares = ?, updated_at = ?
                    WHERE ticker = ?
                ''', (new_shares, datetime.now(), ticker))
            else:
                cursor.execute('DELETE FROM positions WHERE ticker = ?', (ticker,))
                
    def get_account_info(self):
        """Отримує andнформацandю про рахунок"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Баланс
        cursor.execute('SELECT balance FROM account ORDER BY id DESC LIMIT 1')
        balance = cursor.fetchone()[0]
        
        # Поwithицandї
        cursor.execute('SELECT ticker, shares, avg_price FROM positions')
        positions = cursor.fetchall()
        
        # Рахуємо equity
        equity = balance
        position_details = []
        
        for ticker, shares, avg_price in positions:
            current_price = self.get_current_price(ticker)
            if current_price:
                market_value = shares * current_price
                unrealized_pnl = shares * (current_price - avg_price)
                equity += market_value
                
                position_details.append({
                    'ticker': ticker,
                    'shares': shares,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'market_value': market_value,
                    'unrealized_pnl': unrealized_pnl
                })
        
        # P&L andсторandя
        cursor.execute('SELECT SUM(realized_pnl) FROM pnl_history')
        total_realized = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'balance': balance,
            'equity': equity,
            'positions': position_details,
            'total_realized_pnl': total_realized,
            'total_return': ((equity - self.initial_balance) / self.initial_balance) * 100
        }
        
    def print_account_summary(self):
        """Друкує пandдсумок рахунку"""
        info = self.get_account_info()
        
        print("\n" + "="*50)
        print("[DATA] PAPER TRADING ACCOUNT")
        print("="*50)
        print(f"Cash Balance: ${info['balance']:,.2f}")
        print(f"Total Equity: ${info['equity']:,.2f}")
        print(f"Total Return: {info['total_return']:.2f}%")
        print(f"Realized P&L: ${info['total_realized_pnl']:,.2f}")
        
        if info['positions']:
            print(f"\n[UP] POSITIONS ({len(info['positions'])}):")
            for pos in info['positions']:
                pnl_color = "" if pos['unrealized_pnl'] >= 0 else ""
                print(f"  {pos['ticker']}: {pos['shares']} shares @ ${pos['avg_price']:.2f}")
                print(f"    Current: ${pos['current_price']:.2f} | P&L: {pnl_color}${pos['unrealized_pnl']:,.2f}")
        else:
            print("\n[UP] POSITIONS: None")
            
        print("="*50)
        return info

# Global paper trader instance
paper_trader = PaperTradingEngine()

def test_paper_trading():
    """Тест paper trading"""
    print(" TESTING PAPER TRADING...")
    
    # Купуємо SPY
    result = paper_trader.place_order("SPY", "BUY", 10)
    print(f"Buy order: {result}")
    
    # Покаwithуємо рахунок
    paper_trader.print_account_summary()
    
    return result

if __name__ == "__main__":
    test_paper_trading()