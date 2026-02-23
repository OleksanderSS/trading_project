#!/usr/bin/env python3
"""
Virtual Portfolio - Віртуальний рахунок для паперової торгівлі з реальними цінами
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class VirtualPortfolio:
    """
    Віртуальний портфель для паперової торгівлі
    Використовує реальні ціни, але віртуальні гроші
    """
    
    def __init__(self, initial_balance: float = 10000.0, portfolio_name: str = "default"):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.portfolio_name = portfolio_name
        
        # Позиції
        self.positions = {}  # {ticker: {'quantity': int, 'avg_price': float, 'entry_time': datetime}}
        
        # Історія транзакцій
        self.transactions = []  # List of transaction dictionaries
        
        # Performance tracking
        self.performance_history = []
        self.daily_pnl = []
        
        # Ризик-менеджмент
        self.max_position_size = 0.1  # 10% від балансу на позицію
        self.max_total_risk = 0.3     # 30% загальний ризик
        self.stop_loss_pct = 0.05     # 5% stop loss
        self.take_profit_pct = 0.10   # 10% take profit
        
        # Файл для збереження
        self.portfolio_file = Path(f"data/portfolios/{portfolio_name}_portfolio.json")
        self.load_portfolio()
        
        logger.info(f"[MONEY] Virtual portfolio '{portfolio_name}' initialized with ${initial_balance:,.2f}")
    
    def load_portfolio(self):
        """Завантаження портфеля з файлу"""
        try:
            if self.portfolio_file.exists():
                with open(self.portfolio_file, 'r') as f:
                    data = json.load(f)
                
                self.current_balance = data.get('current_balance', self.initial_balance)
                self.positions = data.get('positions', {})
                self.transactions = data.get('transactions', [])
                self.performance_history = data.get('performance_history', [])
                
                # Конвертація дат
                for pos in self.positions.values():
                    if 'entry_time' in pos:
                        pos['entry_time'] = datetime.fromisoformat(pos['entry_time'])
                
                for tx in self.transactions:
                    if 'timestamp' in tx:
                        tx['timestamp'] = datetime.fromisoformat(tx['timestamp'])
                
                logger.info(f"[OK] Portfolio loaded from {self.portfolio_file}")
            else:
                self.save_portfolio()
                
        except Exception as e:
            logger.error(f"[ERROR] Error loading portfolio: {e}")
    
    def save_portfolio(self):
        """Збереження портфеля в файл"""
        try:
            # Створюємо директорію якщо не існує
            self.portfolio_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Підготовка data для збереження
            data = {
                'portfolio_name': self.portfolio_name,
                'initial_balance': self.initial_balance,
                'current_balance': self.current_balance,
                'positions': {},
                'transactions': [],
                'performance_history': self.performance_history,
                'last_updated': datetime.now().isoformat()
            }
            
            # Конвертація позицій
            for ticker, pos in self.positions.items():
                pos_copy = pos.copy()
                if 'entry_time' in pos_copy:
                    pos_copy['entry_time'] = pos_copy['entry_time'].isoformat()
                data['positions'][ticker] = pos_copy
            
            # Конвертація транзакцій
            for tx in self.transactions:
                tx_copy = tx.copy()
                if 'timestamp' in tx_copy:
                    tx_copy['timestamp'] = tx_copy['timestamp'].isoformat()
                data['transactions'].append(tx_copy)
            
            with open(self.portfolio_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"[SAVE] Portfolio saved to {self.portfolio_file}")
            
        except Exception as e:
            logger.error(f"[ERROR] Error saving portfolio: {e}")
    
    def get_current_balance(self) -> float:
        """Отримання поточного балансу"""
        return self.current_balance
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Розрахунок загальної вартості портфеля
        """
        total_value = self.current_balance
        
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                position_value = position['quantity'] * current_prices[ticker]
                total_value += position_value
        
        return total_value
    
    def get_position_size(self, ticker: str, price: float, confidence: float = 0.8) -> int:
        """
        Розрахунок розміру позиції на основі ризик-менеджменту
        """
        # Максимальна вартість позиції
        max_position_value = self.current_balance * self.max_position_size
        
        # Коригування на основі confidence
        adjusted_value = max_position_value * confidence
        
        # Розрахунок кількості акцій
        shares = int(adjusted_value / price)
        
        # Мінімум 1 акція, максимум 100
        return max(1, min(shares, 100))
    
    def can_buy(self, ticker: str, quantity: int, price: float) -> bool:
        """
        Перевірка чи можна купити акції
        """
        total_cost = quantity * price
        
        # Перевірка балансу
        if total_cost > self.current_balance:
            return False
        
        # Перевірка максимального ризику
        current_risk = self.get_total_risk({ticker: price})
        if current_risk > self.max_total_risk:
            return False
        
        return True
    
    def can_sell(self, ticker: str, quantity: int) -> bool:
        """
        Перевірка чи можна продати акції
        """
        if ticker not in self.positions:
            return False
        
        return self.positions[ticker]['quantity'] >= quantity
    
    def buy_stock(self, ticker: str, quantity: int, price: float, 
                  reason: str = "", confidence: float = 0.8) -> Dict:
        """
        Купівля акцій
        """
        try:
            if not self.can_buy(ticker, quantity, price):
                return {'success': False, 'error': 'Cannot buy stock'}
            
            total_cost = quantity * price
            
            # Створення транзакції
            transaction = {
                'timestamp': datetime.now(),
                'type': 'BUY',
                'ticker': ticker,
                'quantity': quantity,
                'price': price,
                'total_cost': total_cost,
                'reason': reason,
                'confidence': confidence
            }
            
            # Оновлення балансу
            self.current_balance -= total_cost
            
            # Оновлення позиції
            if ticker in self.positions:
                # Середня ціна для існуючої позиції
                old_quantity = self.positions[ticker]['quantity']
                old_avg_price = self.positions[ticker]['avg_price']
                
                new_quantity = old_quantity + quantity
                new_avg_price = ((old_quantity * old_avg_price) + (quantity * price)) / new_quantity
                
                self.positions[ticker]['quantity'] = new_quantity
                self.positions[ticker]['avg_price'] = new_avg_price
            else:
                self.positions[ticker] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'entry_time': datetime.now(),
                    'confidence': confidence
                }
            
            # Додавання транзакції
            self.transactions.append(transaction)
            
            # Збереження
            self.save_portfolio()
            
            logger.info(f"[OK] Bought {quantity} shares of {ticker} at ${price:.2f}")
            
            return {
                'success': True,
                'transaction': transaction,
                'new_balance': self.current_balance,
                'position': self.positions[ticker]
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Error buying stock: {e}")
            return {'success': False, 'error': str(e)}
    
    def sell_stock(self, ticker: str, quantity: int, price: float, 
                   reason: str = "") -> Dict:
        """
        Продаж акцій
        """
        try:
            if not self.can_sell(ticker, quantity):
                return {'success': False, 'error': 'Cannot sell stock'}
            
            position = self.positions[ticker]
            total_revenue = quantity * price
            
            # Розрахунок PnL
            cost_basis = quantity * position['avg_price']
            pnl = total_revenue - cost_basis
            pnl_pct = (pnl / cost_basis) * 100
            
            # Створення транзакції
            transaction = {
                'timestamp': datetime.now(),
                'type': 'SELL',
                'ticker': ticker,
                'quantity': quantity,
                'price': price,
                'total_revenue': total_revenue,
                'cost_basis': cost_basis,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'reason': reason
            }
            
            # Оновлення балансу
            self.current_balance += total_revenue
            
            # Оновлення позиції
            if quantity == position['quantity']:
                # Продаж всієї позиції
                del self.positions[ticker]
            else:
                # Частковий продаж
                self.positions[ticker]['quantity'] -= quantity
            
            # Додавання транзакції
            self.transactions.append(transaction)
            
            # Збереження
            self.save_portfolio()
            
            logger.info(f"[OK] Sold {quantity} shares of {ticker} at ${price:.2f} (PnL: ${pnl:.2f})")
            
            return {
                'success': True,
                'transaction': transaction,
                'new_balance': self.current_balance,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Error selling stock: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_total_risk(self, current_prices: Dict[str, float]) -> float:
        """
        Розрахунок загального ризику портфеля
        """
        total_risk = 0.0
        portfolio_value = self.get_total_value(current_prices)
        
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                position_value = position['quantity'] * current_prices[ticker]
                position_risk = position_value / portfolio_value
                total_risk += position_risk
        
        return total_risk
    
    def check_stop_loss_take_profit(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Перевірка stop loss та take profit
        """
        signals = []
        
        for ticker, position in self.positions.items():
            if ticker not in current_prices:
                continue
            
            current_price = current_prices[ticker]
            entry_price = position['avg_price']
            
            # Stop loss
            if current_price <= entry_price * (1 - self.stop_loss_pct):
                signals.append({
                    'ticker': ticker,
                    'type': 'STOP_LOSS',
                    'price': current_price,
                    'reason': f'Stop loss triggered at {self.stop_loss_pct*100:.1f}%'
                })
            
            # Take profit
            elif current_price >= entry_price * (1 + self.take_profit_pct):
                signals.append({
                    'ticker': ticker,
                    'type': 'TAKE_PROFIT',
                    'price': current_price,
                    'reason': f'Take profit triggered at {self.take_profit_pct*100:.1f}%'
                })
        
        return signals
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """
        Отримання резюме портфеля
        """
        total_value = self.get_total_value(current_prices)
        total_pnl = total_value - self.initial_balance
        total_pnl_pct = (total_pnl / self.initial_balance) * 100
        
        # Позиції
        positions_summary = []
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                current_price = current_prices[ticker]
                position_value = position['quantity'] * current_price
                pnl = position_value - (position['quantity'] * position['avg_price'])
                pnl_pct = (pnl / (position['quantity'] * position['avg_price'])) * 100
                
                positions_summary.append({
                    'ticker': ticker,
                    'quantity': position['quantity'],
                    'avg_price': position['avg_price'],
                    'current_price': current_price,
                    'position_value': position_value,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
        
        # Performance metrics
        recent_transactions = [tx for tx in self.transactions 
                              if datetime.now() - tx['timestamp'] <= timedelta(days=30)]
        
        wins = len([tx for tx in recent_transactions if tx['type'] == 'SELL' and tx.get('pnl', 0) > 0])
        losses = len([tx for tx in recent_transactions if tx['type'] == 'SELL' and tx.get('pnl', 0) < 0])
        total_trades = wins + losses
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'portfolio_name': self.portfolio_name,
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'positions_count': len(self.positions),
            'positions': positions_summary,
            'transactions_count': len(self.transactions),
            'recent_wins': wins,
            'recent_losses': losses,
            'recent_trades': total_trades,
            'win_rate': win_rate,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_performance_history(self) -> pd.DataFrame:
        """
        Отримання історії продуктивності
        """
        if not self.performance_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.performance_history)
    
    def update_performance(self, current_prices: Dict[str, float]):
        """
        Оновлення історії продуктивності
        """
        total_value = self.get_total_value(current_prices)
        total_pnl = total_value - self.initial_balance
        total_pnl_pct = (total_pnl / self.initial_balance) * 100
        
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'positions_count': len(self.positions)
        }
        
        self.performance_history.append(performance_record)
        
        # Обмежуємо історію до 1000 записів
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        self.save_portfolio()
    
    def reset_portfolio(self):
        """
        Скидання портфеля до початкового стану
        """
        self.current_balance = self.initial_balance
        self.positions = {}
        self.transactions = []
        self.performance_history = []
        self.daily_pnl = []
        
        self.save_portfolio()
        logger.info(f"[RESTART] Portfolio '{self.portfolio_name}' reset to initial state")
