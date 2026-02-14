# utils/trader_logger.py

import os
import datetime
import pandas as pd
import requests
from typing import Optional, Dict
from utils.logger import ProjectLogger
from config.trader_config import TRADER_DEFAULTS

logger = ProjectLogger.get_logger("TradingProjectLogger")

class TraderLogger:
    def __init__(self,
                 initial_balance: float = TRADER_DEFAULTS["initial_balance"],
                 log_path: str = TRADER_DEFAULTS["log_path"],
                 telegram_token: Optional[str] = None,
                 chat_id: Optional[str] = None,
                 log_hold: bool = False,
                 risk_fraction: float = 1.0):
        """Легкий логер for трейwhereра."""
        self.balance = initial_balance
        self.holdings: Dict[str, float] = {}
        self.log_path = log_path
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.log_hold = log_hold
        self.risk_fraction = risk_fraction

        # створюємо CSV, якщо not andснує
        if not os.path.exists(self.log_path):
            df = pd.DataFrame(columns=["timestamp","ticker","action","price","qty","balance"])
            df.to_csv(self.log_path, index=False)
            logger.info(f"[TraderLogger] [OK] Створено новий лог-file: {self.log_path}")

    def _send_telegram(self, message: str):
        if not (self.telegram_token and self.chat_id):
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        try:
            requests.post(url, data={"chat_id": self.chat_id, "text": message})
            logger.info("[TraderLogger]  Повandдомлення вandдправлено у Telegram")
        except Exception as e:
            logger.error(f"[TraderLogger] [ERROR] Telegram Error: {e}")

    def portfolio_value(self, price_feed: Dict[str, float]) -> float:
        value = self.balance
        for ticker, qty in self.holdings.items():
            value += qty * price_feed.get(ticker, 0.0)
        logger.info(f"[TraderLogger] [DATA] Поточна вартandсть портфеля: {value:.2f}")
        return value

    def execute_trade(self, ticker: str, signal: str, price: float, qty: Optional[float] = None):
        action = signal.upper()
        timestamp = datetime.datetime.utcnow().isoformat()

        if qty is None and action == "BUY" and price > 0:
            invest_amount = self.balance * self.risk_fraction
            qty = invest_amount / price
        elif qty is None:
            qty = 0.0

        executed = False

        if action == "BUY" and qty > 0:
            cost = price * qty
            if self.balance >= cost:
                self.balance -= cost
                self.holdings[ticker] = self.holdings.get(ticker, 0.0) + qty
                executed = True
                logger.info(f"[TraderLogger] [OK] BUY {qty:.4f} {ticker} @ {price:.2f}")
            else:
                logger.warning(f"[TraderLogger] [WARN] Недосandтньо балансу for покупки {ticker}")

        elif action == "SELL":
            held_qty = self.holdings.get(ticker, 0.0)
            sell_qty = min(qty, held_qty)
            if sell_qty > 0:
                self.balance += price * sell_qty
                self.holdings[ticker] -= sell_qty
                if self.holdings[ticker] <= 0:
                    del self.holdings[ticker]
                qty = sell_qty
                executed = True
                logger.info(f"[TraderLogger] [OK] SELL {qty:.4f} {ticker} @ {price:.2f}")
            else:
                logger.warning(f"[TraderLogger] [WARN] Немає активandв for продажу {ticker}")

        elif action == "HOLD":
            if self.log_hold:
                executed = True
                logger.info(f"[TraderLogger] [OK] HOLD {ticker}")
            qty = 0.0

        if not executed:
            return

        # Лог у CSV
        df = pd.DataFrame([{
            "timestamp": timestamp,
            "ticker": ticker,
            "action": action,
            "price": price,
            "qty": qty,
            "balance": self.balance
        }])
        df.to_csv(self.log_path, mode='a', header=False, index=False)

        # Telegram
        self._send_telegram(f"{timestamp} | {action} {qty:.4f} {ticker} @ {price:.2f} | Balance: {self.balance:.2f}")

    def get_history(self, as_dataframe: bool = True):
        try:
            df = pd.read_csv(self.log_path)
        except Exception as e:
            logger.error(f"[TraderLogger] [ERROR] Error чиandння лог-fileу: {e}")
            return pd.DataFrame()
        if as_dataframe:
            return df
        else:
            return df.to_dict(orient="records")