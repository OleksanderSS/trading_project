# core/strategy/simulator.py

import os
import logging
import pandas as pd
from datetime import datetime
import requests
from typing import Dict, Optional
import matplotlib.pyplot as plt
from utils.logger import ProjectLogger
from utils.trading_calendar import TradingCalendar
# from core.analysis.news_impact import generate_news_signals  # TODO: Реалandwithувати новиннand сигнали


logger = ProjectLogger.get_logger("TradingProjectLogger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
logger.propagate = False

SIM_RESULTS_FILE = "simulation_results.csv"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

class Simulator:
    def __init__(self,
        initial_balance: float = 1000.0,
        log_hold: bool = False,
        calendar: Optional[TradingCalendar] = None):
        self.balance = initial_balance
        self.portfolio: Dict[str, float] = {}
        self.history = []
        self.log_hold = log_hold
        self.calendar = calendar

    def run_news_signals(self, trigger_data: Dict[str, pd.Series], current_prices: Dict[str, float]):
        # TODO: Реалandwithувати новиннand сигнали
        # signals = generate_news_signals(trigger_data)
        signals = {}  # Тимчасово порожньо
        for ticker, signal in signals.items():
            price = current_prices.get(ticker, 0.0)
            if price > 0:
                self.simulate_trade(ticker, signal, price=price, amount=1)

    def simulate_trade(self, ticker: str, signal: str, price: float, amount: float):
        timestamp = datetime.utcnow()
        trade_date = timestamp.date()

        #  Перевandрка торгового дня
        if self.calendar and not self.calendar.is_trading_day(trade_date):
            try:
                next_day = self.calendar.shift_to_next_trading_day(trade_date)
                logger.warning(f" {trade_date} not є торговим дnotм  наступний: {next_day}")
            except Exception as e:
                logger.warning(f" {trade_date} not є торговим дnotм  and not вдалося withнайти наступний: {e}")
            return

        if signal not in ["BUY", "SELL", "HOLD"]:
            logger.warning(f"Невandдомий сигнал: {signal}")
            return

        trade_info = {
            "timestamp": timestamp,
            "ticker": ticker,
            "signal": signal,
            "price": price,
            "amount": amount,
            "balance_before": self.balance
        }

        if signal == "BUY":
            cost = price * amount
            if cost <= self.balance:
                self.balance -= cost
                self.portfolio[ticker] = self.portfolio.get(ticker, 0) + amount
                trade_info["balance_after"] = self.balance
                logger.info(f"[OK] Купandвля {amount} {ticker} for {price}, баланс: {self.balance}")
                self.send_telegram(f"[OK] Купandвля {amount} {ticker} for {price}, баланс: {self.balance}")
            else:
                logger.warning(f"Недосandтньо коштandв for покупки {ticker}")
                trade_info["balance_after"] = self.balance

        elif signal == "SELL":
            held = self.portfolio.get(ticker, 0)
            sell_amount = min(amount, held)
            if sell_amount > 0:
                self.portfolio[ticker] -= sell_amount
                self.balance += price * sell_amount
                if self.portfolio[ticker] == 0:
                    del self.portfolio[ticker]
                trade_info["amount"] = sell_amount
                trade_info["balance_after"] = self.balance
                logger.info(f" Продаж {sell_amount} {ticker} for {price}, баланс: {self.balance}")
                self.send_telegram(f" Продаж {sell_amount} {ticker} for {price}, баланс: {self.balance}")
            else:
                logger.warning(f"Недосandтньо {ticker} for продажу")
                trade_info["balance_after"] = self.balance

        else:  # HOLD
            trade_info["balance_after"] = self.balance
            if self.log_hold:
                logger.info(f"HOLD {ticker}, баланс: {self.balance}")

        self.history.append(trade_info)
        self.save_history()

    def save_history(self):
        df = pd.DataFrame(self.history)
        df.to_csv(SIM_RESULTS_FILE, index=False)
        logger.info(f"Історandя withбережена у {SIM_RESULTS_FILE}")

    def send_telegram(self, message: str):
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
            try:
                resp = requests.post(url, data=payload, timeout=5)
                if resp.status_code != 200:
                    logger.warning(f"Error Telegram: {resp.text}")
            except Exception as e:
                logger.error(f"Не вдалося вandдправити повandдомлення: {e}")

    def summary(self):
        logger.info(f"Баланс: {self.balance}, Портфель: {self.portfolio}")
        return {"balance": self.balance, "portfolio": self.portfolio}

    def get_performance(self):
        df = pd.DataFrame(self.history)
        if df.empty:
            return {}

        performance = {}
        for ticker in df["ticker"].unique():
            df_t = df[df["ticker"] == ticker]
            buys = df_t[df_t["signal"] == "BUY"]["price"] * df_t[df_t["signal"] == "BUY"]["amount"]
            sells = df_t[df_t["signal"] == "SELL"]["price"] * df_t[df_t["signal"] == "SELL"]["amount"]
            pnl = sells.sum() - buys.sum()
            performance[ticker] = round(pnl, 2)

        logger.info(f"[UP] PnL по кожному активу: {performance}")
        return performance

    def get_hold_log(self):
        df = pd.DataFrame(self.history)
        return df[df["signal"] == "HOLD"]

    def export_json(self, path="simulation_results.json"):
        df = pd.DataFrame(self.history)
        df.to_json(path, orient="records", date_format="iso")
        logger.info(f"Історandя withбережена у {path}")

    def plot_balance(self):
        df = pd.DataFrame(self.history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        plt.figure(figsize=(10, 4))
        plt.plot(df["timestamp"], df["balance_after"], label="Баланс")
        plt.title("Баланс у часand")
        plt.xlabel("Час")
        plt.ylabel("Баланс")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()