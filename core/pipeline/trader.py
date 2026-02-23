# core/pipeline/trader.py

import os
import datetime
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("TradingProjectLogger")


class DummyTraderCSV:
    def __init__(self,
                 initial_balance: float = 10000.0,
                 csv_path: Optional[str] = None,
                 risk_fraction: float = 1.0,
                 batch_size: int = 20,
                 fail_fast: bool = True):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.history: List[Dict[str, Any]] = []
        self.csv_path = csv_path
        self.risk_fraction = risk_fraction
        self.batch_size = batch_size
        self.buffer: List[Dict[str, Any]] = []
        self.fail_fast = fail_fast

        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            if not os.path.exists(self.csv_path):
                pd.DataFrame(columns=[
                    'time', 'ticker', 'signal', 'ensemble_signal',
                    'price', 'qty', 'balance', 'pnl', 'extra_signals'
                ]).to_csv(self.csv_path, index=False)
                logger.info(f"[Trader] Created CSV at {self.csv_path}")

    def flush(self):
        if self.csv_path and self.buffer:
            try:
                pd.DataFrame(self.buffer).to_csv(self.csv_path, mode='a', header=False, index=False)
                self.buffer.clear()
                logger.info(f"[Trader] Flushed buffer to CSV")
            except Exception as e:
                if self.fail_fast:
                    raise e
                else:
                    logger.warning(f"[Trader] Failed to flush buffer: {e}")

    def execute_trade(self, ticker: str, signal: Any, price: float, extra_signals: Optional[Dict] = None):
        if not isinstance(price, (int, float)) or price <= 0:
            msg = f"Invalid price for ticker {ticker}: {price}"
            if self.fail_fast:
                raise ValueError(msg)
            logger.warning(f"[Trader] {msg}")
            return

        trade_signal = "HOLD"
        ensemble_signal = "HOLD"
        if isinstance(signal, dict):
            trade_signal = signal.get("final_signal", "HOLD")
            ensemble_signal = signal.get("ensemble", "HOLD")
        else:
            trade_signal = signal

        if trade_signal not in {"BUY", "SELL", "HOLD"}:
            msg = f"Unknown signal '{trade_signal}' for {ticker}"
            if self.fail_fast:
                raise ValueError(msg)
            logger.warning(f"[Trader] {msg}")
            return

        trade_record = {
            'time': datetime.datetime.now().isoformat(),
            'ticker': ticker,
            'signal': trade_signal,
            'ensemble_signal': ensemble_signal,
            'price': price,
            'qty': 0.0,
            'balance': self.balance,
            'pnl': 0.0,
            'extra_signals': extra_signals or {}
        }

        if trade_signal == 'BUY':
            if self.balance <= 0:
                msg = f"No balance to BUY {ticker}"
                if self.fail_fast:
                    raise ValueError(msg)
                logger.warning(f"[Trader] {msg}")
                return
            qty = (self.balance * self.risk_fraction) / price
            self.balance -= qty * price
            trade_record['qty'] = qty
            logger.info(f"[Trader] BUY {ticker} | Qty: {qty:.4f} | Price: {price:.2f} | Balance: {self.balance:.2f}")

        elif trade_signal == 'SELL':
            positions = [h for h in reversed(self.history)
                         if h['ticker'] == ticker and h['signal'] == 'BUY' and h['qty'] > 0]
            if not positions:
                msg = f"No open position to SELL {ticker}"
                if self.fail_fast:
                    raise ValueError(msg)
                logger.warning(f"[Trader] {msg}")
                return

            qty_to_sell = sum(h['qty'] for h in positions)
            avg_price = np.average([h['price'] for h in positions], weights=[h['qty'] for h in positions])
            pnl = (price - avg_price) * qty_to_sell
            self.balance += qty_to_sell * price

            for h in positions:
                h['qty'] = 0

            trade_record['qty'] = qty_to_sell
            trade_record['pnl'] = pnl
            trade_record['balance'] = self.balance
            logger.info(f"[Trader] SELL {ticker} | Qty: {qty_to_sell:.4f} | Price: {price:.2f} | PnL: {pnl:.2f} | Balance: {self.balance:.2f}")

        self.history.append(trade_record)

        if self.csv_path:
            self.buffer.append(trade_record)
            if len(self.buffer) >= self.batch_size:
                self.flush()

    def update(self, ticker: str, signal: Any, price: Optional[float] = None, extra_signals: Optional[Dict] = None):
        if price is None:
            last_price = next((h['price'] for h in reversed(self.history) if h['ticker'] == ticker), None)
            if last_price is None:
                msg = f"No price provided and no history for {ticker}"
                if self.fail_fast:
                    raise ValueError(msg)
                logger.warning(f"[Trader] {msg}")
                return
            price = last_price

        self.execute_trade(ticker, signal, price, extra_signals)

    def get_total_pnl(self) -> float:
        """Поверandє сумарний PnL по allх угодах"""
        return sum(h.get('pnl', 0.0) for h in self.history)

    def close(self):
        """Закриває трейwhereра and withберandгає буфер у CSV"""
        self.flush()
        logger.info("[Trader] Trader closed, buffer flushed")