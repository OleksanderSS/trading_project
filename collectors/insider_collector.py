# collectors/insider_collector.py

import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import logging
from collectors.base_collector import BaseCollector
from utils.news_harmonizer import harmonize_batch

logger = logging.getLogger("trading_project.insider_collector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
logger.propagate = False


class InsiderCollector(BaseCollector):
    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        min_usd: float = 100_000.0,
        roles: Optional[List[str]] = None,  # ["CEO","CFO","CTO","Chair","Director"]
        table_name: str = "insider_trades",
        db_path: str = ":memory:",
        strict: bool = True,
        **kwargs
    ):
        super().__init__(db_path=db_path, table_name=table_name, strict=strict, **kwargs)
        self.tickers = [t.upper() for t in (tickers or [])]
        self.min_usd = float(min_usd)
        self.roles = roles or ["CEO", "CFO", "CTO", "Chair", "Director"]

    def _parse_openinsider_table(self, url: str) -> List[Dict[str, Any]]:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            table = soup.find("table")  # беремо першу andблицю на сторandнцand
            if not table:
                logger.warning(f"[InsiderCollector] Не withнайwhereно andблицю на {url}")
                return []
            rows = table.select("tr")[1:]  # пропускаємо forголовок
            entries = []

            for tr in rows:
                td = [t.get_text(strip=True) for t in tr.find_all("td")]
                if len(td) < 10:
                    continue

                insider_name = td[1]
                role = td[2]
                ticker = td[3]
                trade_date = td[5]
                trade_type = td[6]  # "P" (Purchase) or "S" (Sale)
                shares_str = td[7].replace(",", "")
                price_str = td[8].replace("$", "").replace(",", "")
                try:
                    shares = float(shares_str) if shares_str else 0.0
                    price = float(price_str) if price_str else 0.0
                except Exception:
                    shares, price = 0.0, 0.0
                total_value = shares * price

                # Фandльтри
                if total_value < self.min_usd:
                    continue
                if self.roles and not any(r.lower() in role.lower() for r in self.roles):
                    continue
                if self.tickers and ticker.upper() not in self.tickers:
                    continue

                title = "Insider Buy" if trade_type.upper().startswith("P") else "Insider Sell"
                description = (
                    f"{insider_name} ({role}) {title.lower()} {int(shares)} {ticker} "
                    f"at ${price:.2f} (~${total_value:,.0f})"
                )

                entries.append({
                    "title": title,
                    "description": description,
                    "summary": role,
                    "published_at": trade_date,
                    "url": url,
                    "type": "insider",
                    "source": "OpenInsider",
                    "value": total_value,
                    "sentiment": None,
                    "result": "buy" if title == "Insider Buy" else "sell",
                    "ticker": ticker,
                    "raw_data_fields": {"row": td}
                })

            return entries

        except Exception as e:
            logger.warning(f"[InsiderCollector] OpenInsider parse error for {url}: {e}")
            return []

    def fetch(self) -> pd.DataFrame:
        # Актуальнand робочand фandди OpenInsider
        urls = [
            "http://openinsider.com/top-insider-purchases",
            "http://openinsider.com/top-insider-sales",
            "http://openinsider.com/latest-insider-trading"
        ]
        all_entries: List[Dict[str, Any]] = []
        for u in urls:
            all_entries.extend(self._parse_openinsider_table(u))

        if not all_entries:
            # ВИПРАВЛЕНО - зменшено рівень логування
            logger.info("[InsiderCollector] No entries fetched from OpenInsider (normal for some periods)")
            return pd.DataFrame()

        df = pd.DataFrame(harmonize_batch(all_entries, source="OpenInsider"))
        return df

    def collect(self) -> List[Dict[str, Any]]:
        df = self.fetch()
        if df.empty:
            return []
        records = df.to_dict(orient="records")
        self.save(records, strict=self.strict)
        return records

    def collect_data(self) -> pd.DataFrame:
        """Метод для сумісності з іншими колекторами"""
        return self.fetch()