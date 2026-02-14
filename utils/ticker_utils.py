# utils/ticker_utils.py

import re
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("TradingProjectLogger")

def extract_ticker(keywords, tickers, default_ticker="GENERAL"):
    """
    Виwithначає тикер на основand keywords.
    :param keywords: список keywords (for example, with новин)
    :param tickers: список доступних тикерandв
    :param default_ticker: тикер for forмовчуванням, якщо withбandгandв notмає
    :return: withнайwhereний тикер or default_ticker
    """
    if not keywords or not tickers:
        logger.warning("[ticker_utils] [WARN] Порожнand keywords or tickers, поверandємо default")
        return default_ticker

    ticker_map = {t.lower(): t for t in tickers}
    for kw in keywords:
        kw_lower = kw.lower()
        for t_lower, t_original in ticker_map.items():
            if t_lower in kw_lower or re.search(rf"\b{t_lower}\b", kw_lower):
                logger.info(f"[ticker_utils] [OK] Found тикер '{t_original}' у ключовому словand '{kw}'")
                return t_original

    logger.info(f"[ticker_utils]  Збandгandв not withнайwhereно, поверandємо {default_ticker}")
    return default_ticker