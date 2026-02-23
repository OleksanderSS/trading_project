# utils/news_harmonizer.py

import pandas as pd
from utils.logger import ProjectLogger
from config.news_config import NEWS_FIELDS


logger = ProjectLogger.get_logger("TradingProjectLogger")

def detect_news_format(entry: dict) -> dict:
    date_fields = NEWS_FIELDS["date"]
    text_fields = NEWS_FIELDS["text"]

    date_field = next((f for f in date_fields if f in entry and entry[f]), None)
    text_field = next((f for f in text_fields if f in entry and entry[f]), None)

    logger.debug(f"[news_harmonizer] Викорисandно поля: date={date_field}, text={text_field}")
    return {
        "date_field": date_field or "unknown",
        "text_field": text_field or "unknown"
    }

def harmonize_entry(entry: dict, source: str, default_type: str = "qualitative") -> dict:
    format_info = detect_news_format(entry)

    raw_date = entry.get(format_info["date_field"])
    if format_info["date_field"] == "created_utc":
        pub_dt = pd.to_datetime(raw_date, unit="s", errors="coerce")
    else:
        pub_dt = pd.to_datetime(raw_date, errors="coerce")

    if pd.isna(pub_dt):
        # ВИПРАВЛЕНО - зменшено рівень логування для частої проблеми
        logger.debug("[news_harmonizer] [DEBUG] Некоректна дата, використано поточний час")
        pub_dt = pd.Timestamp.now()
    elif pub_dt.tzinfo:
        pub_dt = pub_dt.tz_convert(None)

    title = str(entry.get("title", "")).strip()
    summary = str(entry.get("summary", "")).strip()
    body = str(entry.get(format_info["text_field"], "")).strip()

    if summary:
        description = f"{title}. {summary}".strip()
    else:
        description = title or body

    if not description:
        description = str(entry.get("theme", "")) or str(entry.get("source", "")) or "No description"

    url = entry.get("link") or entry.get("url") or "unknown"

    return {
        "published_at": pub_dt,
        "description": description,
        "type": entry.get("type", default_type),
        "value": entry.get("value", None),
        "sentiment": entry.get("sentiment", None),
        "source": source,
        "url": url
    }

def harmonize_batch(entries: list, source: str, default_type: str = "qualitative") -> list:
    harmonized = []
    for entry in entries:
        item = harmonize_entry(entry, source, default_type)
        if item["description"]:
            harmonized.append(item)
        else:
            logger.warning("[news_harmonizer] [WARN] Пропущено forпис беwith опису")
    logger.info(f"[news_harmonizer] [OK] Гармонandwithовано {len(harmonized)} forписandв andwith {len(entries)}")
    return harmonized