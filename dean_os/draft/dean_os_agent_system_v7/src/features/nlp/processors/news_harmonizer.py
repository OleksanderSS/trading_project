import logging

import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('TradingProjectLogger')


def _get_news_fields():
    """Lazy load news fields configuration."""
    try:
        config = get_current_config().get_config('enrichment')
        return config.get('news_fields', {})
    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.exception(f'Виникла помилка: {e}')
        logger.warning(
            f'[news_harmonizer] Failed to load news fields config: {e}')
        raise RuntimeError("Failed to load news fields configuration") from e


def detect_news_format(entry: dict) ->dict:
    news_fields = _get_news_fields()
    date_fields = news_fields.get('date', [])
    text_fields = news_fields.get('text', [])
    date_field = next((f for f in date_fields if f in entry and entry[f]), None
        )
    text_field = next((f for f in text_fields if f in entry and entry[f]), None
        )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f'[news_harmonizer] Using fields: date={date_field}, text={text_field}'
            )
    return {'date_field': date_field or 'unknown', 'text_field': text_field or
        'unknown'}


def harmonize_entry(entry: dict, source: str, default_type: str='qualitative'
    ) ->dict:
    format_info = detect_news_format(entry)
    raw_date = entry.get(format_info['date_field'])
    if format_info['date_field'] == 'created_utc':
        pub_dt = pd.to_datetime(raw_date, unit='s', errors='coerce')
    else:
        pub_dt = pd.to_datetime(raw_date, errors='coerce')
    if pd.isna(pub_dt):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                '[news_harmonizer] [DEBUG] Invalid date, using current time')
        pub_dt = pd.Timestamp.now()
    elif pub_dt.tzinfo:
        pub_dt = pub_dt.tz_convert(None)
    title = str(entry.get('title', '')).strip()
    summary = str(entry.get('summary', '')).strip()
    body = str(entry.get(format_info['text_field'], '')).strip()
    if summary:
        description = f'{title}. {summary}'.strip()
    else:
        description = title or body
    if not description:
        description = str(entry.get('theme', '')) or str(entry.get('source',
            '')) or 'No description'
    url = entry.get('link') or entry.get('url') or 'unknown'
    return {'published_at': pub_dt, 'description': description, 'type':
        entry.get('type', default_type), 'value': entry.get('value', None),
        'sentiment': entry.get('sentiment', None), 'source': source, 'url': url
        }


def harmonize_batch(entries: list, source: str, default_type: str='qualitative'
    ) ->list:
    harmonized = []
    for entry in entries:
        item = harmonize_entry(entry, source, default_type)
        if item['description']:
            harmonized.append(item)
        else:
            logger.warning(
                '[news_harmonizer] [WARN] Skipping entry without description')
    logger.info(
        f'[news_harmonizer] [OK] Harmonized {len(harmonized)} of {len(entries)} entries'
        )
    return harmonized
