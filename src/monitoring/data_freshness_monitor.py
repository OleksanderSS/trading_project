"""
Моніторинг свіжості даних
"""

from datetime import datetime, timedelta
from typing import Any


class DataFreshnessMonitor:
    """Моніторинг свіжості даних"""

    def __init__(self, max_age_hours: int = 24):
        self.max_age_hours = max_age_hours
        self.last_update_times = {}

    def update_timestamp(self, data_source: str, timestamp: datetime = None):
        """Оновлення часу останнього оновлення"""
        if timestamp is None:
            timestamp = datetime.now()
        self.last_update_times[data_source] = timestamp

    def is_data_fresh(self, data_source: str) -> bool:
        """Перевірка чи дані свіжі"""
        if data_source not in self.last_update_times:
            return False

        last_update = self.last_update_times[data_source]
        max_age = datetime.now() - timedelta(hours=self.max_age_hours)

        return last_update > max_age

    def get_freshness_status(self) -> dict[str, dict[str, Any]]:
        """Отримати статус свіжості для всіх джерел"""
        status = {}

        for source, last_update in self.last_update_times.items():
            is_fresh = self.is_data_fresh(source)
            age_hours = (datetime.now() - last_update).total_seconds() / 3600

            status[source] = {
                'last_update': last_update,
                'is_fresh': is_fresh,
                'age_hours': age_hours,
                'max_age_hours': self.max_age_hours
            }

        return status

    def get_stale_sources(self) -> list[str]:
        """Отримати список застарілих джерел"""
        stale = []

        for source in self.last_update_times:
            if not self.is_data_fresh(source):
                stale.append(source)

        return stale

# Глобальний екземпляр
_global_monitor = None

def get_data_freshness_monitor(max_age_hours: int = 24) -> DataFreshnessMonitor:
    """Отримати глобальний монітор свіжості даних"""
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = DataFreshnessMonitor(max_age_hours)

    return _global_monitor

def check_freshness_quick(data_source: str, max_age_hours: int = 24) -> bool:
    """Швидка перевірка свіжості даних

    Args:
        data_source: Назва джерела даних
        max_age_hours: Максимальний вік даних у годинах

    Returns:
        True якщо дані свіжі, False якщо застарілі
    """
    try:
        monitor = get_data_freshness_monitor(max_age_hours)
        return monitor.is_data_fresh(data_source)
    except Exception:
        # Якщо щось пішло не так, повертаємо True (вважаємо що дані свіжі)
        return True
