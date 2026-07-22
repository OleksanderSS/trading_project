"""
dean_os/evals/time_leakage_guard.py

Перевірка витоку часу (time leakage). Виявляє, чи не використовуються майбутні джерела
в аналізі станом на певну дату (as_of). Відповідає шаблону TIME_LEAKAGE_TESTS з Codex Phase 6.
"""
from __future__ import annotations

import datetime
from typing import Any


class TimeLeak(Exception):
    """Виняток, що сигналізує про виявлений витік часу."""
    pass


def parse_date(date_str: str | None) -> datetime.date | None:
    if not date_str:
        return None
    try:
        return datetime.date.fromisoformat(str(date_str)[:10])
    except ValueError:
        return None


class TimeLeakageGuard:
    """
    Захист від витоку часу.

    Перевіряє, що жоден документ або подія не містять дату публікації/події
    пізніше дати as_of — тобто не використовується "майбутня" інформація
    у ретроспективному аналізі.
    """

    def __init__(self, as_of: str):
        self.as_of_date = parse_date(as_of)
        self.violations: list[dict[str, Any]] = []

    def check_item(self, item_id: str, publication_date: str | None, event_date: str | None) -> bool:
        """
        Перевіряє один запис. Повертає True якщо чисто, False якщо є витік.
        Додає знайдене порушення до self.violations.
        """
        if self.as_of_date is None:
            return True  # не можемо перевірити без опорної дати

        pub = parse_date(publication_date)
        evt = parse_date(event_date)

        leaks = []
        if pub and pub > self.as_of_date:
            leaks.append(f"publication_date={pub} > as_of={self.as_of_date}")
        if evt and evt > self.as_of_date:
            leaks.append(f"event_date={evt} > as_of={self.as_of_date}")

        if leaks:
            self.violations.append({"item_id": item_id, "leaks": leaks})
            return False
        return True

    def check_news_list(self, news: list[dict]) -> int:
        """
        Перевіряє список новинних подій. Повертає кількість знайдених порушень.
        Очікує, що кожен елемент має поля: id, publication_date (опціонально), event_date (опціонально).
        """
        count = 0
        for item in news:
            ok = self.check_item(
                item_id=item.get("id", "unknown"),
                publication_date=item.get("publication_date"),
                event_date=item.get("event_date"),
            )
            if not ok:
                count += 1
        return count

    def summary(self) -> dict:
        return {
            "as_of": str(self.as_of_date),
            "violations_count": len(self.violations),
            "is_clean": len(self.violations) == 0,
            "violations": self.violations,
        }
