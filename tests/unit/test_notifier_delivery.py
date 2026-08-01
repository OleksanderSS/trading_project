"""The channel that reports failures must not fail silently itself.

ErrorHandler routes every critical and error through UniversalNotifier. Four
defects, each verified against the real code:

1. The 'log' backend wrote everything with logger.info, so a CRITICAL alert
   was recorded at INFO -- invisible under any filter above INFO, in the one
   component whose job is visibility.

2. Attachments were read with `await aiofiles.open(path, 'rb').read()`, which
   calls .read() on the context manager, not the file:
   AttributeError: 'AiofilesContextManager' object has no attribute 'read'.
   AttributeError IS in the project's five-tuple, so it was caught and logged
   as a delivery failure -- every report with an image arrived without one.

3. Delivery errors are network errors, and aiohttp.ClientError and
   asyncio.TimeoutError both inherit straight from Exception, so that same
   five-tuple caught neither. gather(return_exceptions=True) then turned the
   escape into a return value nobody inspected: a notification that never
   arrived looked exactly like one that did.

4. UniversalNotifier() with no arguments raised TypeError, which is how
   data_freshness_monitor.py:380 constructs it.

Credentials come from the environment, as in every collector. They used to be
read from a YAML section named `secrets` that exists in no file -- and should
not, since config files are committed.
"""
from __future__ import annotations

import asyncio
import logging

import aiohttp
import pytest

from src.core.logging.notifier import UniversalNotifier


@pytest.fixture()
def notifier(monkeypatch):
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    return UniversalNotifier()


def test_constructs_with_no_arguments():
    """data_freshness_monitor does exactly this."""
    assert UniversalNotifier() is not None


def test_credentials_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "token-from-env")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    assert UniversalNotifier().tokens["telegram_token"] == "token-from-env"


@pytest.mark.parametrize("level,expected", [
    ("CRITICAL", logging.CRITICAL),
    ("ERROR", logging.ERROR),
    ("WARNING", logging.WARNING),
    ("INFO", logging.INFO),
])
def test_the_log_backend_uses_the_severity_it_was_given(notifier, caplog, level, expected):
    with caplog.at_level(logging.DEBUG):
        asyncio.run(notifier.send_message("something happened", level=level))

    records = [r for r in caplog.records if "NOTIFIER" in r.getMessage()]
    assert records, "nothing was logged at all"
    assert records[-1].levelno == expected


def test_an_unknown_level_still_gets_logged(notifier, caplog):
    with caplog.at_level(logging.DEBUG):
        asyncio.run(notifier.send_message("odd", level="LOUD"))

    assert any("NOTIFIER" in r.getMessage() for r in caplog.records)


def test_an_attachment_can_actually_be_read(tmp_path):
    """The exact expression that used to raise AttributeError."""
    payload = b"\x89PNG\r\n\x1a\n fake equity curve"
    image = tmp_path / "equity.png"
    image.write_bytes(payload)

    assert asyncio.run(UniversalNotifier._read_image(str(image))) == payload


def test_missing_credentials_are_reported_not_silently_skipped(notifier, caplog):
    notifier.enabled_backends = ["telegram"]

    with caplog.at_level(logging.WARNING):
        asyncio.run(notifier.send_message("hello"))

    assert any("TELEGRAM_TOKEN" in r.getMessage() for r in caplog.records)


def test_a_delivery_failure_is_reported(notifier, caplog):
    """A backend that raised used to vanish into gather(return_exceptions)."""
    async def failing(*_args, **_kwargs):
        raise aiohttp.ClientError("connection reset")

    notifier.enabled_backends = ["telegram"]
    notifier._send_telegram = failing

    with caplog.at_level(logging.ERROR):
        asyncio.run(notifier.send_message("hello"))

    assert any("failed via" in r.getMessage() for r in caplog.records), (
        "a notification that never arrived looked exactly like one that did"
    )


def test_aiohttp_errors_are_inside_the_handler_now():
    """aiohttp.ClientError and asyncio.TimeoutError inherit from Exception,
    not from ValueError, so the old five-tuple caught neither."""
    from src.core.logging.notifier import _DELIVERY_ERRORS

    assert issubclass(aiohttp.ClientError, _DELIVERY_ERRORS)
    assert issubclass(asyncio.TimeoutError, _DELIVERY_ERRORS)


def test_sync_send_works_outside_an_event_loop(notifier, caplog):
    with caplog.at_level(logging.DEBUG):
        notifier.sync_send("from sync context", level="ERROR")

    assert any("NOTIFIER" in r.getMessage() for r in caplog.records)


def test_sync_send_inside_a_running_loop_keeps_the_task_alive(notifier, caplog):
    """A bare create_task result can be garbage collected before it runs."""
    async def scenario():
        notifier.sync_send("from async context", level="ERROR")
        assert notifier._pending, "the scheduled task was not retained"
        await asyncio.gather(*list(notifier._pending))

    with caplog.at_level(logging.DEBUG):
        asyncio.run(scenario())

    assert any("from async context" in r.getMessage() for r in caplog.records)


def test_completed_tasks_do_not_accumulate(notifier):
    async def scenario():
        for _ in range(5):
            notifier.sync_send("x")
        await asyncio.gather(*list(notifier._pending))

    asyncio.run(scenario())
    assert not notifier._pending, "finished tasks were never released"
