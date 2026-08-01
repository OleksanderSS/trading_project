import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp

logger = logging.getLogger(__name__)

# Notification level -> logging level. The 'log' backend used to emit
# everything at INFO, so a CRITICAL alert raised by ErrorHandler was written
# at the same level as routine chatter and disappeared under any filter above
# INFO -- in the one place whose whole job is to make failures visible.
_LEVELS = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
}

# Delivery failures are network failures. aiohttp.ClientError and
# asyncio.TimeoutError both inherit straight from Exception, so the project's
# usual (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError)
# tuple never caught either of them.
_DELIVERY_ERRORS = (
    aiohttp.ClientError,
    asyncio.TimeoutError,
    OSError,
    ValueError,
    TypeError,
)


class UniversalNotifier:
    """
    Universal notification hub supporting Telegram, Discord, and local logging.

    Settings come from the `notifications` config section; credentials come
    from the environment (loaded from .env by SecureSecretsManager), the same
    way every collector reads its API key. They are deliberately NOT read from
    a YAML section -- config files are committed.
    """

    def __init__(self, config_manager: Any | None = None):
        if config_manager is None:
            # data_freshness_monitor constructs this with no arguments, which
            # used to raise TypeError.
            from src.config.unified_config_manager import get_current_config
            config_manager = get_current_config()

        self.config_manager = config_manager
        self.notify_config = self.config_manager.get_config('notifications', default={}) or {}
        self.enabled_backends = self.notify_config.get('enabled_backends', ['log'])

        # Was config_manager.get_config('secrets'), a section that exists in
        # no YAML and should not: tokens belong in the environment.
        # TELEGRAM_TOKEN, not TELEGRAM_BOT_TOKEN: that is the name
        # SecretsManager.FORMAT_PATTERNS validates and log_secrets_status
        # reports on. Two names for one credential is how a key gets set and
        # still appears missing.
        self.tokens = {
            'telegram_token': os.getenv('TELEGRAM_TOKEN'),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
            'discord_webhook_url': os.getenv('DISCORD_WEBHOOK_URL'),
        }
        self._pending: set[asyncio.Task] = set()

    async def _deliver(self, tasks: list, what: str) -> None:
        """Run delivery coroutines and report the ones that failed.

        gather(return_exceptions=True) turns a failure into a return value;
        without inspecting them, a notification that never arrived looked
        exactly like one that did.
        """
        if not tasks:
            return

        names = [getattr(task, '__qualname__', 'backend') for task in tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for backend, result in zip(names, results):
            if isinstance(result, BaseException):
                logger.error("Notification (%s) failed via %s: %s", what, backend, result)

    async def send_message(self, message: str, level: str = "INFO"):
        """Sends a text message to all enabled backends."""
        formatted_msg = f"[{level}] {message}"

        tasks = []
        if 'log' in self.enabled_backends:
            logger.log(_LEVELS.get(level.upper(), logging.INFO), f"NOTIFIER: {formatted_msg}")

        if 'telegram' in self.enabled_backends:
            tasks.append(self._send_telegram(message))

        if 'discord' in self.enabled_backends:
            tasks.append(self._send_discord(message))

        await self._deliver(tasks, 'message')

    async def send_report(self, message: str, image_path: str | None = None):
        """Sends a text message along with an optional image (e.g., Equity Curve)."""
        tasks = []

        if 'telegram' in self.enabled_backends:
            tasks.append(self._send_telegram(message, image_path))

        if 'discord' in self.enabled_backends:
            tasks.append(self._send_discord(message, image_path))

        await self._deliver(tasks, 'report')

    @staticmethod
    async def _read_image(image_path: str) -> bytes:
        """Read an attachment.

        This was `await aiofiles.open(path, 'rb').read()`, which calls .read()
        on the context manager rather than on the file and raises
        AttributeError -- caught by the handler below and logged as a delivery
        failure, so every report WITH an image silently arrived without one.
        """
        async with aiofiles.open(image_path, 'rb') as handle:
            return await handle.read()

    async def _send_telegram(self, message: str, image_path: str | None = None):
        """Telegram-specific delivery logic."""
        bot_token = self.tokens.get('telegram_token')
        chat_id = self.tokens.get('telegram_chat_id')

        if not bot_token or not chat_id:
            logger.warning(
                "Telegram notifications are enabled but TELEGRAM_TOKEN / "
                "TELEGRAM_CHAT_ID are not set in the environment."
            )
            return

        url_base = f"https://api.telegram.org/bot{bot_token}"

        try:
            async with aiohttp.ClientSession() as session:
                if image_path and os.path.exists(image_path):
                    url = f"{url_base}/sendPhoto"
                    data = aiohttp.FormData()
                    data.add_field('chat_id', str(chat_id))
                    data.add_field('caption', message)
                    data.add_field('photo', await self._read_image(image_path))
                    async with session.post(url, data=data) as resp:
                        if resp.status != 200:
                            logger.error(f"Telegram photo failed: {await resp.text()}")
                else:
                    url = f"{url_base}/sendMessage"
                    payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'}
                    async with session.post(url, json=payload) as resp:
                        if resp.status != 200:
                            logger.error(f"Telegram message failed: {await resp.text()}")
        except _DELIVERY_ERRORS as e:
            logger.exception(f"Error sending to Telegram: {e}")

    async def _send_discord(self, message: str, image_path: str | None = None):
        """Discord-specific delivery logic using Webhooks."""
        webhook_url = self.tokens.get('discord_webhook_url')

        if not webhook_url:
            logger.warning(
                "Discord notifications are enabled but DISCORD_WEBHOOK_URL is "
                "not set in the environment."
            )
            return

        try:
            async with aiohttp.ClientSession() as session:
                if image_path and os.path.exists(image_path):
                    data = aiohttp.FormData()
                    data.add_field('payload_json', '{"content": "' + message + '"}')
                    data.add_field(
                        'file',
                        await self._read_image(image_path),
                        filename=Path(image_path).name,
                    )
                    async with session.post(webhook_url, data=data) as resp:
                        if resp.status not in [200, 204]:
                            logger.error(f"Discord file failed: {await resp.text()}")
                else:
                    payload = {"content": message}
                    async with session.post(webhook_url, json=payload) as resp:
                        if resp.status not in [200, 204]:
                            logger.error(f"Discord message failed: {await resp.text()}")
        except _DELIVERY_ERRORS as e:
            logger.exception(f"Error sending to Discord: {e}")

    def sync_send(self, message: str, level: str = "INFO"):
        """Synchronous wrapper for async send_message.

        ErrorHandler calls this from wherever an error happened, which
        includes worker threads. Scheduling onto a loop owned by another
        thread requires run_coroutine_threadsafe; create_task from a foreign
        thread is not safe. A reference to the task is also kept, because a
        bare create_task result can be garbage-collected before it runs.
        """
        coroutine = self.send_message(message, level)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is not None:
            task = running.create_task(coroutine)
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
            return

        try:
            asyncio.run(coroutine)
        except RuntimeError as e:
            # A loop exists in this thread but is not running, or the
            # interpreter is shutting down. Say so rather than dropping the
            # notification without trace.
            logger.error("Could not deliver notification synchronously: %s", e)
            coroutine.close()
