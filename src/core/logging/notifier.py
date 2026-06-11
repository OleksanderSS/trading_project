import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp

logger = logging.getLogger(__name__)

class UniversalNotifier:
    """
    Universal notification hub supporting Telegram, Discord, and local logging.
    Credentials and settings are fetched from UnifiedConfigManager.
    """

    def __init__(self, config_manager: Any):
        self.config_manager = config_manager
        self.notify_config = self.config_manager.get_config('notifications', default={})
        self.enabled_backends = self.notify_config.get('enabled_backends', ['log'])

        # Secrets/Tokens from secure storage
        self.tokens = self.config_manager.get_config('secrets', default={})

    async def send_message(self, message: str, level: str = "INFO"):
        """Sends a text message to all enabled backends."""
        formatted_msg = f"[{level}] {message}"

        tasks = []
        if 'log' in self.enabled_backends:
            logger.info(f"NOTIFIER: {formatted_msg}")

        if 'telegram' in self.enabled_backends:
            tasks.append(self._send_telegram(message))

        if 'discord' in self.enabled_backends:
            tasks.append(self._send_discord(message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send_report(self, message: str, image_path: str | None = None):
        """Sends a text message along with an optional image (e.g., Equity Curve)."""
        tasks = []

        if 'telegram' in self.enabled_backends:
            tasks.append(self._send_telegram(message, image_path))

        if 'discord' in self.enabled_backends:
            tasks.append(self._send_discord(message, image_path))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_telegram(self, message: str, image_path: str | None = None):
        """Telegram-specific delivery logic."""
        bot_token = self.tokens.get('telegram_token')
        chat_id = self.tokens.get('telegram_chat_id')

        if not bot_token or not chat_id:
            logger.warning("Telegram credentials missing in config.")
            return

        url_base = f"https://api.telegram.org/bot{bot_token}"

        try:
            async with aiohttp.ClientSession() as session:
                if image_path and os.path.exists(image_path):
                    url = f"{url_base}/sendPhoto"
                    data = aiohttp.FormData()
                    data.add_field('chat_id', str(chat_id))
                    data.add_field('caption', message)
                    data.add_field('photo', await aiofiles.open(image_path, 'rb').read())
                    async with session.post(url, data=data) as resp:
                        if resp.status != 200:
                            logger.error(f"Telegram photo failed: {await resp.text()}")
                else:
                    url = f"{url_base}/sendMessage"
                    payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'}
                    async with session.post(url, json=payload) as resp:
                        if resp.status != 200:
                            logger.error(f"Telegram message failed: {await resp.text()}")
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Error sending to Telegram: {e}")

    async def _send_discord(self, message: str, image_path: str | None = None):
        """Discord-specific delivery logic using Webhooks."""
        webhook_url = self.tokens.get('discord_webhook_url')

        if not webhook_url:
            logger.warning("Discord webhook URL missing in config.")
            return

        try:
            async with aiohttp.ClientSession() as session:
                if image_path and os.path.exists(image_path):
                    data = aiohttp.FormData()
                    data.add_field('payload_json', '{"content": "' + message + '"}')
                    data.add_field('file', await aiofiles.open(image_path, 'rb').read(), filename=Path(image_path).name)
                    async with session.post(webhook_url, data=data) as resp:
                        if resp.status not in [200, 204]:
                            logger.error(f"Discord file failed: {await resp.text()}")
                else:
                    payload = {"content": message}
                    async with session.post(webhook_url, json=payload) as resp:
                        if resp.status not in [200, 204]:
                            logger.error(f"Discord message failed: {await resp.text()}")
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Error sending to Discord: {e}")

    def sync_send(self, message: str, level: str = "INFO"):
        """Synchronous wrapper for async send_message."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.send_message(message, level))
            else:
                loop.run_until_complete(self.send_message(message, level))
        except RuntimeError:
            asyncio.run(self.send_message(message, level))
