# src/core/clients/http_client_factory.py

import asyncio
import random
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse

import httpx

from src.config.unified_config_manager import UnifiedConfigManager, get_current_config
from src.core.error_handling.error_handler import IErrorHandler
from src.core.logging.logger import ProjectLogger
from src.utils.rate_limiter import RateLimiter

logger = ProjectLogger.get_logger("HttpClientFactory")

class IHttpClientFactory(ABC):
    """
    Interface for a factory that creates httpx.AsyncClient instances.
    """
    @abstractmethod
    def get_http_client(
        self,
        retries: int | None = None,
        status_forcelist: list[int] | None = None,
        backoff_factor: float | None = None,
        timeout: float | None = None,
        user_agent: str | None = None
    ) -> httpx.AsyncClient:
        """
        Creates and returns an httpx.AsyncClient.
        """
        pass

class HttpClientFactory(IHttpClientFactory):
    """
    A factory for creating and managing httpx.AsyncClient instances with Rate Limiting and Logging.
    """

    # Rate limiting has to be process-wide to mean anything. Five separate
    # HttpClientFactory instances were observed in a single pipeline run
    # (pipeline_factory, pipeline_orchestrator, base_stage's fallback,
    # collector_factory, auto_accumulator), each building its own RateLimiter
    # with its own full bucket -- so a "100 requests per second" limit was
    # really up to 500. These are shared per host so every factory in the
    # process draws from the same buckets.
    _limiters: dict[str, RateLimiter] = {}

    def __init__(
        self,
        config_manager: UnifiedConfigManager | None = None,
        error_handler: IErrorHandler | None = None,
    ):
        """
        Initializes the HttpClientFactory.

        Args:
            config_manager: The application's config manager. Defaults to the
                shared instance -- CollectorFactory calls HttpClientFactory()
                with no arguments, which used to raise TypeError.
            error_handler: Accepted for interface compatibility. This class
                stores it and has never used it.
        """
        self.config_manager = config_manager or get_current_config()
        self.error_handler = error_handler
        self.client_config = self.config_manager.get('http_client', {}) or {}

        rate_limiter_config = self.client_config.get('rate_limiter', {}) or {}
        self._default_rate_limit = rate_limiter_config.get('rate_limit', 10)
        self._default_per_seconds = rate_limiter_config.get('per_seconds', 1.0)
        # {host_substring: {rate_limit, per_seconds}} -- APIs differ by orders
        # of magnitude in what they tolerate, and one global number cannot
        # express that.
        self._per_host_limits: dict[str, dict[str, Any]] = (
            rate_limiter_config.get('per_host', {}) or {}
        )

    def _limiter_for(self, url: str | None) -> RateLimiter:
        """The shared limiter governing this host."""
        host = ''
        if url:
            try:
                host = (urlparse(str(url)).hostname or '').lower()
            except (ValueError, TypeError):
                host = ''

        key = 'default'
        rate_limit = self._default_rate_limit
        per_seconds = self._default_per_seconds
        for pattern, limits in self._per_host_limits.items():
            if pattern.lower() in host:
                key = pattern.lower()
                rate_limit = limits.get('rate_limit', rate_limit)
                per_seconds = limits.get('per_seconds', per_seconds)
                break

        limiter = HttpClientFactory._limiters.get(key)
        if limiter is None:
            limiter = RateLimiter(rate_limit=rate_limit, per_seconds=per_seconds)
            HttpClientFactory._limiters[key] = limiter
        return limiter

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        """A 429 usually says how long to wait. Obeying it beats guessing."""
        raw = response.headers.get('Retry-After')
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            # The header may also be an HTTP-date; not worth parsing here,
            # the caller falls back to exponential backoff.
            return None

    async def get_http_client(
        self,
        retries: int | None = None,
        status_forcelist: list[int] | None = None,
        backoff_factor: float | None = None,
        timeout: float | None = None,
        user_agent: str | None = None
    ) -> httpx.AsyncClient:
        """
        Creates and returns an httpx.AsyncClient configured with a robust retry mechanism.
        """
        retries = retries if retries is not None else self.client_config.get('retries', 3)
        timeout = timeout if timeout is not None else self.client_config.get('timeout', 20.0)
        backoff_factor = backoff_factor if backoff_factor is not None else self.client_config.get('backoff_factor', 0.5)
        status_forcelist = status_forcelist if status_forcelist is not None else self.client_config.get('status_forcelist', [429, 500, 502, 503, 504])
        user_agent = user_agent if user_agent is not None else self.client_config.get('user_agent', 'TradingBot/2.0 (Unified Ecosystem)')

        # httpx's own `retries` covers ONLY connection establishment: httpcore
        # retries on (ConnectError, ConnectTimeout) and nothing else. A 429 or
        # a 503 is a perfectly good response as far as it is concerned, so
        # status_forcelist and backoff_factor were computed here and then
        # never passed anywhere -- the factory named a retry-on-status policy
        # and implemented retry-on-connect-failure. Statuses are retried in
        # the send wrapper below.
        transport = httpx.AsyncHTTPTransport(retries=retries)

        client = httpx.AsyncClient(
            transport=transport,
            timeout=timeout,
            follow_redirects=True,
            headers={'User-Agent': user_agent}
        )
        original_send = client.send
        retry_statuses = set(status_forcelist or [])

        async def rate_limited_send(*args: Any, **kwargs: Any) -> httpx.Response:
            """Rate-limits, then retries retryable statuses with backoff."""
            request = args[0] if args else kwargs.get('request')
            url = getattr(request, 'url', None)
            limiter = self._limiter_for(str(url) if url is not None else None)

            response: httpx.Response | None = None
            for attempt in range(retries + 1):
                await limiter.acquire_async()
                response = await original_send(*args, **kwargs)

                if response.status_code not in retry_statuses or attempt == retries:
                    return response

                delay = self._retry_after_seconds(response)
                if delay is None:
                    # Exponential, with jitter so parallel collectors hitting
                    # the same API do not all come back at the same instant.
                    delay = backoff_factor * (2 ** attempt) + random.uniform(0, backoff_factor)

                logger.warning(
                    "HTTP %s from %s; retrying in %.2fs (attempt %d/%d)",
                    response.status_code, url, delay, attempt + 1, retries,
                )
                await response.aclose()
                await asyncio.sleep(delay)

            return response  # loop always returns or exhausts into this

        client.send = rate_limited_send

        return client

    async def get_session_client(self, **kwargs) -> httpx.AsyncClient:
        """
        Returns a configured client. Use as 'async with await factory.get_session_client() as client:'.
        """
        return await self.get_http_client(**kwargs)
