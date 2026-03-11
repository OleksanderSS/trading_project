# src/core/clients/http_client_factory.py

import asyncio
import time
from abc import ABC, abstractmethod
import httpx
from typing import List, Optional, Any, Dict
from src.config.unified_config_manager import UnifiedConfigManager
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
        retries: Optional[int] = None,
        status_forcelist: Optional[List[int]] = None,
        backoff_factor: Optional[float] = None,
        timeout: Optional[float] = None,
        user_agent: Optional[str] = None
    ) -> httpx.AsyncClient:
        """
        Creates and returns an httpx.AsyncClient.
        """
        pass

class HttpClientFactory(IHttpClientFactory):
    """
    A factory for creating and managing httpx.AsyncClient instances with Rate Limiting and Logging.
    """
    def __init__(self, config_manager: UnifiedConfigManager, error_handler: IErrorHandler):
        """
        Initializes the HttpClientFactory.

        Args:
            config_manager (UnifiedConfigManager): The application's config manager.
            error_handler (IErrorHandler): The application's error handler.
        """
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.client_config = self.config_manager.get('http_client', {})
        
        # Configure RateLimiter from http_client.yaml or use defaults
        rate_limiter_config = self.client_config.get('rate_limiter', {})
        rate_limit = rate_limiter_config.get('rate_limit', 10)
        per_seconds = rate_limiter_config.get('per_seconds', 1.0)
        self.rate_limiter = RateLimiter(rate_limit=rate_limit, per_seconds=per_seconds)

    def get_http_client(
        self,
        retries: Optional[int] = None,
        status_forcelist: Optional[List[int]] = None,
        backoff_factor: Optional[float] = None,
        timeout: Optional[float] = None,
        user_agent: Optional[str] = None
    ) -> httpx.AsyncClient:
        """
        Creates and returns an httpx.AsyncClient configured with a robust retry mechanism.
        """
        retries = retries if retries is not None else self.client_config.get('retries', 3)
        timeout = timeout if timeout is not None else self.client_config.get('timeout', 20.0)
        backoff_factor = backoff_factor if backoff_factor is not None else self.client_config.get('backoff_factor', 0.5)
        status_forcelist = status_forcelist or self.client_config.get('status_forcelist', [429, 500, 502, 503, 504])
        user_agent = user_agent if user_agent is not None else self.client_config.get('user_agent', 'TradingBot/2.0 (Unified Ecosystem)')

        # Use the built-in httpx retry mechanism
        transport = httpx.AsyncHTTPTransport(retries=retries)

        client = httpx.AsyncClient(
            transport=transport,
            timeout=timeout,
            follow_redirects=True,
            headers={'User-Agent': user_agent}
        )
        
        # Wrap the original send method to include rate limiting
        original_send = client.send
        
        async def rate_limited_send(*args, **kwargs):
            await self.rate_limiter.acquire_async()
            return await original_send(*args, **kwargs)
        
        client.send = rate_limited_send

        return client

    def get_session_client(self, **kwargs) -> httpx.AsyncClient:
        """
        Returns a configured client. Use as 'async with factory.get_session_client() as client:'.
        """
        return self.get_http_client(**kwargs)
