from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class LLMClient:
    """Bounded optional client for schema-validated analysis proposals.

    The deterministic pipeline must not depend on this client.  Both an API
    key and an explicit model are required before a request can be made.  A
    missing dependency, credential, model, refusal, or failed request returns
    ``None``; it never manufactures a substitute analysis.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        timeout_seconds: float = 30.0,
        max_input_chars: int = 50_000,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_MODEL")
        self.max_input_chars = max_input_chars
        self.unavailable_reason: str | None = None

        if client is not None:
            self.client = client
        elif not self.api_key:
            self.client = None
            self.unavailable_reason = "api_key_missing"
        elif not self.model:
            self.client = None
            self.unavailable_reason = "explicit_model_missing"
        else:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                self.client = None
                self.unavailable_reason = "openai_sdk_missing"
            else:
                # The application owns a small, explicit retry budget below;
                # disable SDK retries so the two policies cannot multiply.
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    timeout=timeout_seconds,
                    max_retries=0,
                )

        if self.client is not None and not self.model:
            self.unavailable_reason = "explicit_model_missing"
        if self.unavailable_reason:
            logger.info("Optional LLM enrichment unavailable: %s", self.unavailable_reason)

    @property
    def available(self) -> bool:
        return bool(self.client is not None and self.model)

    def availability(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.unavailable_reason,
            "model_explicitly_configured": bool(self.model),
            "proposal_only": True,
            "may_create_evidence": False,
            "may_change_hypothesis_status": False,
            "may_trade": False,
        }

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[ResponseModel],
        system_prompt: str = "Return a review-only analytical proposal.",
        *,
        max_retries: int = 2,
    ) -> ResponseModel | None:
        """Return one parsed proposal or ``None`` with a bounded retry policy."""

        if not self.available:
            return None
        if not 1 <= max_retries <= 3:
            raise ValueError("max_retries must be between 1 and 3")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if len(prompt) > self.max_input_chars:
            raise ValueError("prompt exceeds configured max_input_chars")

        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format=response_model,
                )
                message = response.choices[0].message
                if getattr(message, "refusal", None):
                    logger.warning("Optional LLM enrichment returned a refusal")
                    return None
                parsed = getattr(message, "parsed", None)
                if parsed is None:
                    logger.warning("Optional LLM enrichment returned no parsed payload")
                    return None
                return parsed
            except Exception as exc:  # SDK versions expose different subclasses.
                retryable = _retryable_exception(exc)
                logger.warning(
                    "Optional LLM enrichment failed (%s, attempt %s/%s, retryable=%s)",
                    type(exc).__name__,
                    attempt + 1,
                    max_retries,
                    retryable,
                )
                if not retryable or attempt == max_retries - 1:
                    return None
                await asyncio.sleep(2**attempt)
        return None


def _retryable_exception(exc: Exception) -> bool:
    """Retry only transport, rate-limit, and server-side failures.

    Class-name matching keeps the optional core import-safe when the OpenAI SDK
    is absent and avoids binding deterministic code to one SDK release.
    """

    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }


__all__ = ["LLMClient"]
