# src/data/collectors/newsapi_collector.py

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector


class NewsApiQuotaExhausted(RuntimeError):
    """NewsAPI refused because the account's daily allowance is spent."""


class NewsAPICollector(BaseCollector):
    """Collector for fetching news streams from NewsAPI endpoints.

    NewsAPI developer accounts allow **100 requests per 24 hours**. The
    collector used to build one query per ticker and per keyword -- 552 terms
    on the current preset -- and fire all of them at once through
    `asyncio.gather`, each retried up to three times on failure. One run of
    2026-08-21 issued 1,656 requests, received 1,129 rate limits, and brought
    back zero articles. The first hundred requests spend the day's entire
    allowance; everything after that is refused, and the refusals are counted
    too, so the retries make the next day worse rather than recovering the
    current one.

    The key was never the problem. Asked directly, the service answers
    `code: rateLimited` with the quota spelled out.

    So: a budget that survives restarts, terms ordered so the allowance buys
    the most useful queries first, requests issued one at a time, and a stop
    the moment the service says the day is spent.
    """

    collector_type = "newsapi"
    data_type = "news"

    #: Requests to spend per calendar day (UTC). Deliberately under the
    #: account's 100 so a manual query or another tool is not locked out.
    DEFAULT_DAILY_BUDGET = 90

    def __init__(
        self,
        configs: dict[str, Any],
        http_client_factory: HttpClientFactory,
        db_manager: DataManager,
        cache_manager: CacheManager | None = None,
        **kwargs,
    ):
        super().__init__(
            configs, http_client_factory, db_manager, cache_manager, **kwargs
        )
        self.base_url = self.configs.get(
            "base_url", "https://newsapi.org/v2/everything"
        )
        self.language = self.configs.get("language", "en")
        self.page_size = self.configs.get("page_size", 20)
        self.hash_keys = self.configs.get("hash_keys", ["url", "publishedAt"])

        filter_cfg = self.configs.get("filter", {})
        self.exclude_title_keywords = [
            kw.lower() for kw in filter_cfg.get("exclude_title_keywords", [])
        ]
        # The config file writes `api_key_env`; this read `api_key_name` and
        # worked only because the default happened to equal the configured
        # value. Pointing the config at a different variable would have been
        # silently ignored. Both names are accepted, config's own spelling
        # first.
        api_key_var = (
            self.configs.get("api_key_env")
            or self.configs.get("api_key_name")
            or "NEWS_API_KEY"
        )
        self._api_key: str | None = os.getenv(api_key_var)

        self.daily_budget = int(
            self.configs.get("daily_request_budget", self.DEFAULT_DAILY_BUDGET)
        )
        self._budget_file = Path(
            self.configs.get("budget_state_path", "data/state/newsapi_budget.json")
        )

    # ------------------------------------------------------------------ budget

    def _today(self) -> str:
        """The quota window is 24h rolling; a UTC date is the honest approximation."""
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def _read_budget(self) -> dict[str, Any]:
        try:
            state = json.loads(self._budget_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"date": self._today(), "spent": 0, "exhausted": False}
        if state.get("date") != self._today():
            return {"date": self._today(), "spent": 0, "exhausted": False}
        return state

    def _write_budget(self, spent: int, exhausted: bool) -> None:
        state = {"date": self._today(), "spent": int(spent), "exhausted": bool(exhausted)}
        try:
            self._budget_file.parent.mkdir(parents=True, exist_ok=True)
            self._budget_file.write_text(
                json.dumps(state, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            # A budget we cannot persist is a budget that resets every run,
            # which is how the quota was being spent three times a day.
            self.logger.warning(
                "[NewsAPI] Could not record request budget at %s: %s. "
                "The next run will not know what this one spent.",
                self._budget_file, exc,
            )

    def _remaining_requests(self) -> int:
        state = self._read_budget()
        if state.get("exhausted"):
            return 0
        return max(0, self.daily_budget - int(state.get("spent", 0)))

    @staticmethod
    def _prioritise(tickers: list[str], keywords: list[str]) -> list[str]:
        """Tickers first: a named holding beats a generic market word.

        With ninety requests against five hundred terms the order decides what
        the day buys, so it is chosen rather than left to set iteration.
        """
        ordered: list[str] = []
        seen: set[str] = set()
        for term in [*tickers, *keywords]:
            key = term.strip().lower()
            if key and key not in seen:
                seen.add(key)
                ordered.append(term)
        return ordered

    def _get_api_key(self) -> str | None:
        if self._api_key is None:
            self.logger.error(
                "[NewsAPI] No API key available."
            )
        return self._api_key

    def _check_newsapi_cache(self, cache_key: str, cache_params: dict, table_name: str) -> pd.DataFrame | None:
        """Check cache for existing NewsAPI data and filter new records."""
        if not self.cache_manager:
            return None
        cached = self.cache_manager.get(cache_key, cache_params, namespace="collectors")
        if cached is not None:
            df_cached = pd.DataFrame(cached) if isinstance(cached, list) else cached
            if "hash" in df_cached.columns:
                new_from_cache = self.db_manager.filter_new_records(table_name, df_cached)
                if new_from_cache.empty:
                    self.logger.info("[NewsAPI] Cache hit — no new articles detected.")
                    return None
                return new_from_cache
        return None

    def _process_fetch_results(self, results: list, search_terms: list[str]) -> list[dict]:
        """Process fetch results and extract articles."""
        all_articles = []
        for i, res in enumerate(results):
            if isinstance(res, list):
                all_articles.extend(res)
            elif isinstance(res, Exception):
                self.logger.error(
                    f"[NewsAPI] Failed for term '{search_terms[i]}': "
                    f"{type(res).__name__}: {res}",
                    exc_info=res,
                )
        return all_articles

    def _create_article_hash(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cryptographic hash for deduplication."""
        df["hash"] = df.apply(
            lambda row: hashlib.sha256(
                "|".join(str(row.get(k, "")) for k in self.hash_keys).encode()
            ).hexdigest(),
            axis=1,
        )
        return df

    def _update_cache(self, cache_key: str, cache_params: dict, df: pd.DataFrame, new_df: pd.DataFrame | None = None) -> None:
        """Cache the whole run payload.

        The per-article CacheManager dedup markers (`set(h, True, ttl=3600)`
        in a loop, plus a matching per-article `get(h)` in a now-removed
        `_filter_by_cache`) were deleted: each `set` is a pickle write plus a
        single-row DuckDB upsert into `cache_metadata`, and each `get` is its
        own `SELECT ... WHERE key_hash = ?`. `filter_new_records()` already
        dedups on the same `hash` column in ONE query, and
        `upsert(unique_on=['hash'])` enforces it at write time.
        """
        if not self.cache_manager:
            return
        self.cache_manager.set(cache_key, df.to_dict("records"), cache_params, namespace="collectors")

    async def run(
        self,
        tickers: list[str] | None = None,
        keywords: list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame | None:
        """Fetch news from NewsAPI, filter novel records, commit to DB."""
        api_key = self._get_api_key()
        if not api_key:
            return None

        table_name = self.configs.get("table_name", "newsapi_articles")
        search_terms = self._prioritise(tickers or [], keywords or [])
        if not search_terms:
            self.logger.warning("[NewsAPI] No search terms provided. Skipping execution.")
            return None

        remaining = self._remaining_requests()
        if remaining <= 0:
            self.logger.warning(
                "[NewsAPI] Daily allowance of %d requests is already spent "
                "(state: %s). Skipping without issuing a request -- refused "
                "requests count against the quota too.",
                self.daily_budget, self._budget_file,
            )
            return None

        if len(search_terms) > remaining:
            self.logger.info(
                "[NewsAPI] %d terms, %d requests left today. Querying the "
                "first %d (tickers before generic keywords); the rest wait "
                "for tomorrow's allowance.",
                len(search_terms), remaining, remaining,
            )
            search_terms = search_terms[:remaining]

        cache_key = f"{self.__class__.__name__}_run"
        cache_params = {"terms": sorted(search_terms)}

        # 1. State Verification (Cache lookup)
        cached_result = self._check_newsapi_cache(cache_key, cache_params, table_name)
        if cached_result is not None:
            return cached_result

        # 2. Sequential acquisition, so the run can stop the moment the
        #    service says the day is spent. Firing every term at once through
        #    asyncio.gather left nothing to stop: all 552 were already in
        #    flight before the first refusal came back.
        self.logger.info(
            "[NewsAPI] Issuing up to %d requests (%d of today's %d allowance "
            "already spent)...",
            len(search_terms), self.daily_budget - remaining, self.daily_budget,
        )
        results: list[Any] = []
        attempted: list[str] = []
        spent = self.daily_budget - remaining
        quota_hit = False

        for term in search_terms:
            attempted.append(term)
            spent += 1
            try:
                results.append(await self._fetch_for_term(term, api_key))
            except NewsApiQuotaExhausted:
                quota_hit = True
                self.logger.warning(
                    "[NewsAPI] Allowance exhausted after %d of %d terms. "
                    "Stopping; the remaining %d wait for the next window.",
                    len(attempted), len(search_terms),
                    len(search_terms) - len(attempted),
                )
                results.append([])
                break
            except Exception as exc:  # noqa: BLE001 - one bad term must not end the run
                results.append(exc)

        self._write_budget(spent, exhausted=quota_hit)
        all_articles = self._process_fetch_results(results, attempted)

        if not all_articles:
            self.logger.info("[NewsAPI] Zero articles retrieved from external queries.")
            return None

        df = pd.DataFrame(all_articles)

        # 3. Cryptographic Deduplication Hash
        df = self._create_article_hash(df)

        # 4. Filter against Historical Database
        new_df = self.db_manager.filter_new_records(table_name, df)
        if new_df.empty:
            self.logger.info("[NewsAPI] No novel articles identified against historical database.")
            self._update_cache(cache_key, cache_params, df)
            return None

        # 6. Persistence to Storage
        self.db_manager.upsert(table_name, new_df, unique_on=["hash"])
        self._update_cache(cache_key, cache_params, df, new_df)

        self.logger.info(
            f"[NewsAPI] Successfully persisted {len(new_df)} new articles."
        )
        return new_df

    async def _fetch_for_term(
        self, term: str, api_key: str
    ) -> list[dict[str, Any]]:
        """One query. Raises NewsApiQuotaExhausted when the day is spent.

        The client is asked for zero retries: this endpoint's 429 means a
        24-hour allowance is gone, and a second attempt half a second later
        cannot succeed. It can only spend another request from the allowance
        it is waiting on -- which is what turned 552 queries into 1,656.
        """
        params = {
            "q": f'"{term}"',
            "language": self.language,
            "pageSize": self.page_size,
            "apiKey": api_key,
        }
        try:
            client = await self.http_client_factory.get_http_client(retries=0)
            response = await client.get(self.base_url, params=params)

            if response.status_code == 429:
                detail = ""
                try:
                    body = response.json()
                    detail = f"{body.get('code')}: {body.get('message')}"
                except ValueError:
                    detail = response.text[:200]
                raise NewsApiQuotaExhausted(detail)

            response.raise_for_status()
            articles = response.json().get("articles", [])

            filtered = []
            for a in articles:
                title = (a.get("title") or "").lower()
                if not any(kw in title for kw in self.exclude_title_keywords):
                    a["search_term"] = term
                    filtered.append(a)
            return filtered
        except NewsApiQuotaExhausted:
            raise
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(
                f"[NewsAPI] HTTP context error for '{term}': {e}"
            )
            raise

    async def collect_data(self, **kwargs) -> list[dict[str, Any]] | None:
        """
        UNIFIED data collection - retrieval only, without database storage.
        """
        df = await self.run(**kwargs)
        return df.to_dict("records") if df is not None else None

