# src/data/collectors/sec_fundamentals_collector.py

"""Reported financial facts, each carrying the date it became knowable.

The filings collector next to this one records that a company filed something
and when. This one records WHAT WAS IN IT: assets, liabilities, revenue,
earnings, share count -- the numbers a value screen is built from. Without
them "undervalued" cannot be computed at all, only asserted.

Two decisions carry the whole file.

**`filed`, never `end`.** Every XBRL fact has a period it describes (`end`,
e.g. 2024-06-30) and a date the filing that carried it reached the SEC
(`filed`, e.g. 2024-08-05). Joining on `end` would put the June quarter into
June's bars -- six weeks before anyone could read it. That is the same defect
as `reportDate` versus `filingDate` in the corporate filings enricher, and the
same shape as the persistence baseline that was reading the future: arithmetic
that runs correctly and answers a question nobody can act on.

**Restatements are kept, not collapsed.** The same quarter appears several
times in companyfacts, from the original filing and from every later one that
restated it, each with its own `filed` and accession number. Keeping all of
them is what makes a point-in-time question answerable: what did the numbers
LOOK LIKE then, not what do we now know they were. A screen run on restated
history is a screen that could never have been run.

One request per company. `companyfacts` returns the full reported history in a
single JSON -- years of it -- so history costs nothing extra; what is filtered
is which concepts are stored, since the raw document carries hundreds and most
are irrelevant to a value screen.
"""

import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Any

import httpx
import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

#: The concepts a Graham-style screen needs, by US-GAAP taxonomy name.
#: Alternatives are listed together because companies do not all tag the same
#: way -- a filer may report `Revenues` or `RevenueFromContractWithCustomer...`
#: for the same line. Storing whichever is present keeps the screen's job
#: (choosing between them) out of the collector's.
DEFAULT_CONCEPTS = (
    # What the company owns and owes -- book value, and the current ratio.
    "Assets",
    "AssetsCurrent",
    "Liabilities",
    "LiabilitiesCurrent",
    "StockholdersEquity",
    "LongTermDebtNoncurrent",
    "CashAndCashEquivalentsAtCarryingValue",
    # What it earns.
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
    # What it is divided into, and what it pays out.
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "PaymentsOfDividendsCommonStock",
)

#: SEC EDGAR requires a User-Agent naming a real contact and rate-limits to
#: 10 requests a second.
_SEC_HEADERS = {
    "User-Agent": "DEAN_OS_Agent research@example.com",
    "Accept-Encoding": "gzip, deflate",
}


class SECFundamentalsCollector(BaseCollector):
    """Historical XBRL financial facts, keyed by when they became public."""

    collector_type = "sec_fundamentals"
    data_type = "fundamental"
    collector_name = "sec_fundamentals"

    def __init__(
        self,
        configs: dict[str, Any],
        http_client_factory: HttpClientFactory,
        db_manager: DataManager,
        cache_manager: CacheManager | None = None,
        config_manager: UnifiedConfigManager | None = None,
        **kwargs,
    ):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.config_manager = config_manager or kwargs.get("config_manager")
        self.facts_url_template = self.configs.get(
            "facts_url_template",
            "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        )
        self.table_name = self.configs.get("table_name", "sec_fundamentals")
        # Identity, and nothing derived from the value. A key that contains the
        # number itself turns into a change detector, which is how 22 of 77 VIX
        # dates ended up duplicated. `accn` is what separates an original
        # statement from a restatement of the same quarter.
        # `period_start` belongs here, and leaving it out was a real collision:
        # ONE filing reports NetIncomeLoss twice, once for the quarter and once
        # for the nine months, both ending the same day. Same end, same
        # accession, different start, different number -- 471 of AAPL's 2,939
        # facts collided that way, and deduplication would have thrown away
        # whichever arrived second. With it, zero.
        self.hash_keys = self.configs.get(
            "hash_keys",
            ["cik", "concept", "unit", "period_start", "period_end", "accession"],
        )
        params = self.configs.get("params", {})
        self.concepts = tuple(params.get("concepts") or DEFAULT_CONCEPTS)
        self.taxonomies = tuple(params.get("taxonomies") or ("us-gaap", "dei"))
        self.max_concurrency = int(params.get("max_concurrency", 4))
        self._cik_map: dict[str, str] | None = None

    # ------------------------------------------------------------------ CIK

    def _get_cik_map(self) -> dict[str, str]:
        if self._cik_map is None:
            try:
                assets_config = self.config_manager.get_config("assets")
                details = assets_config.get("details", {})
                self._cik_map = {
                    ticker: str(data["cik"])
                    for ticker, data in details.items()
                    if "cik" in data
                }
                logger.info(
                    "[SEC-XBRL] Loaded CIK map for %d tickers.", len(self._cik_map)
                )
            except (ValueError, TypeError, AttributeError, KeyError) as error:
                logger.error("[SEC-XBRL] Failed to load CIK map: %s", error, exc_info=True)
                self._cik_map = {}
        return self._cik_map

    # -------------------------------------------------------------- parsing

    def _rows_from_facts(self, payload: dict, ticker: str, cik: str) -> list[dict[str, Any]]:
        """Flatten companyfacts into one row per reported fact."""
        rows: list[dict[str, Any]] = []
        facts = payload.get("facts") or {}
        entity = payload.get("entityName")

        for taxonomy in self.taxonomies:
            concepts = facts.get(taxonomy) or {}
            for concept, body in concepts.items():
                if concept not in self.concepts:
                    continue
                label = (body or {}).get("label")
                for unit, entries in ((body or {}).get("units") or {}).items():
                    for entry in entries or []:
                        filed = entry.get("filed")
                        end = entry.get("end")
                        value = entry.get("val")
                        # A fact with no filing date cannot be placed in time,
                        # and a fact placed by its period end is lookahead. So
                        # `filed` is required and the row is dropped without it
                        # rather than guessed at.
                        if filed is None or end is None or value is None:
                            continue
                        rows.append({
                            "ticker": ticker,
                            "cik": cik,
                            "entity": entity,
                            "taxonomy": taxonomy,
                            "concept": concept,
                            "label": label,
                            "unit": unit,
                            "value": float(value),
                            # The period the number describes...
                            "period_start": entry.get("start"),
                            "period_end": end,
                            "fiscal_year": entry.get("fy"),
                            "fiscal_period": entry.get("fp"),
                            "form": entry.get("form"),
                            # ...and the date it could first be read. Joins use
                            # THIS one.
                            "filed": filed,
                            "accession": entry.get("accn"),
                            "frame": entry.get("frame"),
                        })
        return rows

    async def _fetch_one(self, client: httpx.AsyncClient, ticker: str,
                         cik: str) -> list[dict[str, Any]]:
        url = self.facts_url_template.format(cik=cik)
        response = await client.get(url, headers=_SEC_HEADERS, timeout=60.0)
        response.raise_for_status()
        rows = self._rows_from_facts(response.json(), ticker, cik)
        if rows:
            spans = pd.to_datetime([r["filed"] for r in rows], errors="coerce")
            logger.info(
                "[SEC-XBRL] %s: %d facts across %d concepts, filed %s..%s",
                ticker, len(rows), len({r["concept"] for r in rows}),
                spans.min().date(), spans.max().date(),
            )
        else:
            logger.warning(
                "[SEC-XBRL] %s (CIK%s): no facts among the requested concepts. "
                "An ETF or trust reports none of them, which is a data fact "
                "rather than a failure.", ticker, cik,
            )
        return rows

    # ------------------------------------------------------------------ run

    async def run(self, tickers: list[str], **kwargs) -> pd.DataFrame | None:
        if not tickers:
            logger.warning("[SEC-XBRL] No tickers provided. Skipping.")
            return None

        cik_map = self._get_cik_map()
        targets = {
            ticker: str(cik_map[ticker.upper()]).zfill(10)
            for ticker in tickers
            if ticker.upper() in cik_map
        }
        if not targets:
            logger.warning("[SEC-XBRL] None of the requested tickers has a CIK.")
            return None

        logger.info("[SEC-XBRL] Fetching reported facts for %d tickers.", len(targets))
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # `get_client` hands back a COROUTINE -- `HttpClientFactory.get_http_client`
        # is async -- so it has to be awaited before it is a client. Written as
        # `async with self.get_client()` it raised "'coroutine' object does not
        # support the asynchronous context manager protocol" on the first real
        # call, while every test of the parsing passed. The sibling filings
        # collector already does it this way.
        client = await self.get_client()
        async with client:
            async def guarded(ticker: str, cik: str):
                async with semaphore:
                    # SEC allows 10 requests a second; stay well under it.
                    await asyncio.sleep(0.2)
                    return await self._fetch_one(client, ticker, cik)

            results = await asyncio.gather(
                *(guarded(ticker, cik) for ticker, cik in targets.items()),
                return_exceptions=True,
            )

        rows: list[dict[str, Any]] = []
        for (ticker, cik), result in zip(targets.items(), results):
            if isinstance(result, list):
                rows.extend(result)
            elif isinstance(result, httpx.HTTPStatusError) and result.response.status_code == 404:
                # No XBRL document for that CIK. True of most ETFs and trusts,
                # which file but do not report these concepts.
                logger.info("[SEC-XBRL] %s (CIK%s) has no companyfacts.", ticker, cik)
            elif isinstance(result, BaseException):
                # Named, not swallowed: a collector that returns [] after a
                # failure is indistinguishable from one that found nothing.
                logger.error(
                    "[SEC-XBRL] %s (CIK%s) failed with %s: %s",
                    ticker, cik, type(result).__name__, result,
                )

        if not rows:
            logger.warning("[SEC-XBRL] No facts collected for any ticker.")
            return None

        frame = pd.DataFrame(rows)
        frame["filed"] = pd.to_datetime(frame["filed"], errors="coerce")
        frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
        frame["period_start"] = pd.to_datetime(frame["period_start"], errors="coerce")
        frame = frame.dropna(subset=["filed", "period_end"])

        frame["collector_type"] = self.collector_type
        frame["collector_name"] = self.collector_name
        frame["data_type"] = self.data_type
        frame["collected_at"] = datetime.now()
        frame["record_hash"] = frame.apply(self.generate_hash, axis=1)

        logger.info(
            "[SEC-XBRL] Collected %d facts for %d tickers, %d concepts, "
            "filings from %s to %s.",
            len(frame), frame["ticker"].nunique(), frame["concept"].nunique(),
            frame["filed"].min().date(), frame["filed"].max().date(),
        )
        return frame

    def generate_hash(self, row: pd.Series) -> str:
        parts = [str(row.get(key, "")) for key in self.hash_keys]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
