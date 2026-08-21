# src/data/collectors/sdmx_macro_collector.py

import asyncio
import hashlib
from datetime import datetime
from typing import Any

import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector

# `sdmx1` (import name `sdmx`) is the maintained successor to `pandasdmx` and
# exposes the same `Request` / `to_pandas` API this collector uses.
#
# Do NOT "fix" a missing dependency here by installing pandasdmx: its last
# release (1.6.0) pins pydantic<2, so `pip install pandasdmx` silently
# DOWNGRADES pydantic to 1.7.4 and breaks all of dean_os, which is built on
# pydantic v2 (`model_validate`, v2-style validators). `sdmx1` installs with
# no such conflict. The pandasdmx fallback below is kept only so an
# already-provisioned legacy environment keeps working.
try:
    import sdmx
except ImportError:
    try:
        import pandasdmx as sdmx
    except ImportError:
        sdmx = None


class SDMXMacroCollector(BaseCollector):
    """
    Universal SDMX Collector for Macroeconomic indicators.
    Can query World Bank (WB), IMF, ECB, Eurostat, ILO, OECD.
    """
    collector_type = "sdmx_macro"
    data_type = "macro_context"

    def __init__(
        self,
        configs: dict[str, Any],
        http_client_factory: HttpClientFactory,
        db_manager: DataManager,
        cache_manager: CacheManager | None = None,
        **kwargs,
    ):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.enabled = self.configs.get("enabled", True)
        self.table_name = self.configs.get("table_name", "macro_sdmx_data")
        self.agencies = self.configs.get("agencies", ["WB", "ECB"])
        
        # Mapping of indicators per agency.
        # NOTE for WB_WDI: SDMX series codes use underscores, not the dots used
        # by the World Bank's plain REST API -- 'FP_CPI_TOTL_ZG', not
        # 'FP.CPI.TOTL.ZG'. The source id is WB_WDI; plain 'WB' in sdmx1 is
        # World Bank WITS, which only publishes trade/tariff dataflows
        # (DF_WITS_Tariff_TRAINS etc.) and has no CPI/GDP indicators at all.
        self.indicators = self.configs.get("indicators", {
            "WB_WDI": ["FP_CPI_TOTL_ZG", "NY_GDP_MKTP_KD_ZG"],
            "ECB": ["ICP"]
        })
        # Reference areas for sources keyed by country (WB_WDI). Ignored by
        # sources whose dataflow has no REF_AREA dimension.
        self.ref_areas = self.configs.get(
            "ref_areas", ["USA", "CHN", "EMU", "JPN", "DEU", "GBR"]
        )
        # Hard ceiling per SDMX request, see run(). SDMX services are slow and
        # an unkeyed request can stream a whole dataflow indefinitely.
        self.request_timeout = float(self.configs.get("request_timeout", 90))

        self.logger.info(f"SDMXMacroCollector initialized for agencies: {self.agencies}")

    def _generate_hash(self, row: pd.Series) -> str:
        hash_string = f"{row.get('date', '')}|{row.get('agency', '')}|{row.get('indicator', '')}|{row.get('country', '')}"
        return hashlib.sha256(hash_string.encode()).hexdigest()

    async def run(self, **kwargs) -> pd.DataFrame | None:
        if not self.enabled:
            return None

        if sdmx is None:
            self.logger.error(
                "No SDMX library installed. Please run: pip install sdmx1 "
                "(do NOT install pandasdmx -- it pins pydantic<2 and would "
                "downgrade pydantic, breaking dean_os)."
            )
            return None

        self.logger.info("Fetching macroeconomic data via SDMX...")
        
        all_data = []

        # SDMX is inherently synchronous and can be slow, so we run it in a thread
        for agency in self.agencies:
            if agency not in self.indicators:
                continue
                
            indicators_to_fetch = self.indicators[agency]
            for ind in indicators_to_fetch:
                try:
                    # Run sync SDMX request in background thread, under a hard
                    # per-request timeout. Without this, one unbounded request
                    # (e.g. an unkeyed ECB `ICP` pull, which is the entire HICP
                    # dataflow for every country) blocks for many minutes and
                    # starves every collector scheduled after this one in the
                    # same stage.
                    agency_data = await asyncio.wait_for(
                        asyncio.to_thread(self._fetch_sdmx_sync, agency, ind),
                        timeout=self.request_timeout,
                    )
                    if agency_data:
                        all_data.extend(agency_data)
                except TimeoutError:
                    # Do not assert a cause. This line used to say the request
                    # was "probably unkeyed", which sent the reader off to add
                    # a key that WB_WDI already has: the request is
                    # A.<indicator>.USA+CHN+EMU+JPN+DEU+GBR. Measured
                    # 2026-08-21, after that timeout, both configured WDI
                    # indicators returned 396 rows in under a second -- so the
                    # 90s was a one-off, most likely the first request of the
                    # process paying for dataflow-structure discovery that is
                    # cached afterwards. A wrong diagnosis in an error message
                    # costs more than no diagnosis.
                    self.logger.error(
                        "Timed out after %ss fetching %s from %s. Two causes "
                        "are worth checking, in this order: whether the "
                        "request carries an explicit key (an unkeyed dataflow "
                        "pull streams every country and series), and whether "
                        "this was simply the first request of the run, which "
                        "also fetches the dataflow structure. The stage "
                        "continues without this indicator.",
                        self.request_timeout, ind, agency,
                    )
                except Exception as e:
                    self.logger.error(f"Failed to fetch {ind} from {agency}: {e}")

        if not all_data:
            self.logger.warning("No SDMX macro data retrieved.")
            return None

        df = pd.DataFrame(all_data)
        
        # Generate hashes
        df["record_hash"] = df.apply(self._generate_hash, axis=1)

        # Filter and save
        new_df = self.db_manager.filter_new_records(self.table_name, df, unique_cols=["record_hash"])
        if new_df.empty:
            self.logger.info("No new SDMX macro data found.")
            return None

        self.db_manager.upsert(self.table_name, new_df, unique_on=["record_hash"])
        self.logger.info(f"Successfully saved {len(new_df)} new SDMX records.")
        return new_df

    def _fetch_sdmx_sync(self, agency: str, indicator: str) -> list[dict[str, Any]]:
        """Synchronous fetch using pandasdmx."""
        results = []
        try:
            # Create request instance for the specific agency.
            # sdmx1 renamed Request -> Client and will drop Request in v3.0;
            # getattr keeps the legacy pandasdmx fallback working.
            client_cls = getattr(sdmx, "Client", None) or sdmx.Request
            req = client_cls(agency)
            
            # This is highly dependent on the agency's data structure.
            # We use a generic approach or specific known parameters per source.
            if agency == "WB_WDI":
                # The WDI dataflow is keyed FREQ.SERIES.REF_AREA. The key MUST
                # be passed as a string, not a dict: given a dict, sdmx1 first
                # issues a `?detail=serieskeysonly` validation request, and the
                # World Bank answers that with 403 Forbidden -- which is what
                # made every WB fetch fail regardless of the codes used.
                key = f"A.{indicator}.{'+'.join(self.ref_areas)}"
                msg = req.data(
                    resource_id="WDI", key=key, params={"startPeriod": "2000"}
                )
            elif agency == "ECB":
                msg = req.data(resource_id=indicator, params={'startPeriod': '2020-01'})
            else:
                msg = req.data(resource_id=indicator)

            # Convert SDMX message to pandas Series/DataFrame
            data = sdmx.to_pandas(msg)
            
            # Because SDMX returns multi-index series, we flatten it
            if isinstance(data, pd.Series):
                df = data.reset_index()
            else:
                df = data.stack().reset_index()

            # The last column is usually the value, the others are dimensions (TIME_PERIOD, REF_AREA, etc.)
            val_col = df.columns[-1]
            df = df.rename(columns={val_col: "value"})
            
            for _, row in df.iterrows():
                # Extract time and geography heuristically
                row_dict = row.to_dict()
                time_val = row_dict.get('TIME_PERIOD', row_dict.get('time', None))
                geo_val = row_dict.get('REF_AREA', row_dict.get('geo', 'GLOBAL'))
                
                if time_val and not pd.isna(row_dict['value']):
                    results.append({
                        "date": str(time_val),
                        "agency": agency,
                        "indicator": indicator,
                        "country": str(geo_val),
                        "value": float(row_dict['value']),
                        "dimensions": str(row_dict)
                    })
        except Exception as e:
            self.logger.error(f"SDMX Error parsing {agency} {indicator}: {e}")
            
        return results

    async def collect_data(self, **kwargs) -> list[dict[str, Any]] | None:
        df = await self.run(**kwargs)
        return df.to_dict('records') if df is not None else None
