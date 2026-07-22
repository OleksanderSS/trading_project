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
        # Example: WB -> 'FP.CPI.TOTL.ZG' (Inflation), ECB -> 'ICP' (HICP Inflation)
        self.indicators = self.configs.get("indicators", {
            "WB": ["FP.CPI.TOTL.ZG", "NY.GDP.MKTP.KD.ZG"],
            "ECB": ["ICP"]
        })
        
        self.logger.info(f"SDMXMacroCollector initialized for agencies: {self.agencies}")

    def _generate_hash(self, row: pd.Series) -> str:
        hash_string = f"{row.get('date', '')}|{row.get('agency', '')}|{row.get('indicator', '')}|{row.get('country', '')}"
        return hashlib.sha256(hash_string.encode()).hexdigest()

    async def run(self, **kwargs) -> pd.DataFrame | None:
        if not self.enabled:
            return None

        if sdmx is None:
            self.logger.error("pandasdmx is not installed. Please run: pip install pandasdmx")
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
                    # Run sync SDMX request in background thread
                    agency_data = await asyncio.to_thread(self._fetch_sdmx_sync, agency, ind)
                    if agency_data:
                        all_data.extend(agency_data)
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
            # Create request instance for the specific agency
            req = sdmx.Request(agency)
            
            # This is highly dependent on the agency's data structure.
            # We use a generic approach or specific known parameters for WB/ECB.
            if agency == "WB":
                # World Bank SDMX structure typically uses 'INDICATOR'
                msg = req.data(resource_id='WDI', key={'INDICATOR': indicator}, params={'startPeriod': '2020'})
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
