from __future__ import annotations

import asyncio
import logging
from urllib.parse import parse_qs, urlparse
from pathlib import Path

import pandas as pd
import pytest

from src.core.file_management.file_manager import FileManager
from src.data.collectors.fred_collector import FredCollector
from src.pipeline.stages.processing.data_handler import ProcessingDataHandler
from src.pipeline.stages.processing.storage import ProcessingStorage


def test_macro_normalizer_preserves_point_in_time_schema():
    handler = ProcessingDataHandler(normalization_manager=None, data_filter=None)
    source = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "series": ["DGS10"],
            "value": ["4.1"],
            "realtime_start": ["2025-01-03"],
            "source_locator": ["https://fred.stlouisfed.org/series/DGS10"],
        }
    )

    result = handler.clean_and_normalize_macro_data(source)

    assert {"datetime", "series_id", "value", "realtime_start", "source_locator"}.issubset(result.columns)
    assert result.loc[0, "realtime_start"] == "2025-01-03"


def test_macro_normalizer_rejects_nonempty_data_without_availability():
    handler = ProcessingDataHandler(normalization_manager=None, data_filter=None)
    source = pd.DataFrame(
        {"datetime": ["2025-01-01"], "series_id": ["DGS10"], "value": [4.1]}
    )

    with pytest.raises(ValueError, match="point-in-time availability"):
        handler.clean_and_normalize_macro_data(source)


def test_fred_collector_preserves_vintage_and_source_locator():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "observations": [
                    {
                        "date": "2025-01-01",
                        "value": "4.1",
                        "realtime_start": "2025-01-03",
                        "realtime_end": "2025-02-01",
                    }
                ]
            }

    class Client:
        async def get(self, *_args, **_kwargs):
            return Response()

    collector = object.__new__(FredCollector)
    collector.start_date = "2025-01-01"
    collector.timeout = 1
    collector.logger = logging.getLogger("test_fred_contract")

    rows = asyncio.run(collector._fetch_series("DGS10", Client(), "key"))

    assert rows[0]["series_id"] == "DGS10"
    assert rows[0]["realtime_start"] == "2025-01-03"
    assert rows[0]["source_locator"] == "https://fred.stlouisfed.org/series/DGS10"


def test_fred_collector_rejects_observation_without_vintage():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"observations": [{"date": "2025-01-01", "value": "4.1"}]}

    class Client:
        async def get(self, *_args, **_kwargs):
            return Response()

    collector = object.__new__(FredCollector)
    collector.start_date = "2025-01-01"
    collector.timeout = 1
    collector.logger = logging.getLogger("test_fred_contract")

    with pytest.raises(RuntimeError, match="Failed to fetch FRED series"):
        asyncio.run(collector._fetch_series("DGS10", Client(), "key"))


def test_fred_runtime_scope_overrides_static_config(monkeypatch):
    collector = object.__new__(FredCollector)
    collector.configs = {"params": {"series_ids": ["DGS10"]}}
    collector.logger = logging.getLogger("test_fred_contract")
    monkeypatch.setenv("FRED_API_KEY", "key")

    api_key, series_ids = collector._validate_config(
        series_ids=["DCOILWTICO", "INDPRO", "DCOILWTICO"]
    )

    assert api_key == "key"
    assert series_ids == ["DCOILWTICO", "INDPRO"]


def test_fred_fetch_encodes_vintage_and_observation_cutoff():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "observations": [
                    {
                        "date": "2025-01-01",
                        "value": "4.1",
                        "realtime_start": "2026-07-14",
                    }
                ]
            }

    class Client:
        url = None

        async def get(self, url, **_kwargs):
            self.url = url
            return Response()

    collector = object.__new__(FredCollector)
    collector.start_date = "2024-01-01"
    collector.timeout = 1
    collector.logger = logging.getLogger("test_fred_contract")
    client = Client()

    asyncio.run(
        collector._fetch_series(
            "DGS10",
            client,
            "key",
            observation_start="2024-01-01",
            observation_end="2026-07-14",
            vintage_date="2026-07-14",
        )
    )
    query = parse_qs(urlparse(client.url).query)

    assert query["observation_end"] == ["2026-07-14"]
    assert query["vintage_dates"] == ["2026-07-14"]


def test_fred_hash_changes_when_vintage_or_value_changes():
    collector = object.__new__(FredCollector)
    collector.hash_keys = ["series_id", "date", "realtime_start", "value"]
    base = pd.Series(
        {
            "series_id": "DGS10",
            "date": "2025-01-01",
            "realtime_start": "2025-01-03",
            "value": "4.1",
        }
    )
    revised_vintage = base.copy()
    revised_vintage["realtime_start"] = "2025-02-01"
    revised_value = base.copy()
    revised_value["value"] = "4.2"

    assert collector._generate_hash(base) != collector._generate_hash(revised_vintage)
    assert collector._generate_hash(base) != collector._generate_hash(revised_value)


def test_processing_storage_atomically_writes_point_in_time_macro_snapshot(tmp_path):
    file_manager = FileManager(base_dir=tmp_path)
    storage = ProcessingStorage(file_manager)
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-01"], utc=True),
            "series_id": ["DGS10"],
            "value": [4.1],
            "realtime_start": ["2025-01-03"],
            "source_locator": ["https://fred.stlouisfed.org/series/DGS10"],
        }
    )

    path = storage._save_persistent_macro_snapshot(frame)
    saved = pd.read_parquet(tmp_path / path)
    file_manager._executor.shutdown(wait=True)

    assert Path(path) == Path("data/processed/features/macro_data.parquet")
    assert "realtime_start" in saved.columns
    assert saved.loc[0, "realtime_start"] == "2025-01-03"
