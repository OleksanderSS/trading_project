from __future__ import annotations

import json

import pandas as pd

from dean_os.clean_yahoo_market_snapshot import (
    CLEAN_YAHOO_MARKET_SNAPSHOT_CONTRACT,
    _frame_sha256,
    _lane_summaries,
    _normalize_collected_frame,
    _selected_timeframes,
    load_verified_clean_yahoo_market_snapshot,
)


def test_normalize_collected_frame_preserves_source_and_canonical_timeframe():
    frame = _normalize_collected_frame(
        [
            {
                "datetime": "2025-01-01T10:00:00Z",
                "ticker": "nvda",
                "interval": "1h",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
                "hash": "old-source-hash",
            }
        ]
    )

    assert frame.loc[0, "ticker"] == "NVDA"
    assert frame.loc[0, "source_interval"] == "1h"
    assert frame.loc[0, "interval"] == "60m"
    assert frame.loc[0, "hash"] != "old-source-hash"


def test_selected_timeframes_matches_canonical_60m_to_source_1h():
    configured = {
        "15m": {"period": "60d"},
        "1h": {"period": "60d"},
        "1d": {"period": "2y"},
    }

    selected = _selected_timeframes(configured, ["60m"])

    assert selected == {"1h": {"period": "60d"}}


def test_normalized_hash_is_stable_for_same_canonical_identity():
    records = [
        {
            "datetime": pd.Timestamp("2025-01-01T10:00:00Z"),
            "ticker": "NVDA",
            "interval": "1h",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        }
    ]

    first = _normalize_collected_frame(records)
    second = _normalize_collected_frame(records)

    assert first.loc[0, "hash"] == second.loc[0, "hash"]


def test_saved_clean_snapshot_loader_rechecks_parquet_and_manifest(tmp_path):
    frame = _normalize_collected_frame(
        [
            {
                "datetime": "2025-01-01T10:00:00Z",
                "ticker": "NVDA",
                "interval": "15m",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
            }
        ]
    )
    snapshot = tmp_path / "snapshot.parquet"
    frame.to_parquet(snapshot, index=False)
    frame_sha = _frame_sha256(frame)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "contract": CLEAN_YAHOO_MARKET_SNAPSHOT_CONTRACT,
                "mode": "clean_yahoo_market_snapshot",
                "inputs": {
                    "tickers": ["NVDA"],
                    "end_date": "2025-01-01T11:00:00+00:00",
                },
                "summary": {
                    "status": "clean_market_snapshot_validated",
                    "row_count": 1,
                    "ticker_count": 1,
                    "timeframe_count": 1,
                    "snapshot_sha256": frame_sha,
                    "source_gate_issues": [],
                },
                "snapshot": {
                    "path": str(snapshot),
                    "format": "parquet",
                    "sha256": frame_sha,
                },
                "lanes": _lane_summaries(frame),
                "safety": {
                    "source_ticker_validated_before_relabel": True,
                    "cross_identity_exact_ohlcv_gate": True,
                    "cadence_gate": True,
                    "finite_ohlcv_gate": True,
                    "database_write_performed": False,
                    "legacy_artifact_write_performed": False,
                    "learning_write_performed": False,
                    "broker_access_performed": False,
                },
            },
            default=str,
        ),
        encoding="utf-8",
    )

    verified = load_verified_clean_yahoo_market_snapshot(manifest)

    assert verified["tickers"] == ["NVDA"]
    assert verified["timeframes"] == ["15m"]
    assert verified["snapshot_frame_sha256"] == frame_sha
