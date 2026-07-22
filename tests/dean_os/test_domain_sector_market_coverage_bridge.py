from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dean_os.clean_yahoo_market_snapshot import (
    CLEAN_YAHOO_MARKET_SNAPSHOT_CONTRACT,
    _frame_sha256,
    _lane_summaries,
    _normalize_collected_frame,
)
from dean_os.domain_sector_market_coverage_bridge import (
    CONTRACT,
    DomainSectorMarketCoverageBridge,
    load_verified_domain_sector_market_coverage_bridge,
)
from dean_os.pipeline_control.pipeline_control_saved_price_repair import (
    PipelineControlSavedPriceRepair,
)


DOMAIN = "semiconductor_ai_infrastructure"
CUTOFF = "2026-07-10T19:50:45.683169+00:00"
TICKERS = [
    "AMAT",
    "AMD",
    "ARM",
    "ASML",
    "AVGO",
    "INTC",
    "KLAC",
    "LRCX",
    "MU",
    "NVDA",
    "QCOM",
    "SOXX",
    "TSM",
]


def _manifest(tmp_path: Path, tickers: list[str]) -> Path:
    records = []
    timestamps = pd.date_range(
        "2026-07-01T10:00:00Z", periods=190, freq="15min"
    )
    for ticker_index, ticker in enumerate(tickers):
        for index, timestamp in enumerate(timestamps):
            price = 100.0 + ticker_index * 20.0 + index * 0.01
            records.append(
                {
                    "datetime": timestamp,
                    "ticker": ticker,
                    "interval": "15m",
                    "open": price,
                    "high": price + 0.2,
                    "low": price - 0.2,
                    "close": price + 0.05,
                    "volume": 1000.0 + ticker_index,
                }
            )
    frame = _normalize_collected_frame(records)
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
                    "tickers": sorted(tickers),
                    "end_date": "2026-07-10T17:48:00+00:00",
                },
                "summary": {
                    "status": "clean_market_snapshot_validated",
                    "row_count": len(frame),
                    "ticker_count": len(tickers),
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
                    "network_access_performed": True,
                    "database_write_performed": False,
                    "legacy_artifact_write_performed": False,
                    "learning_write_performed": False,
                    "broker_access_performed": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_complete_clean_snapshot_creates_verified_domain_coverage(tmp_path):
    payload = DomainSectorMarketCoverageBridge(tmp_path / "reports").build(
        domain_id=DOMAIN,
        analysis_cutoff=CUTOFF,
        snapshot_manifest_path=_manifest(tmp_path, TICKERS),
        min_rows=180,
        save=False,
    )

    assert payload["contract"] == CONTRACT
    assert payload["status"] == "domain_sector_market_coverage_ready"
    assert payload["summary"]["eligible_15m_ticker_count"] == 13
    assert payload["summary"][
        "candidate_ready_for_saved_price_repair"
    ] is True
    assert payload["summary"]["collector_run_performed"] is False

    artifact = tmp_path / "coverage_bridge.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    verified = load_verified_domain_sector_market_coverage_bridge(
        artifact,
        expected_domain_id=DOMAIN,
    )
    assert len(verified["eligible_contexts"]) == 13


def test_incomplete_snapshot_is_blocked_without_repair_contexts(tmp_path):
    payload = DomainSectorMarketCoverageBridge(tmp_path / "reports").build(
        domain_id=DOMAIN,
        analysis_cutoff=CUTOFF,
        snapshot_manifest_path=_manifest(tmp_path, TICKERS[:-1]),
        min_rows=180,
        save=False,
    )

    assert payload["status"] == "domain_sector_market_coverage_blocked"
    assert "domain_market_ticker_scope_mismatch" in payload["summary"][
        "structural_blockers"
    ]
    assert payload["eligible_contexts"] == []
    assert payload["summary"][
        "candidate_ready_for_saved_price_repair"
    ] is False


def test_verified_domain_bridge_is_accepted_by_repair(tmp_path):
    payload = DomainSectorMarketCoverageBridge(tmp_path / "bridge_reports").build(
        domain_id=DOMAIN,
        analysis_cutoff=CUTOFF,
        snapshot_manifest_path=_manifest(tmp_path, TICKERS),
        min_rows=180,
        save=False,
    )
    coverage = tmp_path / "coverage_bridge.json"
    coverage.write_text(json.dumps(payload), encoding="utf-8")

    repaired = PipelineControlSavedPriceRepair(tmp_path / "repair_reports").build(
        coverage_json=coverage,
        required_model_rows=20,
        min_daily_source_bars=24,
        domain_id=DOMAIN,
        save=False,
    )

    assert repaired["summary"]["repair_status"] == (
        "non_destructive_price_candidates_ready"
    )
    assert repaired["summary"]["source_ticker_count"] == 13
    assert repaired["source_provenance"]["coverage_bridge_verified"] is True
    assert repaired["inputs"]["domain_id"] == DOMAIN
