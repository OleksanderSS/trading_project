from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from dean_os.analysts._producers.sector_market import (
    SavedSectorMarketEvidenceProducer,
    load_verified_sector_market_context_fragment,
)
from dean_os.structured_context_provenance import (
    audit_structured_context,
)


AS_OF = "2026-06-30T21:00:00+00:00"
TICKERS = ["NVDA", "AMD", "INTC", "TSM", "QQQ"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repair_artifact(
    tmp_path: Path,
    *,
    omitted_ticker: str | None = None,
) -> Path:
    raw_path = tmp_path / "prices_15m.parquet"
    raw_path.write_bytes(b"observed saved 15m source")
    daily_path = tmp_path / "prices_1d_resampled.parquet"
    dates = pd.date_range(
        "2026-05-27",
        periods=21,
        freq="B",
        tz="UTC",
    )
    rows = []
    for ticker_index, ticker in enumerate(TICKERS):
        if ticker == omitted_ticker:
            continue
        for index, timestamp in enumerate(dates):
            close = 100.0 + ticker_index * 5.0 + index
            rows.append(
                {
                    "datetime": timestamp,
                    "ticker": ticker,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000_000.0 + index,
                    "interval": "1d",
                    "source_bar_count": 26,
                    "hash": f"{ticker}-{index}",
                }
            )
    pd.DataFrame(rows).to_parquet(daily_path, index=False)

    exact_path = tmp_path / "repair_run.json"
    payload = {
        "run_id": "repair_run",
        "created_at": "2026-06-27T12:00:00+00:00",
        "mode": "pipeline_control_saved_price_repair",
        "summary": {
            "repair_status": "non_destructive_price_candidates_ready",
            "cross_ticker_identity_groups": 0,
            "can_train": False,
            "can_trade": False,
        },
        "source_provenance": {
            "path": str(raw_path),
            "sha256": _sha256(raw_path),
            "synthetic": False,
        },
        "artifacts": {
            "prices_1d_resampled": {
                "path": str(daily_path),
                "sha256": _sha256(daily_path),
                "synthetic": False,
                "derived_from_observed_bars": True,
            }
        },
        "saved_paths": {"json": str(exact_path)},
        "artifact_safety": {
            "learning_write_performed": False,
            "live_execution_performed": False,
        },
    }
    exact_path.write_text(json.dumps(payload), encoding="utf-8")
    latest_path = tmp_path / "latest.json"
    latest_path.write_text(json.dumps(payload), encoding="utf-8")
    return latest_path


def test_sector_market_producer_builds_explicit_market_confirmation(
    tmp_path,
):
    repair_path = _repair_artifact(tmp_path)
    output_dir = tmp_path / "output"
    payload = SavedSectorMarketEvidenceProducer(
        output_dir=output_dir
    ).build(
        repair_artifact_path=repair_path,
        as_of=AS_OF,
        sector_tickers=TICKERS[:-1],
        benchmark="QQQ",
    )

    assert payload["status"] == "sector_market_evidence_ready"
    assert payload["summary"]["sector_ticker_coverage_ratio"] == 1.0
    assert payload["summary"]["required_market_confirmation_ready"]
    assert payload["summary"]["can_train"] is False
    assert payload["summary"]["can_trade"] is False
    assert payload["summary"]["accepted_metric_count"] == 11

    fragment = load_verified_sector_market_context_fragment(
        output_dir / "latest.json",
        expected_as_of=AS_OF,
    )
    audit = audit_structured_context(
        fundamentals={},
        macro={},
        sector_data=fragment["sector_data"],
        as_of=AS_OF,
    )
    eligible = [
        item
        for item in audit["accepted_observations"]
        if item.get("required_lane_eligible") is True
    ]
    assert len(eligible) == 3
    assert {
        item["evidence_type"] for item in eligible
    } == {"market_confirmation"}


@pytest.mark.parametrize(
    ("sector_tickers", "benchmark", "message"),
    [
        (None, "SOXX", "sector_tickers must be explicitly supplied"),
        (["NVDA"], None, "benchmark must be explicitly supplied"),
    ],
)
def test_sector_market_producer_requires_explicit_domain_scope(
    tmp_path,
    sector_tickers,
    benchmark,
    message,
):
    with pytest.raises(ValueError, match=message):
        SavedSectorMarketEvidenceProducer(
            output_dir=tmp_path / "output"
        ).build(
            repair_artifact_path=tmp_path / "unused.json",
            as_of=AS_OF,
            sector_tickers=sector_tickers,
            benchmark=benchmark,
            save=False,
        )


def test_sector_market_producer_fails_closed_on_missing_sector_ticker(
    tmp_path,
):
    repair_path = _repair_artifact(tmp_path, omitted_ticker="INTC")
    payload = SavedSectorMarketEvidenceProducer(
        output_dir=tmp_path / "output"
    ).build(
        repair_artifact_path=repair_path,
        as_of=AS_OF,
        sector_tickers=TICKERS[:-1],
        benchmark="QQQ",
        save=False,
    )

    assert payload["status"] == "blocked_sector_market_evidence"
    assert (
        payload["summary"]["required_market_confirmation_ready"]
        is False
    )
    reasons = payload["summary"]["reason_counts"]
    assert "required_market_ticker_coverage_incomplete" in reasons


def test_sector_market_loader_detects_changed_daily_source(tmp_path):
    repair_path = _repair_artifact(tmp_path)
    output_dir = tmp_path / "output"
    payload = SavedSectorMarketEvidenceProducer(
        output_dir=output_dir
    ).build(
        repair_artifact_path=repair_path,
        as_of=AS_OF,
        sector_tickers=TICKERS[:-1],
        benchmark="QQQ",
    )
    daily = Path(payload["lineage"]["daily_artifact"]["path"])
    daily.write_bytes(daily.read_bytes() + b"tamper")

    with pytest.raises(
        ValueError,
        match="source lineage invalid",
    ):
        load_verified_sector_market_context_fragment(
            output_dir / "latest.json"
        )
