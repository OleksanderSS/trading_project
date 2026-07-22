from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from dean_os.agent_lab import AgentLabRunner
from dean_os.agents.research_agents import (
    metric_patterns_from_context,
)
from dean_os.analysts._producers.macro import (
    SavedMacroEvidenceProducer,
    load_verified_macro_context_fragment,
)
from dean_os.schemas import MarketContext
from dean_os.structured_context_provenance import (
    audit_structured_context,
)


AS_OF = "2026-07-01T00:00:00+00:00"


def _write_csv(tmp_path, rows):
    path = tmp_path / "macro.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_saved_macro_producer_selects_latest_admissible_observation(
    tmp_path,
):
    source = _write_csv(
        tmp_path,
        [
            {
                "date": "2026-03-01",
                "series_id": "CPIAUCSL",
                "value": 331.0,
                "realtime_start": "2026-05-15",
            },
            {
                "date": "2026-04-01",
                "series_id": "CPIAUCSL",
                "value": 332.4,
                "realtime_start": "2026-06-04",
            },
            {
                "date": "2026-06-01",
                "series_id": "DGS10",
                "value": 4.25,
                "realtime_start": "2026-06-30",
            },
        ],
    )

    payload = SavedMacroEvidenceProducer(
        tmp_path / "reports"
    ).build(
        source_path=source,
        as_of=AS_OF,
        save=False,
    )

    assert payload["status"] == "macro_evidence_ready"
    assert payload["summary"]["eligible_row_count"] == 3
    assert payload["summary"]["selected_series_count"] == 2
    assert payload["summary"]["not_selected_eligible_row_count"] == 1
    selected = {
        item["series_id"]: item
        for item in payload["selected_observations"]
    }
    assert selected["CPIAUCSL"]["period"] == "2026-04-01"
    assert selected["CPIAUCSL"]["unit"] == "index_1982_1984_100"
    assert selected["CPIAUCSL"]["available_at"] == (
        "2026-06-04T23:59:59.999999+00:00"
    )
    fragment = payload["market_context_fragment"]
    reaudit = audit_structured_context(
        fundamentals={},
        macro=fragment["macro"],
        sector_data={},
        as_of=fragment["as_of"],
    )
    assert reaudit["accepted_count"] == 2
    assert reaudit["accepted_fingerprint"] == payload["summary"][
        "accepted_fingerprint"
    ]


def test_saved_macro_producer_rejects_source_without_vintage(
    tmp_path,
):
    source = _write_csv(
        tmp_path,
        [
            {
                "datetime": "2026-04-01T00:00:00+00:00",
                "series": "CPIAUCSL",
                "value": 332.4,
            }
        ],
    )

    payload = SavedMacroEvidenceProducer().build(
        source_path=source,
        as_of=AS_OF,
        save=False,
    )

    assert payload["status"] == (
        "blocked_no_admissible_macro_evidence"
    )
    assert payload["schema_mapping"]["availability_column"] is None
    assert payload["summary"]["accepted_series_count"] == 0
    assert payload["integration_boundary"][
        "missing_vintage_fallback_to_file_mtime"
    ] is False


def test_saved_macro_producer_excludes_future_vintage(tmp_path):
    source = _write_csv(
        tmp_path,
        [
            {
                "date": "2026-06-01",
                "series_id": "FEDFUNDS",
                "value": 4.5,
                "realtime_start": "2026-07-02",
            }
        ],
    )

    payload = SavedMacroEvidenceProducer().build(
        source_path=source,
        as_of=AS_OF,
        save=False,
    )

    assert payload["summary"]["accepted_series_count"] == 0
    assert payload["summary"]["reason_counts"][
        "macro_vintage_after_as_of"
    ] == 1


def test_saved_macro_producer_excludes_unregistered_series(tmp_path):
    source = _write_csv(
        tmp_path,
        [
            {
                "date": "2026-06-01",
                "series_id": "UNKNOWN_SERIES",
                "value": 1.0,
                "realtime_start": "2026-06-20",
            }
        ],
    )

    payload = SavedMacroEvidenceProducer().build(
        source_path=source,
        as_of=AS_OF,
        save=False,
    )

    assert payload["summary"]["accepted_series_count"] == 0
    assert payload["summary"]["reason_counts"][
        "macro_series_registry_entry_missing"
    ] == 1


def test_saved_macro_producer_blocks_ohlcv_shaped_file(tmp_path):
    source = _write_csv(
        tmp_path,
        [
            {
                "datetime": "2026-06-01T00:00:00+00:00",
                "ticker": "AMD",
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100.5,
                "volume": 1000,
            }
        ],
    )

    payload = SavedMacroEvidenceProducer().build(
        source_path=source,
        as_of=AS_OF,
        save=False,
    )

    assert payload["status"] == (
        "blocked_no_admissible_macro_evidence"
    )
    reasons = payload["summary"]["reason_counts"]
    assert reasons["macro_schema_missing_series_column"] == 1
    assert reasons["macro_schema_missing_value_column"] == 1
    assert reasons["macro_schema_missing_availability_column"] == 1


def test_current_saved_macro_snapshot_produces_review_fragment():
    source = Path(
        "data/processed/macro_data_20260629_184812.parquet"
    )
    if not source.exists():
        pytest.skip("current saved macro snapshot is absent")

    payload = SavedMacroEvidenceProducer().build(
        source_path=source,
        as_of=AS_OF,
        save=False,
    )

    assert payload["status"] == "macro_evidence_ready"
    assert payload["summary"]["source_row_count"] == 454
    assert payload["summary"]["accepted_series_count"] == 27
    assert payload["summary"]["can_enter_market_context_review"] is True
    assert payload["summary"]["can_become_pipeline_feature"] is False
    assert payload["summary"]["can_trade"] is False


def test_verified_fragment_loader_rechecks_lineage(tmp_path):
    source = _write_csv(
        tmp_path,
        [
            {
                "date": "2026-04-01",
                "series_id": "CPIAUCSL",
                "value": 332.4,
                "realtime_start": "2026-06-04",
            }
        ],
    )
    payload = SavedMacroEvidenceProducer(
        tmp_path / "reports"
    ).build(
        source_path=source,
        as_of=AS_OF,
        save=True,
    )

    fragment = load_verified_macro_context_fragment(
        payload["saved_paths"]["latest_json"],
        expected_as_of=AS_OF,
    )

    assert fragment["metadata"]["saved_macro_verified"] is True
    assert fragment["macro"]["cpi"] == 332.4
    patterns = metric_patterns_from_context(
        MarketContext(
            as_of=fragment["as_of"],
            macro=fragment["macro"],
        )
    )
    assert patterns["policy_easing"] == 0
    source.write_text(
        source.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source artifact hash mismatch"):
        load_verified_macro_context_fragment(
            payload["saved_paths"]["latest_json"],
            expected_as_of=AS_OF,
        )


def test_verified_fragment_loader_rejects_payload_tampering(tmp_path):
    source = _write_csv(
        tmp_path,
        [
            {
                "date": "2026-04-01",
                "series_id": "CPIAUCSL",
                "value": 332.4,
                "realtime_start": "2026-06-04",
            }
        ],
    )
    payload = SavedMacroEvidenceProducer(
        tmp_path / "reports"
    ).build(
        source_path=source,
        as_of=AS_OF,
        save=True,
    )
    artifact = Path(payload["saved_paths"]["latest_json"])
    tampered = json.loads(artifact.read_text(encoding="utf-8"))
    tampered["market_context_fragment"]["macro"]["cpi"] = 999.0
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(
        json.dumps(tampered),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fingerprint"):
        load_verified_macro_context_fragment(
            tampered_path,
            expected_as_of=AS_OF,
        )


def test_verified_fragment_enters_agent_lab_as_review_context(
    tmp_path,
):
    source = _write_csv(
        tmp_path,
        [
            {
                "date": "2026-04-01",
                "series_id": "CPIAUCSL",
                "value": 332.4,
                "realtime_start": "2026-06-04",
            }
        ],
    )
    payload = SavedMacroEvidenceProducer(
        tmp_path / "producer"
    ).build(
        source_path=source,
        as_of=AS_OF,
        save=True,
    )
    fragment = load_verified_macro_context_fragment(
        payload["saved_paths"]["latest_json"],
        expected_as_of=AS_OF,
    )

    report = asyncio.run(
        AgentLabRunner(
            corpus_path=tmp_path / "corpus.sqlite",
            learning_path=tmp_path / "learning.sqlite",
            output_dir=tmp_path / "agent_lab",
            memory_path=tmp_path / "memory.sqlite",
            log_path=None,
        ).run(
            documents=[],
            macro=fragment["macro"],
            macro_provenance=fragment["metadata"],
            as_of=fragment["as_of"],
            include_financial_nlp=False,
            include_synthesis=False,
            create_learning_records=False,
            include_operations_proposals=False,
        )
    )

    structured = report.summary[
        "structured_context_point_in_time_audit"
    ]
    assert structured["accepted_count"] == 1
    assert report.summary["macro_evidence_provenance"][
        "saved_macro_verified"
    ] is True
    macro_report = next(
        item
        for item in report.reports
        if item.agent_name == "macro_policy"
    )
    assert macro_report.verdict == "neutral"
    assert macro_report.position_bias == "neutral"
    assert "policy_easing" not in report.summary["top_patterns"]
