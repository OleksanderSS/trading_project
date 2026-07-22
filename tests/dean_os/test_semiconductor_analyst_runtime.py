from __future__ import annotations

import json

from dean_os.analysts._producers.runtime import (
    SemiconductorAnalystRuntime,
)
from dean_os.structured_context_provenance import (
    audit_structured_context,
)


AS_OF = "2026-06-30T21:00:00+00:00"


def _fundamental_fragment():
    fundamentals = {
        ticker: {
            "revenue": {
                "value": 100.0,
                "unit": "USD",
                "period": "2026-Q1",
                "available_at": "2026-06-20T12:00:00+00:00",
                "source_locator": f"https://example.test/{ticker}",
            }
        }
        for ticker in ("NVDA", "AMD", "INTC", "TSM")
    }
    accepted = audit_structured_context(
        fundamentals=fundamentals,
        macro={},
        sector_data={},
        as_of=AS_OF,
    )["accepted_context"]["fundamentals"]
    return {
        "as_of": AS_OF,
        "fundamentals": accepted,
        "metadata": {"verified": True},
    }


def _macro_fragment():
    macro = {
        "policy_rate": {
            "value": 4.0,
            "unit": "percent",
            "period": "2026-06",
            "available_at": "2026-06-20T12:00:00+00:00",
            "source_locator": "https://example.test/macro",
        }
    }
    accepted = audit_structured_context(
        fundamentals={},
        macro=macro,
        sector_data={},
        as_of=AS_OF,
    )["accepted_context"]["macro"]
    return {
        "as_of": AS_OF,
        "macro": accepted,
        "metadata": {"verified": True},
    }


def _sector_fragment():
    sector = {
        "sector_median_return": {
            "value": 2.0,
            "unit": "percent",
            "period": "2026-05-27/2026-06-26",
            "available_at": "2026-06-27T12:00:00+00:00",
            "source_locator": "prices_1d_resampled.parquet",
            "metadata": {
                "evidence_type": "market_confirmation",
                "required_lane_eligible": True,
                "stance_hint": "positive",
            },
        }
    }
    accepted = audit_structured_context(
        fundamentals={},
        macro={},
        sector_data=sector,
        as_of=AS_OF,
    )["accepted_context"]["sector_data"]
    return {
        "as_of": AS_OF,
        "sector_data": accepted,
        "metadata": {"verified": True},
    }


def _news_fragment():
    def record(
        evidence_type: str,
        title: str,
        matched_terms: list[str],
    ):
        return {
            "title": title,
            "summary": title,
            "source": "Reuters",
            "published_at": "2026-06-20T12:00:00+00:00",
            "url": f"https://example.test/{evidence_type}",
            "_dean_semantic_evidence": {
                "producer_contract": (
                    "dean_saved_semiconductor_news_evidence_"
                    "producer_v1"
                ),
                "evidence_type": evidence_type,
                "required_lane_eligible": True,
                "source_tier": "tier_2_strong_context",
                "source_identity": "reuters",
                "matched_terms": matched_terms,
                "candidate_sha256": f"candidate-{evidence_type}",
                "stance_hint": "unknown",
            },
        }

    return {
        "as_of": AS_OF,
        "news": [
            record(
                "sector_demand",
                "Nvidia AI demand remains strong",
                ["ai demand"],
            ),
            record(
                "capex_cycle",
                "Data center capital spending expands",
                ["capital spending"],
            ),
            record(
                "supply_chain",
                "Memory supply constraints continue",
                ["supply constraints"],
            ),
        ],
        "metadata": {
            "verified": True,
            "ready_required_lanes": [
                "sector_demand",
                "capex_cycle",
                "supply_chain",
            ],
        },
    }


def _policy_fragment():
    return {
        "as_of": AS_OF,
        "news": [
            {
                "title": "BIS advanced computing license guidance",
                "summary": "A license requirement continues to apply.",
                "source": "U.S. Bureau of Industry and Security",
                "published_at": "2026-05-31T00:00:00+00:00",
                "url": "https://bis.test/guidance.pdf",
                "_dean_semantic_evidence": {
                    "producer_contract": (
                        "dean_saved_official_policy_evidence_producer_v1"
                    ),
                    "evidence_type": "policy_or_geopolitical",
                    "required_lane_eligible": True,
                    "source_tier": "tier_1_core_evidence",
                    "source_identity": "us_bureau_industry_security",
                    "matched_terms": ["license requirement"],
                    "candidate_sha256": "policy-candidate",
                    "stance_hint": "unknown",
                },
            }
        ],
        "metadata": {"verified": True},
    }


def _patch_loaders(monkeypatch):
    monkeypatch.setattr(
        "dean_os.semiconductor_analyst_runtime."
        "load_verified_merged_fundamental_context_fragment",
        lambda *args, **kwargs: _fundamental_fragment(),
    )
    monkeypatch.setattr(
        "dean_os.semiconductor_analyst_runtime."
        "load_verified_macro_context_fragment",
        lambda *args, **kwargs: _macro_fragment(),
    )
    monkeypatch.setattr(
        "dean_os.semiconductor_analyst_runtime."
        "load_verified_sector_market_context_fragment",
        lambda *args, **kwargs: _sector_fragment(),
    )


def test_runtime_combines_context_but_blocks_incomplete_sector_thesis(
    tmp_path,
    monkeypatch,
):
    _patch_loaders(monkeypatch)
    sources = []
    for name in ("fundamental.json", "macro.json", "sector.json"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        sources.append(path)

    payload = SemiconductorAnalystRuntime(
        output_dir=tmp_path / "output"
    ).run(
        fundamental_artifact_path=sources[0],
        macro_artifact_path=sources[1],
        sector_market_artifact_path=sources[2],
        as_of=AS_OF,
        save=False,
    )

    assert payload["status"] == "semiconductor_analysis_needs_more_data"
    assert payload["summary"]["recommendation"] == "needs_more_data"
    assert payload["summary"]["market_confirmation_ready"] is True
    assert payload["summary"]["satisfied_required_lane_count"] == 1
    assert payload["summary"]["missing_required_evidence"] == [
        "sector_demand",
        "capex_cycle",
        "supply_chain",
        "policy_or_geopolitical",
    ]
    assert payload["summary"]["thesis_confidence"] == 0.0
    assert payload["summary"]["sector_thesis_ready"] is False
    assert payload["summary"]["can_create_ticker_forecast"] is False
    assert payload["integration_boundary"]["amd_is_sector_proxy"] is False
    evidence_types = {
        item["evidence_type"]
        for item in payload["analyst_report"]["evidence"]
    }
    assert {
        "fundamental_context",
        "macro_context",
        "market_confirmation",
    }.issubset(evidence_types)


def test_runtime_records_amd_pipeline_case_as_excluded(
    tmp_path,
    monkeypatch,
):
    _patch_loaders(monkeypatch)
    sources = []
    for name in ("fundamental.json", "macro.json", "sector.json"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        sources.append(path)

    exact_case = tmp_path / "pipeline_case_run.json"
    case = {
        "created_at": "2026-06-29T14:00:00+00:00",
        "mode": "pipeline_model_case_packet",
        "summary": {
            "case_scope": "ticker_model_evaluation_only",
            "eligible_as_domain_evidence": False,
            "can_trade": False,
            "case_id": "amd-case",
            "case_classification": "negative_evaluation_block_case",
        },
        "case": {
            "lineage": {
                "ticker": "AMD",
                "model": "random_forest",
                "target_name": "target_intraday_up_15m",
                "timeframe": "15m",
            }
        },
        "saved_paths": {"json": str(exact_case)},
    }
    exact_case.write_text(json.dumps(case), encoding="utf-8")
    latest_case = tmp_path / "pipeline_case_latest.json"
    latest_case.write_text(json.dumps(case), encoding="utf-8")

    payload = SemiconductorAnalystRuntime(
        output_dir=tmp_path / "output"
    ).run(
        fundamental_artifact_path=sources[0],
        macro_artifact_path=sources[1],
        sector_market_artifact_path=sources[2],
        pipeline_case_artifact_path=latest_case,
        as_of=AS_OF,
        save=False,
    )

    boundary = payload["pipeline_case_boundary"]
    assert boundary["status"] == "excluded_from_domain_evidence"
    assert boundary["ticker"] == "AMD"
    assert boundary["target"] == "target_intraday_up_15m"
    assert (
        payload["integration_boundary"][
            "amd_pipeline_case_can_close_market_confirmation"
        ]
        is False
    )


def test_runtime_accepts_only_verified_news_lane_eligibility(
    tmp_path,
    monkeypatch,
):
    _patch_loaders(monkeypatch)
    monkeypatch.setattr(
        "dean_os.semiconductor_analyst_runtime."
        "load_verified_semiconductor_news_context_fragment",
        lambda *args, **kwargs: _news_fragment(),
    )
    sources = []
    for name in (
        "fundamental.json",
        "macro.json",
        "sector.json",
        "news.json",
    ):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        sources.append(path)

    payload = SemiconductorAnalystRuntime(
        output_dir=tmp_path / "output"
    ).run(
        fundamental_artifact_path=sources[0],
        macro_artifact_path=sources[1],
        sector_market_artifact_path=sources[2],
        news_artifact_path=sources[3],
        as_of=AS_OF,
        save=False,
    )

    assert payload["summary"]["satisfied_required_lane_count"] == 4
    assert payload["summary"]["missing_required_evidence"] == [
        "policy_or_geopolitical",
    ]
    assert payload["summary"]["news_ready_required_lanes"] == [
        "sector_demand",
        "capex_cycle",
        "supply_chain",
    ]
    assert (
        payload["integration_boundary"][
            "keyword_only_news_can_close_required_lane"
        ]
        is False
    )


def test_runtime_reaches_five_lanes_with_verified_official_policy(
    tmp_path,
    monkeypatch,
):
    _patch_loaders(monkeypatch)
    monkeypatch.setattr(
        "dean_os.semiconductor_analyst_runtime."
        "load_verified_semiconductor_news_context_fragment",
        lambda *args, **kwargs: _news_fragment(),
    )
    monkeypatch.setattr(
        "dean_os.semiconductor_analyst_runtime."
        "load_verified_official_policy_context_fragment",
        lambda *args, **kwargs: _policy_fragment(),
    )
    sources = []
    for name in ("fundamental", "macro", "sector", "news", "policy"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}", encoding="utf-8")
        sources.append(path)

    payload = SemiconductorAnalystRuntime().run(
        fundamental_artifact_path=sources[0],
        macro_artifact_path=sources[1],
        sector_market_artifact_path=sources[2],
        news_artifact_path=sources[3],
        official_policy_artifact_path=sources[4],
        as_of=AS_OF,
        save=False,
    )

    assert payload["summary"]["satisfied_required_lane_count"] == 5
    assert payload["summary"]["missing_required_evidence"] == []
    assert payload["summary"]["sector_thesis_ready"] is True
    assert payload["summary"]["can_trade"] is False
