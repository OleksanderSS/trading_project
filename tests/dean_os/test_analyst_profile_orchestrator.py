from __future__ import annotations

import asyncio

import pandas as pd

from dean_os.analyst_evidence_pack import AnalystEvidencePackRunner
from dean_os.analyst_profile_orchestrator import AnalystProfileOrchestrator


def _pack(tmp_path):
    news_path = tmp_path / "news.csv"
    pd.DataFrame(
        [
            {
                "title": "MSFT AI partnership expands",
                "summary": "MSFT cloud AI demand and partnership news improve data center outlook.",
                "published_at": "2026-01-05T00:00:00+00:00",
                "ticker": "MSFT",
            }
        ]
    ).to_csv(news_path, index=False)
    return AnalystEvidencePackRunner(output_dir=tmp_path / "pack").run(
        news_data_paths=[news_path],
        tickers=["MSFT"],
        sectors=["software"],
        tags=["ai_cycle"],
    )


def test_analyst_profile_orchestrator_runs_base_profile_and_review(tmp_path):
    payload = _pack(tmp_path)
    pack_path = tmp_path / "pack" / "latest.json"

    result = asyncio.run(
        AnalystProfileOrchestrator(output_dir=tmp_path / "profiles").run(
            evidence_pack_path=pack_path,
            build_review_snapshot=True,
        )
    )

    assert result["profile_plan"]["profiles_to_run"] == ["generalist_base_analyst"]
    assert result["profile_runs"][0]["status"] == "completed"
    assert result["profile_runs"][0]["runner"] == "agent_lab"
    assert result["profile_runs"][0]["document_count"] == payload["coverage"]["document_count"]
    assert result["review_snapshot"] is not None
    assert (tmp_path / "profiles" / "latest.json").exists()


def test_analyst_profile_orchestrator_skips_candidate_without_permission(tmp_path):
    _pack(tmp_path)
    pack_path = tmp_path / "pack" / "latest.json"

    result = asyncio.run(
        AnalystProfileOrchestrator(output_dir=tmp_path / "profiles").run(
            evidence_pack_path=pack_path,
            profiles=["generalist_base_analyst", "news_catalyst"],
            allow_candidate_profiles=False,
            build_review_snapshot=False,
        )
    )

    assert result["profile_plan"]["profiles_to_run"] == ["generalist_base_analyst"]
    assert result["profile_plan"]["skipped_profiles"][0]["profile"] == "news_catalyst"
    assert "allow-candidate-profiles" in result["profile_plan"]["skipped_profiles"][0]["reason"]


def test_analyst_profile_orchestrator_can_run_allowed_candidate_profile(tmp_path):
    _pack(tmp_path)
    pack_path = tmp_path / "pack" / "latest.json"

    result = asyncio.run(
        AnalystProfileOrchestrator(output_dir=tmp_path / "profiles").run(
            evidence_pack_path=pack_path,
            profiles=["news_catalyst"],
            allow_candidate_profiles=True,
            build_review_snapshot=False,
        )
    )

    assert result["profile_plan"]["profiles_to_run"] == ["news_catalyst"]
    assert result["profile_runs"][0]["profile"] == "news_catalyst"
    assert result["profile_runs"][0]["runner"] == "domain_agent"
    assert result["analytical_reports"][0]["agent_name"] == "news_catalyst"

