from pathlib import Path

import pytest

from dean_os.analysts._producers.policy import (
    SavedOfficialPolicyEvidenceProducer,
    load_verified_official_policy_context_fragment,
)


AS_OF = "2026-06-30T21:00:00+00:00"


def test_current_official_policy_source_closes_corroboration_gap(tmp_path):
    snapshot = Path(
        "reports/dean_os/bis_policy_snapshot_current/latest.json"
    )
    news = Path(
        "reports/dean_os/"
        "saved_semiconductor_news_evidence_producer_current/latest.json"
    )
    if not snapshot.exists() or not news.exists():
        pytest.skip("current policy source artifacts are absent")
    output = tmp_path / "output"
    payload = SavedOfficialPolicyEvidenceProducer(
        output_dir=output
    ).build(
        snapshot_artifact_path=snapshot,
        corroborating_news_artifact_path=news,
        as_of=AS_OF,
    )

    assert payload["summary"]["policy_lane_ready"] is True
    assert payload["corroboration"][
        "combined_independent_source_count"
    ] == 2
    assert set(
        payload["corroboration"]["combined_independent_sources"]
    ) == {"bloomberg", "us_bureau_industry_security"}
    fragment = load_verified_official_policy_context_fragment(
        output / "latest.json",
        expected_as_of=AS_OF,
    )
    assert fragment["metadata"]["saved_official_policy_verified"]
    assert fragment["news"][0]["_dean_semantic_evidence"][
        "required_lane_eligible"
    ]
