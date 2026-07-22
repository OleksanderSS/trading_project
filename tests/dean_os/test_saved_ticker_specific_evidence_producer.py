from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dean_os.analysts._producers.news import (
    SavedSemiconductorNewsEvidenceProducer,
)
from dean_os.analysts._producers.ticker import (
    SavedTickerSpecificEvidenceProducer,
    load_verified_ticker_specific_evidence_fragment,
)


AS_OF = "2026-06-30T21:00:00+00:00"


def _news_source(tmp_path: Path) -> Path:
    rows = [
        {
            "title": (
                "AMD shares soar after strong AI demand sales forecast"
            ),
            "description": None,
            "published_date": "2026-05-05T20:19:47+00:00",
            "publishedAt": None,
            "link": "https://bloomberg.test/amd-demand",
            "url": None,
            "source": "Bloomberg.com",
        },
        {
            "title": (
                "AMD forecasts revenue above expectations on strong "
                "AI demand"
            ),
            "description": None,
            "published_date": "2026-05-05T22:28:48+00:00",
            "publishedAt": None,
            "link": "https://reuters.test/amd-demand",
            "url": None,
            "source": "Reuters",
        },
        {
            "title": (
                "NVIDIA faces supply constraints and China uncertainty"
            ),
            "description": None,
            "published_date": "2026-05-17T14:48:40+00:00",
            "publishedAt": None,
            "link": "https://bloomberg.test/nvda-supply",
            "url": None,
            "source": "Bloomberg.com",
        },
        {
            "title": (
                "sAMD supplier sees strong AI demand for chips"
            ),
            "description": None,
            "published_date": "2026-05-18T12:00:00+00:00",
            "publishedAt": None,
            "link": "https://cnbc.test/not-amd",
            "url": None,
            "source": "CNBC",
        },
    ]
    path = tmp_path / "news.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _build_news(tmp_path: Path) -> tuple[Path, Path]:
    source = _news_source(tmp_path)
    output = tmp_path / "news_output"
    SavedSemiconductorNewsEvidenceProducer(
        output_dir=output
    ).build(
        source_path=source,
        as_of=AS_OF,
    )
    return source, output / "latest.json"


def test_ticker_specific_producer_requires_exact_alias_and_corroboration(
    tmp_path,
):
    _, news_artifact = _build_news(tmp_path)

    payload = SavedTickerSpecificEvidenceProducer(
        tmp_path / "ticker_output"
    ).build(
        news_artifact_path=news_artifact,
        as_of=AS_OF,
    )

    assert payload["status"] == (
        "ticker_specific_evidence_ready_with_gaps"
    )
    assert payload["summary"]["eligible_tickers"] == ["AMD"]
    assert payload["summary"][
        "missing_corroborated_tickers"
    ] == ["INTC", "NVDA", "TSM"]
    amd_demand = next(
        item
        for item in payload["lane_review"]
        if item["ticker"] == "AMD"
        and item["evidence_type"] == "sector_demand"
    )
    assert amd_demand["status"] == "corroborated"
    assert amd_demand["independent_strong_source_count"] == 2
    assert set(
        amd_demand["independent_strong_sources"]
    ) == {"bloomberg", "reuters"}
    assert amd_demand["dominant_stance"] == "positive"
    assert all(
        item["ticker_thesis_eligible"]
        for item in payload["records"]
        if item["ticker"] == "AMD"
        and item["source_identity"]
        in {"bloomberg", "reuters"}
    )
    assert not any(
        item["summary"].startswith("sAMD")
        for item in payload["records"]
    )
    assert payload["summary"]["can_create_ticker_thesis"] is False
    assert payload["summary"]["can_create_ticker_forecast"] is False
    assert payload["integration_boundary"][
        "sector_context_can_close_ticker_lane"
    ] is False

    fragment = load_verified_ticker_specific_evidence_fragment(
        tmp_path / "ticker_output" / "latest.json",
        expected_as_of=AS_OF,
    )
    assert fragment["metadata"]["verified"] is True
    assert {
        item["ticker"]
        for item in fragment["records"]
        if item["ticker_thesis_eligible"]
    } == {"AMD"}


def test_ticker_specific_loader_rejects_changed_news_source(
    tmp_path,
):
    source, news_artifact = _build_news(tmp_path)
    output = tmp_path / "ticker_output"
    SavedTickerSpecificEvidenceProducer(output).build(
        news_artifact_path=news_artifact,
        as_of=AS_OF,
    )
    source.write_bytes(source.read_bytes() + b"tamper")

    with pytest.raises(
        ValueError,
        match="semiconductor news source_provenance hash mismatch",
    ):
        load_verified_ticker_specific_evidence_fragment(
            output / "latest.json"
        )


def test_ticker_specific_producer_rejects_unknown_ticker(tmp_path):
    _, news_artifact = _build_news(tmp_path)

    with pytest.raises(
        ValueError,
        match="issuer registry missing tickers",
    ):
        SavedTickerSpecificEvidenceProducer(
            tmp_path / "ticker_output"
        ).build(
            news_artifact_path=news_artifact,
            as_of=AS_OF,
            tickers=["UNKNOWN"],
            save=False,
        )
