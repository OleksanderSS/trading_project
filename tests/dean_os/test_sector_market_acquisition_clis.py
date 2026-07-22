from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import run_agent_clean_yahoo_market_snapshot
import run_agent_domain_sector_market_coverage_bridge
import run_agent_pipeline_control_saved_data_coverage
import run_agent_pipeline_control_saved_price_repair
import run_agent_saved_sector_market_evidence


DOMAIN_TICKERS = {
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
}


def test_all_sector_market_clis_import_and_render_help(capsys):
    """Every CLI in the acquisition-to-envelope chain remains importable."""
    clis = [
        run_agent_clean_yahoo_market_snapshot,
        run_agent_domain_sector_market_coverage_bridge,
        run_agent_pipeline_control_saved_data_coverage,
        run_agent_pipeline_control_saved_price_repair,
        run_agent_saved_sector_market_evidence,
    ]

    for cli_module in clis:
        with patch.object(sys, "argv", [cli_module.__file__, "--help"]):
            with pytest.raises(SystemExit) as exc:
                cli_module.main()
        assert exc.value.code == 0
        assert "usage:" in capsys.readouterr().out.lower()


@patch(
    "run_agent_clean_yahoo_market_snapshot.CleanYahooMarketSnapshot.build",
    new_callable=AsyncMock,
)
def test_clean_snapshot_cli_is_offline_when_builder_is_mocked(
    mock_build: AsyncMock,
    tmp_path: Path,
):
    """The smoke test itself performs no network or filesystem write."""
    mock_build.return_value = {"status": "mocked"}
    artifact_dir = tmp_path / "artifacts"
    output_dir = tmp_path / "reports"

    with patch.object(
        sys,
        "argv",
        [
            "run_agent_clean_yahoo_market_snapshot.py",
            "--domain-id",
            "semiconductor_ai_infrastructure",
            "--end-date",
            "2026-07-22T00:00:00Z",
            "--artifact-dir",
            str(artifact_dir),
            "--output-dir",
            str(output_dir),
        ],
    ):
        assert run_agent_clean_yahoo_market_snapshot.main() == 0

    mock_build.assert_awaited_once()
    assert not artifact_dir.exists()
    assert not output_dir.exists()


@patch(
    "run_agent_clean_yahoo_market_snapshot.CleanYahooMarketSnapshot.build",
    new_callable=AsyncMock,
)
def test_semiconductor_domain_resolves_exact_ticker_scope(mock_build: AsyncMock):
    """The domain resolves to the exact 12-name universe plus SOXX."""
    mock_build.return_value = {"status": "mocked"}

    with patch.object(
        sys,
        "argv",
        [
            "run_agent_clean_yahoo_market_snapshot.py",
            "--domain-id",
            "semiconductor_ai_infrastructure",
            "--end-date",
            "2026-07-22T00:00:00Z",
        ],
    ):
        assert run_agent_clean_yahoo_market_snapshot.main() == 0

    tickers = mock_build.call_args.kwargs["tickers"]
    assert len(tickers) == 13
    assert set(tickers) == DOMAIN_TICKERS


@patch(
    "run_agent_clean_yahoo_market_snapshot.CleanYahooMarketSnapshot.build",
    new_callable=AsyncMock,
)
def test_clean_snapshot_cli_defaults_to_native_15m_only(mock_build: AsyncMock):
    mock_build.return_value = {"status": "mocked"}

    with patch.object(
        sys,
        "argv",
        [
            "run_agent_clean_yahoo_market_snapshot.py",
            "--domain-id",
            "semiconductor_ai_infrastructure",
            "--end-date",
            "2026-07-22T00:00:00+00:00",
        ],
    ):
        assert run_agent_clean_yahoo_market_snapshot.main() == 0

    assert mock_build.call_args.kwargs["timeframes"] == ["15m"]


@patch("run_agent_saved_sector_market_evidence.SavedSectorMarketEvidenceProducer.build")
def test_saved_evidence_cli_passes_domain_profile_scope(mock_build):
    mock_build.return_value = {"status": "mocked"}

    with patch.object(
        sys,
        "argv",
        [
            "run_agent_saved_sector_market_evidence.py",
            "unused.json",
            "--domain-id",
            "semiconductor_ai_infrastructure",
            "--as-of",
            "2026-07-22T00:00:00+00:00",
            "--no-save",
        ],
    ):
        assert run_agent_saved_sector_market_evidence.main() == 0

    kwargs = mock_build.call_args.kwargs
    assert set(kwargs["sector_tickers"]) == DOMAIN_TICKERS - {"SOXX"}
    assert kwargs["benchmark"] == "SOXX"


@pytest.mark.parametrize(
    "extra_args",
    [
        [],
        ["--end-date", "2026-07-22T00:00:00"],
    ],
)
def test_missing_or_naive_end_date_is_rejected(extra_args: list[str]):
    argv = [
        "run_agent_clean_yahoo_market_snapshot.py",
        "--domain-id",
        "semiconductor_ai_infrastructure",
        *extra_args,
    ]
    with patch.object(sys, "argv", argv):
        with pytest.raises(SystemExit) as exc:
            run_agent_clean_yahoo_market_snapshot.main()
    assert exc.value.code != 0
