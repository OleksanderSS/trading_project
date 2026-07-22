"""Tests for CLI entry point run_analyst.py.

Uses synthetic artifact fixtures (same as artifact loader tests).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_analyst import main, _render_markdown, _build_argparser


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures (reuse pattern from test_analyst_core_artifact_loader.py)
# ──────────────────────────────────────────────────────────────────────────────


def _make_news_artifact(tmp: Path) -> Path:
    artifact_dir = tmp / "news_artifact"
    artifact_dir.mkdir()
    data = {
        "run_id": "test_run_001",
        "created_at": "2026-07-01T00:00:00Z",
        "producer_contract": "dean_saved_semiconductor_news_evidence_producer_v1",
        "status": "semiconductor_news_evidence_ready_with_gaps",
        "market_context_fragment": {
            "news": [
                {
                    "title": "AI demand accelerates",
                    "summary": "NVIDIA reports record GPU orders",
                    "source": "reuters",
                    "published_at": "2026-06-28T10:00:00Z",
                    "_dean_semantic_evidence": {
                        "producer_contract": "dean_saved_semiconductor_news_evidence_producer_v1",
                        "evidence_type": "sector_demand",
                        "required_lane_eligible": True,
                        "source_tier": "tier_2_strong_context",
                        "source_identity": "reuters",
                        "matched_terms": ["gpu orders"],
                        "candidate_sha256": "abc123",
                        "stance_hint": "positive",
                    },
                },
            ],
        },
        "safety": {"review_only": True},
    }
    with open(artifact_dir / "latest.json", "w") as f:
        json.dump(data, f)
    return artifact_dir


def _make_macro_artifact(tmp: Path) -> Path:
    artifact_dir = tmp / "macro_artifact"
    artifact_dir.mkdir()
    data = {
        "run_id": "test_macro_001",
        "created_at": "2026-07-01T00:00:00Z",
        "producer_contract": "dean_saved_macro_evidence_producer_v1",
        "status": "macro_evidence_ready_with_exclusions",
        "selected_observations": [
            {
                "context_key": "fed_funds_rate",
                "value": 5.25,
                "unit": "percent",
                "period": "2026-06-26",
                "available_at": "2026-06-29T23:59:59Z",
                "source_locator": "https://fred.stlouisfed.org/series/FEDFUNDS",
                "required_lane_eligible": False,
                "stance_hint": "unknown",
            },
        ],
        "safety": {"review_only": True},
    }
    with open(artifact_dir / "latest.json", "w") as f:
        json.dump(data, f)
    return artifact_dir


# ──────────────────────────────────────────────────────────────────────────────
# CLI Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestCLIListDomains:
    def test_list_domains_exits_cleanly(self, capsys):
        code = main(["--list-domains"])
        assert code == 0
        captured = capsys.readouterr()
        assert "semiconductor_ai_infrastructure" in captured.out

    def test_list_domains_contains_energy(self, capsys):
        code = main(["--list-domains"])
        captured = capsys.readouterr()
        assert "energy" in captured.out


class TestCLINoArgs:
    def test_no_args_exits_with_error(self):
        with pytest.raises(SystemExit):
            main([])


class TestCLIWithProducerArtifacts:
    def test_news_only(self, tmp_path, capsys):
        news = _make_news_artifact(tmp_path)
        output_dir = tmp_path / "output"

        code = main([
            "--domain", "semiconductor_ai_infrastructure",
            "--news-artifact", str(news),
            "--output-dir", str(output_dir),
            "--format", "json",
        ])

        assert code == 0
        captured = capsys.readouterr()
        assert "Loaded 1 evidence items" in captured.out

        json_file = output_dir / "semiconductor_ai_infrastructure_report.json"
        assert json_file.exists()
        report = json.loads(json_file.read_text())
        assert report["report_type"] == "sector_analysis"
        assert report["domain_id"] == "semiconductor_ai_infrastructure"
        assert report["stats"]["evidence_count"] == 1

    def test_multiple_producers(self, tmp_path, capsys):
        news = _make_news_artifact(tmp_path)
        macro = _make_macro_artifact(tmp_path)
        output_dir = tmp_path / "output"

        code = main([
            "--domain", "semiconductor_ai_infrastructure",
            "--news-artifact", str(news),
            "--macro-artifact", str(macro),
            "--output-dir", str(output_dir),
            "--format", "both",
        ])

        assert code == 0
        captured = capsys.readouterr()
        assert "Loaded 2 evidence items" in captured.out
        assert (output_dir / "semiconductor_ai_infrastructure_report.json").exists()
        assert (output_dir / "semiconductor_ai_infrastructure_report.md").exists()

    def test_markdown_format(self, tmp_path, capsys):
        news = _make_news_artifact(tmp_path)
        output_dir = tmp_path / "output"

        code = main([
            "--domain", "semiconductor_ai_infrastructure",
            "--news-artifact", str(news),
            "--output-dir", str(output_dir),
            "--format", "markdown",
        ])

        assert code == 0
        md_file = output_dir / "semiconductor_ai_infrastructure_report.md"
        assert md_file.exists()
        content = md_file.read_text()
        assert "# Sector Analysis" in content
        assert "semiconductor_ai_infrastructure" in content


class TestCLIWithRuntimeArtifact:
    def test_runtime_artifact_no_evidence(self, tmp_path):
        """Runtime artifact without adapter.evidence should fail gracefully."""
        artifact_dir = tmp_path / "runtime"
        artifact_dir.mkdir()
        data = {
            "adapter": {"status": "review_context_ready"},
            "safety": {"review_only": True},
        }
        with open(artifact_dir / "latest.json", "w") as f:
            json.dump(data, f)

        code = main([
            "--domain", "semiconductor_ai_infrastructure",
            "--runtime-artifact", str(artifact_dir),
        ])

        assert code == 1


class TestCLIMissingArtifact:
    def test_nonexistent_artifact_exits_with_error(self):
        code = main([
            "--domain", "semiconductor_ai_infrastructure",
            "--runtime-artifact", "/nonexistent/path",
        ])
        assert code == 1


class TestCLIArtifactConflict:
    def test_runtime_and_producer_conflict(self):
        with pytest.raises(SystemExit):
            main([
                "--domain", "semiconductor_ai_infrastructure",
                "--runtime-artifact", "/some/path",
                "--news-artifact", "/some/other/path",
            ])


class TestCLIQuietMode:
    def test_quiet_suppresses_stdout(self, tmp_path, capsys):
        news = _make_news_artifact(tmp_path)
        output_dir = tmp_path / "output"

        code = main([
            "--domain", "semiconductor_ai_infrastructure",
            "--news-artifact", str(news),
            "--output-dir", str(output_dir),
            "--quiet",
        ])

        assert code == 0
        captured = capsys.readouterr()
        assert captured.out == ""
        assert (output_dir / "semiconductor_ai_infrastructure_report.json").exists()


class TestRenderMarkdown:
    def test_render_markdown_from_report(self, tmp_path):
        from dean_os.analyst_core.sector_analyst import SectorAnalyst
        from dean_os.analyst_core.artifact_evidence_loader import load_evidence_from_artifacts

        news = _make_news_artifact(tmp_path)
        evidence = load_evidence_from_artifacts(
            artifact_paths={"news": str(news)},
            domain_id="semiconductor_ai_infrastructure",
        )

        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        report = analyst.run_from_evidence(evidence=evidence, as_of="2026-07-01")

        md = _render_markdown(report)
        assert "# Sector Analysis" in md
        assert "review_only: True" in md
        assert "live_execution_allowed: False" in md


class TestArgParser:
    def test_parser_has_all_flags(self):
        parser = _build_argparser()
        args = parser.parse_args([
            "--domain", "energy",
            "--as-of", "2026-07-01",
            "--news-artifact", "/path/to/news",
            "--macro-artifact", "/path/to/macro",
            "--output-dir", "/output",
            "--format", "json",
            "--quiet",
        ])
        assert args.domain == "energy"
        assert args.as_of == "2026-07-01"
        assert args.news_artifact == "/path/to/news"
        assert args.macro_artifact == "/path/to/macro"
        assert args.output_dir == "/output"
        assert args.format == "json"
        assert args.quiet is True
