#!/usr/bin/env python3
"""CLI entry point for SectorAnalyst.

Runs a domain analysis from saved producer artifacts and writes the report
as JSON + markdown summary.

Examples:
    # From full runtime artifact (152 adapted evidence items)
    python run_analyst.py --domain semiconductor_ai_infrastructure --runtime-artifact reports/dean_os/semiconductor_analyst_runtime_current

    # From individual producer artifacts
    python run_analyst.py --domain semiconductor_ai_infrastructure \\
        --news-artifact reports/dean_os/saved_semiconductor_news_evidence_producer_current \\
        --macro-artifact reports/dean_os/saved_macro_evidence_producer_current \\
        --sector-market-artifact reports/dean_os/saved_sector_market_evidence_producer_current

    # List available domains
    python run_analyst.py --list-domains
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_project_root() -> None:
    """Add project root to sys.path so dean_os imports resolve."""
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_add_project_root()

from dean_os.analyst_core.artifact_evidence_loader import load_evidence_from_artifacts
from dean_os.analyst_core.sector_analyst import SectorAnalyst, SectorReport
from dean_os.analysts.profiles import list_domain_profiles
from dean_os.analysts.domain_feeder import DomainDataFeeder


def _render_markdown(report: SectorReport) -> str:
    """Render a human-readable markdown summary of the report."""
    lines: list[str] = []
    t = report.thesis

    lines.append(f"# Sector Analysis: {report.domain_id}")
    lines.append(f"**As of:** {report.as_of}")
    lines.append(f"**Recommendation:** `{report.recommendation}`")
    lines.append(f"**Review required:** {report.review_required}")
    lines.append(f"**Live execution allowed:** {report.live_execution_allowed}")
    lines.append("")

    # Thesis
    lines.append("## Thesis")
    lines.append(f"- **Stance:** {t.stance}")
    lines.append(f"- **Direction:** {t.expected_direction}")
    lines.append(f"- **Confidence:** {t.confidence:.2f}")
    lines.append(f"- **Thesis:** {t.thesis}")
    lines.append("")

    if t.key_drivers:
        lines.append("### Key Drivers")
        for d in t.key_drivers:
            lines.append(f"- {d}")
        lines.append("")

    if t.risks:
        lines.append("### Risks")
        for r in t.risks:
            lines.append(f"- {r}")
        lines.append("")

    if t.blind_spots:
        lines.append("### Blind Spots")
        for b in t.blind_spots:
            lines.append(f"- {b}")
        lines.append("")

    # Ticker basket
    basket = report.ticker_basket
    lines.append("## Ticker Basket")
    lines.append(f"**Status:** {basket.basket_status}")
    for c in basket.candidates:
        blocked = f" [BLOCKED: {', '.join(c.blocked_reasons)}]" if c.blocked_reasons else ""
        lines.append(
            f"- `{c.ticker}` — {c.candidate_status} | "
            f"direction={c.expected_direction} | confidence={c.confidence:.2f}{blocked}"
        )
    lines.append("")

    # Lens analysis
    if report.regime_context:
        lines.append("## Regime Context")
        ctx = report.regime_context
        if hasattr(ctx, "model_dump"):
            ctx_dict = ctx.model_dump()
        else:
            ctx_dict = ctx
        for dim, val in ctx_dict.items():
            lines.append(f"- **{dim}:** {val}")
        lines.append("")

    if report.hypotheses:
        lines.append("## Hypotheses")
        for h in report.hypotheses:
            lines.append(f"- `{h.hypothesis_id}` [{h.status}] — {h.hypothesis}")
            if h.invalidation_signals:
                lines.append(f"  - Invalidation: {', '.join(h.invalidation_signals)}")
        lines.append("")

    if report.evidence_gaps:
        lines.append("## Evidence Gaps")
        for g in report.evidence_gaps:
            lines.append(f"- [{g.priority}] {g.expected_source_type}: {g.description}")
        lines.append("")

    if report.watch_signals:
        lines.append("## Watch Signals")
        for w in report.watch_signals:
            signal_type = w.get("signal_type", "unknown")
            reason = w.get("reason", "")
            lines.append(f"- **{signal_type}:** {reason}")
        lines.append("")

    if report.transmission_channels:
        lines.append("## Transmission Channels")
        for ch in report.transmission_channels:
            name = ch.get("channel_name", ch.get("name", "unknown"))
            lines.append(f"- {name}")
        lines.append("")

    # Stats
    lines.append("## Stats")
    lines.append(f"- Evidence items: {report.evidence_count}")
    lines.append(f"- Evidence exclusions: {report.evidence_exclusion_count}")
    lines.append(f"- Lens deltas: {report.lens_count}")
    lines.append("")

    # Safety
    lines.append("## Safety")
    lines.append("- review_only: True")
    lines.append("- live_execution_allowed: False")
    lines.append("- can_trade: False")

    return "\n".join(lines)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SectorAnalyst from saved producer artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List available domain IDs and exit.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="semiconductor_ai_infrastructure",
        help="Domain ID (default: semiconductor_ai_infrastructure).",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Point-in-time cutoff (ISO format). Defaults to now.",
    )

    # Artifact paths (mutually exclusive groups)
    artifact_group = parser.add_argument_group("artifact paths (use one mode)")
    artifact_group.add_argument(
        "--runtime-artifact",
        type=str,
        default=None,
        help="Path to full runtime artifact directory (contains latest.json with adapter.evidence).",
    )
    artifact_group.add_argument(
        "--news-artifact",
        type=str,
        default=None,
        help="Path to news producer artifact directory.",
    )
    artifact_group.add_argument(
        "--macro-artifact",
        type=str,
        default=None,
        help="Path to macro producer artifact directory.",
    )
    artifact_group.add_argument(
        "--sector-market-artifact",
        type=str,
        default=None,
        help="Path to sector market producer artifact directory.",
    )
    artifact_group.add_argument(
        "--policy-artifact",
        type=str,
        default=None,
        help="Path to policy producer artifact directory.",
    )
    artifact_group.add_argument(
        "--fundamental-artifact",
        type=str,
        default=None,
        help="Path to fundamental (SEC) producer artifact directory.",
    )

    # Output options
    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="reports/dean_os/analyst_output",
        help="Directory to write output files (default: reports/dean_os/analyst_output).",
    )
    output_group.add_argument(
        "--format",
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format (default: both).",
    )
    output_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout output, only write files.",
    )

    # Feeding Context (Plain Text/JSON files)
    feed_group = parser.add_argument_group("feed custom context (files)")
    feed_group.add_argument(
        "--feed-theory",
        type=str,
        default=None,
        help="Path to text/md file with economic or domain theory.",
    )
    feed_group.add_argument(
        "--feed-history",
        type=str,
        default=None,
        help="Path to text/md file with historical context.",
    )
    feed_group.add_argument(
        "--feed-stats",
        type=str,
        default=None,
        help="Path to JSON or text file with industry statistics.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    # List domains
    if args.list_domains:
        domains = list_domain_profiles()
        print("Available domain profiles:")
        for d in domains:
            print(f"  - {d}")
        return 0

    # Validate artifact arguments
    has_runtime = args.runtime_artifact is not None
    has_producers = any([
        args.news_artifact,
        args.macro_artifact,
        args.sector_market_artifact,
        args.policy_artifact,
        args.fundamental_artifact,
    ])

    has_fed_docs = any([
        args.feed_theory,
        args.feed_history,
        args.feed_stats,
    ])

    if not has_runtime and not has_producers and not has_fed_docs:
        parser.error(
            "Provide either --runtime-artifact, at least one producer artifact, "
            "or at least one fed document (--feed-theory, etc)."
        )

    if has_runtime and has_producers:
        parser.error(
            "Cannot use --runtime-artifact together with individual "
            "producer artifacts. Use one mode."
        )

    # Build artifact_paths dict
    artifact_paths: dict[str, str] = {}
    if has_runtime:
        artifact_paths["runtime"] = args.runtime_artifact
    else:
        if args.news_artifact:
            artifact_paths["news"] = args.news_artifact
        if args.macro_artifact:
            artifact_paths["macro"] = args.macro_artifact
        if args.sector_market_artifact:
            artifact_paths["sector_market"] = args.sector_market_artifact
        if args.policy_artifact:
            artifact_paths["policy"] = args.policy_artifact
        if args.fundamental_artifact:
            artifact_paths["fundamental"] = args.fundamental_artifact

    from dean_os.schemas import utc_now_iso
    active_as_of = args.as_of or utc_now_iso()

    # Load evidence
    evidence = []
    if has_runtime or has_producers:
        try:
            evidence = load_evidence_from_artifacts(
                artifact_paths=artifact_paths,
                domain_id=args.domain,
                as_of=active_as_of,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading evidence: {e}", file=sys.stderr)
            return 1

    if not args.quiet and (has_runtime or has_producers):
        print(f"Loaded {len(evidence)} evidence items from saved artifacts for domain={args.domain}")

    # Build a MarketContext to hold any fed raw documents
    from dean_os.schemas import MarketContext
    fed_context = MarketContext(as_of=active_as_of)
    feeder = DomainDataFeeder(args.domain)
    
    if args.feed_theory:
        feeder.feed_theory(fed_context, args.feed_theory)
    if args.feed_history:
        feeder.feed_history(fed_context, args.feed_history)
    if args.feed_stats:
        feeder.feed_stats(fed_context, args.feed_stats)
        
    # Adapt fed documents to evidence
    from dean_os.analysts.context_adapter import MarketContextEvidenceAdapter
    if fed_context.research_documents:
        adapter = MarketContextEvidenceAdapter(domain_id=args.domain)
        fed_evidence = adapter.from_context(fed_context, active_as_of)
        evidence.extend(fed_evidence)
        if not args.quiet:
            print(f"Adapted {len(fed_evidence)} evidence items from fed files.")

    # Run analyst
    if not evidence:
        print("No evidence to analyze.", file=sys.stderr)
        return 1

    analyst = SectorAnalyst(domain_id=args.domain)
    report = analyst.run_from_evidence(
        evidence=evidence,
        as_of=active_as_of,
    )

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_slug = args.domain.replace("/", "_").replace(" ", "_")

    if args.format in ("json", "both"):
        json_path = output_dir / f"{domain_slug}_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        if not args.quiet:
            print(f"JSON report written to {json_path}")

    if args.format in ("markdown", "both"):
        md_path = output_dir / f"{domain_slug}_report.md"
        md_content = _render_markdown(report)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        if not args.quiet:
            print(f"Markdown report written to {md_path}")

    # Print summary
    if not args.quiet:
        print("")
        print(report.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
