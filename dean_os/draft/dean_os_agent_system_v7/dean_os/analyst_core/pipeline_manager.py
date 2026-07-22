"""SectorPipelineManager — minimal pipeline orchestrator for any domain.

Discovers evidence artifacts → runs analysis → evaluates outcomes → builds knowledge.
Each step is optional; smarter models will refine the wiring later.

Usage:
    pm = SectorPipelineManager(domain_id="energy")
    result = pm.run_analysis(
        artifact_dirs={...},
        as_of="2026-07-01",
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dean_os.analyst_core.domain_analyst_runtime import DomainAnalystRuntime


def _render_report_markdown(report) -> str:
    """Render a SectorReport as human-readable markdown.

    Uses report.to_dict() to stay consistent with the Pydantic model schema.
    """
    d = report.to_dict()
    lines: list[str] = []

    lines.append(f"# Sector Analysis: {d['domain_id']}")
    lines.append(f"**As of:** {d['as_of']}")
    lines.append(f"**Recommendation:** `{d['recommendation']}`")
    lines.append(f"**Review required:** {d['review_required']}")
    lines.append(f"**Live execution allowed:** {d['live_execution_allowed']}")
    lines.append("")

    thesis = d.get("thesis", {})
    lines.append("## Thesis")
    lines.append(f"- **Stance:** {thesis.get('stance', 'unknown')}")
    lines.append(f"- **Direction:** {thesis.get('expected_direction', 'unknown')}")
    lines.append(f"- **Confidence:** {thesis.get('confidence', 0.0):.2f}")
    lines.append(f"- **Thesis:** {thesis.get('thesis', '')}")
    lines.append("")

    for section, label in [("key_drivers", "Key Drivers"), ("risks", "Risks"), ("blind_spots", "Blind Spots")]:
        items = thesis.get(section, [])
        if items:
            lines.append(f"### {label}")
            for item in items:
                lines.append(f"- {item}")
            lines.append("")

    basket = d.get("ticker_basket", {})
    lines.append("## Ticker Basket")
    lines.append(f"**Status:** {basket.get('basket_status', 'unknown')}")
    for c in basket.get("candidates", []):
        blocked = c.get("blocked_reasons", [])
        blocked_str = f" [BLOCKED: {', '.join(blocked)}]" if blocked else ""
        lines.append(
            f"- `{c.get('ticker', '?')}` — {c.get('status', c.get('candidate_status', '?'))} | "
            f"direction={c.get('direction', c.get('expected_direction', '?'))} | "
            f"confidence={c.get('confidence', 0.0):.2f}{blocked_str}"
        )
    lines.append("")

    rc = d.get("regime_context")
    if rc:
        lines.append("## Regime Context")
        for dim, val in rc.items() if isinstance(rc, dict) else []:
            lines.append(f"- **{dim}:** {val}")
        lines.append("")

    for h in d.get("hypotheses", []):
        lines.append(f"- `{h.get('hypothesis_id', '?')}` [{h.get('status', '?')}] — {h.get('hypothesis', '')}")
        invalidation = h.get("invalidation_signals", [])
        if invalidation:
            lines.append(f"  - Invalidation: {', '.join(invalidation)}")

    for g in d.get("evidence_gaps", []):
        lines.append(f"- [{g.get('priority', '?')}] {g.get('expected_source_type', '?')}: {g.get('description', '')}")
        lines.append(f"  - Status: {g.get('current_status', '?')}")

    for w in d.get("watch_signals", []):
        lines.append(f"- **{w.get('signal_type', 'unknown')}:** {w.get('reason', '')}")

    for ch in d.get("transmission_channels", []):
        lines.append(f"- {ch.get('channel_name', ch.get('name', 'unknown'))}")

    stats = d.get("stats", {})
    lines.append("## Stats")
    lines.append(f"- Evidence items: {stats.get('evidence_count', 0)}")
    lines.append(f"- Evidence exclusions: {stats.get('evidence_exclusion_count', 0)}")
    lines.append(f"- Lens deltas: {stats.get('lens_count', 0)}")
    lines.append("")

    lines.append("## Safety")
    lines.append("- review_only: True")
    lines.append("- live_execution_allowed: False")
    lines.append("- can_trade: False")

    return "\n".join(lines)


@dataclass
class PipelineRunResult:
    """Result of a single pipeline run."""
    domain_id: str
    as_of: str
    analysis_result: dict[str, Any] | None = None
    evaluation_result: dict[str, Any] | None = None
    knowledge_result: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _discover_artifact_dir(base: Path, name: str) -> Path | None:
    """Check if a named subdirectory has latest.json."""
    candidate = base / name
    if candidate.is_dir() and (candidate / "latest.json").exists():
        return candidate
    return None


class SectorPipelineManager:
    """Minimal pipeline manager for any domain.

    Args:
        domain_id: Sector identifier (e.g. "energy", "semiconductor_ai_infrastructure").
    """

    def __init__(self, domain_id: str):
        self.domain_id = domain_id
        self.runtime = DomainAnalystRuntime(domain_id=domain_id)

    def discover_artifacts(
        self,
        base_path: str | Path,
    ) -> dict[str, Path | None]:
        """Discover saved producer artifacts by convention.

        Looks for subdirectories named news, macro, sector_market, policy,
        fundamental under *base_path*.
        """
        base = Path(base_path)
        if not base.is_dir():
            return {}
        return {
            "news": _discover_artifact_dir(base, "news"),
            "macro": _discover_artifact_dir(base, "macro"),
            "sector_market": _discover_artifact_dir(base, "sector_market"),
            "policy": _discover_artifact_dir(base, "policy"),
            "fundamental": _discover_artifact_dir(base, "fundamental"),
            "runtime": _discover_artifact_dir(base, "runtime"),
        }

    def run_analysis(
        self,
        *,
        artifact_dirs: dict[str, Path | str | None] | None = None,
        news_path: str | Path | None = None,
        macro_path: str | Path | None = None,
        sector_market_path: str | Path | None = None,
        policy_path: str | Path | None = None,
        fundamental_path: str | Path | None = None,
        runtime_artifact: str | Path | None = None,
        as_of: str,
        tickers: list[str] | None = None,
        horizon_days: int | None = None,
        output_dir: str | Path | None = None,
    ) -> PipelineRunResult:
        """Run the analyst pipeline: load evidence → analyze → report.

        Accepts either explicit paths or a dict of discovered artifact dirs.
        """
        result = PipelineRunResult(domain_id=self.domain_id, as_of=as_of)

        # Merge artifact_dirs with explicit paths
        if artifact_dirs:
            news_path = news_path or artifact_dirs.get("news")
            macro_path = macro_path or artifact_dirs.get("macro")
            sector_market_path = sector_market_path or artifact_dirs.get("sector_market")
            policy_path = policy_path or artifact_dirs.get("policy")
            fundamental_path = fundamental_path or artifact_dirs.get("fundamental")
            runtime_artifact = runtime_artifact or artifact_dirs.get("runtime")

        try:
            if runtime_artifact:
                analysis = self._run_from_runtime(
                    runtime_artifact=Path(runtime_artifact),
                    as_of=as_of,
                    tickers=tickers,
                    horizon_days=horizon_days,
                )
            else:
                analysis = self._run_from_producers(
                    news_path=Path(news_path) if news_path else None,
                    macro_path=Path(macro_path) if macro_path else None,
                    sector_market_path=Path(sector_market_path) if sector_market_path else None,
                    policy_path=Path(policy_path) if policy_path else None,
                    fundamental_path=Path(fundamental_path) if fundamental_path else None,
                    as_of=as_of,
                    tickers=tickers,
                    horizon_days=horizon_days,
                )
            result.analysis_result = analysis
        except Exception as e:
            result.errors.append(f"Analysis failed: {e}")

        if output_dir and result.analysis_result:
            self._save_report(result.analysis_result, output_dir)

        return result

    def _run_from_runtime(
        self,
        runtime_artifact: Path,
        as_of: str,
        tickers: list[str] | None = None,
        horizon_days: int | None = None,
    ) -> dict[str, Any]:
        from dean_os.analyst_core.artifact_evidence_loader import ArtifactEvidenceLoader

        loader = ArtifactEvidenceLoader()
        evidence = loader.from_runtime_artifact(
            runtime_artifact,
            domain_id=self.domain_id,
            as_of=as_of,
        )
        report = self.runtime.analyst.run_from_evidence(
            evidence=evidence,
            as_of=as_of,
            tickers=tickers,
            horizon_days=horizon_days,
        )
        return {
            "domain_id": self.domain_id,
            "as_of": as_of,
            "evidence_source": "runtime_artifact",
            "evidence_count": len(evidence),
            "report": report,
            "status": report.recommendation,
        }

    def _run_from_producers(
        self,
        news_path: Path | None,
        macro_path: Path | None,
        sector_market_path: Path | None,
        policy_path: Path | None,
        fundamental_path: Path | None,
        as_of: str,
        tickers: list[str] | None = None,
        horizon_days: int | None = None,
    ) -> dict[str, Any]:
        return self.runtime.run(
            news_path=news_path,
            macro_path=macro_path,
            sector_market_path=sector_market_path,
            policy_path=policy_path,
            fundamental_path=fundamental_path,
            as_of=as_of,
            tickers=tickers,
            horizon_days=horizon_days,
        )

    def evaluate(
        self,
        *,
        price_data_path: str | Path,
        analysis_result: dict[str, Any] | None = None,
        report=None,
        as_of: str,
        horizons: list[int] | None = None,
    ) -> PipelineRunResult:
        """Evaluate analyst outcomes against actual prices."""
        from dean_os.analyst_core.outcome_evaluator import OutcomeEvaluator

        evaluator = OutcomeEvaluator(price_data_path=Path(price_data_path))
        report_obj = report or (analysis_result or {}).get("report")
        if report_obj is None:
            return PipelineRunResult(
                domain_id=self.domain_id,
                as_of=as_of,
                errors=["No report to evaluate"],
            )

        try:
            eval_result = evaluator.evaluate(report_obj, as_of=as_of, horizons=horizons)
            return PipelineRunResult(
                domain_id=self.domain_id,
                as_of=as_of,
                analysis_result=analysis_result,
                evaluation_result={
                    "summary": eval_result.summary,
                    "horizon_count": len(eval_result.horizons),
                },
            )
        except Exception as e:
            return PipelineRunResult(
                domain_id=self.domain_id,
                as_of=as_of,
                errors=[f"Evaluation failed: {e}"],
            )

    def build_knowledge(
        self,
        *,
        artifact_dirs: dict[str, Path | str | None],
        output_dir: str | Path,
        as_of: str,
    ) -> PipelineRunResult:
        """Build knowledge pack from producer artifact directories."""
        import sys
        _root = str(Path(__file__).resolve().parent.parent.parent)
        if _root not in sys.path:
            sys.path.insert(0, _root)

        from build_knowledge_pack import build_knowledge_pack
        from dean_os.analyst_knowledge.pack_loader import save_knowledge_pack

        # Convert Paths to strings for the function
        str_dirs: dict[str, str] = {}
        for key in ("news", "macro", "sector_market", "policy", "fundamental", "runtime"):
            val = artifact_dirs.get(key)
            if val is not None:
                str_dirs[key] = str(val)

        if not str_dirs:
            return PipelineRunResult(
                domain_id=self.domain_id,
                as_of=as_of,
                errors=["No artifact directories provided"],
            )

        try:
            pack = build_knowledge_pack(self.domain_id, str_dirs, as_of=as_of)
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            pack_path = save_knowledge_pack(pack, output / "pack.json")

            return PipelineRunResult(
                domain_id=self.domain_id,
                as_of=as_of,
                knowledge_result={
                    "pack_path": str(pack_path),
                    "item_count": len(pack.items),
                    "source_count": len(pack.sources),
                },
            )
        except Exception as e:
            return PipelineRunResult(
                domain_id=self.domain_id,
                as_of=as_of,
                errors=[f"Knowledge build failed: {e}"],
            )

    @staticmethod
    def save_report(
        analysis_result: dict[str, Any],
        output_dir: str | Path,
        fmt: str = "both",
    ) -> dict[str, str]:
        """Save analysis report to disk."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        report = analysis_result.get("report")
        if report is None:
            return {}

        paths = {}
        if fmt in ("json", "both"):
            import json
            json_path = output / f"{analysis_result.get('domain_id', 'analyst')}_report.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict() if hasattr(report, "to_dict") else report, f, indent=2, default=str)
            paths["json"] = str(json_path)

        if fmt in ("markdown", "both"):
            md_path = output / f"{analysis_result.get('domain_id', 'analyst')}_report.md"
            md_content = _render_report_markdown(report)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            paths["markdown"] = str(md_path)

        return paths

    def _save_report(self, analysis_result: dict[str, Any], output_dir: str | Path) -> None:
        self.save_report(analysis_result, output_dir)
