"""DomainAnalystRuntime — minimal universal analyst runtime.

Works with any economic sector. Clone by changing domain_id and tickers.
Codex will later enhance with detailed settings per domain.

Usage:
    runtime = DomainAnalystRuntime(domain_id="energy")
    result = runtime.run(
        news_path="path/to/news",
        macro_path="path/to/macro",
        as_of="2026-07-01",
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from dean_os.analyst_core.artifact_evidence_loader import ArtifactEvidenceLoader
from dean_os.analyst_core.sector_analyst import SectorAnalyst
from dean_os.analysts.profiles import get_domain_profile, list_domain_profiles


class DomainAnalystRuntime:
    """Minimal universal analyst runtime.
    
    Works with any economic sector. Clone by changing domain_id.
    
    Args:
        domain_id: Sector identifier (e.g. "energy", "semiconductor_ai_infrastructure").
        agent_name: Optional name for this analyst instance.
    """

    def __init__(
        self,
        domain_id: str,
        agent_name: str | None = None,
    ):
        self.domain_id = domain_id
        self.profile = get_domain_profile(domain_id)
        self.agent_name = agent_name or f"{domain_id}_analyst"

        # Core analyst
        self.analyst = SectorAnalyst(
            domain_id=domain_id,
            agent_name=self.agent_name,
        )

        # Evidence loader
        self.evidence_loader = ArtifactEvidenceLoader()

    def run(
        self,
        *,
        news_path: str | Path | None = None,
        macro_path: str | Path | None = None,
        sector_market_path: str | Path | None = None,
        policy_path: str | Path | None = None,
        fundamental_path: str | Path | None = None,
        as_of: str,
        tickers: list[str] | None = None,
        horizon_days: int | None = None,
    ) -> dict[str, Any]:
        """Run the domain analyst.
        
        Args:
            news_path: Path to news producer artifact.
            macro_path: Path to macro producer artifact.
            sector_market_path: Path to sector market producer artifact.
            policy_path: Path to policy producer artifact.
            fundamental_path: Path to fundamental producer artifact.
            as_of: Point-in-time cutoff (ISO format).
            tickers: Optional ticker override.
            horizon_days: Optional horizon override.
            
        Returns:
            Dict with report, metadata, and status.
        """
        # Load evidence
        evidence = self.evidence_loader.from_producer_artifacts(
            news_path=Path(news_path) if news_path else None,
            macro_path=Path(macro_path) if macro_path else None,
            sector_market_path=Path(sector_market_path) if sector_market_path else None,
            policy_path=Path(policy_path) if policy_path else None,
            fundamental_path=Path(fundamental_path) if fundamental_path else None,
            domain_id=self.domain_id,
            as_of=as_of,
        )

        # Run analyst
        report = self.analyst.run_from_evidence(
            evidence=evidence,
            as_of=as_of,
            tickers=tickers,
            horizon_days=horizon_days,
        )

        # Build result
        return {
            "domain_id": self.domain_id,
            "as_of": as_of,
            "report": report,
            "evidence_count": len(evidence),
            "status": report.recommendation,
            "review_required": report.review_required,
            "live_execution_allowed": report.live_execution_allowed,
        }

    def run_from_artifacts(
        self,
        artifact_dir: str | Path,
        as_of: str,
        tickers: list[str] | None = None,
        horizon_days: int | None = None,
    ) -> dict[str, Any]:
        """Run from a directory containing multiple artifacts.
        
        Expects:
            - news/latest.json (optional)
            - macro/latest.json (optional)
            - sector_market/latest.json (optional)
            - policy/latest.json (optional)
            - fundamental/latest.json (optional)
        """
        artifact_dir = Path(artifact_dir)

        # _load_*_evidence methods expect a directory containing latest.json
        news_dir = artifact_dir / "news"
        macro_dir = artifact_dir / "macro"
        sector_market_dir = artifact_dir / "sector_market"
        policy_dir = artifact_dir / "policy"
        fundamental_dir = artifact_dir / "fundamental"

        return self.run(
            news_path=news_dir if (news_dir / "latest.json").exists() else None,
            macro_path=macro_dir if (macro_dir / "latest.json").exists() else None,
            sector_market_path=sector_market_dir if (sector_market_dir / "latest.json").exists() else None,
            policy_path=policy_dir if (policy_dir / "latest.json").exists() else None,
            fundamental_path=fundamental_dir if (fundamental_dir / "latest.json").exists() else None,
            as_of=as_of,
            tickers=tickers,
            horizon_days=horizon_days,
        )

    def clone(
        self,
        domain_id: str,
        **kwargs,
    ) -> DomainAnalystRuntime:
        """Clone this runtime for a different domain.
        
        Args:
            domain_id: New domain identifier.
            **kwargs: Additional overrides (tickers, keywords, etc.)
            
        Returns:
            New DomainAnalystRuntime instance.
        """
        new_runtime = DomainAnalystRuntime(
            domain_id=domain_id,
            agent_name=kwargs.get("agent_name"),
        )

        # Clone analyst with overrides
        new_runtime.analyst = self.analyst.clone(
            domain_id=domain_id,
            ticker_universe=kwargs.get("ticker_universe"),
            sector_keywords=kwargs.get("sector_keywords"),
            core_questions=kwargs.get("core_questions"),
            required_evidence_types=kwargs.get("required_evidence_types"),
        )

        return new_runtime


def list_available_domains() -> list[str]:
    """List all available domain IDs."""
    return list_domain_profiles()


def create_analyst(domain_id: str, **kwargs) -> DomainAnalystRuntime:
    """Factory function to create an analyst for any domain."""
    return DomainAnalystRuntime(domain_id=domain_id, **kwargs)
