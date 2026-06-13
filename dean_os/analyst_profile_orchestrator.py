from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from dean_os.agent_lab import AgentLabRunner
from dean_os.analyst_evidence_pack import documents_from_evidence_pack
from dean_os.agents.domain_research import MacroPolicyAgent, NewsCatalystAgent, SectorCycleAgent, ValueScreeningAgent
from dean_os.review import AgentReviewBuilder
from dean_os.schemas import AnalyticalReport, MarketContext, ResearchDocument, utc_now_iso
from dean_os.utils import json_ready


BASE_PROFILE = "generalist_base_analyst"
DOMAIN_PROFILE_AGENTS = {
    "macro_policy": MacroPolicyAgent,
    "news_catalyst": NewsCatalystAgent,
    "sector_cycle": SectorCycleAgent,
    "value_screening": ValueScreeningAgent,
}


class AnalystProfileOrchestrator:
    """Central manager for analyst subprocesses.

    It starts with one real base analyst profile and only runs specialized
    profiles when explicitly allowed by the caller and supported by the
    evidence-pack manager plan.
    """

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/analyst_profiles",
        corpus_path: str | Path | None = None,
        learning_path: str | Path | None = None,
        operations_path: str | Path | None = None,
        review_actions_path: str | Path | None = None,
        memory_path: str | Path | None = None,
        log_path: str | Path | None = None,
    ):
        self.output_dir = Path(output_dir)
        state_dir = self.output_dir / "state"
        self.corpus_path = Path(corpus_path) if corpus_path else state_dir / "research_corpus.sqlite"
        self.learning_path = Path(learning_path) if learning_path else state_dir / "agent_learning.sqlite"
        self.operations_path = Path(operations_path) if operations_path else state_dir / "operation_queue.sqlite"
        self.review_actions_path = Path(review_actions_path) if review_actions_path else state_dir / "review_actions.sqlite"
        self.memory_path = Path(memory_path) if memory_path else state_dir / "recommendation_memory.sqlite"
        self.log_path = Path(log_path) if log_path else self.output_dir / "events.jsonl"

    async def run(
        self,
        evidence_pack_path: str | Path,
        profiles: list[str] | None = None,
        tickers: list[str] | None = None,
        sectors: list[str] | None = None,
        tags: list[str] | None = None,
        allow_candidate_profiles: bool = False,
        create_learning_records: bool = False,
        include_operation_proposals: bool = False,
        build_review_snapshot: bool = True,
    ) -> dict[str, Any]:
        evidence_pack_path = Path(evidence_pack_path)
        evidence_pack = _load_evidence_pack(evidence_pack_path)
        documents = documents_from_evidence_pack(evidence_pack_path)
        coverage = evidence_pack.get("coverage", {})
        manager_plan = evidence_pack.get("analyst_inputs", {}).get("manager_plan", {})
        resolved_tickers = _normalize_tickers(tickers or coverage.get("tickers", []) or _document_tickers(documents))
        resolved_sectors = _normalize_strings(sectors or coverage.get("sectors", []) or _document_sectors(documents))
        resolved_tags = _normalize_strings(tags or list(coverage.get("by_tag", {}).keys()))
        requested_profiles = _requested_profiles(profiles, manager_plan, bool(documents))
        profile_plan = _profile_plan(
            requested_profiles=requested_profiles,
            manager_plan=manager_plan,
            allow_candidate_profiles=allow_candidate_profiles,
            has_documents=bool(documents),
        )

        run_id = _run_id("analyst_profiles")
        runs: list[dict[str, Any]] = []
        analytical_reports: list[AnalyticalReport] = []
        review_snapshot: dict[str, Any] | None = None

        if BASE_PROFILE in profile_plan["profiles_to_run"]:
            base_run = await self._run_base_profile(
                documents=documents,
                tickers=resolved_tickers,
                sectors=resolved_sectors,
                tags=[*resolved_tags, BASE_PROFILE],
                create_learning_records=create_learning_records,
                include_operation_proposals=include_operation_proposals,
            )
            runs.append(base_run)
            if build_review_snapshot and base_run.get("report_json"):
                review_snapshot = self._build_review_snapshot(base_run["report_json"])

        domain_profiles = [profile for profile in profile_plan["profiles_to_run"] if profile in DOMAIN_PROFILE_AGENTS]
        if domain_profiles:
            context = _domain_context(documents, resolved_tickers, resolved_sectors, evidence_pack)
            for profile in domain_profiles:
                report = await DOMAIN_PROFILE_AGENTS[profile](name=profile, config={"horizon_years": _profile_horizon(profile)}).run(context)
                analytical_reports.append(report)
                runs.append(
                    {
                        "profile": profile,
                        "status": "completed",
                        "runner": "domain_agent",
                        "report": report.model_dump(mode="json"),
                    }
                )

        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "analyst_profile_orchestrator",
            "inputs": {
                "evidence_pack_path": str(evidence_pack_path),
                "profiles": profiles or [],
                "tickers": resolved_tickers,
                "sectors": resolved_sectors,
                "tags": resolved_tags,
                "allow_candidate_profiles": allow_candidate_profiles,
                "create_learning_records": create_learning_records,
                "include_operation_proposals": include_operation_proposals,
            },
            "evidence_pack": {
                "run_id": evidence_pack.get("run_id"),
                "path": str(evidence_pack_path),
                "coverage": coverage,
                "manager_plan": manager_plan,
            },
            "profile_plan": profile_plan,
            "profile_runs": runs,
            "analytical_reports": [report.model_dump(mode="json") for report in analytical_reports],
            "review_snapshot": review_snapshot,
            "recommendations": _recommendations(profile_plan, runs, review_snapshot),
        }
        self.save(payload)
        return payload

    async def _run_base_profile(
        self,
        documents: list[ResearchDocument],
        tickers: list[str],
        sectors: list[str],
        tags: list[str],
        create_learning_records: bool,
        include_operation_proposals: bool,
    ) -> dict[str, Any]:
        if not documents:
            return {"profile": BASE_PROFILE, "status": "skipped", "reason": "No evidence documents available."}
        agent_lab_dir = self.output_dir / "agent_lab"
        runner = AgentLabRunner(
            corpus_path=self.corpus_path,
            learning_path=self.learning_path,
            output_dir=agent_lab_dir,
            operation_queue_path=self.operations_path,
            memory_path=self.memory_path,
            log_path=self.log_path,
        )
        report = await runner.run(
            documents=documents,
            tickers=tickers,
            sectors=sectors,
            tags=tags,
            create_learning_records=create_learning_records,
            include_operations_proposals=include_operation_proposals,
        )
        return {
            "profile": BASE_PROFILE,
            "status": "completed",
            "runner": "agent_lab",
            "agent_lab_run_id": report.run_id,
            "report_json": str(agent_lab_dir / f"{report.run_id}.json"),
            "report_markdown": str(agent_lab_dir / f"{report.run_id}.md"),
            "summary": report.summary,
            "document_count": report.document_count,
            "note_count": report.note_count,
            "learning_record_count": len(report.learning_records),
            "proposal_count": len(report.action_proposals),
        }

    def _build_review_snapshot(self, report_json: str | Path) -> dict[str, Any]:
        builder = AgentReviewBuilder(
            reports_dir=self.output_dir / "agent_lab",
            learning_path=self.learning_path,
            operations_path=self.operations_path,
            review_actions_path=self.review_actions_path,
            memory_path=self.memory_path,
            log_path=self.log_path,
            output_dir=self.output_dir / "review",
        )
        snapshot = builder.build(report_path=report_json)
        json_path, md_path = builder.save(snapshot)
        snapshot["saved_paths"] = {"json": str(json_path), "markdown": str(md_path)}
        return snapshot

    def save(self, payload: dict[str, Any]) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{payload['run_id']}.json"
        md_path = self.output_dir / f"{payload['run_id']}.md"
        latest_json = self.output_dir / "latest.json"
        latest_md = self.output_dir / "latest.md"
        payload["saved_paths"] = {
            "json": str(json_path),
            "markdown": str(md_path),
            "latest_json": str(latest_json),
            "latest_markdown": str(latest_md),
        }
        rendered_json = json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n"
        rendered_md = render_analyst_profile_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def run_sync(*args, **kwargs) -> dict[str, Any]:
    return asyncio.run(AnalystProfileOrchestrator(**kwargs.pop("orchestrator_kwargs", {})).run(*args, **kwargs))


def render_analyst_profile_markdown(payload: dict[str, Any]) -> str:
    plan = payload.get("profile_plan", {})
    lines = [
        "# DEAN-OS Analyst Profile Orchestrator",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Evidence pack: `{payload.get('evidence_pack', {}).get('path')}`",
        f"- Profiles to run: {', '.join(plan.get('profiles_to_run', [])) or 'none'}",
        f"- Skipped profiles: {len(plan.get('skipped_profiles', []))}",
        f"- Completed runs: {sum(1 for item in payload.get('profile_runs', []) if item.get('status') == 'completed')}",
        "",
        "## Profile Runs",
        "",
    ]
    for item in payload.get("profile_runs", []):
        lines.append(f"- `{item.get('profile')}`: {item.get('status')} via {item.get('runner', 'none')}")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _load_evidence_pack(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Evidence pack must be a JSON object: {path}")
    return payload


def _requested_profiles(profiles: list[str] | None, manager_plan: dict[str, Any], has_documents: bool) -> list[str]:
    if profiles:
        return _normalize_strings(profiles)
    active = manager_plan.get("active_profiles") or []
    if active:
        return _normalize_strings(active)
    return [BASE_PROFILE] if has_documents else []


def _profile_plan(
    requested_profiles: list[str],
    manager_plan: dict[str, Any],
    allow_candidate_profiles: bool,
    has_documents: bool,
) -> dict[str, Any]:
    blocked_profiles = dict(manager_plan.get("blocked_profiles") or {})
    candidate_profiles = set(manager_plan.get("candidate_profiles") or [])
    profiles_to_run: list[str] = []
    skipped_profiles: list[dict[str, str]] = []

    for profile in requested_profiles:
        if profile == BASE_PROFILE:
            if has_documents:
                profiles_to_run.append(profile)
            else:
                skipped_profiles.append({"profile": profile, "reason": "No evidence documents available."})
            continue
        if profile in blocked_profiles:
            skipped_profiles.append({"profile": profile, "reason": blocked_profiles[profile]})
            continue
        if profile not in DOMAIN_PROFILE_AGENTS:
            skipped_profiles.append({"profile": profile, "reason": "Unsupported profile runner."})
            continue
        if profile not in candidate_profiles and not allow_candidate_profiles:
            skipped_profiles.append({"profile": profile, "reason": "Profile is not listed as a candidate in the evidence pack."})
            continue
        if not allow_candidate_profiles:
            skipped_profiles.append({"profile": profile, "reason": "Candidate profiles require --allow-candidate-profiles."})
            continue
        profiles_to_run.append(profile)

    return {
        "requested_profiles": requested_profiles,
        "profiles_to_run": profiles_to_run,
        "skipped_profiles": skipped_profiles,
        "allow_candidate_profiles": allow_candidate_profiles,
        "source_manager_plan": manager_plan,
    }


def _domain_context(
    documents: list[ResearchDocument],
    tickers: list[str],
    sectors: list[str],
    evidence_pack: dict[str, Any],
) -> MarketContext:
    news_items = [
        {
            "title": document.title,
            "content": document.text,
            "published_at": document.published_at,
            "tickers": document.tickers,
            "sectors": document.sectors,
            "tags": document.tags,
            "uri": document.uri,
        }
        for document in documents
    ]
    macro_docs = [document for document in documents if "macro" in document.tags or document.source_type == "report"]
    sector_data = {sector: {"evidence_document_count": len(documents)} for sector in sectors}
    return MarketContext(
        tickers=tickers,
        news=news_items,
        research_documents=documents,
        macro={"evidence_document_count": len(macro_docs)} if macro_docs else {},
        sector_data=sector_data,
        metadata={"evidence_pack_run_id": evidence_pack.get("run_id")},
    )


def _recommendations(
    profile_plan: dict[str, Any],
    runs: list[dict[str, Any]],
    review_snapshot: dict[str, Any] | None,
) -> list[str]:
    recommendations: list[str] = []
    if not runs:
        recommendations.append("No analyst profile ran; build a stronger evidence pack first.")
    if profile_plan.get("skipped_profiles"):
        recommendations.append("Review skipped_profiles before enabling specialized analysts.")
    if review_snapshot:
        recommendations.append("Use the linked review snapshot before creating paper decisions or learning records.")
    if any(item.get("proposal_count", 0) for item in runs):
        recommendations.append("Dry-run queued operation proposals before approval.")
    if not recommendations:
        recommendations.append("Base analyst run completed; review citations and notes before specialization.")
    return recommendations


def _profile_horizon(profile: str) -> float:
    return {"news_catalyst": 0.5, "macro_policy": 1.5, "sector_cycle": 1.0, "value_screening": 3.0}.get(profile, 1.0)


def _document_tickers(documents: list[ResearchDocument]) -> list[str]:
    return sorted({ticker for document in documents for ticker in document.tickers})


def _document_sectors(documents: list[ResearchDocument]) -> list[str]:
    return sorted({sector for document in documents for sector in document.sectors})


def _normalize_tickers(values: list[str]) -> list[str]:
    return sorted({str(value).strip().upper() for value in values if str(value).strip()})


def _normalize_strings(values: list[str]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"

