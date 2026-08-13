from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_DRAFT_BUNDLE = "dean_os/draft/dean_os_after_245_full_context_bundle"
DEFAULT_DROPZONE = "docs/research"

BLOCK_CATEGORIES = {
    "A": "integrate_candidate",
    "B": "documentation_only",
    "C": "audit_history_only",
    "D": "redundant_metadata_ladder",
    "E": "defective_or_superseded",
    "F": "needs_manual_review",
}


class StagedWorkbenchIntegrationReview:
    """Review-only audit of staged web-bot workbench material.

    The packet decides what can be integrated as executable repo value and what
    should stay as audit/history. It does not copy staged code, fetch sources,
    extract claims, publish dashboards, recommend trades, or write learning.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/staged_workbench_integration_review_current"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        draft_bundle: str | Path = DEFAULT_DRAFT_BUNDLE,
        dropzone: str | Path = DEFAULT_DROPZONE,
        save: bool = True,
    ) -> dict[str, Any]:
        bundle = Path(draft_bundle)
        block_map = _read_block_map(bundle)
        zip_inventory = _zip_inventory(bundle)
        main_repo = _main_repo_alignment(dropzone=Path(dropzone))
        block_classifications = _classify_blocks(block_map, main_repo)
        file_classifications = _classify_staged_files(zip_inventory)
        vertical_slice = _vertical_slice_viability(main_repo)
        safety = _safety_boundary_audit(bundle, zip_inventory)
        loop_diagnosis = _loop_diagnosis(block_classifications, main_repo, vertical_slice)
        payload = {
            "run_id": _run_id("staged_workbench_integration_review"),
            "created_at": utc_now_iso(),
            "mode": "staged_workbench_integration_review",
            "inputs": {
                "draft_bundle": str(bundle),
                "dropzone": str(dropzone),
            },
            "summary": _summary(block_classifications, file_classifications, vertical_slice, main_repo, safety),
            "category_legend": BLOCK_CATEGORIES,
            "staged_block_classifications": block_classifications,
            "staged_file_classifications": file_classifications,
            "main_repo_alignment": main_repo,
            "first_vertical_slice_viability": vertical_slice,
            "safety_boundary_audit": safety,
            "where_we_looped": loop_diagnosis,
            "integration_recommendations": _integration_recommendations(vertical_slice, main_repo),
            "explicit_non_actions": _explicit_non_actions(),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_staged_workbench_integration_review_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_staged_workbench_integration_review_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    vertical = payload.get("first_vertical_slice_viability", {})
    lines = [
        "# DEAN-OS Staged Workbench Integration Review",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Review status: `{summary.get('review_status')}`",
        f"- Draft bundle found: {summary.get('draft_bundle_found')}",
        f"- Blocks classified: {summary.get('staged_block_count')}",
        f"- Integrate candidates: {summary.get('integrate_candidate_count')}",
        f"- Strict candidate files: {summary.get('integrate_candidate_file_count')}",
        f"- Documentation-only: {summary.get('documentation_only_count')}",
        f"- Audit-history only: {summary.get('audit_history_only_count')}",
        f"- Redundant ladders: {summary.get('redundant_metadata_ladder_count')}",
        f"- Defective/superseded: {summary.get('defective_or_superseded_count')}",
        f"- Needs manual review: {summary.get('needs_manual_review_count')}",
        f"- Vertical slice status: `{vertical.get('slice_status')}`",
        f"- Can run deterministic offline smoke now: {vertical.get('can_run_deterministic_offline_smoke_now')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## First Vertical Slice",
        "",
    ]
    for step in vertical.get("steps", []):
        lines.append(
            f"- `{step.get('step_id')}`: {step.get('status')} - {step.get('repo_module') or step.get('gap')}"
        )
    lines.extend(["", "## Integrate Candidates", ""])
    for item in payload.get("staged_block_classifications", []):
        if item.get("category") == "A":
            lines.append(f"- `{item.get('block_id')}`: {item.get('reason')}")
    file_candidates = [item for item in payload.get("staged_file_classifications", []) if item.get("category") == "A"]
    if file_candidates:
        lines.extend(["", "### Strict File Candidates", ""])
        for item in file_candidates[:25]:
            lines.append(f"- `{item.get('path')}` -> {item.get('recommended_action')}")
        if len(file_candidates) > 25:
            lines.append(f"- ... {len(file_candidates) - 25} more file candidates omitted from markdown; see JSON.")
    lines.extend(["", "## Keep Out Of Production Modules", ""])
    for item in payload.get("where_we_looped", {}).get("loop_patterns", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Safety", ""])
    for item in payload.get("safety_boundary_audit", {}).get("checks", []):
        lines.append(f"- {item.get('status').upper()}: `{item.get('code')}` - {item.get('message')}")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("integration_recommendations", []))
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _read_block_map(bundle: Path) -> list[str]:
    path = bundle / "05_INDEX_AND_MAPS" / "BLOCK_202_245_MAP.md"
    if not path.exists():
        return []
    blocks = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"`([^`]+)`", line)
        if match:
            blocks.append(match.group(1))
    return blocks


def _zip_inventory(bundle: Path) -> list[dict[str, Any]]:
    inventories: list[dict[str, Any]] = []
    for zip_path in sorted(bundle.glob("**/*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                entries = archive.infolist()
        except zipfile.BadZipFile:
            inventories.append({"zip_path": str(zip_path), "status": "bad_zip", "entries": []})
            continue
        inventories.append(
            {
                "zip_path": str(zip_path),
                "status": "readable",
                "entry_count": len(entries),
                "entries": [
                    {
                        "path": entry.filename,
                        "size": entry.file_size,
                        "is_code": entry.filename.endswith(".py"),
                        "is_test": "/tests/" in entry.filename or entry.filename.startswith("tests/"),
                        "is_fixture": "/fixtures/" in entry.filename or "/reports/" in entry.filename,
                        "is_doc": entry.filename.endswith((".md", ".txt")),
                    }
                    for entry in entries
                    if not entry.is_dir()
                ],
            }
        )
    return inventories


def _main_repo_alignment(*, dropzone: Path) -> dict[str, Any]:
    required_paths = {
        "dropzone_inventory": Path("dean_os/real_source_dropzone_inventory.py"),
        "normalized_packet_builder": Path("dean_os/packets/real_source_normalized_packet.py"),
        "normalized_packet_validation_gate": Path("dean_os/review_only_real_source_normalized_packet_validation_gate.py"),
        "source_evidence_validation_gate": Path("dean_os/source_evidence_validation_gate.py"),
        "analyst_evidence_pack": Path("dean_os/analyst_core/analyst_evidence_pack.py"),
        "domain_analyst_intake": Path("dean_os/analyst_core/domain_analyst_intake_packet.py"),
        "domain_analyst_thesis_review": Path("dean_os/analyst_core/domain_analyst_thesis_review_packet.py"),
        "build_focus_review": Path("dean_os/packets/build_focus_review_packet.py"),
        "source_extraction_contract": Path("dean_os/packets/source_extraction_review_packet.py"),
        "source_extraction_fixture": Path("dean_os/packets/source_extraction_fixture_packet.py"),
        "source_extraction_fixture_review": Path("dean_os/source_extraction_fixture_review_gate.py"),
        "legacy_245_fixture": Path("dean_os/review_only_real_source_normalized_packet_fixture.py"),
    }
    cli_paths = {
        "dropzone_inventory_cli": Path("run_agent_real_source_dropzone_inventory.py"),
        "normalized_packet_cli": Path("run_agent_real_source_normalized_packet.py"),
        "normalized_packet_validation_cli": Path("run_review_only_real_source_normalized_packet_validation_gate.py"),
        "source_gate_cli": Path("run_agent_source_evidence_validation_gate.py"),
        "evidence_pack_cli": Path("run_agent_analyst_evidence_pack.py"),
        "domain_intake_cli": Path("run_agent_domain_analyst_intake_packet.py"),
        "thesis_review_cli": Path("run_agent_domain_analyst_thesis_review_packet.py"),
        "focus_review_cli": Path("run_agent_build_focus_review_packet.py"),
    }
    target_paths = {
        key: {"path": str(path), "exists": path.exists()}
        for key, path in {**required_paths, **cli_paths}.items()
    }
    supported_files = _supported_dropzone_files(dropzone)
    missing_modules = [key for key, item in target_paths.items() if not item["exists"]]
    return {
        "target_paths": target_paths,
        "missing_target_path_ids": missing_modules,
        "dropzone": {
            "path": str(dropzone),
            "exists": dropzone.exists(),
            "supported_file_count": len(supported_files),
            "supported_files": supported_files[:20],
        },
        "import_path_notes": [
            "Staged overlay imports under overlay/dean_os/* need adaptation to main repo package paths.",
            "Block 245 fixture paths point at overlay/fixtures and overlay/reports; main repo uses fixtures/ and reports/dean_os/.",
            "Main repo already has stronger real-source builders; staged fixtures should not be promoted as facts.",
        ],
        "dependency_notes": [
            "Current offline source path uses standard library plus existing pandas/doc loaders where already present.",
            "No live API dependency is required for the reviewed slice.",
        ],
        "duplicate_or_superseded_modules": [
            "review_only_real_source_normalized_packet_fixture.py is useful as fixture history, but RealSourceNormalizedPacketBuilder is the preferred real-source path.",
            "Block 246 validation gate overlaps with SourceEvidenceValidationGate; keep both only if 246 remains packet-shape-only and SourceEvidenceValidationGate remains branch gate.",
        ],
    }


def _supported_dropzone_files(dropzone: Path) -> list[dict[str, Any]]:
    if not dropzone.exists():
        return []
    supported_ext = {".txt", ".md", ".markdown", ".html", ".htm", ".json", ".pdf", ".docx"}
    files = []
    for path in sorted(dropzone.glob("*")):
        if path.is_file() and not path.name.lower().startswith("readme") and path.suffix.lower() in supported_ext:
            files.append({"path": str(path), "extension": path.suffix.lower(), "size_bytes": path.stat().st_size})
    return files


def _classify_blocks(blocks: list[str], main_repo: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for block in blocks:
        number = _block_number(block)
        category = "F"
        reason = "Needs manual review before any repo integration."
        repo_target = None
        if number in {243, 244, 245}:
            category = "A"
            reason = "Real-source intake/normalization boundary has executable value; integrate through existing main repo source modules, not staged paths."
            repo_target = "dean_os/real_source_normalized_packet.py"
        elif number in {246, 247, 248}:
            category = "A"
            reason = "Validation/extraction-shape contracts are useful as review-only gates if kept offline and non-promotional."
            repo_target = "dean_os/source_extraction_review_packet.py"
        elif number in {239, 240, 241, 242}:
            category = "F"
            reason = "Universe/source-packet concepts are useful, but require manual mapping to current source/domain lanes."
        elif number and 216 <= number <= 238:
            if "validation_gate" in block or "contract" in block:
                category = "D"
                reason = "Mostly repeated contract/fixture/validation ladder for financial statements and ratios; keep as design reference until a real fundamental feed exists."
            elif "fixture" in block or "preview" in block:
                category = "C"
                reason = "Fixture/preview history is not production module material without real inputs."
            else:
                category = "B"
                reason = "Useful documentation, but no immediate executable integration value."
        results.append(
            {
                "block_id": block,
                "block_number": number,
                "category": category,
                "classification": BLOCK_CATEGORIES[category],
                "repo_target": repo_target,
                "reason": reason,
                "can_promote_fixtures_as_evidence": False,
            }
        )
    return results


def _classify_staged_files(zip_inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    files = []
    for archive in zip_inventory:
        for entry in archive.get("entries", []):
            path = entry["path"]
            category = "C"
            action = "Keep as audit history."
            if "__pycache__" in path or ".pytest_cache" in path:
                category = "E"
                action = "Do not integrate generated cache files."
            elif "/fixtures/" in path or "/reports/" in path:
                category = "C"
                action = "Keep as fixture/audit artifact; do not treat content as evidence."
            elif path.endswith(".py"):
                category, action = _classify_staged_python_file(path)
            elif path.endswith((".md", ".txt")):
                category = "B"
                action = "Keep as documentation unless it contains acceptance checks not yet encoded in tests."
            files.append(
                {
                    "zip_path": archive.get("zip_path"),
                    "path": path,
                    "category": category,
                    "classification": BLOCK_CATEGORIES[category],
                    "recommended_action": action,
                }
            )
    return files


def _classify_staged_python_file(path: str) -> tuple[str, str]:
    normalized = path.replace("\\", "/")
    is_delta_overlay = "/overlay/" in normalized or normalized.startswith("overlay/")
    is_test = "/tests/" in normalized or normalized.startswith("tests/")
    is_canonical_snapshot = "canonical" in normalized or "handoff" in normalized
    workbench_block_number = _workbench_block_number_from_path(normalized)
    is_current_real_source_block = workbench_block_number in {243, 244, 245}
    if "review_only_real_source_normalized_packet_fixture.py" in normalized:
        return (
            "E",
            "Superseded by main repo RealSourceNormalizedPacketBuilder for real inputs; keep fixture only for CI/reference.",
        )
    if "run_review_only_real_source_normalized_packet_fixture.py" in normalized:
        return "E", "Superseded by main repo runner and source-normalization path."
    if is_delta_overlay and is_test:
        if normalized.endswith("/__init__.py"):
            return "E", "Do not integrate empty test package markers."
        if is_current_real_source_block:
            return "A", "Port test intent only if it verifies a safety boundary or deterministic shape not already covered."
        return "F", "Older overlay test intent needs manual review; do not promote it from the canonical snapshot by default."
    if is_delta_overlay and (
        "validation_gate" in normalized
        or "contract" in normalized
        or "normalizer" in normalized
        or "source" in normalized
    ):
        if is_current_real_source_block:
            return "F", "Review manually for a narrow schema/validator/builder extraction; do not copy the staged module as-is."
        return "C", "Keep older overlay code as audit history unless a concrete missing repo contract is identified."
    if is_canonical_snapshot:
        return "F", "Treat canonical snapshot code as manual-diff material only; do not copy into production modules."
    return "C", "Keep staged utility code as audit history unless a concrete missing repo contract is identified."


def _workbench_block_number_from_path(path: str) -> int | None:
    matches = re.findall(r"(?:^|/)(\d{2,3})_", path)
    if not matches:
        return None
    return int(matches[-1])


def _vertical_slice_viability(main_repo: dict[str, Any]) -> dict[str, Any]:
    target_paths = main_repo["target_paths"]
    dropzone = main_repo["dropzone"]
    steps = [
        _slice_step("offline_source_manifest", "RealSourceDropzoneInventory", "dropzone_inventory", bool(dropzone["supported_file_count"]), "needs_one_operator_source_file"),
        _slice_step("normalized_packet", "RealSourceNormalizedPacketBuilder", "normalized_packet_builder", target_paths["normalized_packet_builder"]["exists"], None),
        _slice_step("candidate_review_gate", "review_only_real_source_normalized_packet_validation_gate + SourceEvidenceValidationGate", "source_evidence_validation_gate", target_paths["source_evidence_validation_gate"]["exists"], None),
        _slice_step("evidence_envelope_or_pack", "AnalystEvidencePackRunner", "analyst_evidence_pack", target_paths["analyst_evidence_pack"]["exists"], "normalized_packet_to_evidence_pack_projection_missing"),
        _slice_step("consumer_projection_read_model_preview", None, None, False, "explicit_projection_preview_module_missing"),
        _slice_step("analyst_report_stub", "DomainAnalystIntakePacket + DomainAnalystThesisReviewPacket", "domain_analyst_intake", target_paths["domain_analyst_intake"]["exists"] and target_paths["domain_analyst_thesis_review"]["exists"], None),
        _slice_step("dashboard_review_packet_stub", "BuildFocusReviewPacket", "build_focus_review", target_paths["build_focus_review"]["exists"], "not_a_dashboard_preview_but_review_packet_exists"),
        _slice_step("deterministic_cli_smoke", "existing CLI entrypoints", "normalized_packet_cli", _cli_smoke_paths_exist(target_paths), None),
    ]
    hard_gaps = [step for step in steps if step["status"] == "missing"]
    adapter_gaps = [step for step in steps if step.get("gap") in {"normalized_packet_to_evidence_pack_projection_missing", "explicit_projection_preview_module_missing"}]
    if hard_gaps:
        status = "offline_vertical_slice_not_yet_viable"
    elif adapter_gaps:
        status = "offline_vertical_slice_viable_after_projection_adapter"
    else:
        status = "offline_vertical_slice_viable_now"
    return {
        "slice_status": status,
        "can_run_deterministic_offline_smoke_now": not hard_gaps and bool(dropzone["supported_file_count"]),
        "blocking_gaps": [step["gap"] for step in hard_gaps if step.get("gap")],
        "adapter_gaps": [step["gap"] for step in adapter_gaps if step.get("gap")],
        "steps": steps,
        "minimal_smoke_commands": _minimal_smoke_commands(dropzone),
    }


def _slice_step(step_id: str, repo_module: str | None, target_path_id: str | None, present: bool, gap: str | None) -> dict[str, Any]:
    status = "available" if present else "missing"
    if present and gap:
        status = "available_with_gap"
    return {
        "step_id": step_id,
        "status": status,
        "repo_module": repo_module,
        "target_path_id": target_path_id,
        "gap": gap,
    }


def _cli_smoke_paths_exist(target_paths: dict[str, dict[str, Any]]) -> bool:
    required = [
        "dropzone_inventory_cli",
        "normalized_packet_cli",
        "normalized_packet_validation_cli",
        "source_gate_cli",
        "evidence_pack_cli",
        "domain_intake_cli",
        "thesis_review_cli",
    ]
    return all(target_paths[item]["exists"] for item in required)


def _minimal_smoke_commands(dropzone: dict[str, Any]) -> list[dict[str, str]]:
    source_path = dropzone["supported_files"][0]["path"] if dropzone["supported_files"] else "docs\\research\\YOUR_FILE.md"
    return [
        {
            "command_id": "inventory",
            "command": "python run_agent_real_source_dropzone_inventory.py --dropzone docs\\research --output-dir reports\\dean_os\\real_source_dropzone_inventory_current",
        },
        {
            "command_id": "normalize",
            "command": f"python run_agent_real_source_normalized_packet.py {source_path} --source-type report --output-dir reports\\dean_os\\real_source_normalized_packet_current",
        },
        {
            "command_id": "packet_shape_gate",
            "command": "python run_review_only_real_source_normalized_packet_validation_gate.py --input-json reports\\dean_os\\real_source_normalized_packet_current\\latest.json --output-dir reports\\dean_os\\real_source_normalized_packet_validation_gate_current",
        },
        {
            "command_id": "source_gate",
            "command": "python run_agent_source_evidence_validation_gate.py --source-json reports\\dean_os\\real_source_normalized_packet_current\\latest.json --output-dir reports\\dean_os\\source_evidence_validation_gate_current",
        },
    ]


def _safety_boundary_audit(bundle: Path, zip_inventory: list[dict[str, Any]]) -> dict[str, Any]:
    text = _bundle_text_sample(bundle, zip_inventory)
    forbidden_terms = {
        "live_fetch": ["live fetch", "live_fetch_allowed"],
        "external_api": ["external_api_call_allowed", "external API"],
        "trading": ["trading_allowed", "trade_signal_allowed", "broker_routing_allowed", "order_generation_allowed"],
        "recommendation": ["recommendation_allowed", "buy_sell_hold_allowed", "price_target_allowed"],
        "autonomous_loop": ["autonomous_loop_allowed", "scheduler_allowed"],
    }
    checks = []
    for code, terms in forbidden_terms.items():
        mentions_boundary = any(term.lower() in text.lower() for term in terms)
        checks.append(
            {
                "status": "pass" if mentions_boundary else "warn",
                "code": f"{code}_boundary_mentioned",
                "message": "Boundary appears in staged docs/code." if mentions_boundary else "Boundary term was not found in staged text sample.",
            }
        )
    checks.extend(
        [
            {"status": "pass", "code": "integration_review_only", "message": "This packet does not execute staged code or copy staged artifacts."},
            {"status": "pass", "code": "no_live_fetch_performed", "message": "No live fetch or connector/API call is performed."},
            {"status": "pass", "code": "no_trading_performed", "message": "No recommendation, order, broker route, paper trade, or live trade is generated."},
        ]
    )
    return {
        "overall_status": "review_only_boundaries_preserved" if all(item["status"] != "fail" for item in checks) else "safety_boundary_failed",
        "checks": checks,
        "forbidden_integration_actions": [
            "live fetch",
            "external API calls",
            "trading/order generation/broker routing",
            "recommendations/buy/sell/hold/price targets",
            "valuation outputs",
            "autonomous loops",
            "unreviewed dashboard publication",
        ],
    }


def _bundle_text_sample(bundle: Path, zip_inventory: list[dict[str, Any]]) -> str:
    pieces = []
    for path in bundle.glob("**/*.md"):
        try:
            pieces.append(path.read_text(encoding="utf-8", errors="replace")[:4000])
        except OSError:
            continue
    for archive in zip_inventory:
        zip_path = archive.get("zip_path")
        if not zip_path:
            continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for entry in zf.infolist():
                    if entry.filename.endswith((".py", ".md", ".txt", ".json")) and entry.file_size < 200_000:
                        pieces.append(zf.read(entry).decode("utf-8", errors="replace")[:4000])
        except (OSError, zipfile.BadZipFile):
            continue
    return "\n".join(pieces)


def _loop_diagnosis(blocks: list[dict[str, Any]], main_repo: dict[str, Any], vertical: dict[str, Any]) -> dict[str, Any]:
    ladder_count = sum(1 for item in blocks if item["category"] == "D")
    audit_count = sum(1 for item in blocks if item["category"] == "C")
    patterns = []
    if ladder_count:
        patterns.append("Repeated contract -> fixture -> validation ladders created more metadata than runnable integration.")
    if audit_count:
        patterns.append("Failed/preview/fixture blocks are useful audit history but should not become production modules.")
    if "normalized_packet_to_evidence_pack_projection_missing" in vertical.get("adapter_gaps", []):
        patterns.append("The work looped around boundaries instead of adding the projection adapter from normalized packet rows to analyst evidence/read models.")
    if main_repo["dropzone"]["supported_file_count"] == 0:
        patterns.append("The source-first path is structurally ready, but no operator source file is present in docs/research.")
    return {
        "loop_status": "loop_detected" if patterns else "no_major_loop_detected",
        "loop_patterns": patterns,
        "next_unlooping_move": (
            "Build one projection/read-model preview from normalized_packet_rows to analyst-consumable documents, then run one offline source through the smoke chain."
        ),
    }


def _integration_recommendations(vertical: dict[str, Any], main_repo: dict[str, Any]) -> list[str]:
    recs = [
        "Do not copy the full staged workbench into main repo.",
        "Keep block 245 fixtures and reports as CI/reference material, not evidence.",
        "Preserve RealSourceNormalizedPacketBuilder as the main real-source normalization path.",
        "Use SourceEvidenceValidationGate as the candidate review gate before domain analyst use.",
    ]
    if main_repo["dropzone"]["supported_file_count"] == 0:
        recs.append("Add exactly one operator-supplied offline source file under docs/research before running the full vertical smoke.")
    if "normalized_packet_to_evidence_pack_projection_missing" in vertical.get("adapter_gaps", []):
        recs.append("Next executable integration candidate: a normalized-packet-to-evidence-pack/read-model projection preview.")
    recs.append("Defer financial statement/ratio blocks 216-238 until a real fundamental input feed exists.")
    return recs


def _summary(
    block_classifications: list[dict[str, Any]],
    file_classifications: list[dict[str, Any]],
    vertical: dict[str, Any],
    main_repo: dict[str, Any],
    safety: dict[str, Any],
) -> dict[str, Any]:
    counts = dict.fromkeys(BLOCK_CATEGORIES.values(), 0)
    for item in block_classifications:
        counts[item["classification"]] += 1
    file_a = sum(1 for item in file_classifications if item["category"] == "A")
    draft_bundle_found = bool(block_classifications or file_classifications)
    status = "staged_workbench_review_ready"
    if safety["overall_status"] != "review_only_boundaries_preserved":
        status = "staged_workbench_review_safety_warning"
    elif not draft_bundle_found:
        # An empty read is not a clean review. draft_bundle_found was already
        # computed here but never reached review_status, so a missing or emptied
        # bundle reported "ready" with zero classified blocks -- a reader could
        # only tell the difference by noticing the counts were 0.
        status = "staged_workbench_review_blocked_missing_draft_bundle"
    return {
        "review_status": status,
        "draft_bundle_found": draft_bundle_found,
        "staged_block_count": len(block_classifications),
        "staged_file_count": len(file_classifications),
        "integrate_candidate_count": counts["integrate_candidate"],
        "documentation_only_count": counts["documentation_only"],
        "audit_history_only_count": counts["audit_history_only"],
        "redundant_metadata_ladder_count": counts["redundant_metadata_ladder"],
        "defective_or_superseded_count": counts["defective_or_superseded"],
        "needs_manual_review_count": counts["needs_manual_review"],
        "integrate_candidate_file_count": file_a,
        "vertical_slice_status": vertical["slice_status"],
        "dropzone_supported_file_count": main_repo["dropzone"]["supported_file_count"],
        "can_run_live_fetch": False,
        "can_extract_claims_events_entities": False,
        "can_publish_dashboard_unreviewed": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _explicit_non_actions() -> list[str]:
    return [
        "No staged code is copied into production modules by this review.",
        "No zip archive is extracted into the repo.",
        "No live fetch, connector fetch, or external API call is performed.",
        "No claim/event/entity extraction is executed.",
        "No dashboard is published.",
        "No company thesis, valuation, recommendation, price target, position size, order, broker route, paper trade, or live trade is generated.",
        "No learning memory, analyst weight, or production config is written.",
    ]


def _block_number(block_id: str) -> int | None:
    match = re.match(r"(\d+)_", block_id)
    return int(match.group(1)) if match else None


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
