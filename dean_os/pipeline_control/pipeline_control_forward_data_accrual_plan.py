from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


class PipelineControlForwardDataAccrualPlan:
    """Register a prospective boundary for the next development refresh.

    This component does not load prices, generate labels, train models, or
    create a locked holdout. It records what the current development review
    has already seen so a later artifact can prove that it is genuinely new.
    """

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/pipeline_control_forward_data_accrual_plan_current"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        walk_forward_json: str | Path,
        acknowledge_development_refresh_only: bool,
        save: bool = True,
    ) -> dict[str, Any]:
        if not acknowledge_development_refresh_only:
            raise ValueError(
                "Explicit development-refresh-only acknowledgement is required."
            )

        source_path = Path(walk_forward_json)
        loaded = _load_json(source_path)
        created_at = utc_now_iso()
        checks = _checks(loaded)
        ready = all(check["status"] == "pass" for check in checks)
        plan = (
            _build_plan(
                loaded,
                source_path=source_path,
                created_at=created_at,
            )
            if ready
            else None
        )
        status = (
            "forward_development_accrual_plan_ready"
            if ready
            else "blocked_invalid_walk_forward_development_boundary"
        )
        run_id = _run_id("pipeline_control_forward_data_accrual_plan")
        payload = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "pipeline_control_forward_data_accrual_plan",
            "inputs": {
                "walk_forward_json": str(source_path),
                "acknowledge_development_refresh_only": (
                    acknowledge_development_refresh_only
                ),
            },
            "summary": {
                "plan_status": status,
                "check_pass_count": sum(
                    check["status"] == "pass" for check in checks
                ),
                "check_fail_count": sum(
                    check["status"] == "fail" for check in checks
                ),
                "development_candidate_was_blocked": bool(
                    plan and plan["baseline"]["development_contract_passed"] is False
                ),
                "can_accept_existing_artifact_as_new": False,
                "can_call_next_data_virgin_holdout": False,
                "can_train": False,
                "can_promote_model": False,
                "can_write_production_config": False,
                "can_trade": False,
            },
            "checks": checks,
            "accrual_plan": plan,
            "next_step": (
                "Collect or save a new immutable source artifact after "
                "registered_at, then validate both artifact acquisition time "
                "and observation timestamps before any next development run."
                if ready
                else (
                    "Repair the walk-forward development boundary; do not "
                    "register forward data or a holdout from this artifact."
                )
            ),
            "explicit_non_actions": [
                "No price, feature, target, test, or past-evaluation rows are loaded.",
                "No collector or external API is started.",
                "No model training, evaluation, replay, backtest, or tuning is run.",
                "No locked or virgin holdout is created.",
                "No production config, recommendation, allocation, order, or trade is generated.",
            ],
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_forward_data_accrual_plan_markdown(payload),
                run_id=run_id,
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_forward_data_accrual_plan_markdown(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary", {})
    plan = payload.get("accrual_plan") or {}
    baseline = plan.get("baseline", {})
    boundary = plan.get("acceptance_boundary", {})
    lines = [
        "# DEAN-OS Forward Data Accrual Plan",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('plan_status')}`",
        f"- Context: `{baseline.get('context_key')}`",
        f"- Last used validation timestamp: `{baseline.get('last_used_validation_timestamp')}`",
        f"- New artifact must be acquired after: `{boundary.get('source_artifact_acquired_after')}`",
        f"- Can call next data virgin holdout: {summary.get('can_call_next_data_virgin_holdout')}",
        f"- Can train: {summary.get('can_train')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Checks",
        "",
    ]
    for check in payload.get("checks", []):
        lines.append(
            f"- `{check.get('check_id')}`: {check.get('status')} — "
            f"{check.get('detail')}"
        )
    lines.extend(["", "## Acceptance Boundary", ""])
    for key, value in boundary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _checks(payload: dict[str, Any] | None) -> list[dict[str, str]]:
    if payload is None:
        return [
            _check(
                "fail",
                "walk_forward_artifact_available",
                "Walk-forward JSON is missing, unreadable, or not an object.",
            )
        ]

    candidate = payload.get("walk_forward_candidate")
    candidate = candidate if isinstance(candidate, dict) else {}
    metrics = candidate.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    test_contract = candidate.get("test_contract")
    test_contract = test_contract if isinstance(test_contract, dict) else {}
    source_lineage = candidate.get("source_lineage")
    source_lineage = source_lineage if isinstance(source_lineage, dict) else {}
    folds = candidate.get("folds")
    folds = folds if isinstance(folds, list) else []

    development_sources = source_lineage.get("development_artifacts")
    development_sources = (
        development_sources if isinstance(development_sources, dict) else {}
    )
    source_contract_ok = bool(development_sources) and all(
        isinstance(item, dict)
        and item.get("partition") == "development"
        and bool(item.get("sha256"))
        for item in development_sources.values()
    )
    temporal_ends = _validation_end_values(folds)
    no_test_access = (
        test_contract.get("eligible_as_locked_test_evidence") is False
        and int(test_contract.get("test_rows_loaded", -1)) == 0
        and int(test_contract.get("past_evaluation_rows_loaded", -1)) == 0
        and test_contract.get("frozen_test_windows_accessed") is False
    )
    context_complete = all(
        bool(candidate.get(key))
        for key in ("ticker", "timeframe", "target_name", "context_fingerprint")
    )

    return [
        _check(
            "pass"
            if payload.get("mode")
            == "pipeline_control_walk_forward_validation_run"
            else "fail",
            "walk_forward_artifact_class",
            f"mode={payload.get('mode')}.",
        ),
        _check(
            "pass"
            if candidate.get("artifact_class")
            == "pipeline_control_walk_forward_validation_candidate"
            and candidate.get("evidence_class")
            == "development_train_validation_only"
            else "fail",
            "development_only_candidate",
            (
                f"artifact_class={candidate.get('artifact_class')}, "
                f"evidence_class={candidate.get('evidence_class')}."
            ),
        ),
        _check(
            "pass" if no_test_access else "fail",
            "test_and_past_evaluation_untouched",
            (
                f"test_rows={test_contract.get('test_rows_loaded')}, "
                "past_evaluation_rows="
                f"{test_contract.get('past_evaluation_rows_loaded')}, "
                "frozen_test_accessed="
                f"{test_contract.get('frozen_test_windows_accessed')}."
            ),
        ),
        _check(
            "pass" if metrics.get("contract_passed") is False else "fail",
            "blocked_candidate_requires_development_refresh",
            f"development_contract_passed={metrics.get('contract_passed')}.",
        ),
        _check(
            "pass" if context_complete else "fail",
            "context_identity_complete",
            (
                f"context={candidate.get('ticker')}/"
                f"{candidate.get('timeframe')}/"
                f"{candidate.get('target_name')}."
            ),
        ),
        _check(
            "pass" if bool(temporal_ends) else "fail",
            "validation_watermark_available",
            (
                f"validation_window_count={len(temporal_ends)}, "
                f"latest={max(temporal_ends) if temporal_ends else None}."
            ),
        ),
        _check(
            "pass" if source_contract_ok else "fail",
            "development_source_hashes_available",
            f"development_source_count={len(development_sources)}.",
        ),
    ]


def _build_plan(
    payload: dict[str, Any],
    *,
    source_path: Path,
    created_at: str,
) -> dict[str, Any]:
    candidate = payload["walk_forward_candidate"]
    folds = candidate["folds"]
    validation_ends = _validation_end_values(folds)
    source_lineage = candidate["source_lineage"]["development_artifacts"]
    source_hashes = sorted(
        str(item["sha256"])
        for item in source_lineage.values()
        if isinstance(item, dict) and item.get("sha256")
    )
    validation_sizes = [
        int(fold.get("validation_window", {}).get("sample_count", 0))
        for fold in folds
        if isinstance(fold, dict)
    ]
    minimum_new_rows = max(validation_sizes) if validation_sizes else 1
    context_key = (
        f"{candidate['ticker']}/{candidate['timeframe']}/"
        f"{candidate['target_name']}"
    )
    registration_key = "|".join(
        [
            str(candidate["context_fingerprint"]),
            context_key,
            max(validation_ends),
        ]
    )
    return {
        "artifact_class": "pipeline_control_forward_data_accrual_plan",
        "evidence_class": "prospective_development_data_boundary",
        "plan_id": hashlib.sha256(registration_key.encode("utf-8")).hexdigest(),
        "registered_at": created_at,
        "lane": "development_refresh_only",
        "baseline": {
            "walk_forward_json": str(source_path),
            "walk_forward_sha256": _sha256_file(source_path),
            "context_key": context_key,
            "context_fingerprint": candidate["context_fingerprint"],
            "development_contract_passed": candidate["metrics"][
                "contract_passed"
            ],
            "last_used_validation_timestamp": max(validation_ends),
            "seen_development_source_sha256": source_hashes,
        },
        "acceptance_boundary": {
            "source_artifact_acquired_after": created_at,
            "observation_timestamp_strictly_after": max(validation_ends),
            "minimum_new_base_timeframe_rows": minimum_new_rows,
            "source_sha256_must_be_new": True,
            "source_artifact_must_be_immutable": True,
            "ticker_must_equal": candidate["ticker"],
            "timeframe_must_equal": candidate["timeframe"],
            "target_contract_must_equal": candidate["target_name"],
            "partition_name": "forward_development_candidate",
            "may_be_used_as_locked_test_evidence": False,
            "may_be_called_virgin_holdout": False,
        },
        "future_transition": {
            "next_development_run_requires_predeclared_contract": True,
            "a_passing_development_candidate_must_be_frozen_before_holdout_registration": True,
            "virgin_holdout_registration_is_a_separate_future_gate": True,
        },
    }


def _validation_end_values(folds: list[Any]) -> list[str]:
    values: list[str] = []
    for fold in folds:
        if not isinstance(fold, dict):
            continue
        window = fold.get("validation_window")
        if not isinstance(window, dict):
            continue
        value = window.get("end")
        if not isinstance(value, str):
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            continue
        values.append(parsed.astimezone(UTC).isoformat())
    return values


def _check(status: str, check_id: str, detail: str) -> dict[str, str]:
    return {"status": status, "check_id": check_id, "detail": detail}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id(prefix: str) -> str:
    stamp = utc_now_iso().replace("-", "").replace(":", "").replace("+", "")
    return f"{prefix}_{stamp}"
