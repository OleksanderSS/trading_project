from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.relative_return_direction_policy import (
    calibrate_relative_return_direction_contract,
    validate_relative_return_direction_contract,
)
from dean_os.schemas import utc_now_iso
from dean_os.world_model.world_model_review_resolution import (
    HYPOTHESIS_RESOLUTION_SPECS_CONTRACT_V2,
)


HYPOTHESIS_MEASUREMENT_POLICY_PREPARATION_CONTRACT = (
    "dean_hypothesis_measurement_policy_preparation_v1"
)


class HypothesisMeasurementPolicyPreparer:
    """Prepare governed measurement contracts before hypothesis resolution.

    The source draft is never mutated. Pipeline artifacts may contribute exact
    historical price rows, while explicit verified price artifacts take
    precedence on duplicate ticker/timestamps.
    """

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/hypothesis_measurement_policy_prepared_current"
        ),
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        resolution_specs_json: str | Path,
        *,
        price_paths: list[str | Path] | None = None,
        pipeline_paths: list[str | Path] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        source_path = Path(resolution_specs_json)
        source = _load(source_path)
        if source.get("contract") != HYPOTHESIS_RESOLUTION_SPECS_CONTRACT_V2:
            raise ValueError("measurement policy preparer requires v2 resolution specs")
        prepared = copy.deepcopy(source)
        explicit_prices = [Path(item) for item in price_paths or []]
        pipelines = [Path(item) for item in pipeline_paths or _default_pipeline_paths()]
        # Explicit verified prices are last and therefore win exact duplicate
        # ticker/timestamp rows inside the calibrator.
        calibration_paths = pipelines + explicit_prices
        rows: list[dict[str, Any]] = []
        for hypothesis_id, raw_spec in (prepared.get("resolutions") or {}).items():
            spec = raw_spec if isinstance(raw_spec, dict) else {}
            measurement = spec.get("measurement_spec")
            if not isinstance(measurement, dict):
                rows.append(
                    _blocked_row(
                        str(hypothesis_id), "measurement_spec_missing"
                    )
                )
                continue
            metrics = [str(item) for item in measurement.get("target_metrics") or []]
            relative_metrics = [
                item
                for item in metrics
                if "relative" in item.lower() and "return" in item.lower()
            ]
            if not relative_metrics:
                rows.append(
                    {
                        "hypothesis_id": str(hypothesis_id),
                        "status": "not_applicable_no_relative_return_metric",
                        "relative_return_metrics": [],
                        "blockers_added": [],
                    }
                )
                continue
            existing = measurement.get("relative_return_direction_contract")
            if isinstance(existing, dict):
                try:
                    validate_relative_return_direction_contract(
                        existing,
                        primary_horizon_days=int(
                            measurement.get("primary_horizon_days") or 0
                        ),
                    )
                except ValueError as exc:
                    _add_blocker(spec, "relative_return_direction_contract_invalid")
                    rows.append(
                        _blocked_row(
                            str(hypothesis_id),
                            "relative_return_direction_contract_invalid",
                            detail=str(exc),
                            metrics=relative_metrics,
                        )
                    )
                else:
                    rows.append(
                        {
                            "hypothesis_id": str(hypothesis_id),
                            "status": "existing_calibrated_contract_preserved",
                            "relative_return_metrics": relative_metrics,
                            "neutral_band_absolute_return": existing.get(
                                "neutral_band_absolute_return"
                            ),
                            "blockers_added": [],
                        }
                    )
                continue
            direction = str(
                measurement.get("relative_return_expected_direction") or ""
            ).strip().lower()
            context = dict(measurement.get("measurement_context") or {})
            universe = dict(
                measurement.get("relative_return_universe")
                or context.get("capital_equipment_basket")
                or {}
            )
            members = list(universe.get("members") or [])
            benchmark = str(universe.get("benchmark") or "")
            cutoff = context.get("trigger_event_at") or context.get("context_as_of")
            horizon = int(measurement.get("primary_horizon_days") or 0)
            missing: list[str] = []
            if direction not in {"positive", "negative"}:
                missing.append("relative_return_expected_direction_missing")
            if not members or not benchmark:
                missing.append("relative_return_universe_or_benchmark_missing")
            if not cutoff:
                missing.append("relative_return_calibration_cutoff_missing")
            if horizon < 1:
                missing.append("relative_return_primary_horizon_missing")
            if missing:
                for blocker in missing:
                    _add_blocker(spec, blocker)
                rows.append(
                    {
                        "hypothesis_id": str(hypothesis_id),
                        "status": "blocked_missing_direction_policy_inputs",
                        "relative_return_metrics": relative_metrics,
                        "blockers_added": missing,
                    }
                )
                continue
            try:
                contract = calibrate_relative_return_direction_contract(
                    price_paths=calibration_paths,
                    members=members,
                    benchmark=benchmark,
                    calibration_cutoff_at=str(cutoff),
                    horizon_days=horizon,
                    expected_direction=direction,
                )
            except (ValueError, OSError) as exc:
                blocker = "relative_return_direction_calibration_failed"
                _add_blocker(spec, blocker)
                rows.append(
                    _blocked_row(
                        str(hypothesis_id),
                        blocker,
                        detail=str(exc),
                        metrics=relative_metrics,
                    )
                )
                continue
            if contract.get("status") != "calibrated_pre_outcome_direction_contract":
                blocker = "relative_return_direction_history_insufficient"
                _add_blocker(spec, blocker)
                measurement["relative_return_direction_contract_candidate"] = contract
                rows.append(
                    {
                        "hypothesis_id": str(hypothesis_id),
                        "status": "blocked_insufficient_pre_outcome_history",
                        "relative_return_metrics": relative_metrics,
                        "historical_sample_count": contract["calibration"][
                            "historical_sample_count"
                        ],
                        "blockers_added": [blocker],
                    }
                )
                continue
            measurement["relative_return_direction_contract"] = contract
            rows.append(
                {
                    "hypothesis_id": str(hypothesis_id),
                    "status": "calibrated_contract_attached",
                    "relative_return_metrics": relative_metrics,
                    "expected_direction": direction,
                    "historical_sample_count": contract["calibration"][
                        "historical_sample_count"
                    ],
                    "neutral_band_absolute_return": contract[
                        "neutral_band_absolute_return"
                    ],
                    "blockers_added": [],
                }
            )

        ready = sum(
            row["status"]
            in {
                "calibrated_contract_attached",
                "existing_calibrated_contract_preserved",
            }
            for row in rows
        )
        blocked = sum(bool(row.get("blockers_added")) for row in rows)
        prepared.update(
            {
                "run_id": "hypothesis_measurement_policy_preparation_"
                + utc_now_iso().replace(":", "").replace("+00:00", "Z"),
                "created_at": utc_now_iso(),
                "measurement_policy_preparation": {
                    "contract": HYPOTHESIS_MEASUREMENT_POLICY_PREPARATION_CONTRACT,
                    "source_resolution_specs": _binding(source_path),
                    "price_inputs": [
                        _binding(path) for path in explicit_prices if path.is_file()
                    ],
                    "pipeline_inputs": [
                        _binding(path) for path in pipelines if path.is_file()
                    ],
                    "input_precedence": "explicit_verified_price_over_pipeline_on_exact_duplicate",
                    "hypothesis_rows": rows,
                    "summary": {
                        "hypothesis_count": len(rows),
                        "relative_return_contract_ready_count": ready,
                        "blocked_hypothesis_count": blocked,
                        "source_draft_mutated": False,
                        "manual_review_and_resolution_still_required": True,
                        "replay_registration_performed": False,
                        "learning_memory_write_performed": False,
                        "can_trade": False,
                    },
                },
                "safety": {
                    **dict(prepared.get("safety") or {}),
                    "source_draft_mutated": False,
                    "automatic_hypothesis_approval_performed": False,
                    "replay_registration_performed": False,
                    "learning_memory_write_performed": False,
                    "production_rule_update_performed": False,
                    "can_trade": False,
                },
            }
        )
        if save:
            prepared["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=prepared,
                markdown=_markdown(prepared),
                run_id=prepared["run_id"],
            )
        return prepared


def _add_blocker(spec: dict[str, Any], blocker: str) -> None:
    blockers = list(spec.get("registration_blockers") or [])
    if blocker not in blockers:
        blockers.append(blocker)
    spec["registration_blockers"] = blockers


def _blocked_row(
    hypothesis_id: str,
    blocker: str,
    *,
    detail: str | None = None,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "hypothesis_id": hypothesis_id,
        "status": "blocked",
        "relative_return_metrics": metrics or [],
        "blockers_added": [blocker],
        "detail": detail,
    }


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _default_pipeline_paths() -> list[Path]:
    return [
        path
        for path in (
            Path("data/colab/accumulated/main_database/features.parquet"),
            Path(
                "data/colab/regenerated/semiconductor_clean_1d_stage23/features.parquet"
            ),
        )
        if path.is_file()
    ]


def _markdown(payload: dict[str, Any]) -> str:
    preparation = payload["measurement_policy_preparation"]
    summary = preparation["summary"]
    lines = [
        "# Hypothesis Measurement Policy Preparation",
        "",
        f"- Hypotheses: {summary['hypothesis_count']}",
        f"- Relative-return contracts ready: {summary['relative_return_contract_ready_count']}",
        f"- Blocked: {summary['blocked_hypothesis_count']}",
        "- Source draft mutated: false",
        "- Registration performed: false",
        "- Can trade: false",
        "",
    ]
    for row in preparation["hypothesis_rows"]:
        lines.extend(
            [
                f"## `{row['hypothesis_id']}`",
                "",
                f"- Status: `{row['status']}`",
                f"- Direction: `{row.get('expected_direction') or 'not applicable/missing'}`",
                f"- Neutral band: `{row.get('neutral_band_absolute_return')}`",
                f"- Blockers: {', '.join(row.get('blockers_added') or []) or 'none'}",
                "",
            ]
        )
    return "\n".join(lines)


__all__ = [
    "HYPOTHESIS_MEASUREMENT_POLICY_PREPARATION_CONTRACT",
    "HypothesisMeasurementPolicyPreparer",
]
