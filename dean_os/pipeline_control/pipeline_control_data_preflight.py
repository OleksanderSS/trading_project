from __future__ import annotations

from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.pipeline_control.pipeline_control_saved_data_coverage import PipelineControlSavedDataCoverage
from dean_os.pipeline_control.pipeline_control_saved_price_repair import PipelineControlSavedPriceRepair
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


class PipelineControlDataPreflight:
    """Run saved-data coverage and non-destructive repair as one offline command."""

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/pipeline_control_data_preflight_current",
    ):
        self.output_dir = Path(output_dir)
        self.base_reports_dir = self.output_dir.parent

    def build(
        self,
        *,
        assets_yaml: str | Path = "src/config/assets.yaml",
        price_paths: list[str | Path] | None = None,
        macro_paths: list[str | Path] | None = None,
        required_model_rows: int = 180,
        min_daily_source_bars: int = 24,
        save: bool = True,
    ) -> dict[str, Any]:
        coverage = PipelineControlSavedDataCoverage(
            self.base_reports_dir / "pipeline_control_saved_data_coverage_current"
        ).build(
            assets_yaml=assets_yaml,
            price_paths=price_paths,
            macro_paths=macro_paths,
            min_rows=required_model_rows,
        )
        coverage_json = coverage.get("saved_paths", {}).get("latest_json")
        if not coverage_json:
            raise ValueError("Coverage report was not saved; repair cannot establish lineage.")
        repair = PipelineControlSavedPriceRepair(
            self.base_reports_dir / "pipeline_control_saved_price_repair_current"
        ).build(
            coverage_json=coverage_json,
            required_model_rows=required_model_rows,
            min_daily_source_bars=min_daily_source_bars,
        )
        coverage_summary = coverage.get("summary", {})
        repair_summary = repair.get("summary", {})
        summary = {
            "preflight_status": (
                "saved_data_preflight_ready_15m_only"
                if repair_summary.get("timeframes_ready_for_required_rows") == ["15m"]
                else "saved_data_preflight_ready"
            ),
            "configured_asset_count": coverage_summary.get("configured_asset_count"),
            "eligible_context_count": coverage_summary.get("eligible_context_count"),
            "recommended_macro_source": coverage_summary.get("recommended_macro_source"),
            "latest_processed_macro_snapshot_empty": coverage_summary.get(
                "latest_processed_macro_snapshot_empty"
            ),
            "timeframes_ready_for_required_rows": repair_summary.get(
                "timeframes_ready_for_required_rows",
                [],
            ),
            "timeframes_still_short": repair_summary.get("timeframes_still_short", []),
            "can_start_bounded_15m_review": "15m"
            in repair_summary.get("timeframes_ready_for_required_rows", []),
            "can_start_60m_review": "60m"
            in repair_summary.get("timeframes_ready_for_required_rows", []),
            "can_start_1d_review": "1d"
            in repair_summary.get("timeframes_ready_for_required_rows", []),
            "can_run_collectors": False,
            "can_train": False,
            "can_trade": False,
        }
        payload = {
            "run_id": _run_id("pipeline_control_data_preflight"),
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_data_preflight",
            "inputs": {
                "assets_yaml": str(assets_yaml),
                "price_paths": [str(path) for path in price_paths] if price_paths else None,
                "macro_paths": [str(path) for path in macro_paths] if macro_paths else None,
                "required_model_rows": required_model_rows,
                "min_daily_source_bars": min_daily_source_bars,
            },
            "summary": summary,
            "steps": [
                {
                    "step_id": "saved_data_coverage",
                    "status": "completed",
                    "primary_status": coverage_summary.get("coverage_status"),
                    "report_json": coverage_json,
                },
                {
                    "step_id": "saved_price_repair",
                    "status": "completed",
                    "primary_status": repair_summary.get("repair_status"),
                    "report_json": repair.get("saved_paths", {}).get("latest_json"),
                },
            ],
            "reports": {
                "coverage": coverage.get("saved_paths", {}),
                "repair": repair.get("saved_paths", {}),
            },
            "repaired_artifacts": repair.get("artifacts", {}),
            "operator_next_steps": repair.get("operator_next_steps", []),
            "explicit_non_actions": [
                "No live collector or external API was called.",
                "No source parquet or DuckDB table was modified.",
                "No model was trained, tuned, promoted, or evaluated.",
                "No recommendation, order, paper trade, or live trade was created.",
            ],
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_data_preflight_markdown(payload),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def render_data_preflight_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Pipeline Control Data Preflight",
        "",
        f"- Status: `{summary.get('preflight_status')}`",
        f"- Configured assets: {summary.get('configured_asset_count')}",
        f"- Eligible contexts: {summary.get('eligible_context_count')}",
        f"- Macro source: `{summary.get('recommended_macro_source')}`",
        f"- Ready timeframes: {summary.get('timeframes_ready_for_required_rows')}",
        f"- Still short: {summary.get('timeframes_still_short')}",
        f"- Can start bounded 15m review: {summary.get('can_start_bounded_15m_review')}",
        f"- Can train: {summary.get('can_train')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Steps",
        "",
    ]
    for step in payload.get("steps", []):
        lines.append(
            f"- `{step.get('step_id')}`: {step.get('status')} ({step.get('primary_status')})"
        )
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
