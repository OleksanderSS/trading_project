from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.pipeline_control.pipeline_control_bounded_evidence_run import PipelineControlBoundedEvidenceRun
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


class PipelineControlBoundedEvidenceBatch:
    """Run a predeclared set of offline bounded evidence contexts."""

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/pipeline_control_bounded_evidence_batch_current",
    ):
        self.output_dir = Path(output_dir)

    async def run(
        self,
        *,
        coverage_json: str | Path,
        tickers: list[str],
        macro_source_path: str | Path | None = None,
        rows_per_context: int = 480,
        max_features: int = 40,
        gap_size: int = 5,
        min_rows: int = 180,
        transaction_cost_per_turn: float = 0.0025,
        run_real_metric_review: bool = True,
        frozen_contexts: list[str] | None = None,
        max_contexts: int = 8,
        input_is_enriched: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        coverage_path = Path(coverage_json)
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        selected_tickers = list(
            dict.fromkeys(str(ticker).upper() for ticker in tickers if str(ticker).strip())
        )
        frozen = {
            _normalize_context_key(context)
            for context in (frozen_contexts or [])
            if str(context).strip()
        }
        contexts = _select_contexts(
            coverage,
            tickers=selected_tickers,
            frozen_contexts=frozen,
            rows_per_context=rows_per_context,
            max_contexts=max_contexts,
        )
        if not contexts:
            raise ValueError("No eligible, non-frozen contexts were selected from coverage.")

        resolved_macro = (
            str(macro_source_path)
            if macro_source_path
            else coverage.get("summary", {}).get("recommended_macro_source")
        )
        run_id = _run_id("pipeline_control_bounded_evidence_batch")
        run_dir = self.output_dir / run_id
        manifest = _build_manifest(
            run_id=run_id,
            coverage_path=coverage_path,
            contexts=contexts,
            macro_source_path=resolved_macro,
            rows_per_context=rows_per_context,
            max_features=max_features,
            gap_size=gap_size,
            min_rows=min_rows,
            transaction_cost_per_turn=transaction_cost_per_turn,
            run_real_metric_review=run_real_metric_review,
            frozen_contexts=sorted(frozen),
        )
        manifest_paths = ReviewArtifactWriter(run_dir / "manifest").write(
            payload=manifest,
            markdown=render_batch_manifest_markdown(manifest),
            run_id=f"{run_id}_manifest",
        )

        results: list[dict[str, Any]] = []
        for context in contexts:
            context_key = _context_key(context["ticker"], context["timeframe"])
            context_dir = run_dir / "contexts" / context_key.replace("/", "_")
            try:
                payload = await PipelineControlBoundedEvidenceRun(context_dir).run(
                    source_path=context["source_path"],
                    macro_source_path=resolved_macro,
                    ticker=context["ticker"],
                    timeframe=context["timeframe"],
                    target_name=context["target_name"],
                    start=context.get("effective_start"),
                    max_rows=context["max_rows"],
                    max_features=max_features,
                    gap_size=gap_size,
                    min_rows=min_rows,
                    input_is_enriched=input_is_enriched,
                    transaction_cost_per_turn=transaction_cost_per_turn,
                    run_real_metric_review=run_real_metric_review,
                )
                results.append(_compact_result(context, payload))
            except Exception as exc:
                results.append(
                    {
                        "context_key": context_key,
                        "ticker": context["ticker"],
                        "timeframe": context["timeframe"],
                        "status": "failed_exception",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "can_use_as_metric_evidence": False,
                        "can_trade": False,
                    }
                )

        summary = _batch_summary(results)
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_bounded_evidence_batch",
            "manifest_fingerprint": manifest["manifest_fingerprint"],
            "inputs": {
                "coverage_json": str(coverage_path),
                "tickers": selected_tickers,
                "macro_source_path": resolved_macro,
                "rows_per_context": rows_per_context,
                "max_features": max_features,
                "gap_size": gap_size,
                "min_rows": min_rows,
                "transaction_cost_per_turn": transaction_cost_per_turn,
                "run_real_metric_review": run_real_metric_review,
                "frozen_contexts": sorted(frozen),
                "max_contexts": max_contexts,
                "input_is_enriched": input_is_enriched,
            },
            "summary": summary,
            "manifest": manifest,
            "manifest_paths": manifest_paths,
            "results": results,
            "test_window_policy": [
                "Every context and row budget is fixed in the manifest before a model is fit.",
                "A context listed as frozen is excluded from this batch.",
                "No model variant may be selected by comparing results on these test windows.",
                "Future feature or model changes require new predeclared test windows.",
            ],
            "explicit_non_actions": [
                "No collector or external API was called.",
                "No tuning loop or model promotion was run.",
                "No learning-memory or production-config write was made.",
                "No recommendation, order, broker route, paper trade, or live trade was created.",
            ],
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_bounded_evidence_batch_markdown(payload),
                run_id=run_id,
            )
        return json_ready(payload)


def render_batch_manifest_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Pipeline Control Bounded Evidence Batch Manifest",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Manifest fingerprint: `{payload.get('manifest_fingerprint')}`",
        f"- Macro source: `{payload.get('macro_source_path')}`",
        f"- Frozen contexts: {payload.get('frozen_contexts')}",
        "",
        "## Predeclared Contexts",
        "",
    ]
    for context in payload.get("contexts", []):
        lines.append(
            f"- `{context.get('ticker')}/{context.get('timeframe')}`: "
            f"start={context.get('effective_start')} rows={context.get('max_rows')} "
            f"target={context.get('target_name')}"
        )
    return "\n".join(lines).strip() + "\n"


def render_bounded_evidence_batch_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Pipeline Control Bounded Evidence Batch",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('batch_status')}`",
        f"- Contexts: {summary.get('context_count')}",
        f"- Completed: {summary.get('completed_context_count')}",
        f"- Failed or blocked: {summary.get('failed_or_blocked_context_count')}",
        f"- Real metric evidence accepted: {summary.get('metric_evidence_accepted_count')}",
        f"- Mean validation score: {summary.get('mean_validation_score')}",
        f"- Mean test score: {summary.get('mean_test_score')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Results",
        "",
    ]
    for result in payload.get("results", []):
        lines.append(
            f"- `{result.get('context_key')}`: status={result.get('status')} "
            f"validation={result.get('validation_score')} test={result.get('test_score')} "
            f"stability={result.get('feature_stability_score')} "
            f"metric_evidence={result.get('can_use_as_metric_evidence')}"
        )
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _select_contexts(
    coverage: dict[str, Any],
    *,
    tickers: list[str],
    frozen_contexts: set[str],
    rows_per_context: int,
    max_contexts: int,
) -> list[dict[str, Any]]:
    requested = set(tickers)
    selected = []
    for context in coverage.get("eligible_contexts", []):
        ticker = str(context.get("ticker", "")).upper()
        timeframe = str(context.get("timeframe", "")).lower()
        key = _context_key(ticker, timeframe)
        if ticker not in requested or key in frozen_contexts:
            continue
        available_rows = int(context.get("rows_after_effective_start", 0) or 0)
        selected.append(
            {
                "source_path": context.get("source_path"),
                "ticker": ticker,
                "timeframe": timeframe,
                "target_name": context.get("target_name"),
                "effective_start": context.get("effective_start"),
                "available_rows": available_rows,
                "max_rows": min(max(1, int(rows_per_context)), available_rows),
            }
        )
    ticker_order = {ticker: index for index, ticker in enumerate(tickers)}
    selected.sort(
        key=lambda context: (
            ticker_order.get(context["ticker"], len(ticker_order)),
            context["timeframe"],
        )
    )
    return selected[: max(1, int(max_contexts))]


def _build_manifest(
    *,
    run_id: str,
    coverage_path: Path,
    contexts: list[dict[str, Any]],
    macro_source_path: str | None,
    rows_per_context: int,
    max_features: int,
    gap_size: int,
    min_rows: int,
    transaction_cost_per_turn: float,
    run_real_metric_review: bool,
    frozen_contexts: list[str],
) -> dict[str, Any]:
    fixed = {
        "coverage_json": str(coverage_path),
        "contexts": contexts,
        "macro_source_path": macro_source_path,
        "rows_per_context": rows_per_context,
        "max_features": max_features,
        "gap_size": gap_size,
        "min_rows": min_rows,
        "transaction_cost_per_turn": transaction_cost_per_turn,
        "run_real_metric_review": run_real_metric_review,
        "frozen_contexts": frozen_contexts,
        "model_contract": {
            "type": "random_forest",
            "n_estimators": 128,
            "max_depth": 8,
            "min_samples_leaf": 5,
            "random_state": 42,
        },
    }
    encoded = json.dumps(fixed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "mode": "pipeline_control_bounded_evidence_batch_manifest",
        **fixed,
        "manifest_fingerprint": hashlib.sha256(encoded).hexdigest(),
        "locked_before_fit": True,
        "can_trade": False,
    }


def _compact_result(context: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    return {
        "context_key": _context_key(context["ticker"], context["timeframe"]),
        "ticker": context["ticker"],
        "timeframe": context["timeframe"],
        "target_name": context["target_name"],
        "status": summary.get("bounded_evidence_status"),
        "model_row_count": summary.get("model_row_count"),
        "train_sample_count": summary.get("train_sample_count"),
        "validation_sample_count": summary.get("validation_sample_count"),
        "test_sample_count": summary.get("test_sample_count"),
        "validation_score": summary.get("validation_score"),
        "test_score": summary.get("test_score"),
        "test_balanced_accuracy": summary.get("test_balanced_accuracy"),
        "max_drawdown": summary.get("max_drawdown"),
        "feature_stability_score": summary.get("feature_stability_score"),
        "blocked_metric_planes": summary.get("blocked_metric_planes", []),
        "caution_metric_planes": summary.get("caution_metric_planes", []),
        "macro_used_in_stage_3": summary.get("macro_used_in_stage_3"),
        "selected_macro_feature_count": summary.get("selected_macro_feature_count"),
        "locked_model_evaluation_ready": summary.get("locked_model_evaluation_ready", False),
        "locked_feature_stability_ready": summary.get("locked_feature_stability_ready", False),
        "can_use_as_metric_evidence": summary.get("can_use_as_metric_evidence", False),
        "can_clear_current_real_cautions": summary.get(
            "can_clear_current_real_cautions",
            False,
        ),
        "can_trade": False,
        "report_json": payload.get("saved_paths", {}).get("latest_json"),
        "artifacts": payload.get("artifacts", {}),
    }


def _batch_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        result
        for result in results
        if result.get("locked_model_evaluation_ready")
        and result.get("locked_feature_stability_ready")
    ]
    failed_or_blocked = len(results) - len(completed)
    return {
        "batch_status": (
            "bounded_evidence_batch_complete"
            if completed and not failed_or_blocked
            else "bounded_evidence_batch_partial"
            if completed
            else "bounded_evidence_batch_blocked"
        ),
        "context_count": len(results),
        "completed_context_count": len(completed),
        "failed_or_blocked_context_count": failed_or_blocked,
        "metric_evidence_accepted_count": sum(
            1 for result in completed if result.get("can_use_as_metric_evidence")
        ),
        "cautions_cleared_count": sum(
            1 for result in completed if result.get("can_clear_current_real_cautions")
        ),
        "mean_validation_score": _mean(completed, "validation_score"),
        "mean_test_score": _mean(completed, "test_score"),
        "mean_test_balanced_accuracy": _mean(completed, "test_balanced_accuracy"),
        "mean_feature_stability_score": _mean(completed, "feature_stability_score"),
        "can_write_learning_memory": False,
        "can_write_production_config": False,
        "can_tune": False,
        "can_trade": False,
    }


def _mean(items: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(item[key])
        for item in items
        if item.get(key) is not None
    ]
    return sum(values) / len(values) if values else None


def _normalize_context_key(value: str) -> str:
    normalized = str(value).strip().replace(":", "/")
    ticker, _, timeframe = normalized.partition("/")
    return _context_key(ticker, timeframe)


def _context_key(ticker: str, timeframe: str) -> str:
    return f"{str(ticker).upper()}/{str(timeframe).lower()}"


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
