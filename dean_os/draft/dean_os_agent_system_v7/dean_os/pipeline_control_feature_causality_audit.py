from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_bounded_evidence_run import (
    _bound_source_frame,
    _load_source_frame,
    _prepare_macro_input,
    _resolve_enriched_datetime,
    _run_stage_3_enrichment,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready
from src.pipeline.target_column_utils import is_target_like_column

NUMERIC_RTOL = 1e-7
NUMERIC_ATOL = 1e-7


class PipelineControlFeatureCausalityAudit:
    """Compare Stage 3 prefix features with and without a future suffix."""

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/pipeline_control_feature_causality_audit_current",
    ):
        self.output_dir = Path(output_dir)

    async def run(
        self,
        *,
        batch_json: str | Path,
        tickers: list[str] | None = None,
        max_contexts: int = 2,
        mismatch_ratio_limit: float = 0.01,
        save: bool = True,
    ) -> dict[str, Any]:
        batch_path = Path(batch_json)
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        requested = {
            str(ticker).upper() for ticker in (tickers or []) if str(ticker).strip()
        }
        selected = [
            item
            for item in batch.get("results", [])
            if item.get("report_json")
            and (not requested or str(item.get("ticker", "")).upper() in requested)
        ][: max(1, int(max_contexts))]
        if not selected:
            raise ValueError("No batch contexts selected for feature causality audit.")

        contexts = []
        family_counts: Counter[str] = Counter()
        for item in selected:
            report = json.loads(Path(item["report_json"]).read_text(encoding="utf-8"))
            context = await _audit_context(
                item,
                report,
                mismatch_ratio_limit=mismatch_ratio_limit,
            )
            contexts.append(context)
            family_counts.update(context["noncausal_feature_family_counts"])
        noncausal_union = sorted(
            {
                feature["feature"]
                for context in contexts
                for feature in context["noncausal_features"]
            }
        )
        summary = {
            "causality_audit_status": (
                "feature_causality_violations_found"
                if noncausal_union
                else "feature_prefix_invariance_passed"
            ),
            "context_count": len(contexts),
            "contexts_with_violations": sum(
                1 for context in contexts if context["noncausal_feature_count"]
            ),
            "noncausal_feature_union_count": len(noncausal_union),
            "noncausal_feature_families": dict(family_counts.most_common()),
            "test_metrics_read": False,
            "models_trained": 0,
            "can_use_features_for_new_locked_evidence": not noncausal_union,
            "can_trade": False,
        }
        payload = {
            "run_id": _run_id("pipeline_control_feature_causality_audit"),
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_feature_causality_audit",
            "inputs": {
                "batch_json": str(batch_path),
                "tickers": sorted(requested) if requested else None,
                "max_contexts": max_contexts,
                "mismatch_ratio_limit": mismatch_ratio_limit,
                "numeric_rtol": NUMERIC_RTOL,
                "numeric_atol": NUMERIC_ATOL,
            },
            "summary": summary,
            "contexts": contexts,
            "noncausal_feature_union": noncausal_union,
            "operator_next_steps": _next_steps(summary, family_counts),
            "explicit_non_actions": [
                "No test labels, predictions, scores, returns, drawdown, or profitability metrics were read.",
                "The future suffix was used only to test whether earlier feature values changed.",
                "No model was trained, tuned, evaluated, or promoted.",
                "No source artifact, production config, recommendation, order, or trade was written.",
            ],
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_feature_causality_audit_markdown(payload),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def render_feature_causality_audit_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Pipeline Control Feature Causality Audit",
        "",
        f"- Status: `{summary.get('causality_audit_status')}`",
        f"- Contexts: {summary.get('context_count')}",
        f"- Contexts with violations: {summary.get('contexts_with_violations')}",
        f"- Noncausal feature union: {summary.get('noncausal_feature_union_count')}",
        f"- Test metrics read: {summary.get('test_metrics_read')}",
        f"- Models trained: {summary.get('models_trained')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Contexts",
        "",
    ]
    for context in payload.get("contexts", []):
        lines.append(
            f"- `{context.get('context_key')}`: compared={context.get('compared_feature_count')} "
            f"noncausal={context.get('noncausal_feature_count')} "
            f"rows={context.get('compared_row_count')}"
        )
        for feature in context.get("noncausal_features", [])[:10]:
            lines.append(
                f"  - `{feature.get('feature')}`: mismatch={feature.get('mismatch_ratio')} "
                f"max_abs_diff={feature.get('max_abs_diff')}"
            )
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


async def _audit_context(
    item: dict[str, Any],
    report: dict[str, Any],
    *,
    mismatch_ratio_limit: float,
) -> dict[str, Any]:
    inputs = report["inputs"]
    source_path = Path(inputs["source_path"])
    source = _load_source_frame(source_path)
    full_source = _bound_source_frame(
        source,
        ticker=inputs["ticker"],
        timeframe=inputs["timeframe"],
        start=inputs.get("start"),
        end=inputs.get("end"),
        max_rows=int(inputs.get("max_rows", 480)),
    )
    datetime_column = next(
        column
        for column in ("datetime", "timestamp", "date")
        if column in full_source.columns
    )
    test_start = pd.to_datetime(report["split_windows"]["test"]["start"], utc=True)
    prefix_source = full_source.loc[
        pd.to_datetime(full_source[datetime_column], errors="coerce", utc=True)
        < test_start
    ].copy()
    macro_path = report.get("macro_provenance", {}).get("path")
    full_macro, _, full_checks = _prepare_macro_input(
        Path(macro_path) if macro_path else None,
        bounded_frame=full_source,
    )
    prefix_macro, _, prefix_checks = _prepare_macro_input(
        Path(macro_path) if macro_path else None,
        bounded_frame=prefix_source,
    )
    if any(
        check["status"] == "fail"
        for check in [*full_checks, *prefix_checks]
    ):
        raise ValueError(f"Macro input failed for causality audit: {item.get('context_key')}")
    full_enriched = await _run_stage_3_enrichment(
        full_source,
        timeframe=inputs["timeframe"],
        macro_frame=full_macro,
    )
    prefix_enriched = await _run_stage_3_enrichment(
        prefix_source,
        timeframe=inputs["timeframe"],
        macro_frame=prefix_macro,
    )
    train_start = pd.to_datetime(report["split_windows"]["train"]["start"], utc=True)
    validation_end = pd.to_datetime(
        report["split_windows"]["validation"]["end"],
        utc=True,
    )
    comparison = compare_feature_prefix_invariance(
        full_enriched,
        prefix_enriched,
        start=train_start,
        end=validation_end,
        mismatch_ratio_limit=mismatch_ratio_limit,
    )
    family_counts = Counter(
        _feature_family(item["feature"])
        for item in comparison["noncausal_features"]
    )
    return {
        "context_key": item.get("context_key"),
        "ticker": inputs["ticker"],
        "timeframe": inputs["timeframe"],
        "prefix_boundary": test_start.isoformat(),
        "comparison_start": train_start.isoformat(),
        "comparison_end": validation_end.isoformat(),
        **comparison,
        "noncausal_feature_family_counts": dict(family_counts.most_common()),
        "test_metrics_read": False,
        "models_trained": 0,
    }


def compare_feature_prefix_invariance(
    full: pd.DataFrame,
    prefix: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    mismatch_ratio_limit: float,
) -> dict[str, Any]:
    full_frame, full_datetime_source = _indexed_enriched_frame(full)
    prefix_frame, prefix_datetime_source = _indexed_enriched_frame(prefix)
    common_index = full_frame.index.intersection(prefix_frame.index)
    common_index = common_index[(common_index >= start) & (common_index <= end)]
    if common_index.empty:
        raise ValueError("Full and prefix enrichment have no overlapping audit rows.")
    service_columns = {
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    service_invariance = _compare_numeric_columns(
        full_frame,
        prefix_frame,
        common_index=common_index,
        columns=sorted(service_columns & set(full_frame.columns) & set(prefix_frame.columns)),
        mismatch_ratio_limit=mismatch_ratio_limit,
    )
    common_columns = sorted(
        {
            str(column)
            for column in full_frame.select_dtypes(include=[np.number, "bool"]).columns
        }
        & {
            str(column)
            for column in prefix_frame.select_dtypes(include=[np.number, "bool"]).columns
        }
    )
    common_columns = [
        column
        for column in common_columns
        if column not in service_columns and not is_target_like_column(column)
    ]
    rows = _compare_numeric_columns(
        full_frame,
        prefix_frame,
        common_index=common_index,
        columns=common_columns,
        mismatch_ratio_limit=mismatch_ratio_limit,
    )
    rows.sort(key=lambda item: (-item["mismatch_ratio"], -item["max_abs_diff"], item["feature"]))
    noncausal = [item for item in rows if not item["prefix_invariant"]]
    return {
        "compared_row_count": len(common_index),
        "compared_feature_count": len(rows),
        "datetime_sources": {
            "full": full_datetime_source,
            "prefix": prefix_datetime_source,
        },
        "service_column_invariance": service_invariance,
        "service_columns_invariant": all(
            item["prefix_invariant"] for item in service_invariance
        ),
        "noncausal_feature_count": len(noncausal),
        "noncausal_features": noncausal,
        "highest_difference_features": rows[:20],
    }


def _compare_numeric_columns(
    full_frame: pd.DataFrame,
    prefix_frame: pd.DataFrame,
    *,
    common_index: pd.Index,
    columns: list[str],
    mismatch_ratio_limit: float,
) -> list[dict[str, Any]]:
    rows = []
    for column in columns:
        full_values = pd.to_numeric(
            full_frame.loc[common_index, column],
            errors="coerce",
        )
        prefix_values = pd.to_numeric(
            prefix_frame.loc[common_index, column],
            errors="coerce",
        )
        valid = full_values.notna() & prefix_values.notna()
        if not valid.any():
            continue
        left = full_values.loc[valid].to_numpy(dtype=float)
        right = prefix_values.loc[valid].to_numpy(dtype=float)
        differences = np.abs(left - right)
        close = np.isclose(
            left,
            right,
            rtol=NUMERIC_RTOL,
            atol=NUMERIC_ATOL,
            equal_nan=True,
        )
        mismatch_ratio = float((~close).mean())
        rows.append(
            {
                "feature": column,
                "compared_value_count": int(valid.sum()),
                "mismatch_count": int((~close).sum()),
                "mismatch_ratio": mismatch_ratio,
                "max_abs_diff": float(differences.max()) if len(differences) else 0.0,
                "mean_abs_diff": float(differences.mean()) if len(differences) else 0.0,
                "prefix_invariant": mismatch_ratio <= mismatch_ratio_limit,
            }
        )
    return rows


def _indexed_enriched_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    result = frame.copy()
    datetime_source = next(
        (
            column
            for column in ("datetime", "timestamp", "date")
            if column in result.columns
            and pd.to_datetime(result[column], errors="coerce", utc=True).notna().any()
        ),
        None,
    )
    result["_audit_datetime"] = _resolve_enriched_datetime(result)
    result = result.dropna(subset=["_audit_datetime"]).sort_values(
        "_audit_datetime"
    ).drop_duplicates("_audit_datetime", keep="last")
    if datetime_source is None:
        datetime_source = next(
            (
                str(column)
                for column in result.columns
                if str(column).lower().startswith(("datetime_", "timestamp_", "date_"))
                and pd.to_datetime(result[column], errors="coerce", utc=True).notna().any()
            ),
            "DatetimeIndex",
        )
    return result.set_index("_audit_datetime"), str(datetime_source)


def _feature_family(feature: str) -> str:
    lowered = feature.lower()
    if "context" in lowered or lowered.startswith("state_"):
        return "context_state"
    if any(token in lowered for token in ("sma", "ema", "macd", "bb_")):
        return "trend_technical"
    if any(token in lowered for token in ("volatility", "atr", "drawdown", "sharpe", "sortino")):
        return "risk_volatility"
    if any(token in lowered for token in ("day_", "week_", "hour_", "month_")):
        return "calendar_time"
    if lowered.startswith("fred_"):
        return "macro"
    if "volume" in lowered or "obv" in lowered:
        return "volume"
    return "other"


def _next_steps(
    summary: dict[str, Any],
    family_counts: Counter[str],
) -> list[str]:
    if not summary.get("noncausal_feature_union_count"):
        return [
            "Prefix invariance passed for audited contexts; use new rolling train-only folds before another candidate."
        ]
    dominant = family_counts.most_common(1)
    dominant_text = dominant[0][0] if dominant else "unknown"
    return [
        f"Trace and repair the dominant noncausal feature family: {dominant_text}.",
        "Add prefix-invariance tests at the responsible enricher boundary.",
        "Invalidate locked evidence that selected a causality-violating feature until rebuilt.",
        "Do not run another model/feature variant on frozen validation or test windows.",
    ]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
