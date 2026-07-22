from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.structured_context_provenance import (
    AVAILABILITY_FIELDS,
    audit_structured_context,
)
from dean_os.utils import json_ready

DEFAULT_FUNDAMENTALS_JSON = "reports/dean_os/fundamentals_input/latest.json"
SUPPORTED_UNITS = {
    "USD",
    "TWD",
    "ratio",
    "percent",
    "shares",
}
RATIO_LIKE_METRICS = {
    "pe",
    "price_to_earnings",
    "pb",
    "price_to_book",
    "debt_to_equity",
    "current_ratio",
    "quick_ratio",
}
PERCENT_LIKE_METRICS = {
    "fcf_yield",
    "free_cash_flow_yield",
    "roe",
    "roic",
    "gross_margin",
    "operating_margin",
    "net_margin",
}


class FundamentalInputReadinessGate:
    """Review-only gate for caller-supplied fundamental inputs before value agents."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/fundamental_input_readiness_gate"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        fundamentals_json: str | Path = DEFAULT_FUNDAMENTALS_JSON,
        as_of: str | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        artifact = _load_json(fundamentals_json)
        rows = _fundamental_rows(artifact)
        checks = _readiness_checks(rows)
        structured_audit = _structured_metric_audit(
            rows,
            as_of=as_of,
        )
        guidance = _bind_structured_guidance(
            _decision_guidance(checks),
            structured_audit,
            row_count=len(rows),
        )
        payload = {
            "run_id": _run_id("fundamental_input_readiness_gate"),
            "created_at": utc_now_iso(),
            "mode": "fundamental_input_readiness_gate",
            "inputs": {
                "fundamentals_json": str(fundamentals_json),
                "artifact_shape": _artifact_shape(artifact),
                "as_of": as_of,
            },
            "summary": {
                "readiness_status": guidance["status"],
                "recommended_action": guidance["recommended_action"],
                "metric_count": len(rows),
                "ticker_count": len({row.get("ticker") for row in rows if row.get("ticker")}),
                "source_citation_missing_count": sum(1 for row in rows if not row.get("source_citation_present")),
                "period_missing_count": sum(1 for row in rows if not row.get("period")),
                "availability_timestamp_missing_count": sum(
                    1 for row in rows if not row.get("available_at")
                ),
                "structured_point_in_time_status": structured_audit[
                    "status"
                ],
                "structured_accepted_metric_count": structured_audit[
                    "accepted_count"
                ],
                "structured_accepted_fingerprint": (
                    structured_audit["accepted_fingerprint"]
                ),
                "can_enter_manual_fundamental_review": guidance["can_enter_manual_fundamental_review"],
                "can_feed_value_screening_after_manual_review": guidance["can_feed_value_screening_after_manual_review"],
                "can_compute_ratios_now": False,
                "can_interpret_ratios_now": False,
                "can_generate_valuation_now": False,
                "can_create_recommendation": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "metric_rows": rows,
            "structured_context_audit": {
                key: value
                for key, value in structured_audit.items()
                if key
                not in {
                    "accepted_context",
                    "accepted_observations",
                }
            },
            "ticker_metric_summary": _ticker_metric_summary(rows),
            "readiness_checks": checks,
            "decision_guidance": guidance,
            "output_boundary": _output_boundary(),
            "explicit_non_actions": _explicit_non_actions(),
            "commands": _commands(fundamentals_json, as_of=as_of),
            "recommendations": _recommendations(guidance),
        }
        if save:
            self.save(payload)
        return payload

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
        rendered_md = render_fundamental_input_readiness_gate_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_fundamental_input_readiness_gate_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    guidance = payload.get("decision_guidance", {})
    lines = [
        "# DEAN-OS Fundamental Input Readiness Gate",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('readiness_status')}`",
        f"- Recommended action: `{summary.get('recommended_action')}`",
        f"- Metrics: {summary.get('metric_count')}",
        f"- Tickers: {summary.get('ticker_count')}",
        f"- Missing citations: {summary.get('source_citation_missing_count')}",
        f"- Missing periods: {summary.get('period_missing_count')}",
        f"- Missing availability timestamps: {summary.get('availability_timestamp_missing_count')}",
        f"- Point-in-time status: `{summary.get('structured_point_in_time_status')}`",
        f"- Accepted metric fingerprint: `{summary.get('structured_accepted_fingerprint')}`",
        f"- Can enter manual fundamental review: {summary.get('can_enter_manual_fundamental_review')}",
        f"- Can feed value screening after manual review: {summary.get('can_feed_value_screening_after_manual_review')}",
        f"- Can compute ratios now: {summary.get('can_compute_ratios_now')}",
        f"- Can generate valuation now: {summary.get('can_generate_valuation_now')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Metric Samples",
        "",
    ]
    lines.extend(_render_metric_samples(payload.get("metric_rows", [])))
    lines.extend(["", "## Readiness Checks", ""])
    lines.extend(_render_check_samples(payload.get("readiness_checks", [])))
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    lines.extend(["", "## Decision Rationale", ""])
    lines.extend(_render_reason_samples(guidance.get("reasons", [])))
    return "\n".join(lines).strip() + "\n"


def _fundamental_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("extracted_fundamental_metrics", "fundamental_metric_rows", "metrics"):
        values = artifact.get(key)
        if isinstance(values, list):
            for index, item in enumerate(values, start=1):
                if isinstance(item, dict):
                    rows.append(_row_from_metric_row(item, index=index, source_kind=key))
            return rows

    fundamentals = artifact.get("fundamentals")
    if isinstance(fundamentals, dict):
        return _rows_from_fundamentals_map(fundamentals)

    if _looks_like_fundamentals_map(artifact):
        return _rows_from_fundamentals_map(artifact)
    return rows


def _rows_from_fundamentals_map(fundamentals: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ticker, metrics in sorted(fundamentals.items()):
        if not isinstance(metrics, dict):
            continue
        for metric_name, raw in sorted(metrics.items()):
            if metric_name.startswith("_") or metric_name in {"metadata", "source", "source_citation", "period"}:
                continue
            rows.append(_row_from_map_metric(str(ticker), str(metric_name), raw, metrics))
    return rows


def _row_from_map_metric(ticker: str, metric_name: str, raw: Any, parent: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        value = raw.get("value")
        unit = raw.get("unit") or _infer_unit(metric_name)
        period = raw.get("period") or parent.get("period")
        citation = raw.get("source_citation") or raw.get("citation") or raw.get("source") or parent.get("source_citation") or parent.get("source")
        available_at = _first_available_at(raw) or _first_available_at(
            parent
        )
    else:
        value = raw
        unit = _infer_unit(metric_name)
        period = parent.get("period")
        citation = parent.get("source_citation") or parent.get("source")
        available_at = _first_available_at(parent)
    return _normalized_row(
        ticker=ticker,
        metric_name=metric_name,
        value=value,
        unit=unit,
        period=period,
        citation=citation,
        available_at=available_at,
        source_kind="fundamentals_map",
    )


def _row_from_metric_row(item: dict[str, Any], index: int, source_kind: str) -> dict[str, Any]:
    metric_name = item.get("metric_name") or item.get("name") or item.get("metric") or f"metric_{index}"
    citation = item.get("source_citation") or item.get("citation") or item.get("source")
    return _normalized_row(
        ticker=item.get("ticker"),
        metric_name=metric_name,
        value=item.get("value"),
        unit=item.get("unit") or _infer_unit(str(metric_name)),
        period=item.get("period"),
        citation=citation,
        available_at=_first_available_at(item),
        source_kind=source_kind,
    )


def _normalized_row(
    *,
    ticker: Any,
    metric_name: Any,
    value: Any,
    unit: Any,
    period: Any,
    citation: Any,
    available_at: Any,
    source_kind: str,
) -> dict[str, Any]:
    numeric_value = _as_float(value)
    normalized_unit = str(unit) if unit else None
    return {
        "ticker": str(ticker).upper().strip() if ticker not in {None, ""} else None,
        "metric_name": str(metric_name).strip() if metric_name not in {None, ""} else None,
        "value": numeric_value,
        "raw_value": value,
        "unit": normalized_unit,
        "period": str(period).strip() if period not in {None, ""} else None,
        "source_citation_present": bool(citation),
        "source_locator": (
            str(citation).strip()
            if citation is not None and citation != ""
            else None
        ),
        "available_at": (
            str(available_at).strip()
            if available_at is not None and available_at != ""
            else None
        ),
        "source_kind": source_kind,
        "review_boundary": "Metric is caller-supplied input only; no ratio computation, interpretation, valuation, recommendation, or trading action is performed.",
    }


def _readiness_checks(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    _add_check(checks, bool(rows), "fundamental_metrics_present", "Fundamental metric rows are present.")
    for index, row in enumerate(rows, start=1):
        label = f"metric_{index}_{row.get('ticker') or 'unknown'}_{row.get('metric_name') or 'unknown'}"
        _add_check(checks, bool(row.get("ticker")), f"{label}_ticker_present", "Metric row has a ticker.")
        _add_check(checks, bool(row.get("metric_name")), f"{label}_metric_name_present", "Metric row has a metric name.")
        _add_check(checks, row.get("value") is not None, f"{label}_numeric_value_present", "Metric row has a numeric value.")
        _add_check(checks, row.get("unit") in SUPPORTED_UNITS, f"{label}_unit_supported", "Metric row has a supported unit.")
        if not row.get("period"):
            _add_check(checks, False, f"{label}_period_missing", "Metric row has no period.", status_if_false="warn")
        if not row.get("source_citation_present"):
            _add_check(checks, False, f"{label}_source_citation_missing", "Metric row has no source citation.", status_if_false="warn")
        if not row.get("available_at"):
            _add_check(
                checks,
                False,
                f"{label}_availability_timestamp_missing",
                "Metric row has no point-in-time availability timestamp.",
                status_if_false="warn",
            )
    for code, message in [
        ("ratio_computation_not_performed", "This gate does not compute ratios."),
        ("ratio_interpretation_not_performed", "This gate does not interpret ratios."),
        ("valuation_not_generated", "This gate does not generate valuation."),
        ("recommendation_not_created", "This gate does not create recommendations."),
        ("trading_not_allowed", "This gate does not allow trading."),
        ("learning_write_not_allowed", "This gate does not write learning memory."),
    ]:
        checks.append(_check("pass", code, message))
    return checks


def _decision_guidance(checks: list[dict[str, str]]) -> dict[str, Any]:
    fails = [check for check in checks if check.get("status") == "fail"]
    warnings = [check for check in checks if check.get("status") == "warn"]
    if fails:
        status = "blocked_fundamental_input"
        action = "fix_fundamental_input_shape_before_review"
        can_review = False
        can_feed = False
    elif warnings:
        status = "fundamental_input_ready_with_warnings"
        action = "manual_fundamental_review_with_limitations"
        can_review = True
        can_feed = False
    else:
        status = "fundamental_input_ready_for_manual_review"
        action = "manual_fundamental_review"
        can_review = True
        can_feed = True
    return {
        "status": status,
        "recommended_action": action,
        "can_enter_manual_fundamental_review": can_review,
        "can_feed_value_screening_after_manual_review": can_feed,
        "fail_count": len(fails),
        "warning_count": len(warnings),
        "pass_count": sum(1 for check in checks if check.get("status") == "pass"),
        "reasons": [check["message"] for check in [*fails, *warnings] if check.get("message")],
    }


def _structured_metric_audit(
    rows: list[dict[str, Any]],
    *,
    as_of: str | None,
) -> dict[str, Any]:
    if not as_of:
        return {
            "status": "blocked_gate_as_of_missing",
            "as_of": None,
            "input_count": len(rows),
            "accepted_count": 0,
            "excluded_count": len(rows),
            "accepted_fingerprint": None,
            "reason_counts": {
                "gate_as_of_missing": len(rows)
            }
            if rows
            else {},
            "exclusions": [
                {
                    "family": "fundamental",
                    "scope": row.get("ticker"),
                    "name": row.get("metric_name"),
                    "status": "excluded",
                    "reasons": ["gate_as_of_missing"],
                }
                for row in rows
            ],
        }
    fundamentals: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = row.get("ticker")
        metric_name = row.get("metric_name")
        if not ticker or not metric_name:
            continue
        ticker_payload = fundamentals.setdefault(
            str(ticker),
            {"metrics": {}},
        )
        ticker_payload["metrics"][str(metric_name)] = {
            "value": row.get("value"),
            "unit": row.get("unit"),
            "period": row.get("period"),
            "available_at": row.get("available_at"),
            "source_url": row.get("source_locator"),
        }
    try:
        return audit_structured_context(
            fundamentals=fundamentals,
            macro={},
            sector_data={},
            as_of=as_of,
        )
    except ValueError:
        return {
            "status": "blocked_gate_as_of_invalid",
            "as_of": as_of,
            "input_count": len(rows),
            "accepted_count": 0,
            "excluded_count": len(rows),
            "accepted_fingerprint": None,
            "reason_counts": {
                "gate_as_of_invalid": len(rows)
            }
            if rows
            else {},
            "exclusions": [
                {
                    "family": "fundamental",
                    "scope": row.get("ticker"),
                    "name": row.get("metric_name"),
                    "status": "excluded",
                    "reasons": ["gate_as_of_invalid"],
                }
                for row in rows
            ],
        }


def _bind_structured_guidance(
    guidance: dict[str, Any],
    audit: dict[str, Any],
    *,
    row_count: int,
) -> dict[str, Any]:
    bound = dict(guidance)
    compatible = (
        row_count > 0
        and audit.get("accepted_count") == row_count
        and audit.get("excluded_count") == 0
        and bool(audit.get("accepted_fingerprint"))
    )
    bound["structured_point_in_time_compatible"] = compatible
    bound["structured_fingerprint_match_required"] = True
    if compatible:
        return bound
    bound["can_feed_value_screening_after_manual_review"] = False
    if bound.get("can_enter_manual_fundamental_review"):
        bound["status"] = (
            "fundamental_input_structured_contract_blocked"
        )
        bound["recommended_action"] = (
            "fix_point_in_time_fundamental_semantics_before_value_screening"
        )
    reasons = list(bound.get("reasons", []))
    reasons.append(
        "Value screening requires every metric to pass the point-in-time "
        "semantic audit and bind to its accepted fingerprint."
    )
    bound["reasons"] = reasons
    return bound


def _first_available_at(payload: dict[str, Any]) -> Any | None:
    layers = [payload]
    for key in ("provenance", "_provenance", "metadata"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            layers.append(nested)
    for layer in layers:
        for field in AVAILABILITY_FIELDS:
            value = layer.get(field)
            if value not in {None, ""}:
                return value
    return None


def _ticker_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_ticker: dict[str, list[str]] = defaultdict(list)
    unit_counts = Counter()
    source_counts = Counter()
    for row in rows:
        ticker = row.get("ticker") or "unknown"
        metric_name = row.get("metric_name") or "unknown"
        by_ticker[ticker].append(metric_name)
        unit_counts[str(row.get("unit") or "unknown")] += 1
        source_counts[str(row.get("source_kind") or "unknown")] += 1
    return {
        "metrics_by_ticker": {ticker: sorted(metrics) for ticker, metrics in sorted(by_ticker.items())},
        "unit_counts": dict(sorted(unit_counts.items())),
        "source_kind_counts": dict(sorted(source_counts.items())),
    }


def _artifact_shape(artifact: dict[str, Any]) -> str:
    if isinstance(artifact.get("fundamentals"), dict):
        return "fundamentals_map"
    for key in ("extracted_fundamental_metrics", "fundamental_metric_rows", "metrics"):
        if isinstance(artifact.get(key), list):
            return key
    if _looks_like_fundamentals_map(artifact):
        return "top_level_fundamentals_map"
    return "unknown"


def _looks_like_fundamentals_map(value: dict[str, Any]) -> bool:
    if not value:
        return False
    return all(isinstance(metrics, dict) for metrics in value.values())


def _infer_unit(metric_name: str) -> str | None:
    normalized = metric_name.lower().strip()
    if normalized in RATIO_LIKE_METRICS:
        return "ratio"
    if normalized in PERCENT_LIKE_METRICS:
        return "percent"
    if "share" in normalized:
        return "shares"
    if normalized in {"revenue", "net_income", "free_cash_flow", "ebitda", "cash", "debt"}:
        return "USD"
    return None


def _output_boundary() -> dict[str, bool]:
    return {
        "numeric_extraction_performed_now": False,
        "statement_reconciliation_performed_now": False,
        "ratio_computation_performed_now": False,
        "ratio_interpretation_performed_now": False,
        "valuation_generated_now": False,
        "recommendation_output_now": False,
        "price_target_output_now": False,
        "learning_write_performed_now": False,
        "trade_signal_output_now": False,
        "trading_allowed": False,
    }


def _explicit_non_actions() -> list[str]:
    return [
        "No live fetch, connector fetch, or external API call is performed.",
        "No numeric extraction, statement reconciliation, or ratio computation is executed.",
        "No ratio interpretation, valuation, recommendation, price target, allocation, or trade signal is generated.",
        "No learning memory, analyst weight, production config, pipeline, broker, or trading action is written.",
    ]


def _commands(
    fundamentals_json: str | Path,
    *,
    as_of: str | None,
) -> dict[str, str]:
    as_of_arg = f" --as-of {as_of}" if as_of else ""
    return {
        "rerun_gate": (
            "python run_agent_fundamental_input_readiness_gate.py "
            f"--fundamentals-json {fundamentals_json} "
            "--output-dir reports\\dean_os\\fundamental_input_readiness_gate_current"
            f"{as_of_arg}"
        )
    }


def _recommendations(guidance: dict[str, Any]) -> list[str]:
    if guidance["status"] == "blocked_fundamental_input":
        return ["Fix missing ticker, metric name, numeric value, or supported unit before review."]
    recommendations = [
        "Review source citations and reporting periods before using fundamentals in any value screen.",
        "Keep financial statement normalization, ratio computation, ratio interpretation, valuation, and recommendation as separate gates.",
        "Do not mix this fundamental input gate with sector thesis, ticker evidence promotion, learning promotion, or trading.",
    ]
    if guidance["status"] == "fundamental_input_ready_with_warnings":
        recommendations.append("Resolve missing citations or periods before treating this as a stable input contract.")
    return recommendations


def _render_metric_samples(items: list[dict[str, Any]], max_items: int = 12) -> list[str]:
    if not items:
        return ["- None."]
    lines = []
    for item in items[:max_items]:
        lines.append(
            f"- `{item.get('ticker')}` `{item.get('metric_name')}` value=`{item.get('value')}` "
            f"unit=`{item.get('unit')}` period=`{item.get('period')}` citation={item.get('source_citation_present')}"
        )
    remaining = len(items) - len(lines)
    if remaining > 0:
        lines.append(f"- ... {remaining} additional metric(s) in JSON.")
    return lines


def _render_check_samples(items: list[dict[str, str]], max_items: int = 14) -> list[str]:
    if not items:
        return ["- None."]
    failed = [item for item in items if item.get("status") == "fail"]
    warned = [item for item in items if item.get("status") == "warn"]
    passed = [item for item in items if item.get("status") == "pass"]
    selected = [*failed, *warned, *passed]
    lines = [f"- `{item.get('status')}` {item.get('code')}: {item.get('message')}" for item in selected[:max_items]]
    remaining = len(items) - len(lines)
    if remaining > 0:
        lines.append(f"- ... {remaining} additional check(s) in JSON.")
    return lines


def _render_reason_samples(items: list[str], max_items: int = 8) -> list[str]:
    if not items:
        return ["- No blockers."]
    lines = [f"- {item}" for item in items[:max_items]]
    remaining = len(items) - len(lines)
    if remaining > 0:
        lines.append(f"- ... {remaining} additional reason(s) in JSON.")
    return lines


def _add_check(
    checks: list[dict[str, str]],
    passed: bool,
    code: str,
    message: str,
    status_if_false: str = "fail",
) -> None:
    checks.append(_check("pass" if passed else status_if_false, code, message))


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _load_json(path: str | Path) -> dict[str, Any]:
    from dean_os.dean_paths import DeanPaths

    return DeanPaths.load_json(path)


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
