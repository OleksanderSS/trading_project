from __future__ import annotations

__all__ = [
    'FORMULAS',
    'SAVED_SEC_DERIVED_RATIO_CONTRACT',
    'SavedSECDerivedRatioProducer',
    'load_verified_derived_ratio_context_fragment',
    'render_derived_ratio_markdown',
]

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from dean_os.analysts._producers.sec.merger import (
    load_verified_merged_fundamental_context_fragment,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso
from dean_os.draft.dean_os_agent_system_v7.dean_os.structured_context_provenance import audit_structured_context

SAVED_SEC_DERIVED_RATIO_CONTRACT = (
    "dean_saved_sec_derived_ratio_evidence_v1"
)
FORMULAS = {
    "operating_margin": ("operating_income", "revenue"),
    "net_margin": ("net_income", "revenue"),
    "capex_to_revenue": ("capital_expenditure", "revenue"),
    "cash_to_assets": ("cash_and_equivalents", "assets"),
    "equity_to_assets": ("equity", "assets"),
    "liabilities_to_assets": ("liabilities", "assets"),
}


class SavedSECDerivedRatioProducer:
    """Derive formula-bound ratios without mixing source periods."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/saved_sec_derived_ratio_producer"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        merged_fundamental_artifact_path: str | Path,
        as_of: str,
        save: bool = True,
    ) -> dict[str, Any]:
        as_of_dt = parse_timezone_aware(as_of)
        if as_of_dt is None:
            raise ValueError(
                "derived ratio as_of must be timezone-aware"
            )
        source_path = Path(merged_fundamental_artifact_path)
        load_verified_merged_fundamental_context_fragment(
            source_path,
            expected_as_of=as_of_dt.isoformat(),
        )
        source_payload = json.loads(
            source_path.read_text(encoding="utf-8")
        )
        normalized = _derive_ratios(source_payload.get("facts", []))
        ratios = normalized["ratios"]
        fundamentals: dict[str, dict[str, Any]] = defaultdict(dict)
        for ratio in ratios:
            fundamentals[ratio["ticker"]][ratio["ratio_name"]] = {
                "value": ratio["value"],
                "unit": "ratio",
                "period": ratio["period"],
                "available_at": ratio["available_at"],
                "source_locator": ratio["source_locator"],
                "metadata": {
                    "evidence_type": "fundamental_ratio_context",
                    "required_lane_eligible": False,
                    "formula": ratio["formula"],
                    "source_fact_sha256": ratio[
                        "source_fact_sha256"
                    ],
                    "comparison_period_class": ratio[
                        "comparison_period_class"
                    ],
                },
            }
        audit = audit_structured_context(
            fundamentals=dict(fundamentals),
            macro={},
            sector_data={},
            as_of=as_of_dt.isoformat(),
        )
        lanes = _comparison_lanes(ratios)
        full_cohort = [
            lane
            for lane in lanes
            if lane["ticker_count"] == 4
        ]
        status = (
            "derived_ratio_evidence_ready_with_gaps"
            if ratios and not full_cohort
            else "derived_ratio_evidence_ready"
            if ratios
            else "blocked_no_derived_ratio_evidence"
        )
        run_id = _run_id()
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "saved_sec_derived_ratio_producer",
            "producer_contract": SAVED_SEC_DERIVED_RATIO_CONTRACT,
            "inputs": {
                "merged_fundamental_artifact_path": str(source_path),
                "merged_fundamental_artifact_sha256": _sha256_file(
                    source_path
                ),
                "as_of": as_of_dt.isoformat(),
            },
            "status": status,
            "summary": {
                "source_fact_count": len(
                    source_payload.get("facts", [])
                ),
                "derived_ratio_count": len(ratios),
                "derived_tickers": sorted(
                    {ratio["ticker"] for ratio in ratios}
                ),
                "comparison_lane_count": len(lanes),
                "multi_ticker_comparison_lane_count": sum(
                    lane["ticker_count"] >= 2 for lane in lanes
                ),
                "full_cohort_comparison_lane_count": len(full_cohort),
                "can_claim_full_cohort_comparability": bool(
                    full_cohort
                ),
                "accepted_fingerprint": audit[
                    "accepted_fingerprint"
                ],
                "accepted_ratio_count": audit["accepted_count"],
                "can_feed_fundamental_review": bool(ratios),
                "can_become_prediction_feature": False,
                "can_trade": False,
            },
            "formulas": {
                name: {
                    "numerator": values[0],
                    "denominator": values[1],
                    "formula": f"{values[0]} / {values[1]}",
                }
                for name, values in FORMULAS.items()
            },
            "ratios": ratios,
            "comparison_lanes": lanes,
            "exclusions": normalized["exclusions"],
            "reason_counts": normalized["reason_counts"],
            "structured_context_audit": {
                key: value
                for key, value in audit.items()
                if key
                not in {"accepted_context", "accepted_observations"}
            },
            "market_context_fragment": {
                "as_of": as_of_dt.isoformat(),
                "fundamentals": audit["accepted_context"][
                    "fundamentals"
                ],
                "metadata": {
                    "saved_sec_derived_ratio_run_id": run_id,
                    "accepted_fingerprint": audit[
                        "accepted_fingerprint"
                    ],
                    "full_cohort_comparability": bool(full_cohort),
                },
            },
            "integration_boundary": {
                "review_only": True,
                "same_ticker_unit_period_required": True,
                "quarterly_and_annual_lanes_separate": True,
                "currency_conversion_performed": False,
                "cross_period_comparison_allowed": False,
                "valuation_performed": False,
                "prediction_feature_promotion_allowed": False,
                "automatic_trading_allowed": False,
            },
            "safety": _safety(),
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_derived_ratio_markdown(payload),
                run_id=run_id,
            )
        return payload


def load_verified_derived_ratio_context_fragment(
    artifact_path: str | Path,
    *,
    expected_as_of: str | None = None,
) -> dict[str, Any]:
    path = Path(artifact_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("producer_contract") != SAVED_SEC_DERIVED_RATIO_CONTRACT:
        raise ValueError("unsupported derived ratio contract")
    if payload.get("status") not in {
        "derived_ratio_evidence_ready",
        "derived_ratio_evidence_ready_with_gaps",
    }:
        raise ValueError("derived ratio artifact is not ready")
    as_of = parse_timezone_aware(
        payload.get("market_context_fragment", {}).get("as_of")
    )
    if as_of is None:
        raise ValueError("derived ratio as_of invalid")
    if expected_as_of is not None:
        expected = parse_timezone_aware(expected_as_of)
        if expected is None or expected != as_of:
            raise ValueError("derived ratio expected as_of mismatch")
    source = Path(
        payload.get("inputs", {}).get(
            "merged_fundamental_artifact_path", ""
        )
    )
    if (
        not source.exists()
        or _sha256_file(source)
        != payload.get("inputs", {}).get(
            "merged_fundamental_artifact_sha256"
        )
    ):
        raise ValueError("derived ratio source hash mismatch")
    load_verified_merged_fundamental_context_fragment(
        source,
        expected_as_of=as_of.isoformat(),
    )
    fundamentals = payload.get("market_context_fragment", {}).get(
        "fundamentals"
    )
    audit = audit_structured_context(
        fundamentals=fundamentals,
        macro={},
        sector_data={},
        as_of=as_of.isoformat(),
    )
    summary = payload.get("summary", {})
    if (
        audit["excluded_count"] != 0
        or audit["accepted_count"]
        != summary.get("accepted_ratio_count")
        or audit["accepted_fingerprint"]
        != summary.get("accepted_fingerprint")
    ):
        raise ValueError("derived ratio fingerprint mismatch")
    return {
        "as_of": as_of.isoformat(),
        "fundamentals": audit["accepted_context"]["fundamentals"],
        "metadata": {
            **payload.get("market_context_fragment", {}).get(
                "metadata", {}
            ),
            "saved_sec_derived_ratio_verified": True,
            "artifact_path": str(path),
            "artifact_sha256": _sha256_file(path),
        },
    }


def _derive_ratios(facts: list[dict[str, Any]]) -> dict[str, Any]:
    by_ticker_metric = {
        (str(fact.get("ticker")), str(fact.get("metric_name"))): fact
        for fact in facts
    }
    tickers = sorted({key[0] for key in by_ticker_metric})
    ratios = []
    exclusions = []
    for ticker in tickers:
        for ratio_name, (numerator_name, denominator_name) in FORMULAS.items():
            numerator = by_ticker_metric.get((ticker, numerator_name))
            denominator = by_ticker_metric.get((ticker, denominator_name))
            reasons = []
            if numerator is None:
                reasons.append("ratio_numerator_missing")
            if denominator is None:
                reasons.append("ratio_denominator_missing")
            if numerator is not None and denominator is not None:
                if numerator.get("unit") != denominator.get("unit"):
                    reasons.append("ratio_source_unit_mismatch")
                if numerator.get("period") != denominator.get("period"):
                    reasons.append("ratio_source_period_mismatch")
                if (
                    numerator.get("period_type")
                    != denominator.get("period_type")
                ):
                    reasons.append("ratio_source_period_type_mismatch")
                try:
                    denominator_value = float(denominator.get("value"))
                    numerator_value = float(numerator.get("value"))
                    if (
                        not math.isfinite(denominator_value)
                        or denominator_value == 0
                        or not math.isfinite(numerator_value)
                    ):
                        reasons.append("ratio_source_value_invalid")
                except (TypeError, ValueError):
                    reasons.append("ratio_source_value_invalid")
            if reasons:
                exclusions.append(
                    {
                        "ticker": ticker,
                        "ratio_name": ratio_name,
                        "reasons": sorted(set(reasons)),
                    }
                )
                continue
            source_hashes = sorted(
                [
                    str(numerator.get("fact_sha256")),
                    str(denominator.get("fact_sha256")),
                ]
            )
            ratio_hash = _canonical_sha256(
                {
                    "ticker": ticker,
                    "ratio_name": ratio_name,
                    "source_fact_sha256": source_hashes,
                }
            )
            ratios.append(
                {
                    "ticker": ticker,
                    "ratio_name": ratio_name,
                    "value": numerator_value / denominator_value,
                    "unit": "ratio",
                    "period": numerator["period"],
                    "period_type": numerator["period_type"],
                    "comparison_period_class": _period_class(numerator),
                    "source_currency": numerator["unit"],
                    "available_at": max(
                        numerator["available_at"],
                        denominator["available_at"],
                    ),
                    "formula": (
                        f"{numerator_name} / {denominator_name}"
                    ),
                    "source_fact_sha256": source_hashes,
                    "source_accessions": sorted(
                        {
                            numerator["accession_number"],
                            denominator["accession_number"],
                        }
                    ),
                    "source_locator": (
                        f"dean-derived://sec-ratio/{ticker}/"
                        f"{ratio_name}/{ratio_hash}"
                    ),
                    "ratio_sha256": ratio_hash,
                }
            )
    return {
        "ratios": sorted(
            ratios,
            key=lambda item: (item["ratio_name"], item["ticker"]),
        ),
        "exclusions": exclusions,
        "reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for item in exclusions
                    for reason in item["reasons"]
                ).items()
            )
        ),
    }


def _period_class(fact: dict[str, Any]) -> str:
    form = str(fact.get("form") or "")
    fiscal_period = str(fact.get("fiscal_period") or "")
    if form == "10-Q" and fiscal_period:
        return f"quarterly_{fiscal_period}"
    if form in {"10-K", "20-F", "40-F"}:
        return "annual"
    if fact.get("period_type") == "duration":
        try:
            start = date.fromisoformat(str(fact["period_start"]))
            end = date.fromisoformat(str(fact["period_end"]))
            days = (end - start).days
            return "annual" if days >= 330 else "other_duration"
        except (KeyError, TypeError, ValueError):
            return "other_duration"
    return "other_instant"


def _comparison_lanes(ratios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for ratio in ratios:
        grouped[
            (
                ratio["ratio_name"],
                ratio["comparison_period_class"],
            )
        ].append(ratio)
    return [
        {
            "ratio_name": key[0],
            "comparison_period_class": key[1],
            "tickers": sorted(item["ticker"] for item in values),
            "ticker_count": len(values),
            "status": (
                "multi_ticker_review_lane"
                if len(values) >= 2
                else "single_ticker_context_only"
            ),
            "cross_currency_conversion_required": False,
            "source_currencies": sorted(
                {item["source_currency"] for item in values}
            ),
        }
        for key, values in sorted(grouped.items())
    ]


def _safety() -> dict[str, Any]:
    return {
        "review_only": True,
        "network_access_performed": False,
        "database_opened": False,
        "pipeline_run_performed": False,
        "training_run_performed": False,
        "tuning_run_performed": False,
        "learning_write_performed": False,
        "live_execution_performed": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _run_id() -> str:
    return (
        "saved_sec_derived_ratio_"
        + utc_now_iso().replace(":", "").replace("+", "Z")
    )


def render_derived_ratio_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    return "\n".join(
        [
            "# Saved SEC Derived Ratio Evidence",
            "",
            f"- Status: `{payload.get('status')}`",
            f"- Derived ratios: `{summary.get('derived_ratio_count', 0)}`",
            (
                "- Multi-ticker lanes: `"
                + str(summary.get("multi_ticker_comparison_lane_count", 0))
                + "`"
            ),
            (
                "- Full cohort lanes: `"
                + str(summary.get("full_cohort_comparison_lane_count", 0))
                + "`"
            ),
            "",
            "Quarterly and annual ratios remain separate. No currency "
            "conversion, valuation, prediction feature, or trade is created.",
            "",
        ]
    )
