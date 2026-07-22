from __future__ import annotations

__all__ = [
    'DEFAULT_TICKERS',
    'DOMAIN_ID',
    'SEMICONDUCTOR_ANALYST_RUNTIME_CONTRACT',
    'SemiconductorAnalystRuntime',
    'render_semiconductor_runtime_markdown',
]

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.analysts._producers.macro import (
    load_verified_macro_context_fragment,
)
from dean_os.analysts._producers.news import (
    load_verified_semiconductor_news_context_fragment,
)
from dean_os.analysts._producers.policy import (
    load_verified_official_policy_context_fragment,
)
from dean_os.analysts._producers.sec.merger import (
    load_verified_merged_fundamental_context_fragment,
)
from dean_os.analysts._producers.sec.ratios import (
    load_verified_derived_ratio_context_fragment,
)
from dean_os.analysts._producers.sector_market import (
    load_verified_sector_market_context_fragment,
)
from dean_os.analysts.base import BaseAnalystAgent
from dean_os.analysts.context_adapter import MarketContextEvidenceAdapter
from dean_os.analysts.profiles import get_domain_profile
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import MarketContext, utc_now_iso

SEMICONDUCTOR_ANALYST_RUNTIME_CONTRACT = (
    "dean_semiconductor_analyst_runtime_v1"
)
DOMAIN_ID = "semiconductor_ai_infrastructure"
_DOMAIN_TICKERS = get_domain_profile(DOMAIN_ID).ticker_universe_hint
DEFAULT_TICKERS = tuple(_DOMAIN_TICKERS)


class SemiconductorAnalystRuntime:
    """Run one evidence-gated, review-only semiconductor analyst slice."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/semiconductor_analyst_runtime"
        ),
    ):
        self.output_dir = Path(output_dir)

    def run(
        self,
        *,
        fundamental_artifact_path: str | Path,
        macro_artifact_path: str | Path,
        sector_market_artifact_path: str | Path,
        as_of: str,
        tickers: list[str] | None = None,
        news_artifact_path: str | Path | None = None,
        official_policy_artifact_path: str | Path | None = None,
        derived_ratio_artifact_path: str | Path | None = None,
        pipeline_case_artifact_path: str | Path | None = None,
        horizon_days: int | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        as_of_dt = parse_timezone_aware(as_of)
        if as_of_dt is None:
            raise ValueError(
                "semiconductor runtime as_of must be a timezone-aware "
                "ISO-8601 timestamp"
            )
        normalized_as_of = as_of_dt.isoformat()
        requested_tickers = sorted(
            {
                str(ticker).upper().strip()
                for ticker in (tickers or DEFAULT_TICKERS)
                if str(ticker).strip()
            }
        )
        if not requested_tickers:
            raise ValueError("at least one ticker is required")

        fundamental = (
            load_verified_merged_fundamental_context_fragment(
                fundamental_artifact_path,
                expected_as_of=normalized_as_of,
            )
        )
        macro = load_verified_macro_context_fragment(
            macro_artifact_path,
            expected_as_of=normalized_as_of,
        )
        sector = load_verified_sector_market_context_fragment(
            sector_market_artifact_path,
            expected_as_of=normalized_as_of,
        )
        news = (
            load_verified_semiconductor_news_context_fragment(
                news_artifact_path,
                expected_as_of=normalized_as_of,
            )
            if news_artifact_path is not None
            else {
                "as_of": normalized_as_of,
                "news": [],
                "metadata": {"status": "not_supplied"},
            }
        )
        derived_ratios = (
            load_verified_derived_ratio_context_fragment(
                derived_ratio_artifact_path,
                expected_as_of=normalized_as_of,
            )
            if derived_ratio_artifact_path is not None
            else {
                "as_of": normalized_as_of,
                "fundamentals": {},
                "metadata": {"status": "not_supplied"},
            }
        )
        official_policy = (
            load_verified_official_policy_context_fragment(
                official_policy_artifact_path,
                expected_as_of=normalized_as_of,
            )
            if official_policy_artifact_path is not None
            else {
                "as_of": normalized_as_of,
                "news": [],
                "metadata": {"status": "not_supplied"},
            }
        )
        combined_fundamentals = _merge_fundamental_contexts(
            fundamental["fundamentals"],
            derived_ratios["fundamentals"],
        )
        accepted_fundamental_tickers = sorted(
            fundamental["fundamentals"]
        )
        missing_fundamental_tickers = sorted(
            set(requested_tickers) - set(accepted_fundamental_tickers)
        )

        pipeline_case_exclusion = None
        if pipeline_case_artifact_path is not None:
            pipeline_case_exclusion = _pipeline_case_exclusion(
                Path(pipeline_case_artifact_path),
                as_of=as_of_dt,
            )

        context = MarketContext(
            phase="pre_pipeline",
            as_of=normalized_as_of,
            tickers=requested_tickers,
            timeframes=["sector_review"],
            news=[*news["news"], *official_policy["news"]],
            fundamentals=combined_fundamentals,
            macro=macro["macro"],
            sector_data=sector["sector_data"],
            metadata={
                "semiconductor_runtime_contract": (
                    SEMICONDUCTOR_ANALYST_RUNTIME_CONTRACT
                ),
                "verified_fragments": {
                    "fundamental": fundamental["metadata"],
                    "macro": macro["metadata"],
                    "sector_market": sector["metadata"],
                    "semiconductor_news": news["metadata"],
                    "derived_ratios": derived_ratios["metadata"],
                    "official_policy": official_policy["metadata"],
                },
            },
        )
        adapter_packet = MarketContextEvidenceAdapter(DOMAIN_ID).adapt(
            context,
            as_of=normalized_as_of,
        )
        analyst_report = BaseAnalystAgent(
            DOMAIN_ID,
            agent_name="semiconductor_sector_analyst",
        ).run(
            adapter_packet["evidence"],
            tickers=requested_tickers,
            horizon_days=horizon_days,
            as_of=normalized_as_of,
        )
        profile = get_domain_profile(DOMAIN_ID)
        evidence = analyst_report.evidence
        eligible_counts = Counter(
            item.evidence_type
            for item in evidence
            if item.provenance.get("required_lane_eligible")
            is not False
        )
        all_counts = Counter(item.evidence_type for item in evidence)
        missing = list(analyst_report.thesis.blind_spots)
        lanes = [
            {
                "evidence_type": evidence_type,
                "status": (
                    "satisfied"
                    if evidence_type not in missing
                    else "missing"
                ),
                "eligible_evidence_count": eligible_counts.get(
                    evidence_type, 0
                ),
                "all_context_item_count": all_counts.get(
                    evidence_type, 0
                ),
            }
            for evidence_type in profile.required_evidence_types
        ]
        status = (
            "semiconductor_analysis_ready_for_review"
            if analyst_report.recommendation == "ready_for_review"
            else "semiconductor_analysis_partial_ready_for_review"
            if analyst_report.recommendation
            == "partial_ready_for_review"
            else "semiconductor_analysis_needs_more_data"
        )
        run_id = _run_id()
        source_artifacts = {
            "fundamental": _artifact_reference(
                Path(fundamental_artifact_path)
            ),
            "macro": _artifact_reference(Path(macro_artifact_path)),
            "sector_market": _artifact_reference(
                Path(sector_market_artifact_path)
            ),
        }
        if news_artifact_path is not None:
            source_artifacts["semiconductor_news"] = (
                _artifact_reference(Path(news_artifact_path))
            )
        if derived_ratio_artifact_path is not None:
            source_artifacts["derived_ratios"] = (
                _artifact_reference(Path(derived_ratio_artifact_path))
            )
        if official_policy_artifact_path is not None:
            source_artifacts["official_policy"] = (
                _artifact_reference(Path(official_policy_artifact_path))
            )
        if pipeline_case_exclusion is not None:
            source_artifacts["excluded_pipeline_case"] = (
                pipeline_case_exclusion["source_artifact"]
            )
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "semiconductor_analyst_runtime",
            "runtime_contract": SEMICONDUCTOR_ANALYST_RUNTIME_CONTRACT,
            "domain_id": DOMAIN_ID,
            "status": status,
            "inputs": {
                "as_of": normalized_as_of,
                "tickers": requested_tickers,
                "horizon_days": analyst_report.horizon_days,
                "fundamental_artifact_path": str(
                    fundamental_artifact_path
                ),
                "macro_artifact_path": str(macro_artifact_path),
                "sector_market_artifact_path": str(
                    sector_market_artifact_path
                ),
                "news_artifact_path": (
                    str(news_artifact_path)
                    if news_artifact_path is not None
                    else None
                ),
                "derived_ratio_artifact_path": (
                    str(derived_ratio_artifact_path)
                    if derived_ratio_artifact_path is not None
                    else None
                ),
                "official_policy_artifact_path": (
                    str(official_policy_artifact_path)
                    if official_policy_artifact_path is not None
                    else None
                ),
                "pipeline_case_artifact_path": (
                    str(pipeline_case_artifact_path)
                    if pipeline_case_artifact_path is not None
                    else None
                ),
            },
            "source_artifacts": source_artifacts,
            "summary": {
                "recommendation": analyst_report.recommendation,
                "thesis_stance": analyst_report.thesis.stance,
                "thesis_confidence": analyst_report.thesis.confidence,
                "evidence_count": len(evidence),
                "structured_exclusion_count": len(
                    adapter_packet["exclusions"]
                ),
                "required_lane_count": len(
                    profile.required_evidence_types
                ),
                "satisfied_required_lane_count": (
                    len(profile.required_evidence_types) - len(missing)
                ),
                "missing_required_evidence": missing,
                "accepted_fundamental_tickers": (
                    accepted_fundamental_tickers
                ),
                "missing_fundamental_tickers": (
                    missing_fundamental_tickers
                ),
                "fundamentals_are_supporting_context_only": True,
                "macro_is_supporting_context_only": True,
                "derived_ratios_are_supporting_context_only": True,
                "verified_derived_ratios_supplied": (
                    derived_ratio_artifact_path is not None
                ),
                "market_confirmation_ready": (
                    "market_confirmation" not in missing
                ),
                "verified_news_supplied": (
                    news_artifact_path is not None
                ),
                "verified_official_policy_supplied": (
                    official_policy_artifact_path is not None
                ),
                "news_ready_required_lanes": news[
                    "metadata"
                ].get("ready_required_lanes", []),
                "sector_thesis_ready": not missing,
                "can_train": False,
                "can_tune": False,
                "can_create_ticker_forecast": False,
                "can_trade": False,
            },
            "evidence_lane_coverage": {
                "required_lanes": lanes,
                "all_evidence_type_counts": dict(
                    sorted(all_counts.items())
                ),
                "eligible_evidence_type_counts": dict(
                    sorted(eligible_counts.items())
                ),
            },
            "pipeline_case_boundary": (
                pipeline_case_exclusion
                or {
                    "status": "not_supplied",
                    "semantic_rule": (
                        "Ticker/model evaluation cases are never promoted "
                        "to sector evidence by this runtime."
                    ),
                }
            ),
            "adapter": {
                key: value
                for key, value in adapter_packet.items()
                if key != "evidence"
            },
            "analyst_report": analyst_report.model_dump(mode="json"),
            "integration_boundary": {
                "review_only": True,
                "fail_closed_on_missing_required_lanes": True,
                "fundamental_macro_market_context_combined": True,
                "verified_semiconductor_news_optional": True,
                "verified_derived_ratio_context_optional": True,
                "verified_official_policy_context_optional": True,
                "keyword_only_news_can_close_required_lane": False,
                "ticker_model_case_is_sector_evidence": False,
                "amd_is_sector_proxy": False,
                "amd_pipeline_case_can_close_market_confirmation": False,
                "training_allowed": False,
                "tuning_allowed": False,
                "automatic_learning_write_allowed": False,
                "production_config_write_allowed": False,
                "automatic_trading_allowed": False,
            },
            "safety": _safety(),
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_semiconductor_runtime_markdown(
                    payload
                ),
                run_id=run_id,
            )
        return payload


def _merge_fundamental_contexts(
    base: dict[str, Any],
    additional: dict[str, Any],
) -> dict[str, Any]:
    merged = {
        ticker: dict(values) for ticker, values in base.items()
    }
    for ticker, values in additional.items():
        target = merged.setdefault(ticker, {})
        for name, value in values.items():
            if name == "_dean_structured_provenance":
                continue
            if name in target:
                raise ValueError(
                    f"duplicate fundamental context metric: {ticker}.{name}"
                )
            target[name] = value
        base_provenance = dict(
            target.get("_dean_structured_provenance", {})
        )
        additional_provenance = dict(
            values.get("_dean_structured_provenance", {})
        )
        observations = {
            **dict(base_provenance.get("observations", {})),
            **dict(additional_provenance.get("observations", {})),
        }
        if observations:
            target["_dean_structured_provenance"] = {
                "contract": (
                    base_provenance.get("contract")
                    or additional_provenance.get("contract")
                ),
                "as_of": (
                    base_provenance.get("as_of")
                    or additional_provenance.get("as_of")
                ),
                "observations": observations,
            }
    return merged


def _pipeline_case_exclusion(
    path: Path,
    *,
    as_of: Any,
) -> dict[str, Any]:
    if not path.exists():
        raise ValueError("pipeline model case artifact missing")
    latest = json.loads(path.read_text(encoding="utf-8"))
    exact_path = Path(
        str(latest.get("saved_paths", {}).get("json") or path)
    )
    if not exact_path.exists():
        raise ValueError("immutable pipeline model case artifact missing")
    payload = json.loads(exact_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    created_at = parse_timezone_aware(payload.get("created_at"))
    if created_at is None or created_at > as_of:
        raise ValueError("pipeline model case is not point-in-time eligible")
    if (
        payload.get("mode") != "pipeline_model_case_packet"
        or summary.get("case_scope") != "ticker_model_evaluation_only"
        or summary.get("eligible_as_domain_evidence") is not False
        or summary.get("can_trade") is not False
    ):
        raise ValueError(
            "pipeline model case violates sector evidence boundary"
        )
    case = payload.get("case", {})
    lineage = case.get("lineage", {})
    return {
        "status": "excluded_from_domain_evidence",
        "reason": "ticker_model_evaluation_case_not_sector_evidence",
        "ticker": str(lineage.get("ticker") or "").upper() or None,
        "model": lineage.get("model"),
        "target": lineage.get("target_name"),
        "timeframe": lineage.get("timeframe"),
        "case_id": summary.get("case_id"),
        "case_classification": summary.get("case_classification"),
        "source_artifact": _artifact_reference(exact_path),
        "semantic_rule": (
            "The case may inform exact ticker/model pipeline review only; "
            "it cannot satisfy a semiconductor evidence lane."
        ),
    }


def _artifact_reference(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"source artifact missing: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safety() -> dict[str, Any]:
    return {
        "review_only": True,
        "network_access_performed": False,
        "collector_run_performed": False,
        "pipeline_run_performed": False,
        "training_run_performed": False,
        "tuning_run_performed": False,
        "learning_write_performed": False,
        "production_config_write_performed": False,
        "broker_access_performed": False,
        "live_execution_performed": False,
    }


def _run_id() -> str:
    return (
        "semiconductor_analyst_runtime_"
        + utc_now_iso().replace(":", "").replace("+", "Z")
    )


def render_semiconductor_runtime_markdown(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary", {})
    missing = summary.get("missing_required_evidence", [])
    lines = [
        "# Semiconductor Analyst Runtime",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- As of: `{payload.get('inputs', {}).get('as_of')}`",
        (
            "- Tickers: `"
            + ", ".join(payload.get("inputs", {}).get("tickers", []))
            + "`"
        ),
        (
            "- Recommendation: `"
            + str(summary.get("recommendation"))
            + "`"
        ),
        (
            "- Required lanes satisfied: `"
            + str(summary.get("satisfied_required_lane_count", 0))
            + "/"
            + str(summary.get("required_lane_count", 0))
            + "`"
        ),
        (
            "- Missing required evidence: `"
            + (", ".join(missing) if missing else "none")
            + "`"
        ),
        (
            "- Market confirmation ready: `"
            + str(summary.get("market_confirmation_ready", False)).lower()
            + "`"
        ),
        "",
        "The result is review-only. Missing lanes block the sector thesis, "
        "and ticker/model pipeline cases remain outside sector evidence.",
        "",
    ]
    return "\n".join(lines)
