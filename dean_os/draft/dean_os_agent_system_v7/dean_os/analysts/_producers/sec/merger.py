from __future__ import annotations

__all__ = [
    'SAVED_SEC_FUNDAMENTAL_MERGER_CONTRACT',
    'SavedSECFundamentalEvidenceMerger',
    'load_verified_merged_fundamental_context_fragment',
    'render_merged_fundamental_markdown',
]

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dean_os.analysts._producers.sec.companyfacts import (
    load_verified_fundamental_context_fragment,
)
from dean_os.analysts._producers.sec.inline_xbrl import (
    load_verified_inline_xbrl_context_fragment,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso
from dean_os.draft.dean_os_agent_system_v7.dean_os.structured_context_provenance import audit_structured_context

SAVED_SEC_FUNDAMENTAL_MERGER_CONTRACT = (
    "dean_saved_sec_fundamental_evidence_merger_v1"
)


class SavedSECFundamentalEvidenceMerger:
    """Merge verified SEC fact producers without hiding coverage gaps."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/saved_sec_fundamental_evidence_merger"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        companyfacts_artifact_path: str | Path,
        additional_companyfacts_artifact_paths: (
            list[str | Path] | None
        ) = None,
        inline_xbrl_artifact_paths: list[str | Path] | None = None,
        as_of: str | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        company_paths = [
            Path(companyfacts_artifact_path),
            *[
                Path(value)
                for value in (
                    additional_companyfacts_artifact_paths or []
                )
            ],
        ]
        company_fragment = load_verified_fundamental_context_fragment(
            company_paths[0],
            expected_as_of=as_of,
        )
        resolved_as_of = company_fragment["as_of"]
        company_payloads: list[dict[str, Any]] = []
        source_artifacts: list[dict[str, Any]] = []
        source_facts: list[dict[str, Any]] = []
        for company_path in company_paths:
            load_verified_fundamental_context_fragment(
                company_path,
                expected_as_of=resolved_as_of,
            )
            company_payload = json.loads(
                company_path.read_text(encoding="utf-8")
            )
            company_payloads.append(company_payload)
            source_artifacts.append(
                _source_artifact(
                    "sec_companyfacts",
                    company_path,
                    company_payload,
                )
            )
            source_facts.extend(
                {
                    **item,
                    "_producer_family": "sec_companyfacts",
                }
                for item in company_payload.get("facts", [])
            )
        for raw_path in inline_xbrl_artifact_paths or []:
            inline_path = Path(raw_path)
            load_verified_inline_xbrl_context_fragment(
                inline_path,
                expected_as_of=resolved_as_of,
            )
            inline_payload = json.loads(
                inline_path.read_text(encoding="utf-8")
            )
            source_artifacts.append(
                _source_artifact(
                    "sec_inline_xbrl",
                    inline_path,
                    inline_payload,
                )
            )
            source_facts.extend(
                {
                    **item,
                    "_producer_family": "sec_inline_xbrl",
                }
                for item in inline_payload.get("facts", [])
            )

        normalized = _merge_facts(source_facts)
        facts = normalized["facts"]
        fundamentals = _fundamentals_from_facts(facts)
        audit = audit_structured_context(
            fundamentals=fundamentals,
            macro={},
            sector_data={},
            as_of=resolved_as_of,
        )
        accepted = audit["accepted_context"]["fundamentals"]
        requested_tickers = sorted(
            {
                str(value).upper()
                for company_payload in company_payloads
                for value in company_payload.get("inputs", {}).get(
                    "requested_tickers", []
                )
            }
            | {
                str(item["ticker"]).upper()
                for item in facts
            }
        )
        accepted_tickers = sorted(accepted)
        missing_tickers = sorted(
            set(requested_tickers) - set(accepted_tickers)
        )
        comparability = _comparability(
            facts=facts,
            requested_tickers=requested_tickers,
        )
        facts_fingerprint = _canonical_sha256(
            [_fact_fingerprint_row(item) for item in facts]
        )
        status = (
            "merged_fundamental_evidence_ready_with_gaps"
            if facts
            and (
                missing_tickers
                or normalized["exclusions"]
                or comparability["status"] != "comparable"
            )
            else "merged_fundamental_evidence_ready"
            if facts
            else "blocked_no_merged_fundamental_evidence"
        )
        payload = {
            "run_id": _run_id(),
            "created_at": utc_now_iso(),
            "mode": "saved_sec_fundamental_evidence_merger",
            "producer_contract": SAVED_SEC_FUNDAMENTAL_MERGER_CONTRACT,
            "inputs": {
                "companyfacts_artifact_path": str(company_paths[0]),
                "companyfacts_artifact_paths": [
                    str(value) for value in company_paths
                ],
                "inline_xbrl_artifact_paths": [
                    str(Path(value))
                    for value in inline_xbrl_artifact_paths or []
                ],
                "as_of": resolved_as_of,
                "requested_tickers": requested_tickers,
            },
            "status": status,
            "source_artifacts": source_artifacts,
            "summary": {
                "source_artifact_count": len(source_artifacts),
                "source_fact_count": len(source_facts),
                "accepted_fact_count": audit["accepted_count"],
                "accepted_fact_ticker_count": len(accepted_tickers),
                "accepted_fact_tickers": accepted_tickers,
                "requested_ticker_count": len(requested_tickers),
                "missing_tickers": missing_tickers,
                "ticker_coverage_ratio": _ratio(
                    len(accepted_tickers),
                    len(requested_tickers),
                ),
                "ticker_coverage_status": (
                    "complete" if not missing_tickers else "partial"
                ),
                "duplicate_fact_count": normalized[
                    "duplicate_fact_count"
                ],
                "conflicting_fact_count": normalized[
                    "conflicting_fact_count"
                ],
                "excluded_fact_count": len(
                    normalized["exclusions"]
                ),
                "reason_counts": normalized["reason_counts"],
                "accepted_fingerprint": audit[
                    "accepted_fingerprint"
                ],
                "facts_fingerprint": facts_fingerprint,
                "cross_ticker_comparability_status": comparability[
                    "status"
                ],
                "can_enter_fundamental_input_gate": bool(accepted),
                "can_claim_complete_sector_fundamentals": (
                    bool(requested_tickers)
                    and not missing_tickers
                    and comparability["status"] == "comparable"
                    and not normalized["exclusions"]
                ),
                "can_compute_ratios": False,
                "can_translate_currencies": False,
                "can_feed_ticker_prediction_directly": False,
                "can_trade": False,
            },
            "fundamental_metric_rows": [
                _gate_metric_row(item) for item in facts
            ],
            "facts": facts,
            "exclusions": normalized["exclusions"],
            "comparability": comparability,
            "structured_context_audit": {
                key: value
                for key, value in audit.items()
                if key
                not in {
                    "accepted_context",
                    "accepted_observations",
                }
            },
            "market_context_fragment": {
                "as_of": resolved_as_of,
                "fundamentals": accepted,
                "metadata": {
                    "saved_sec_fundamental_merger_run_id": None,
                    "saved_sec_fundamental_merger_facts_fingerprint": (
                        facts_fingerprint
                    ),
                    "saved_sec_fundamental_merger_accepted_fingerprint": (
                        audit["accepted_fingerprint"]
                    ),
                    "requested_tickers": requested_tickers,
                    "accepted_fact_tickers": accepted_tickers,
                    "missing_tickers": missing_tickers,
                    "complete_sector_fundamentals": False,
                    "cross_ticker_comparability_status": comparability[
                        "status"
                    ],
                },
            },
            "integration_boundary": {
                "review_only": True,
                "verified_source_artifacts_only": True,
                "conflicting_duplicate_facts_fail_closed": True,
                "coverage_gaps_remain_explicit": True,
                "period_or_unit_mismatch_blocks_sector_comparison": True,
                "currency_translation_allowed": False,
                "ratio_computation_allowed": False,
                "ticker_prediction_feature_promotion_allowed": False,
            },
            "safety": _safety(),
        }
        payload["market_context_fragment"]["metadata"][
            "saved_sec_fundamental_merger_run_id"
        ] = payload["run_id"]
        payload["market_context_fragment"]["metadata"][
            "complete_sector_fundamentals"
        ] = payload["summary"][
            "can_claim_complete_sector_fundamentals"
        ]
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_merged_fundamental_markdown(payload),
                run_id=payload["run_id"],
            )
        return payload


def load_verified_merged_fundamental_context_fragment(
    artifact_path: str | Path,
    *,
    expected_as_of: str | None = None,
) -> dict[str, Any]:
    path = Path(artifact_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("producer_contract")
        != SAVED_SEC_FUNDAMENTAL_MERGER_CONTRACT
    ):
        raise ValueError("unsupported merged fundamental contract")
    if payload.get("status") not in {
        "merged_fundamental_evidence_ready",
        "merged_fundamental_evidence_ready_with_gaps",
    }:
        raise ValueError("merged fundamental artifact is not ready")
    summary = payload.get("summary", {})
    safety = payload.get("safety", {})
    if (
        safety.get("review_only") is not True
        or safety.get("network_access_performed") is not False
        or safety.get("live_execution_performed") is not False
        or summary.get("can_trade") is not False
    ):
        raise ValueError("merged fundamental safety boundary invalid")
    fragment = payload.get("market_context_fragment")
    if not isinstance(fragment, dict):
        raise ValueError("merged fundamental fragment missing")
    fragment_as_of = parse_timezone_aware(fragment.get("as_of"))
    if fragment_as_of is None:
        raise ValueError("merged fundamental as_of invalid")
    if expected_as_of is not None:
        expected = parse_timezone_aware(expected_as_of)
        if expected is None or expected != fragment_as_of:
            raise ValueError("merged fundamental expected as_of mismatch")

    for source in payload.get("source_artifacts", []):
        source_path = Path(str(source.get("path") or ""))
        if (
            not source_path.exists()
            or _sha256_file(source_path) != source.get("sha256")
        ):
            raise ValueError("merged fundamental source hash mismatch")
        if source.get("family") == "sec_companyfacts":
            load_verified_fundamental_context_fragment(
                source_path,
                expected_as_of=fragment_as_of.isoformat(),
            )
        elif source.get("family") == "sec_inline_xbrl":
            load_verified_inline_xbrl_context_fragment(
                source_path,
                expected_as_of=fragment_as_of.isoformat(),
            )
        else:
            raise ValueError("unsupported merged fundamental source")

    facts_fingerprint = _canonical_sha256(
        [_fact_fingerprint_row(item) for item in payload.get("facts", [])]
    )
    if facts_fingerprint != summary.get("facts_fingerprint"):
        raise ValueError("merged fundamental facts fingerprint mismatch")
    fundamentals = fragment.get("fundamentals")
    if not isinstance(fundamentals, dict):
        raise ValueError("merged fundamental payload invalid")
    audit = audit_structured_context(
        fundamentals=fundamentals,
        macro={},
        sector_data={},
        as_of=fragment_as_of.isoformat(),
    )
    if (
        audit["excluded_count"] != 0
        or audit["accepted_count"] != summary.get("accepted_fact_count")
        or audit["accepted_fingerprint"]
        != summary.get("accepted_fingerprint")
    ):
        raise ValueError("merged fundamental fragment fingerprint mismatch")
    return {
        "as_of": fragment_as_of.isoformat(),
        "fundamentals": audit["accepted_context"]["fundamentals"],
        "metadata": {
            **dict(fragment.get("metadata", {})),
            "saved_sec_fundamental_merger_artifact_path": str(path),
            "saved_sec_fundamental_merger_artifact_sha256": (
                _sha256_file(path)
            ),
            "saved_sec_fundamental_merger_verified": True,
        },
    }


def _merge_facts(source_facts: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for fact in source_facts:
        grouped[
            (
                str(fact.get("ticker") or "").upper(),
                str(fact.get("metric_name") or ""),
            )
        ].append(fact)
    accepted: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    duplicate_count = 0
    conflicting_count = 0
    for (ticker, metric_name), values in sorted(grouped.items()):
        identities: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in values:
            identities[_fact_identity(item)].append(item)
        if len(identities) > 1:
            conflicting_count += len(values)
            exclusions.append(
                {
                    "ticker": ticker,
                    "metric_name": metric_name,
                    "status": "excluded",
                    "reasons": ["conflicting_verified_fundamental_facts"],
                    "candidate_fact_sha256": sorted(
                        str(item.get("fact_sha256") or "")
                        for item in values
                    ),
                }
            )
            continue
        chosen = sorted(
            values,
            key=lambda item: (
                str(item.get("_producer_family") or ""),
                str(item.get("fact_sha256") or ""),
            ),
        )[0]
        duplicate_count += max(0, len(values) - 1)
        accepted.append(
            {
                key: value
                for key, value in chosen.items()
                if key != "_producer_family"
            }
        )
    reason_counts = Counter(
        reason
        for exclusion in exclusions
        for reason in exclusion["reasons"]
    )
    return {
        "facts": accepted,
        "exclusions": exclusions,
        "duplicate_fact_count": duplicate_count,
        "conflicting_fact_count": conflicting_count,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _fact_identity(item: dict[str, Any]) -> str:
    return _canonical_sha256(
        {
            "ticker": item.get("ticker"),
            "metric_name": item.get("metric_name"),
            "value": item.get("value"),
            "unit": item.get("unit"),
            "period": item.get("period"),
            "accepted_at": item.get("accepted_at"),
            "accession_number": item.get("accession_number"),
        }
    )


def _comparability(
    *,
    facts: list[dict[str, Any]],
    requested_tickers: list[str],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        grouped[fact["metric_name"]].append(fact)
    metrics: dict[str, Any] = {}
    comparable_count = 0
    for metric_name, values in sorted(grouped.items()):
        tickers = sorted({item["ticker"] for item in values})
        units = sorted({item["unit"] for item in values})
        periods = sorted({item["period"] for item in values})
        complete = set(tickers) == set(requested_tickers)
        comparable = complete and len(units) == 1 and len(periods) == 1
        if comparable:
            comparable_count += 1
        metrics[metric_name] = {
            "tickers": tickers,
            "missing_tickers": sorted(
                set(requested_tickers) - set(tickers)
            ),
            "units": units,
            "periods": periods,
            "ticker_coverage_ratio": _ratio(
                len(tickers),
                len(requested_tickers),
            ),
            "comparable_without_transformation": comparable,
        }
    return {
        "status": (
            "comparable"
            if metrics and comparable_count == len(metrics)
            else "partial_or_period_unit_mismatch"
            if metrics
            else "unavailable"
        ),
        "requested_tickers": requested_tickers,
        "comparable_metric_count": comparable_count,
        "metric_count": len(metrics),
        "metrics": metrics,
        "rule": (
            "Raw facts are directly comparable only when every requested "
            "ticker, reporting unit, and fiscal period match. No currency "
            "or period transformation is performed."
        ),
    }


def _fundamentals_from_facts(facts: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for fact in facts:
        ticker_payload = output.setdefault(
            fact["ticker"],
            {"metrics": {}},
        )
        ticker_payload["metrics"][fact["metric_name"]] = {
            "value": fact["value"],
            "unit": fact["unit"],
            "period": fact["period"],
            "available_at": fact["available_at"],
            "source_url": fact["source_locator"],
            "metadata": {
                "taxonomy": fact["taxonomy"],
                "concept": fact["concept"],
                "accession_number": fact["accession_number"],
                "source_artifact_sha256": fact[
                    "source_artifact_sha256"
                ],
                "source_fact_sha256": fact["source_fact_sha256"],
            },
        }
    return output


def _source_artifact(
    family: str,
    path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "family": family,
        "path": str(path),
        "sha256": _sha256_file(path),
        "producer_contract": payload.get("producer_contract"),
        "run_id": payload.get("run_id"),
        "status": payload.get("status"),
        "accepted_fingerprint": payload.get("summary", {}).get(
            "accepted_fingerprint"
        ),
        "facts_fingerprint": payload.get("summary", {}).get(
            "facts_fingerprint"
        ),
    }


def _gate_metric_row(fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": fact["ticker"],
        "metric_name": fact["metric_name"],
        "value": fact["value"],
        "unit": fact["unit"],
        "period": fact["period"],
        "available_at": fact["available_at"],
        "source_citation": fact["source_locator"],
        "accession_number": fact["accession_number"],
        "fact_sha256": fact["fact_sha256"],
    }


def _fact_fingerprint_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "ticker",
            "cik",
            "metric_name",
            "taxonomy",
            "concept",
            "value",
            "unit",
            "period",
            "accepted_at",
            "accession_number",
            "source_artifact_sha256",
            "source_fact_sha256",
            "fact_sha256",
        )
    }


def render_merged_fundamental_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Merged SEC Fundamental Evidence",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{payload.get('status')}`",
        f"- As-of: `{payload.get('inputs', {}).get('as_of')}`",
        (
            "- Requested tickers: "
            + (
                ", ".join(
                    payload.get("inputs", {}).get(
                        "requested_tickers",
                        [],
                    )
                )
                or "none"
            )
        ),
        (
            "- Accepted tickers: "
            + (
                ", ".join(summary.get("accepted_fact_tickers", []))
                or "none"
            )
        ),
        (
            "- Missing tickers: "
            + (
                ", ".join(summary.get("missing_tickers", []))
                or "none"
            )
        ),
        f"- Accepted facts: {summary.get('accepted_fact_count', 0)}",
        f"- Ticker coverage: {summary.get('ticker_coverage_ratio', 0)}",
        (
            "- Raw comparability: "
            f"`{summary.get('cross_ticker_comparability_status')}`"
        ),
        (
            "- Complete sector fundamentals: "
            f"{summary.get('can_claim_complete_sector_fundamentals', False)}"
        ),
        f"- Can trade: {summary.get('can_trade', False)}",
        "",
        "## Boundary",
        "",
        "- Only verified producer artifacts are merged.",
        "- Conflicting duplicate facts fail closed.",
        "- Missing tickers, different fiscal periods, and different currencies remain explicit.",
        "- No currency translation, ratio computation, prediction feature, learning, or trading action is performed.",
        "",
    ]
    return "\n".join(lines)


def _safety() -> dict[str, bool]:
    return {
        "review_only": True,
        "network_access_performed": False,
        "ratio_computation_performed": False,
        "valuation_performed": False,
        "pipeline_run_performed": False,
        "training_run_performed": False,
        "learning_write_performed": False,
        "production_config_write_performed": False,
        "paper_execution_performed": False,
        "live_execution_performed": False,
        "can_trade": False,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return (
        round(numerator / denominator, 6)
        if denominator > 0
        else 0.0
    )


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
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _run_id() -> str:
    return (
        "saved_sec_fundamental_merger_"
        f"{utc_now_iso().replace(':', '').replace('+', 'Z')}"
    )
