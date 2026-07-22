"""ArtifactEvidenceLoader — converts saved producer artifacts into AnalystEvidenceItem[].

Two modes:
1. Load from full runtime artifact (already adapted evidence)
2. Load from individual producer artifacts (news, macro, sector market, policy)

All reads are offline — no network, no collectors, no pipeline runs.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dean_os.analysts.schemas import AnalystEvidenceItem
from dean_os.utils import sha256_json

RUNTIME_CONTRACT = "dean_semiconductor_analyst_runtime_v1"
RUNTIME_MODE = "semiconductor_analyst_runtime"

# ──────────────────────────────────────────────────────────────────────────────
# Freshness decay helper
# ──────────────────────────────────────────────────────────────────────────────

# Half-life in days by evidence type: faster decay for news, slower for macro.
_FRESHNESS_HALF_LIFE: dict[str, float] = {
    "news": 3.0,
    "policy_or_geopolitical": 14.0,
    "sector_demand": 7.0,
    "supply_chain": 7.0,
    "capex_cycle": 21.0,
    "market_confirmation": 5.0,
    "macro_context": 30.0,
    "fundamental_context": 90.0,
    "earnings_guidance": 30.0,
    "order_backlog": 14.0,
    "export_control_update": 14.0,
}
_DEFAULT_HALF_LIFE = 7.0  # days


def _compute_freshness(
    item_timestamp: str | None,
    as_of: str | None,
    evidence_type: str = "",
) -> float:
    """Exponential freshness decay based on item age relative to as_of.

    score = 0.5 ^ (age_days / half_life)

    Returns a value in [0.05, 1.0]:
    - 1.0  → published exactly at as_of
    - 0.5  → published one half-life ago
    - 0.05 → very old / unknown timestamp (floor)
    """
    if not item_timestamp or not as_of:
        return 0.5  # neutral default when timestamps are missing
    try:
        item_dt = datetime.fromisoformat(
            str(item_timestamp).replace("Z", "+00:00")
        )
        as_of_dt = datetime.fromisoformat(
            str(as_of).replace("Z", "+00:00")
        )
        age_days = max(0.0, (as_of_dt - item_dt).total_seconds() / 86400.0)
    except (TypeError, ValueError):
        return 0.5
    half_life = _FRESHNESS_HALF_LIFE.get(evidence_type, _DEFAULT_HALF_LIFE)
    score = 0.5 ** (age_days / half_life)
    return max(0.05, min(1.0, score))



def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _find_latest(artifact_dir: Path) -> Path:
    if artifact_dir.is_file():
        return artifact_dir
    latest = artifact_dir / "latest.json"
    if not latest.exists():
        raise FileNotFoundError(f"No latest.json in {artifact_dir}")
    return latest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _resolve_linked_path(raw_path: str, runtime_path: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    for root in [Path.cwd(), *runtime_path.parents]:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return Path.cwd() / candidate


class ArtifactEvidenceLoader:
    """Loads AnalystEvidenceItem[] from saved producer artifacts.

    Usage::

        loader = ArtifactEvidenceLoader()

        # From full runtime (152 adapted evidence items)
        evidence = loader.from_runtime_artifact(
            Path("reports/dean_os/semiconductor_analyst_runtime_current"),
            domain_id="semiconductor_ai_infrastructure",
        )

        # From individual producers
        evidence = loader.from_producer_artifacts(
            news_path=Path("reports/dean_os/saved_semiconductor_news_evidence_producer_current"),
            macro_path=Path("reports/dean_os/saved_macro_evidence_producer_current"),
            sector_market_path=Path("reports/dean_os/saved_sector_market_evidence_producer_current"),
            policy_path=Path("reports/dean_os/saved_official_policy_evidence_producer_current"),
            domain_id="semiconductor_ai_infrastructure",
            as_of="2026-07-01T00:00:00Z",
        )
    """

    def from_runtime_artifact(
        self,
        artifact_dir: Path,
        domain_id: str = "semiconductor_ai_infrastructure",
        as_of: str | None = None,
    ) -> list[AnalystEvidenceItem]:
        """Load adapted evidence from a full runtime artifact.

        The runtime artifact already has evidence items in
        ``adapter.evidence`` — we just parse and validate them.
        """
        runtime_path = _find_latest(artifact_dir)
        data = _load_json(runtime_path)

        report_evidence = data.get("analyst_report", {}).get("evidence", [])
        legacy_evidence = data.get("adapter", {}).get("evidence", [])
        raw_evidence = report_evidence or legacy_evidence

        if not raw_evidence:
            raise ValueError(
                f"Runtime artifact at {artifact_dir} has no adapted evidence "
                "in analyst_report.evidence or legacy adapter.evidence"
            )

        if report_evidence:
            self._validate_current_runtime(
                data=data,
                runtime_path=runtime_path,
                raw_evidence=raw_evidence,
                domain_id=domain_id,
            )

        evidence: list[AnalystEvidenceItem] = []
        for index, item in enumerate(raw_evidence):
            try:
                evidence.append(AnalystEvidenceItem(**item))
            except Exception as exc:
                raise ValueError(
                    f"Invalid runtime evidence item at index {index}: {exc}"
                ) from exc

        if report_evidence:
            self._validate_evidence_items(
                evidence=evidence,
                data=data,
                domain_id=domain_id,
                expected_as_of=as_of,
            )

        return evidence

    def _validate_current_runtime(
        self,
        *,
        data: dict[str, Any],
        runtime_path: Path,
        raw_evidence: list[dict[str, Any]],
        domain_id: str,
    ) -> None:
        if data.get("runtime_contract") != RUNTIME_CONTRACT:
            raise ValueError(
                f"Unexpected runtime_contract {data.get('runtime_contract')!r}; "
                f"expected {RUNTIME_CONTRACT!r}"
            )
        if data.get("mode") != RUNTIME_MODE:
            raise ValueError(
                f"Unexpected runtime mode {data.get('mode')!r}; expected {RUNTIME_MODE!r}"
            )
        if data.get("domain_id") != domain_id:
            raise ValueError(
                f"Runtime domain {data.get('domain_id')!r} does not match "
                f"requested domain {domain_id!r}"
            )
        if "review" not in str(data.get("status", "")).lower():
            raise ValueError(f"Runtime status is not review-routable: {data.get('status')!r}")

        safety = data.get("safety", {})
        if safety.get("review_only") is not True:
            raise ValueError("Runtime safety.review_only must be true")
        forbidden_performed = (
            "pipeline_run_performed",
            "training_run_performed",
            "tuning_run_performed",
            "learning_write_performed",
            "production_config_write_performed",
            "broker_access_performed",
            "live_execution_performed",
        )
        performed = [name for name in forbidden_performed if safety.get(name) is True]
        if performed:
            raise ValueError(
                "Runtime violated review-only safety: " + ", ".join(sorted(performed))
            )

        expected_counts = {
            "summary.evidence_count": data.get("summary", {}).get("evidence_count"),
            "adapter.summary.evidence_count": (
                data.get("adapter", {}).get("summary", {}).get("evidence_count")
            ),
        }
        for label, expected in expected_counts.items():
            if expected is not None and int(expected) != len(raw_evidence):
                raise ValueError(
                    f"Runtime evidence count mismatch: {label}={expected}, "
                    f"analyst_report.evidence={len(raw_evidence)}"
                )

        for name, linked in data.get("source_artifacts", {}).items():
            if not isinstance(linked, dict):
                raise ValueError(f"source_artifacts.{name} must be an object")
            raw_path = str(linked.get("path", "")).strip()
            expected_hash = str(linked.get("sha256", "")).strip().lower()
            if not raw_path or len(expected_hash) != 64:
                raise ValueError(f"source_artifacts.{name} requires path and sha256")
            linked_path = _resolve_linked_path(raw_path, runtime_path)
            if not linked_path.is_file():
                raise FileNotFoundError(
                    f"Linked runtime artifact does not exist: {linked_path}"
                )
            actual_hash = _sha256(linked_path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Linked artifact hash mismatch for {name}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )

    @staticmethod
    def _validate_evidence_items(
        *,
        evidence: list[AnalystEvidenceItem],
        data: dict[str, Any],
        domain_id: str,
        expected_as_of: str | None = None,
    ) -> None:
        runtime_as_of = (
            data.get("inputs", {}).get("as_of")
            or data.get("adapter", {}).get("as_of")
            or data.get("analyst_report", {}).get("as_of")
        )
        if not runtime_as_of:
            raise ValueError("Current runtime artifact is missing its as_of cutoff")
        cutoff = _parse_timestamp(str(runtime_as_of))
        if (
            expected_as_of
            and _parse_timestamp(str(expected_as_of)) != cutoff
        ):
            raise ValueError(
                "Requested as_of does not match runtime cutoff: "
                f"{expected_as_of!r} != {runtime_as_of!r}"
            )

        seen_ids: set[str] = set()
        for index, item in enumerate(evidence):
            if item.evidence_id in seen_ids:
                raise ValueError(f"Duplicate evidence_id {item.evidence_id!r}")
            seen_ids.add(item.evidence_id)
            if item.domain_id != domain_id:
                raise ValueError(
                    f"Evidence item {index} domain {item.domain_id!r} does not "
                    f"match runtime domain {domain_id!r}"
                )
            if item.as_of != runtime_as_of:
                raise ValueError(
                    f"Evidence item {item.evidence_id} as_of {item.as_of!r} "
                    f"does not match runtime cutoff {runtime_as_of!r}"
                )
            if item.published_at and _parse_timestamp(item.published_at) > cutoff:
                raise ValueError(
                    f"Evidence item {item.evidence_id} is future evidence "
                    f"({item.published_at} > {runtime_as_of})"
                )

    def from_producer_artifacts(
        self,
        *,
        news_path: Path | None = None,
        macro_path: Path | None = None,
        sector_market_path: Path | None = None,
        policy_path: Path | None = None,
        fundamental_path: Path | None = None,
        domain_id: str,
        as_of: str = "",
    ) -> list[AnalystEvidenceItem]:
        """Load evidence from individual producer artifacts.

        Each producer produces a ``market_context_fragment`` with its own
        data. We extract the evidence items from each fragment.
        """
        evidence: list[AnalystEvidenceItem] = []

        if news_path is not None:
            evidence.extend(self._load_news_evidence(news_path, domain_id, as_of))

        if policy_path is not None:
            evidence.extend(self._load_news_evidence(policy_path, domain_id, as_of))

        if macro_path is not None:
            evidence.extend(self._load_structured_evidence(macro_path, domain_id, as_of, "macro"))

        if sector_market_path is not None:
            evidence.extend(self._load_structured_evidence(sector_market_path, domain_id, as_of, "sector"))

        if fundamental_path is not None:
            evidence.extend(self._load_fundamental_evidence(fundamental_path, domain_id, as_of))

        return evidence

    def from_signal_bus(
        self,
        domain_id: str,
        as_of: str,
        *,
        bus_dir: Path | None = None,
    ) -> list[AnalystEvidenceItem]:
        """Load explicitly enabled, hash-bound cross-domain signals."""
        bus_dir = bus_dir or Path("reports/dean_os/signal_bus")
        if not bus_dir.is_dir():
            return []

        analysis_cutoff = _aware_timestamp(as_of, "analysis as_of")
        evidence: list[AnalystEvidenceItem] = []
        for path in sorted(bus_dir.glob("signal_*.json")):
            signal = _load_json(path)
            if signal.get("contract") != "dean_cross_domain_signal_v1":
                raise ValueError(f"Unsupported cross-domain signal contract: {path}")
            expected_hash = str(signal.get("signal_sha256") or "").strip().lower()
            hash_payload = {
                key: value
                for key, value in signal.items()
                if key != "signal_sha256"
            }
            if not expected_hash or sha256_json(hash_payload) != expected_hash:
                raise ValueError(f"Cross-domain signal hash mismatch: {path}")
            available_at = str(signal.get("available_at") or "").strip()
            if not available_at:
                raise ValueError(f"Cross-domain signal availability missing: {path}")
            if _aware_timestamp(available_at, "signal available_at") > analysis_cutoff:
                raise ValueError(f"Cross-domain signal is future evidence: {path}")
            source_hash = str(signal.get("source_evidence_sha256") or "").strip().lower()
            if not source_hash:
                raise ValueError(f"Cross-domain signal lineage hash missing: {path}")

            rules = signal.get("propagation_rules", {})
            if domain_id not in rules.get("target_domains", []):
                continue

            source_domain = signal.get("source_domain", "unknown")
            event_class = signal.get("event_class", "unknown")
            title = signal.get("title", f"{event_class} signal from {source_domain}")
            text = signal.get("text", "")
            raw_mat = float(signal.get("materiality", 0.5) or 0.5)
            multiplier = float(rules.get("strength_multiplier", 1.0) or 1.0)
            strength = min(1.0, raw_mat * multiplier)
            source_reliability = float(signal.get("source_reliability", 0.0) or 0.0)
            reliability = min(0.7, source_reliability * multiplier)

            evidence.append(
                AnalystEvidenceItem(
                    evidence_id=f"cross_domain_{expected_hash[:24]}_{domain_id}",
                    evidence_type=rules.get("evidence_type", "macro_context"),
                    source=f"cross_domain_bus:{source_domain}",
                    source_type="signal_bus",
                    summary=f"[CROSS-DOMAIN: {event_class}] {title}\n{text}",
                    directness="indirect",
                    stance_hint=rules.get("stance_hint", "mixed"),
                    reliability_score=reliability,
                    strength=strength,
                    point_in_time={
                        "contract": "dean_cross_domain_signal_v1",
                        "status": "point_in_time_compatible",
                        "available_at": available_at,
                        "analysis_as_of": as_of,
                    },
                    as_of=as_of,
                    published_at=available_at,
                    domain_id=domain_id,
                    tickers=[],
                    sectors=[domain_id],
                    provenance={
                        "source_domain": source_domain,
                        "original_event_class": event_class,
                        "source_evidence_id": signal.get("source_evidence_id"),
                        "source_evidence_sha256": source_hash,
                        "signal_sha256": expected_hash,
                        "signal_file": path.name,
                    },
                )
            )

        return evidence

    def _load_news_evidence(
        self,
        artifact_dir: Path,
        domain_id: str,
        as_of: str,
    ) -> list[AnalystEvidenceItem]:
        """Extract evidence from news/policy producer artifacts.

        Handles both real artifacts (market_context_fragment.news[]) and
        test fixtures (context_adapter.market_context_fragment.news[]).
        """
        data, artifact_path, artifact_sha256, source_as_of = _validated_producer(
            artifact_dir, as_of=as_of
        )
        # Real artifacts: market_context_fragment is top-level
        fragment = data.get("market_context_fragment", {})
        # Test fixtures: nested under context_adapter
        if not fragment.get("news"):
            fragment = data.get("context_adapter", {}).get("market_context_fragment", {})
        news_items = fragment.get("news", [])

        evidence: list[AnalystEvidenceItem] = []
        for item in news_items:
            semantic = item.get("_dean_semantic_evidence", {})
            if not semantic:
                continue

            evidence_type = semantic.get("evidence_type", "other")
            source_tier = semantic.get("source_tier", "tier_3_event_context")
            reliability = {
                "tier_1_core_evidence": 0.9,
                "tier_2_strong_context": 0.75,
                "tier_3_event_context": 0.55,
                "tier_4_weak_or_unverified": 0.3,
            }.get(source_tier, 0.4)

            stance = semantic.get("stance_hint", "unknown")
            if stance not in ("positive", "negative", "neutral", "mixed", "unknown"):
                stance = "unknown"

            summary = item.get("summary") or item.get("title") or ""
            if not summary:
                continue

            published_at = item.get("published_at")
            _validate_item_time(
                published_at, source_as_of=source_as_of, label="news published_at"
            )
            evidence.append(AnalystEvidenceItem(
                evidence_id=_producer_evidence_id(
                    artifact_sha256, "news", semantic.get("candidate_sha256") or item
                ),
                source_type="news",
                source=semantic.get("source_identity", item.get("source", "unknown")),
                published_at=published_at,
                as_of=as_of or data.get("created_at", ""),
                domain_id=domain_id,
                tickers=[],
                sectors=[domain_id],
                evidence_type=evidence_type,
                summary=summary[:260],
                stance_hint=stance,
                strength=min(1.0, 0.5 + 0.1 * len(semantic.get("matched_terms", []))),
                freshness_score=_compute_freshness(published_at, as_of, evidence_type),
                directness="sector",
                reliability_score=reliability,
                limitations=[
                    f"Loaded from saved producer artifact: {data.get('producer_contract', 'unknown')}"
                ],
                provenance={
                    "producer_contract": semantic.get("producer_contract"),
                    "producer_artifact_path": str(artifact_path),
                    "producer_artifact_sha256": artifact_sha256,
                    "producer_created_at": data.get("created_at"),
                    "producer_as_of": source_as_of,
                    "candidate_sha256": semantic.get("candidate_sha256"),
                    "source_tier": source_tier,
                    "required_lane_eligible": semantic.get("required_lane_eligible", False),
                },
                point_in_time={
                    "status": "historical_snapshot_compatible",
                    "available_at": published_at,
                    "producer_as_of": source_as_of,
                    "analysis_as_of": as_of,
                },
            ))

        return evidence

    def _load_structured_evidence(
        self,
        artifact_dir: Path,
        domain_id: str,
        as_of: str,
        family: str,
    ) -> list[AnalystEvidenceItem]:
        """Extract evidence from macro/sector structured producer artifacts."""
        data, artifact_path, artifact_sha256, source_as_of = _validated_producer(
            artifact_dir, as_of=as_of
        )
        observations = data.get("selected_observations", data.get("metrics", []))

        evidence: list[AnalystEvidenceItem] = []
        for obs in observations:
            name = obs.get("context_key") or obs.get("name", "unknown")
            value = obs.get("value")
            unit = obs.get("unit", "")
            period = obs.get("period", "")
            source_locator = obs.get("source_locator", "")
            available_at = obs.get("available_at") or obs.get("observation_at", "")

            if value is None:
                continue

            evidence_type = obs.get("evidence_type", f"{family}_context")
            stance = obs.get("stance_hint", "unknown")
            if stance not in ("positive", "negative", "neutral", "mixed", "unknown"):
                stance = "unknown"

            required_eligible = obs.get("required_lane_eligible", False)

            _validate_item_time(
                available_at,
                source_as_of=source_as_of,
                label=f"{family} available_at",
            )
            evidence.append(AnalystEvidenceItem(
                evidence_id=_producer_evidence_id(
                    artifact_sha256, family, obs
                ),
                source_type="macro" if family == "macro" else "sector",
                source=source_locator,
                published_at=available_at,
                as_of=as_of or data.get("created_at", ""),
                domain_id=domain_id,
                tickers=[],
                sectors=[domain_id],
                evidence_type=evidence_type,
                summary=f"{family} observation {name}={value} {unit} for {period}.",
                stance_hint=stance,
                strength=0.7 if required_eligible else 0.55,
                freshness_score=_compute_freshness(available_at, as_of, evidence_type),
                directness="macro" if family == "macro" else "sector",
                reliability_score=0.6 if required_eligible else 0.5,
                limitations=[
                    f"Structured context from saved {family} producer artifact."
                ],
                provenance={
                    "family": family,
                    "producer_artifact_path": str(artifact_path),
                    "producer_artifact_sha256": artifact_sha256,
                    "producer_created_at": data.get("created_at"),
                    "producer_as_of": source_as_of,
                    "name": name,
                    "unit": unit,
                    "period": period,
                    "required_lane_eligible": required_eligible,
                },
                point_in_time={
                    "status": "historical_snapshot_compatible",
                    "available_at": available_at,
                    "producer_as_of": source_as_of,
                    "analysis_as_of": as_of,
                },
            ))

        return evidence

    def _load_fundamental_evidence(
        self,
        artifact_dir: Path,
        domain_id: str,
        as_of: str,
    ) -> list[AnalystEvidenceItem]:
        """Extract evidence from SEC fundamental merger artifact."""
        data, artifact_path, artifact_sha256, source_as_of = _validated_producer(
            artifact_dir, as_of=as_of
        )
        facts = data.get("fundamental_metric_rows", [])

        evidence: list[AnalystEvidenceItem] = []
        for fact in facts:
            ticker = fact.get("ticker", "")
            metric = fact.get("metric_name", "")
            value = fact.get("value")
            unit = fact.get("unit", "")
            period = fact.get("period", "")

            if value is None or not ticker:
                continue

            available_at = fact.get("available_at")
            _validate_item_time(
                available_at,
                source_as_of=source_as_of,
                label="fundamental available_at",
            )
            evidence.append(AnalystEvidenceItem(
                evidence_id=_producer_evidence_id(
                    artifact_sha256, "fundamental", fact
                ),
                source_type="fundamental",
                source=fact.get("source_citation", ""),
                published_at=available_at,
                as_of=as_of or data.get("created_at", ""),
                domain_id=domain_id,
                tickers=[ticker],
                sectors=[domain_id],
                evidence_type="fundamental_context",
                summary=f"{ticker} {metric}={value:,.0f} {unit} for {period}.",
                stance_hint="unknown",
                strength=0.6,
                freshness_score=_compute_freshness(available_at, as_of, "fundamental_context"),
                directness="ticker",
                reliability_score=0.8,
                limitations=[
                    "SEC Company Facts from saved fundamental merger artifact."
                ],
                provenance={
                    "producer_artifact_path": str(artifact_path),
                    "producer_artifact_sha256": artifact_sha256,
                    "producer_created_at": data.get("created_at"),
                    "producer_as_of": source_as_of,
                    "ticker": ticker,
                    "metric": metric,
                    "unit": unit,
                    "period": period,
                    "accession": fact.get("accession_number", ""),
                },
                point_in_time={
                    "status": "historical_snapshot_compatible",
                    "available_at": available_at,
                    "producer_as_of": source_as_of,
                    "analysis_as_of": as_of,
                },
            ))

        return evidence


def _validated_producer(
    artifact_dir: Path,
    *,
    as_of: str,
) -> tuple[dict[str, Any], Path, str, str]:
    if not as_of:
        raise ValueError("Producer evidence loading requires an analysis as_of")
    analysis_cutoff = _aware_timestamp(as_of, "analysis as_of")
    artifact_path = _find_latest(artifact_dir)
    data = _load_json(artifact_path)
    created_at = str(data.get("created_at") or "").strip()
    created = _aware_timestamp(created_at, "producer created_at")
    if created > analysis_cutoff:
        raise ValueError(
            f"Producer artifact is future evidence: {created_at} > {as_of}"
        )
    source_as_of = str(
        (data.get("inputs") or {}).get("as_of") or created_at
    ).strip()
    source_cutoff = _aware_timestamp(source_as_of, "producer inputs.as_of")
    if source_cutoff > analysis_cutoff:
        raise ValueError(
            f"Producer cutoff is after analysis cutoff: {source_as_of} > {as_of}"
        )
    if "ready" not in str(data.get("status") or "").lower():
        raise ValueError(f"Producer artifact is not review-routable: {data.get('status')!r}")
    safety = data.get("safety") or {}
    if safety.get("review_only") is not True:
        raise ValueError("Producer safety.review_only must be true")
    forbidden = (
        "training_run_performed",
        "tuning_run_performed",
        "learning_write_performed",
        "production_config_write_performed",
        "broker_access_performed",
        "live_execution_performed",
    )
    performed = [key for key in forbidden if safety.get(key) is True]
    if performed:
        raise ValueError(
            "Producer violated review-only safety: " + ", ".join(performed)
        )
    return data, artifact_path, _sha256(artifact_path), source_as_of


def _validate_item_time(
    value: Any,
    *,
    source_as_of: str,
    label: str,
) -> None:
    if value in (None, ""):
        return
    if _aware_timestamp(str(value), label) > _aware_timestamp(
        source_as_of, "producer inputs.as_of"
    ):
        raise ValueError(
            f"{label} is after producer cutoff: {value} > {source_as_of}"
        )


def _aware_timestamp(value: str, label: str) -> datetime:
    parsed = _parse_timestamp(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed


def _producer_evidence_id(
    artifact_sha256: str,
    family: str,
    payload: Any,
) -> str:
    return "producer_evidence_" + sha256_json(
        {
            "artifact_sha256": artifact_sha256,
            "family": family,
            "payload": payload,
        }
    )[:24]


def load_evidence_from_artifacts(
    artifact_paths: dict[str, str],
    domain_id: str,
    as_of: str = "",
) -> list[AnalystEvidenceItem]:
    """Convenience function to load evidence from artifact paths.

    Args:
        artifact_paths: Dict mapping artifact type to directory path.
            Keys: "runtime", "news", "macro", "sector_market", "policy", "fundamental"
        domain_id: Domain identifier.
        as_of: Point-in-time cutoff.

    Returns:
        List of AnalystEvidenceItem instances.
    """
    loader = ArtifactEvidenceLoader()

    if "runtime" in artifact_paths:
        return loader.from_runtime_artifact(
            Path(artifact_paths["runtime"]),
            domain_id=domain_id,
        )

    return loader.from_producer_artifacts(
        news_path=Path(artifact_paths["news"]) if "news" in artifact_paths else None,
        macro_path=Path(artifact_paths["macro"]) if "macro" in artifact_paths else None,
        sector_market_path=Path(artifact_paths["sector_market"]) if "sector_market" in artifact_paths else None,
        policy_path=Path(artifact_paths["policy"]) if "policy" in artifact_paths else None,
        fundamental_path=Path(artifact_paths["fundamental"]) if "fundamental" in artifact_paths else None,
        domain_id=domain_id,
        as_of=as_of,
    )


__all__ = [
    "ArtifactEvidenceLoader",
    "load_evidence_from_artifacts",
]
