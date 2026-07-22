from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dean_os.schemas import MarketContext
from dean_os.utils import sha256_json


STAGE03_PACKET_SCHEMA_VERSION = "dean_pipeline_stage03_packet_v1"
_ALLOWED_STAGES = {0, 1, 2, 3}


class PipelineArtifactReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    stage: int = Field(ge=0, le=3)
    artifact_type: str
    locator: str
    format: str = "unknown"
    content_hash: str | None = None
    row_count: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineStageState(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: int = Field(ge=0, le=3)
    status: Literal["available", "partial", "missing", "failed"] = "missing"
    output_keys: list[str] = Field(default_factory=list)
    row_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PipelineStage03Packet(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    schema_version: str = STAGE03_PACKET_SCHEMA_VERSION
    status: Literal["available", "partial", "missing", "failed"]
    as_of: str
    knowledge_cutoff: str
    stages_present: list[int] = Field(default_factory=list)
    stage_states: list[PipelineStageState] = Field(default_factory=list)
    news_items: list[dict[str, Any]] = Field(default_factory=list)
    macro_payload: dict[str, Any] = Field(default_factory=dict)
    sector_payload: dict[str, Any] = Field(default_factory=dict)
    artifact_references: list[PipelineArtifactReference] = Field(default_factory=list)
    dataframe_keys: list[str] = Field(default_factory=list)
    source_result_hash: str
    warnings: list[str] = Field(default_factory=list)
    safety: dict[str, bool] = Field(
        default_factory=lambda: {
            "stages_above_3_allowed": False,
            "can_trade": False,
            "can_promote_model": False,
            "source_artifacts_mutable": False,
        }
    )
    content_hash: str = ""

    @model_validator(mode="after")
    def validate_packet(self) -> "PipelineStage03Packet":
        if any(stage not in _ALLOWED_STAGES for stage in self.stages_present):
            raise ValueError("PipelineStage03Packet may only contain stages 0-3")
        as_of = datetime.fromisoformat(self.as_of)
        cutoff = datetime.fromisoformat(self.knowledge_cutoff)
        if as_of.tzinfo is None or cutoff.tzinfo is None:
            raise ValueError("as_of and knowledge_cutoff must be timezone-aware")
        if cutoff > as_of:
            raise ValueError("knowledge_cutoff cannot exceed as_of")
        expected = sha256_json(self.model_dump(mode="json", exclude={"content_hash"}))
        if self.content_hash and self.content_hash != expected:
            raise ValueError("PipelineStage03Packet content hash mismatch")
        object.__setattr__(self, "content_hash", expected)
        return self


class PipelineStage03Bridge:
    """Normalize already-produced pipeline stages 0-3 into DEAN-OS context.

    The bridge is intentionally read-only. It accepts an in-memory pipeline
    result or explicit artifact references. It never starts stage 4+ and never
    mutates the original pipeline artifacts.
    """

    def build_packet(
        self,
        result: dict[str, Any] | None,
        *,
        as_of: str | None = None,
        knowledge_cutoff: str | None = None,
        artifact_references: list[PipelineArtifactReference | dict[str, Any]] | None = None,
    ) -> PipelineStage03Packet:
        resolved_as_of = as_of or datetime.now(UTC).isoformat()
        resolved_cutoff = knowledge_cutoff or resolved_as_of
        payload = dict(result or {})
        nested = payload.get("results")
        result_view = dict(nested) if isinstance(nested, dict) else payload
        warnings: list[str] = []

        requested_stages = _normalize_stage_list(
            payload.get("requested_stages") or payload.get("stages_to_run")
        )
        forbidden = sorted(stage for stage in requested_stages if stage not in _ALLOWED_STAGES)
        if forbidden:
            warnings.append(
                "Ignored stage identifiers above the active 0-3 integration boundary: "
                + ", ".join(str(item) for item in forbidden)
            )
        stages_present = sorted(
            _infer_stages(result_view) | {stage for stage in requested_stages if stage in _ALLOWED_STAGES}
        )
        stage_states = [self._stage_state(stage, result_view) for stage in range(4)]

        news_items = _extract_news_records(result_view)
        macro_payload = _extract_mapping(result_view, ("macro", "macro_data", "economic_data"))
        sector_payload = _extract_mapping(result_view, ("sector_data", "sector", "industry_data"))
        dataframe_keys = sorted(
            key for key, value in result_view.items() if _is_dataframe_like(value)
        )
        references = [
            item if isinstance(item, PipelineArtifactReference) else PipelineArtifactReference(**item)
            for item in (artifact_references or [])
        ]
        if not references:
            references = _references_from_result(result_view)

        status = _packet_status(payload, stages_present, stage_states)
        packet = PipelineStage03Packet(
            status=status,
            as_of=resolved_as_of,
            knowledge_cutoff=resolved_cutoff,
            stages_present=stages_present,
            stage_states=stage_states,
            news_items=news_items,
            macro_payload=macro_payload,
            sector_payload=sector_payload,
            artifact_references=references,
            dataframe_keys=dataframe_keys,
            source_result_hash=sha256_json(_result_identity_payload(payload)),
            warnings=warnings,
        )
        return packet

    def attach_to_context(
        self,
        context: MarketContext,
        packet: PipelineStage03Packet,
        *,
        raw_result: dict[str, Any] | None = None,
    ) -> MarketContext:
        context.as_of = context.as_of or packet.as_of
        context.metadata["knowledge_cutoff"] = packet.knowledge_cutoff
        context.metadata["pipeline_stage03_packet"] = packet.model_dump(mode="json")
        context.metadata["pipeline_active_stage_boundary"] = [0, 1, 2, 3]
        context.metadata["pipeline_stage03_source_hash"] = packet.source_result_hash
        context.metadata["pipeline_stage03_artifacts"] = [
            item.model_dump(mode="json") for item in packet.artifact_references
        ]
        if packet.macro_payload:
            context.macro.update(packet.macro_payload)
        if packet.sector_payload:
            context.sector_data.update(packet.sector_payload)
        if packet.news_items:
            context.news.extend(packet.news_items)

        if isinstance(raw_result, dict):
            nested = raw_result.get("results")
            view = nested if isinstance(nested, dict) else raw_result
            for source_key, context_key in (
                ("features_df", "features"),
                ("targets_df", "targets"),
                ("market_data", "market"),
                ("news_data", "news"),
                ("processed_data", "processed"),
                ("raw_data", "raw"),
                ("economic_data", "macro"),
                ("macro_data", "macro"),
            ):
                value = view.get(source_key) if isinstance(view, dict) else None
                if value is not None:
                    context.dataframes[context_key] = value
            context.pipeline_result.update(raw_result)
        context.pipeline_result.setdefault("status", packet.status)
        context.pipeline_result["stage03_packet"] = packet.model_dump(mode="json")
        return context

    def _stage_state(self, stage: int, result: dict[str, Any]) -> PipelineStageState:
        output_keys = sorted(_keys_for_stage(stage, result))
        row_counts: dict[str, int] = {}
        warnings: list[str] = []
        for key in output_keys:
            value = result.get(key)
            count = _safe_len(value)
            if count is not None:
                row_counts[key] = count
        if output_keys:
            status: Literal["available", "partial", "missing", "failed"] = "available"
        else:
            status = "missing"
        explicit = _explicit_stage_payload(stage, result)
        if isinstance(explicit, dict):
            explicit_status = str(explicit.get("status") or "").lower()
            if explicit_status in {"failed", "error"}:
                status = "failed"
                warnings.append(str(explicit.get("error") or explicit.get("reason") or "stage failed"))
            elif explicit_status in {"partial", "warning"}:
                status = "partial"
        return PipelineStageState(
            stage=stage,
            status=status,
            output_keys=output_keys,
            row_counts=row_counts,
            warnings=warnings,
        )


def _normalize_stage_list(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(result))


def _infer_stages(result: dict[str, Any]) -> set[int]:
    stages: set[int] = set()
    keys = set(result)
    if keys & {"setup", "config", "runtime_profile", "stage_0", "stage0"}:
        stages.add(0)
    if keys & {"raw_data", "collection", "collected_data", "stage_1", "stage1"}:
        stages.add(1)
    if keys & {"processed_data", "processing", "normalized_data", "stage_2", "stage2"}:
        stages.add(2)
    if keys & {
        "features_df", "targets_df", "feature_data", "feature_engineering",
        "news_data", "stage_3", "stage3",
    }:
        stages.add(3)
    for stage in range(4):
        if _explicit_stage_payload(stage, result) is not None:
            stages.add(stage)
    return stages


def _explicit_stage_payload(stage: int, result: dict[str, Any]) -> Any:
    for key in (stage, str(stage), f"stage_{stage}", f"stage{stage}"):
        if key in result:
            return result[key]
    stage_results = result.get("stage_results")
    if isinstance(stage_results, dict):
        for key in (stage, str(stage), f"stage_{stage}", f"stage{stage}"):
            if key in stage_results:
                return stage_results[key]
    return None


def _keys_for_stage(stage: int, result: dict[str, Any]) -> set[str]:
    groups = {
        0: {"setup", "config", "runtime_profile", "stage_0", "stage0"},
        1: {"raw_data", "collection", "collected_data", "stage_1", "stage1"},
        2: {"processed_data", "processing", "normalized_data", "stage_2", "stage2"},
        3: {
            "features_df", "targets_df", "feature_data", "feature_engineering",
            "news_data", "stage_3", "stage3",
        },
    }
    present = {key for key in groups[stage] if key in result}
    if _explicit_stage_payload(stage, result) is not None:
        present.add(f"stage_{stage}")
    return present


def _extract_news_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Any] = []
    for key in ("news_data", "news", "news_items", "processed_news"):
        if key in result:
            candidates.append(result[key])
    raw_data = result.get("raw_data")
    if isinstance(raw_data, dict):
        for key, value in raw_data.items():
            if "news" in str(key).lower() or "rss" in str(key).lower():
                candidates.append(value)
    processed = result.get("processed_data")
    if isinstance(processed, dict):
        for key, value in processed.items():
            if "news" in str(key).lower():
                candidates.append(value)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        for item in _to_records(candidate, limit=500):
            normalized = _normalize_news_item(item)
            identity = sha256_json(normalized)
            if identity in seen:
                continue
            seen.add(identity)
            records.append(normalized)
    return records


def _normalize_news_item(item: dict[str, Any]) -> dict[str, Any]:
    payload = {str(key): value for key, value in item.items()}
    payload.setdefault("source_type", "news")
    payload.setdefault("source", payload.get("source_name") or payload.get("publisher") or "pipeline_stage03")
    payload.setdefault("title", payload.get("headline") or payload.get("name") or "Pipeline news item")
    payload.setdefault("text", payload.get("content") or payload.get("summary") or payload.get("description") or "")
    timestamp = (
        payload.get("available_at")
        or payload.get("published_at")
        or payload.get("timestamp")
        or payload.get("date")
    )
    if timestamp is not None:
        payload.setdefault("available_at", str(timestamp))
        payload.setdefault("published_at", str(timestamp))
    payload.setdefault("external_artifact_ref", "pipeline_stage03")
    return payload


def _to_records(value: Any, *, limit: int) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [dict(item) for item in value[:limit] if isinstance(item, dict)]
    if isinstance(value, tuple):
        return [dict(item) for item in list(value)[:limit] if isinstance(item, dict)]
    if isinstance(value, dict):
        if value and all(isinstance(item, dict) for item in value.values()):
            return [dict(item) for item in list(value.values())[:limit]]
        return [dict(value)]
    if hasattr(value, "to_dict"):
        try:
            records = value.to_dict(orient="records")
            return [dict(item) for item in records[:limit] if isinstance(item, dict)]
        except TypeError:
            try:
                converted = value.to_dict()
                if isinstance(converted, dict):
                    return [converted]
            except Exception:
                return []
        except Exception:
            return []
    return []


def _extract_mapping(result: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    for key in keys:
        value = result.get(key)
        if isinstance(value, dict):
            return dict(value)
        records = _to_records(value, limit=200)
        if records:
            return {"records": records, "source_key": key}
    return {}


def _references_from_result(result: dict[str, Any]) -> list[PipelineArtifactReference]:
    references: list[PipelineArtifactReference] = []
    for stage in range(4):
        for key in sorted(_keys_for_stage(stage, result)):
            value = result.get(key)
            locator = None
            if isinstance(value, (str, Path)):
                locator = str(value)
            elif isinstance(value, dict):
                candidate = value.get("path") or value.get("artifact_path") or value.get("output_path")
                locator = str(candidate) if candidate else None
            if not locator:
                continue
            suffix = Path(locator).suffix.lower().lstrip(".") or "unknown"
            references.append(
                PipelineArtifactReference(
                    artifact_id=f"stage{stage}_{key}_{sha256_json(locator)[:12]}",
                    stage=stage,
                    artifact_type=key,
                    locator=locator,
                    format=suffix,
                    row_count=_safe_len(value),
                )
            )
    return references


def _is_dataframe_like(value: Any) -> bool:
    return hasattr(value, "columns") and hasattr(value, "to_dict")


def _safe_len(value: Any) -> int | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        return max(0, int(len(value)))
    except Exception:
        return None


def _packet_status(
    payload: dict[str, Any],
    stages_present: list[int],
    states: list[PipelineStageState],
) -> Literal["available", "partial", "missing", "failed"]:
    explicit = str(payload.get("status") or "").lower()
    if explicit in {"failed", "error"}:
        return "failed"
    if not stages_present:
        return "missing"
    if any(item.status == "failed" for item in states):
        return "failed"
    if explicit in {"partial", "pipeline_skipped"} or any(
        item.status in {"partial", "missing"} for item in states if item.stage in stages_present
    ):
        return "partial"
    return "available"


def _result_identity_payload(result: dict[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    for key, value in result.items():
        if _is_dataframe_like(value):
            identity[key] = {
                "rows": _safe_len(value),
                "columns": [str(item) for item in list(getattr(value, "columns", []))],
            }
        elif isinstance(value, dict):
            identity[key] = _result_identity_payload(value)
        elif isinstance(value, list):
            identity[key] = {
                "length": len(value),
                "sample_hash": sha256_json(value[:5]),
            }
        else:
            identity[key] = value
    return identity


__all__ = [
    "PipelineArtifactReference",
    "PipelineStage03Bridge",
    "PipelineStage03Packet",
    "PipelineStageState",
]
