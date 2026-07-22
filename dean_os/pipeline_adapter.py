from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from dean_os.context_evidence_provenance import audit_news_records
from dean_os.packets.pipeline_prediction_review_packet import (
    PipelinePredictionReviewPacket,
)
from dean_os.schemas import MarketContext
from dean_os.structured_context_provenance import (
    apply_market_context_structured_boundary,
)

HybridMode = Literal["local", "light", "prepare", "full"]


class HybridPipelineAdapter:
    """Adapter that lets DEAN-OS call the existing HybridOrchestrator.

    The adapter imports the project pipeline lazily so tests and lightweight
    agent flows can import DEAN-OS without booting the full stack.
    """

    def __init__(
        self,
        mode: HybridMode = "local",
        batch_name: str = "main_database",
        project_root: str | Path = ".",
        orchestrator: Any | None = None,
        orchestrator_factory: Callable[[], Any] | None = None,
        stages_to_run: list[int] | None = None,
        prepare_kwargs: dict[str, Any] | None = None,
    ):
        self.mode = mode
        self.batch_name = batch_name
        self.project_root = Path(project_root).resolve()
        self._orchestrator = orchestrator
        self._orchestrator_factory = orchestrator_factory
        self.stages_to_run = stages_to_run
        self.prepare_kwargs = prepare_kwargs or {}
        # Set True when the src/ pipeline stack cannot be imported (missing
        # optional deps, e.g. google-cloud). Lets DEAN-OS degrade to a no-op
        # pipeline result instead of crashing the whole orchestrator.
        self._src_unavailable = False
        self._src_unavailable_reason: str | None = None

    async def __call__(self, context: MarketContext) -> dict[str, Any]:
        tickers = self._resolve_tickers(context)
        timeframes = self._resolve_timeframes(context)

        # If the src/ pipeline could not be imported, degrade to a no-op so
        # the agent branches (review/analysis) still run and produce a decision.
        if self._src_unavailable:
            normalized = {
                "status": "pipeline_skipped",
                "tickers": tickers,
                "timeframes": timeframes,
                "timeframe": timeframes[0] if timeframes else None,
                "adapter_mode": self.mode,
                "pipeline_skipped": True,
                "skip_reason": self._src_unavailable_reason,
            }
            review_contract = self._build_review_contract(normalized)
            normalized["dean_os_review_contract"] = review_contract
            context.as_of = (
                context.as_of
                or self._resolve_context_as_of(normalized)
            )
            self._apply_news_point_in_time_boundary(context)
            self._apply_structured_context_boundary(context)
            return normalized

        orchestrator = self._get_orchestrator()

        if self.mode == "local":
            result = await orchestrator.run_local_pipeline(
                tickers=tickers,
                timeframes=timeframes,
                stages_to_run=self.stages_to_run,
            )
        elif self.mode == "light":
            result = await orchestrator.run_light_models(
                tickers=tickers,
                timeframes=timeframes,
            )
        elif self.mode == "prepare":
            result = await self._execute_prepare(orchestrator, tickers, timeframes)
        elif self.mode == "full":
            result = await self._execute_full(orchestrator, tickers, timeframes)
        else:
            raise ValueError(f"Unsupported hybrid adapter mode: {self.mode}")

        normalized = self._normalize_result(result, tickers, timeframes)
        review_contract = self._build_review_contract(normalized)
        normalized["dean_os_review_contract"] = review_contract
        self._enrich_context(context, normalized)
        return normalized

    def _get_orchestrator(self):
        if self._orchestrator is not None:
            return self._orchestrator
        if self._orchestrator_factory is not None:
            self._orchestrator = self._orchestrator_factory()
            return self._orchestrator

        try:
            from src.config.unified_config_manager import UnifiedConfigManager
            from src.pipeline.hybrid_orchestrator import HybridOrchestrator
        except ImportError as exc:
            # Optional dependency missing (e.g. google-cloud, torch). Mark the
            # adapter as unavailable so __call__ returns a no-op instead of
            # raising during orchestrator startup.
            self._src_unavailable = True
            self._src_unavailable_reason = f"src pipeline import failed: {exc}"
            raise

        config_manager = UnifiedConfigManager()
        self._orchestrator = HybridOrchestrator(config_manager, batch_name=self.batch_name)
        return self._orchestrator

    async def _execute_prepare(self, orchestrator, tickers: list[str], timeframes: list[str]) -> dict[str, Any]:
        from src.cli.pipeline_executor import PipelineExecutor

        return await PipelineExecutor.execute_prepare_mode(
            orchestrator,
            tickers,
            timeframes,
            **self.prepare_kwargs,
        )

    async def _execute_full(self, orchestrator, tickers: list[str], timeframes: list[str]) -> dict[str, Any]:
        from src.cli.pipeline_executor import PipelineExecutor

        return await PipelineExecutor.execute_full_mode(orchestrator, tickers, timeframes)

    def _resolve_tickers(self, context: MarketContext) -> list[str]:
        return list(context.tickers or [])

    def _resolve_timeframes(self, context: MarketContext) -> list[str]:
        if context.timeframes:
            return list(context.timeframes)
        if context.timeframe:
            return [context.timeframe]
        return ["1d"]

    def _normalize_result(self, result: Any, tickers: list[str], timeframes: list[str]) -> dict[str, Any]:
        if isinstance(result, dict):
            normalized = dict(result)
        else:
            normalized = {"status": "unknown", "raw_result": result}
        normalized.setdefault("tickers", tickers)
        normalized.setdefault("timeframes", timeframes)
        normalized.setdefault("timeframe", timeframes[0] if timeframes else None)
        normalized.setdefault("adapter_mode", self.mode)
        return normalized

    def _enrich_context(self, context: MarketContext, result: dict[str, Any]) -> None:
        payload = result.get("results", result)
        self._capture_dataframe(context, payload, "features_df", "features")
        self._capture_dataframe(context, payload, "targets_df", "targets")
        self._capture_dataframe(context, payload, "market_data", "market")
        self._capture_dataframe(context, payload, "news_data", "news")
        self._capture_dataframe(context, payload, "economic_data", "macro")
        self._capture_dataframe(context, payload, "macro_data", "macro")

        news_frame = context.dataframes.get("news")
        if news_frame is not None and not context.news:
            context.news = self._records_from_dataframe(news_frame, limit=200)
        context.as_of = context.as_of or self._resolve_context_as_of(
            result
        )
        self._apply_news_point_in_time_boundary(context)

        macro_frame = context.dataframes.get("macro")
        if macro_frame is not None:
            context.metadata["raw_macro_frame_inventory"] = {
                "rows": self._safe_len(macro_frame),
                "columns": self._columns(macro_frame),
                "evidence_status": (
                    "raw_pipeline_table_not_structured_context_evidence"
                ),
            }
        self._apply_structured_context_boundary(context)

        if context.returns is None:
            returns_source = context.dataframes.get("market")
            if returns_source is None:
                returns_source = context.dataframes.get("features")
            returns_series, returns_metadata = self._extract_returns(returns_source)
            context.returns = returns_series

            # Add warnings if target_return_* was used (offline-only data)
            if returns_metadata.get("offline_only", False):
                warnings = context.metadata.setdefault("context_enrichment_warnings", [])
                warnings.append(
                    f"Returns source is {returns_metadata['column_used']} (target_label), "
                    "which is supervised/offline-only. Not suitable for live/paper trading."
                )
                context.metadata["returns_source"] = returns_metadata["returns_source"]
                context.metadata["returns_offline_only"] = True

        context.metadata["pipeline_review_contract"] = result.get(
            "dean_os_review_contract", {}
        )
        context.metadata["stage7_regime_review"] = (
            context.metadata["pipeline_review_contract"].get(
                "stage7_regime_review",
                {},
            )
        )
        context.metadata["stage5_prediction_review"] = (
            context.metadata["pipeline_review_contract"].get(
                "stage5_prediction_review",
                {},
            )
        )
        context.pipeline_result.update(result)

    def _apply_news_point_in_time_boundary(
        self,
        context: MarketContext,
    ) -> None:
        if not context.news:
            context.metadata["news_point_in_time_audit"] = {
                "contract": "dean_context_evidence_point_in_time_v1",
                "status": "no_news_records",
                "as_of": context.as_of,
                "input_count": 0,
                "accepted_count": 0,
                "excluded_count": 0,
            }
            return
        if not context.as_of:
            input_count = len(context.news)
            context.news = []
            context.metadata["news_point_in_time_audit"] = {
                "contract": "dean_context_evidence_point_in_time_v1",
                "status": "blocked_context_as_of_missing",
                "as_of": None,
                "input_count": input_count,
                "accepted_count": 0,
                "excluded_count": input_count,
                "reason_counts": {
                    "context_as_of_missing": input_count
                },
            }
            return
        try:
            audit = audit_news_records(
                list(context.news),
                as_of=context.as_of,
                requested_tickers=context.tickers,
            )
        except ValueError:
            input_count = len(context.news)
            context.news = []
            context.metadata["news_point_in_time_audit"] = {
                "contract": "dean_context_evidence_point_in_time_v1",
                "status": "blocked_context_as_of_invalid",
                "as_of": context.as_of,
                "input_count": input_count,
                "accepted_count": 0,
                "excluded_count": input_count,
                "reason_counts": {
                    "context_as_of_invalid": input_count
                },
            }
            return
        context.news = list(audit["accepted"])
        context.metadata["news_point_in_time_audit"] = {
            key: value
            for key, value in audit.items()
            if key != "accepted"
        }

    def _apply_structured_context_boundary(
        self,
        context: MarketContext,
    ) -> None:
        apply_market_context_structured_boundary(context)

    def _resolve_context_as_of(
        self,
        result: dict[str, Any],
    ) -> str | None:
        payload = result.get("results")
        candidates = [
            result.get("as_of"),
            result.get("timestamp"),
            payload.get("as_of") if isinstance(payload, dict) else None,
            payload.get("timestamp")
            if isinstance(payload, dict)
            else None,
        ]
        for candidate in candidates:
            if candidate is not None and str(candidate).strip():
                return str(candidate)
        return None

    def _capture_dataframe(self, context: MarketContext, payload: dict[str, Any], source_key: str, target_key: str) -> None:
        value = payload.get(source_key)
        if value is not None:
            context.dataframes[target_key] = value

    def _records_from_dataframe(self, frame: Any, limit: int) -> list[dict[str, Any]]:
        if not hasattr(frame, "head") or not hasattr(frame, "to_dict"):
            return []
        try:
            return frame.head(limit).to_dict(orient="records")
        except Exception:
            return []

    def _extract_returns(self, frame: Any) -> tuple[Any, dict[str, Any]]:
        """Extract returns from dataframe with source tracking.

        Returns:
            Tuple of (returns_series, metadata_dict)
            metadata includes:
            - returns_source: "realized_return" or "target_label"
            - offline_only: True if target_return_* was used
            - column_used: the actual column name used
        """
        if frame is None or not hasattr(frame, "columns"):
            return None, {"returns_source": "none", "offline_only": False, "column_used": None}

        try:
            columns = {str(column).lower(): column for column in frame.columns}

            # Realized returns always take precedence over supervised labels.
            for name in ("return", "returns", "close_return", "pct_change"):
                if name in columns:
                    return frame[columns[name]], {
                        "returns_source": "realized_return",
                        "offline_only": False,
                        "column_used": columns[name]
                    }

            # Fallback to close price calculation
            if "close" in columns:
                return frame[columns["close"]].pct_change().dropna(), {
                    "returns_source": "realized_return",
                    "offline_only": False,
                    "column_used": columns["close"]
                }

            # Target returns are accepted only as explicitly offline evidence.
            for name in ("target_return_1d", "target_return_5d", "target_return_10d"):
                if name in columns:
                    return frame[columns[name]], {
                        "returns_source": "target_label",
                        "offline_only": True,
                        "column_used": columns[name]
                    }

        except Exception:
            return None, {"returns_source": "none", "offline_only": False, "column_used": None}

        return None, {"returns_source": "none", "offline_only": False, "column_used": None}

    def _build_review_contract(self, result: dict[str, Any]) -> dict[str, Any]:
        stage4_manifests = self._as_string_list(
            self._find_nested(
                result,
                "pipeline_control_metric_artifact_manifests",
            )
        )
        stage7_artifacts = self._find_nested(
            result,
            "pipeline_control_evaluation_metric_artifacts",
        )
        execution_boundary = self._find_nested(result, "execution_boundary")
        learning_candidate = self._find_nested(
            result,
            "learning_review_candidate",
        )
        analyzer_coverage = self._find_nested(result, "_analysis_coverage")
        stage7_analysis_contract = self._find_nested(
            result,
            "_stage7_analysis_contract",
        )
        return {
            "schema_version": "dean_pipeline_review_contract_v1",
            "adapter_mode": self.mode,
            "pipeline_status": result.get("status", "unknown"),
            "stage4_metric_artifact_manifests": stage4_manifests,
            "stage7_metric_artifacts": (
                stage7_artifacts if isinstance(stage7_artifacts, dict) else {}
            ),
            "execution_status": (
                self._find_nested(result, "execution_status")
                or "not_reported"
            ),
            "execution_boundary": (
                execution_boundary
                if isinstance(execution_boundary, dict)
                else {}
            ),
            "learning_review_status": (
                learning_candidate.get("status")
                if isinstance(learning_candidate, dict)
                else "not_reported"
            ),
            "stage5_prediction_review": (
                PipelinePredictionReviewPacket().build(
                    result,
                    requested_tickers=self._as_string_list(
                        result.get("tickers")
                    ),
                    requested_timeframes=self._as_string_list(
                        result.get("timeframes")
                        or result.get("timeframe")
                    ),
                    save=False,
                )
            ),
            "stage7_analyzer_review": self._build_analyzer_review(
                analyzer_coverage,
                stage7_analysis_contract,
            ),
            "stage7_regime_review": self._build_stage7_regime_review(
                result,
                analyzer_coverage,
                stage7_analysis_contract,
            ),
            "evidence_inventory_required": True,
            "can_write_learning_memory": False,
            "can_write_production_config": False,
            "can_trade": False,
        }

    def _build_stage7_regime_review(
        self,
        result: dict[str, Any],
        coverage: Any,
        stage7_contract: Any,
    ) -> dict[str, Any]:
        coverage = coverage if isinstance(coverage, dict) else {}
        stage7_contract = (
            stage7_contract
            if isinstance(stage7_contract, dict)
            else {}
        )
        executed = self._as_string_list(
            coverage.get(
                "executed_analyzers",
                coverage.get("executed"),
            )
        )
        requested_tickers = self._as_string_list(result.get("tickers"))
        requested_timeframes = self._as_string_list(
            result.get("timeframes")
        )
        if not requested_timeframes and result.get("timeframe"):
            requested_timeframes = [str(result["timeframe"])]
        contexts = []
        analysis_by_context = self._find_nested(
            result,
            "analysis_by_context",
        )
        if isinstance(analysis_by_context, dict):
            for context_key, context_result in sorted(
                analysis_by_context.items()
            ):
                if not isinstance(context_result, dict):
                    continue
                normalized = self._normalize_regime_result(
                    context_key=str(context_key),
                    result=context_result.get("market_regime"),
                    context_window=context_result.get(
                        "_stage7_context_window"
                    ),
                    requested_tickers=requested_tickers,
                    requested_timeframes=requested_timeframes,
                )
                if normalized:
                    contexts.append(normalized)
        else:
            direct_result = self._find_nested(
                result,
                "market_regime",
            )
            normalized = self._normalize_regime_result(
                context_key="all_prices",
                result=direct_result,
                context_window=self._find_nested(
                    result,
                    "_stage7_context_window",
                ),
                requested_tickers=requested_tickers,
                requested_timeframes=requested_timeframes,
            )
            if normalized:
                contexts.append(normalized)

        contract_hashes = self._stage7_analysis_contract_hashes(
            coverage
        )
        if "market_regime" not in executed:
            status = "stage7_market_regime_not_executed"
        elif contexts:
            status = "stage7_regime_contexts_recorded"
        else:
            status = "stage7_market_regime_output_unavailable"
        return {
            "schema_version": "dean_stage7_regime_review_v1",
            "status": status,
            "context_count": len(contexts),
            "contexts": contexts,
            "context_partitioned": bool(
                stage7_contract.get(
                    "price_context_partitioned",
                    False,
                )
            ),
            "price_data_source": stage7_contract.get(
                "price_data_source"
            ),
            "analysis_contract_hash": (
                contract_hashes[0]
                if len(contract_hashes) == 1
                else None
            ),
            "analysis_contract_hashes": contract_hashes,
            "source_analyzer": "market_regime",
            "evidence_class": (
                "supporting_analysis_not_locked_evidence"
            ),
            "can_clear_locked_evidence": False,
            "can_promote_model": False,
            "can_create_recommendation": False,
            "can_trade": False,
        }

    def _normalize_regime_result(
        self,
        *,
        context_key: str,
        result: Any,
        context_window: Any,
        requested_tickers: list[str],
        requested_timeframes: list[str],
    ) -> dict[str, Any] | None:
        if not isinstance(result, dict):
            return None
        analysis_status = str(result.get("status") or "completed")
        if analysis_status != "completed":
            return None
        regime = str(result.get("regime") or "UNKNOWN").upper()
        if regime == "UNKNOWN":
            return None
        try:
            confidence = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        identity = self._parse_context_key(context_key)
        inferred_fields: list[str] = []
        if (
            not identity.get("ticker")
            and len(requested_tickers) == 1
        ):
            identity["ticker"] = requested_tickers[0].upper()
            inferred_fields.append("ticker")
        if (
            not identity.get("timeframe")
            and len(requested_timeframes) == 1
        ):
            identity["timeframe"] = requested_timeframes[0].lower()
            inferred_fields.append("timeframe")
        if identity.get("ticker") and identity.get("timeframe"):
            identity_status = (
                "inferred_from_single_requested_context"
                if inferred_fields
                else "exact_context_key"
            )
        else:
            identity_status = "ambiguous_context_identity"
        metrics = {
            key: value
            for key, value in result.items()
            if key
            not in {
                "status",
                "regime",
                "confidence",
                "supporting_review_only",
            }
        }
        context_window = (
            context_window
            if isinstance(context_window, dict)
            else {}
        )
        return {
            "context_key": context_key,
            "ticker": identity.get("ticker"),
            "timeframe": identity.get("timeframe"),
            "identity_status": identity_status,
            "identity_inferred_fields": inferred_fields,
            "regime": regime,
            "confidence": max(0.0, min(confidence, 1.0)),
            "metrics": metrics,
            "as_of": (
                result.get("as_of")
                or result.get("evaluated_at")
                or result.get("timestamp")
                or context_window.get("end")
            ),
            "price_window": context_window,
            "analysis_status": analysis_status,
            "supporting_review_only": True,
            "decision_influence": False,
            "can_promote_model": False,
            "can_trade": False,
        }

    def _parse_context_key(
        self,
        context_key: str,
    ) -> dict[str, str]:
        identity: dict[str, str] = {}
        for component in context_key.split("|"):
            key, separator, value = component.partition("=")
            if not separator or not value:
                continue
            normalized_key = key.strip().lower()
            if normalized_key in {"ticker", "symbol"}:
                identity["ticker"] = value.strip().upper()
            elif normalized_key in {"interval", "timeframe"}:
                identity["timeframe"] = value.strip().lower()
        return identity

    def _stage7_analysis_contract_hashes(
        self,
        coverage: dict[str, Any],
    ) -> list[str]:
        values: set[str] = set()
        direct = coverage.get("analysis_contract_hash")
        if direct:
            values.add(str(direct))
        context_coverage = coverage.get("context_coverage")
        if isinstance(context_coverage, dict):
            for item in context_coverage.values():
                if not isinstance(item, dict):
                    continue
                value = item.get("analysis_contract_hash")
                if value:
                    values.add(str(value))
        return sorted(values)

    def _build_analyzer_review(
        self,
        coverage: Any,
        stage7_contract: Any,
    ) -> dict[str, Any]:
        coverage = coverage if isinstance(coverage, dict) else {}
        stage7_contract = (
            stage7_contract if isinstance(stage7_contract, dict) else {}
        )
        return {
            "schema_version": "dean_stage7_analyzer_review_v1",
            "status": coverage.get("status", "not_reported"),
            "context_count": coverage.get(
                "context_count",
                stage7_contract.get("price_context_count", 0),
            ),
            "context_partitioned": bool(
                stage7_contract.get("price_context_partitioned", False)
            ),
            "price_data_source": stage7_contract.get("price_data_source"),
            "executed_analyzers": self._as_string_list(
                coverage.get("executed_analyzers", coverage.get("executed"))
            ),
            "failed_analyzers": self._as_string_list(
                coverage.get("failed_analyzers", coverage.get("failed"))
            ),
            "disabled_analyzers": self._as_string_list(
                coverage.get("disabled_analyzers", coverage.get("disabled"))
            ),
            "evidence_class": coverage.get(
                "evidence_class",
                "supporting_analysis_not_locked_evidence",
            ),
            "evidence_inventory_required": True,
            "can_clear_locked_evidence": False,
            "can_promote_model": False,
            "can_trade": False,
        }

    def _find_nested(
        self,
        value: Any,
        key: str,
        depth: int = 0,
    ) -> Any:
        if depth > 5 or not isinstance(value, dict):
            return None
        if key in value:
            return value[key]
        for child in value.values():
            found = self._find_nested(child, key, depth + 1)
            if found is not None:
                return found
        return None

    def _as_string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        values = value if isinstance(value, (list, tuple, set)) else [value]
        return [str(item) for item in values if item]

    def _safe_len(self, value: Any) -> int:
        try:
            return int(len(value))
        except Exception:
            return 0

    def _columns(self, frame: Any) -> list[str]:
        return [str(column) for column in getattr(frame, "columns", [])]
