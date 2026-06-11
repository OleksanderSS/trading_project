from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from dean_os.schemas import MarketContext

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

    async def __call__(self, context: MarketContext) -> dict[str, Any]:
        orchestrator = self._get_orchestrator()
        tickers = self._resolve_tickers(context)
        timeframes = self._resolve_timeframes(context)

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
        self._enrich_context(context, normalized)
        return normalized

    def _get_orchestrator(self):
        if self._orchestrator is not None:
            return self._orchestrator
        if self._orchestrator_factory is not None:
            self._orchestrator = self._orchestrator_factory()
            return self._orchestrator

        from src.config.unified_config_manager import UnifiedConfigManager
        from src.pipeline.hybrid_orchestrator import HybridOrchestrator

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

        macro_frame = context.dataframes.get("macro")
        if macro_frame is not None and not context.macro:
            context.macro = {"rows": self._safe_len(macro_frame), "columns": self._columns(macro_frame)}

        if context.returns is None:
            returns_source = context.dataframes.get("market")
            if returns_source is None:
                returns_source = context.dataframes.get("features")
            context.returns = self._extract_returns(returns_source)

        context.pipeline_result.update(result)

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

    def _extract_returns(self, frame: Any):
        if frame is None or not hasattr(frame, "columns"):
            return None
        try:
            columns = {str(column).lower(): column for column in frame.columns}
            for name in ("return", "returns", "target_return_1d", "close_return", "pct_change"):
                if name in columns:
                    return frame[columns[name]]
            if "close" in columns:
                return frame[columns["close"]].pct_change().dropna()
        except Exception:
            return None
        return None

    def _safe_len(self, value: Any) -> int:
        try:
            return int(len(value))
        except Exception:
            return 0

    def _columns(self, frame: Any) -> list[str]:
        return [str(column) for column in getattr(frame, "columns", [])]
