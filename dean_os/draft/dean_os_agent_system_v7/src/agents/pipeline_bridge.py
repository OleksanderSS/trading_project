"""
Pipeline Bridge: Stage 7 → DEAN-OS MarketContext
=================================================

Reads the output of Stage 7 (EvaluationStage) and translates it into
a ``MarketContext`` that DEAN-OS agents (ModelPerformanceAgent, TuningAgent,
UnifiedResearchAgent, etc.) can consume directly.

Usage
-----
    from src.agents.pipeline_bridge import PipelineBridge

    bridge = PipelineBridge()

    # Option A – from an already-running pipeline result dict
    context = bridge.from_pipeline_result(stage7_result, tickers=["AAPL"])

    # Option B – from a saved evaluation_summary.json on disk
    context = bridge.from_evaluation_file("data/results/evaluation_summary.json")

    # Then hand the context to DEANOrchestrator or individual agents
    from dean_os.agents import ModelPerformanceAgent, TuningAgent
    report = await ModelPerformanceAgent(config={"min_validation_score": 0.55}).run(context)
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_metrics import PipelineMetricNormalizer
from dean_os.schemas import MarketContext

logger = logging.getLogger(__name__)


# Keys Stage 7 writes into evaluation_summary.json / its result dict
_METRIC_KEY_MAP: dict[str, str] = {
    # Stage 7 key              → model_performance sub-key
    "validation_score":        "validation_score",
    "val_score":               "validation_score",
    "sharpe_ratio":            "sharpe",
    "sharpe":                  "sharpe",
    "max_drawdown":            "max_drawdown",
    "win_rate":                "win_rate",
    "total_return":            "total_return",
    "total_trades":            "total_trades",
    "sample_count":            "sample_count",
    "n_samples":               "sample_count",
    "evaluated_at":            "evaluated_at",
    "created_at":              "evaluated_at",
    "timestamp":               "evaluated_at",
}

_FRESHNESS_THRESHOLD_HOURS = 24.0


class PipelineBridge:
    """
    Translates Stage 7 pipeline artefacts into ``dean_os.schemas.MarketContext``.

    No write authority: read-only, review-only adapter.
    """

    def __init__(
        self,
        *,
        evaluation_summary_path: str | Path | None = None,
        project_root: str | Path = ".",
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self._default_eval_path: Path | None = (
            Path(evaluation_summary_path).resolve()
            if evaluation_summary_path
            else self.project_root / "data" / "results" / "evaluation_summary.json"
        )
        self._metric_normalizer = PipelineMetricNormalizer(self.project_root)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def from_pipeline_result(
        self,
        stage7_result: dict[str, Any],
        *,
        tickers: list[str] | None = None,
        timeframes: list[str] | None = None,
        as_of: str | None = None,
    ) -> MarketContext:
        """
        Build a MarketContext from the dict returned by EvaluationStage.run().

        Parameters
        ----------
        stage7_result:
            The raw dict from Stage 7 (keys: 'financial_metrics', 'signals',
            'model_results', 'training_metrics', 'validation_metrics', …).
        tickers:
            Optional override; if absent the bridge tries to infer from signals.
        timeframes:
            Optional override.
        as_of:
            ISO-8601 timestamp; defaults to now.
        """
        as_of = as_of or _utc_now_iso()
        tickers = tickers or _infer_tickers(stage7_result)

        model_performance = self._extract_model_performance(stage7_result)
        data_freshness    = self._build_data_freshness(stage7_result)
        pipeline_result   = self._build_pipeline_result(stage7_result)
        regime_context    = self._extract_regime(stage7_result)
        metric_snapshot = self._metric_normalizer.from_pipeline_result(
            stage7_result,
            as_of=as_of,
            tickers=tickers,
            timeframes=timeframes or [],
        ).model_dump(mode="json")
        pipeline_result["dean_os_pipeline_metric_snapshot"] = metric_snapshot

        context = MarketContext(
            phase="post_pipeline",
            as_of=as_of,
            tickers=tickers,
            timeframes=timeframes or [],
            pipeline_result=pipeline_result,
            metadata={
                "model_performance":       model_performance,
                "data_freshness":          data_freshness,
                "regime_context":          regime_context,
                "pipeline_metric_snapshot": metric_snapshot,
                "pipeline_review_contract": {
                    "stage7_analyzer_review": stage7_result.get("analyzer_review", {}),
                },
                "source": "pipeline_bridge.from_pipeline_result",
            },
        )
        logger.info(
            "PipelineBridge: context built for tickers=%s  verdict_hint=%s",
            tickers,
            model_performance.get("verdict", "?"),
        )
        return context

    def from_evaluation_file(
        self,
        path: str | Path | None = None,
        *,
        tickers: list[str] | None = None,
        timeframes: list[str] | None = None,
        as_of: str | None = None,
    ) -> MarketContext:
        """
        Build a MarketContext by loading a saved evaluation_summary.json.

        Falls back to ``self._default_eval_path`` when ``path`` is None.
        """
        resolved = Path(path).resolve() if path else self._default_eval_path
        if not resolved or not resolved.exists():
            logger.warning(
                "PipelineBridge: evaluation file not found at %s — returning empty context",
                resolved,
            )
            return self._empty_context(as_of=as_of or _utc_now_iso(), tickers=tickers or [])

        try:
            raw = json.loads(resolved.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("PipelineBridge: failed to read %s: %s", resolved, exc)
            return self._empty_context(as_of=as_of or _utc_now_iso(), tickers=tickers or [])

        # Inject file-level metadata before building
        raw.setdefault("_source_file", str(resolved))
        raw.setdefault("_file_mtime", _file_mtime_iso(resolved))
        context = self.from_pipeline_result(raw, tickers=tickers, timeframes=timeframes, as_of=as_of)
        logger.info("PipelineBridge: loaded evaluation from %s", resolved)
        return context

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _extract_model_performance(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Extract and normalise model performance metrics that ModelPerformanceAgent
        and TuningAgent expect under ``context.metadata["model_performance"]``.
        """
        # Flatten from nested dicts that Stage 7 may produce
        flat: dict[str, Any] = {}
        for src in (
            result.get("financial_metrics") or {},
            result.get("training_metrics") or {},
            result.get("validation_metrics") or {},
            result,  # top-level fallback
        ):
            if isinstance(src, dict):
                for raw_key, value in src.items():
                    mapped = _METRIC_KEY_MAP.get(raw_key)
                    if mapped and mapped not in flat:
                        flat[mapped] = value

        # Threshold failures (consumed by TuningAgent)
        failures: list[str] = []
        val_score = _safe_float(flat.get("validation_score"))
        sharpe    = _safe_float(flat.get("sharpe"))
        drawdown  = _safe_float(flat.get("max_drawdown"))
        n_samples = _safe_int(flat.get("sample_count"))

        if val_score is not None and val_score < 0.55:
            failures.append("validation_score_below_threshold")
        if sharpe is not None and sharpe < 0.0:
            failures.append("sharpe_below_threshold")
        if drawdown is not None and drawdown > 0.25:
            failures.append("drawdown_above_threshold")
        if n_samples is not None and n_samples < 50:
            failures.append("sample_count_below_threshold")

        # Staleness check
        eval_ts = flat.get("evaluated_at") or result.get("_file_mtime")
        if eval_ts:
            age_h = _age_hours(eval_ts)
            if age_h > _FRESHNESS_THRESHOLD_HOURS:
                failures.append("evaluation_artifact_stale")
            flat["evaluation_age_hours"] = round(age_h, 2)

        verdict = "caution" if failures else "clear"
        flat.update({
            "threshold_failures": failures,
            "verdict": verdict,
            "performance_score": val_score,
            # Scope fields TuningAgent uses for exact lineage
            "evaluation_scope": {
                "ticker":              result.get("ticker") or (result.get("tickers") or [None])[0],
                "model":               result.get("model_name") or result.get("best_model"),
                "target_name":         result.get("target_name"),
                "timeframe":           result.get("timeframe"),
                "context_fingerprint": result.get("context_fingerprint"),
            },
        })
        return flat

    def _build_data_freshness(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Build the ``data_freshness`` dict that DataQualityAgent and TuningAgent
        check for stale sources.
        """
        sources: dict[str, dict[str, Any]] = {}
        signal_count = len(result.get("signals") or [])
        sources["stage7_signals"] = {
            "stale": signal_count == 0,
            "row_count": signal_count,
        }
        # If Stage 7 passed through news/macro counts, track them
        for key in ("news_data", "macro_data", "price_data"):
            val = result.get(key)
            if val is not None:
                count = len(val) if hasattr(val, "__len__") else 0
                sources[key] = {"stale": count == 0, "row_count": count}
        return sources

    def _build_pipeline_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Serialize Stage 7's result safely (no DataFrames) for pipeline_result."""
        safe: dict[str, Any] = {}
        for key, value in result.items():
            if key.startswith("_"):
                continue
            if hasattr(value, "to_dict"):       # pandas DataFrame/Series
                try:
                    safe[key] = value.head(10).to_dict(orient="records")
                except Exception:
                    safe[key] = f"<DataFrame: {getattr(value, 'shape', '?')}>"
            elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                safe[key] = value
            else:
                safe[key] = str(value)
        return safe

    def _extract_regime(self, result: dict[str, Any]) -> dict[str, Any]:
        """Try to extract a regime hint from Stage 7 analysis results."""
        analysis = result.get("analysis_results") or {}
        if isinstance(analysis, dict):
            regime = analysis.get("regime") or analysis.get("market_regime")
            if regime:
                return {"regime": str(regime), "source": "stage7_analysis"}
        return {"regime": "UNKNOWN", "source": "pipeline_bridge_default"}

    @staticmethod
    def _empty_context(*, as_of: str, tickers: list[str]) -> MarketContext:
        return MarketContext(
            phase="post_pipeline",
            as_of=as_of,
            tickers=tickers,
            metadata={
                "model_performance": {
                    "threshold_failures": ["missing_evaluation_metrics"],
                    "verdict": "caution",
                },
                "pipeline_metric_snapshot": PipelineMetricNormalizer().from_pipeline_result(
                    {"status": "pipeline_skipped", "pipeline_skipped": True},
                    as_of=as_of,
                    tickers=tickers,
                ).model_dump(mode="json"),
                "source": "pipeline_bridge.empty_fallback",
            },
        )


# ------------------------------------------------------------------ #
# Module-level helpers
# ------------------------------------------------------------------ #

def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _file_mtime_iso(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
    except OSError:
        return _utc_now_iso()


def _age_hours(iso_str: str) -> float:
    try:
        dt = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return (datetime.now(UTC) - dt).total_seconds() / 3600
    except (TypeError, ValueError):
        return 0.0


def _infer_tickers(result: dict[str, Any]) -> list[str]:
    for key in ("tickers", "ticker"):
        val = result.get(key)
        if isinstance(val, list):
            return [str(t).upper() for t in val if t]
        if isinstance(val, str) and val:
            return [val.upper()]
    signals = result.get("signals") or []
    seen: list[str] = []
    for sig in signals[:50]:
        t = str(sig.get("ticker") or "").upper()
        if t and t not in seen:
            seen.append(t)
    return seen


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
