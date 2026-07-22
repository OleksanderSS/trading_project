from __future__ import annotations

from pathlib import Path
from typing import Any

from dean_os.base import BaseAgent
from dean_os.regime_context import RegimeContextBuilder, normalize_context_tags
from dean_os.schemas import MarketContext, MarketRegimeSnapshot, PipelineReport


class RegimeAgent(BaseAgent):
    """Turns local market regime context into a soft pipeline report."""

    version = "0.2.0"
    branch = "pipeline"

    def check_prerequisites(self, context: MarketContext) -> bool:
        if not super().check_prerequisites(context):
            return False
        if not self.config.get(
            "require_stage7_regime_review",
            False,
        ):
            return True
        run_phases = {
            str(item)
            for item in self.config.get(
                "run_phases",
                ["pre_trade"],
            )
        }
        review = context.metadata.get("stage7_regime_review")
        return (
            context.phase in run_phases
            and isinstance(review, dict)
            and review.get("schema_version")
            == "dean_stage7_regime_review_v1"
            and review.get("status")
            == "stage7_regime_contexts_recorded"
            and bool(review.get("contexts"))
        )

    async def run(self, context: MarketContext) -> PipelineReport:
        snapshot = build_regime_snapshot(
            context=context,
            engine=str(self.config.get("engine", "fallback")),
            market_data_path=self.config.get("market_data_path"),
            latest_processed_prices=self.config.get("latest_processed_prices"),
            ticker=(
                self.config.get("ticker")
                or (
                    context.tickers[0]
                    if len(context.tickers) == 1
                    else None
                )
            ),
            close_col=str(self.config.get("close_col", "close")),
            volume_col=self.config.get("volume_col", "volume"),
            manual_regime=self.config.get("manual_regime"),
            manual_tags=self.config.get("manual_tags", []),
            require_stage7_review=bool(
                self.config.get(
                    "require_stage7_regime_review",
                    False,
                )
            ),
        )

        context.metadata["regime_context"] = snapshot.model_dump(mode="json")
        regime_tags = normalize_context_tags(
            [*context.metadata.get("regime_tags", []), *snapshot.context_tags]
        )
        context.metadata["regime_tags"] = regime_tags

        shadow_mode = bool(self.config.get("shadow_mode", False))
        observed_signal = _regime_signal(snapshot)
        signal_strength = 0.0 if shadow_mode else observed_signal
        verdict = "clear" if snapshot.regime != "UNKNOWN" and not snapshot.warnings else "caution"
        reasons = [
            _regime_reason(
                snapshot,
                observed_signal,
                shadow_mode=shadow_mode,
            )
        ]
        risks = []
        if snapshot.warnings:
            risks.extend(snapshot.warnings)
        if snapshot.regime in {"CRISIS", "VOLATILE"}:
            risks.append("Risk sizing and model interpretation should account for elevated regime stress.")

        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=snapshot.confidence,
            data_quality_score=0.75 if verdict == "clear" else 0.45,
            signal_strength=signal_strength,
            reasons=reasons,
            risks=risks,
            blind_spots=[
                "Regime classification is a context signal only; it does not validate model forecasts or execute trades."
            ],
            evidence=[
                self.evidence("metric", snapshot.source, "regime", snapshot.regime),
                self.evidence("metric", snapshot.source, "confidence", snapshot.confidence),
                self.evidence("metric", snapshot.source, "context_tags", snapshot.context_tags),
                self.evidence("metric", snapshot.source, "metrics", snapshot.metrics),
            ],
            input_hash=self.context_hash(context),
            metrics_snapshot={
                **snapshot.model_dump(mode="json"),
                "observed_signal_strength": observed_signal,
                "decision_influence": not shadow_mode,
                "supporting_review_only": shadow_mode,
                "can_promote_model": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
        )


def build_regime_snapshot(
    context: MarketContext,
    engine: str = "fallback",
    market_data_path: str | Path | None = None,
    latest_processed_prices: str | None = None,
    ticker: str | None = None,
    close_col: str = "close",
    volume_col: str | None = "volume",
    manual_regime: str | None = None,
    manual_tags: list[str] | tuple[str, ...] | None = None,
    require_stage7_review: bool = False,
) -> MarketRegimeSnapshot:
    builder = RegimeContextBuilder()
    if manual_regime:
        snapshot = builder.from_analyzer_result(
            {"regime": manual_regime, "confidence": 0.5, "tags": list(manual_tags or []), "manual": True},
            source="manual",
        )
        snapshot.warnings.append("Manual regime context. Use for review/smoke runs, not as market evidence.")
        return snapshot

    stage7_review = context.metadata.get("stage7_regime_review")
    if isinstance(stage7_review, dict):
        return _snapshot_from_stage7_review(
            builder=builder,
            review=stage7_review,
            ticker=ticker,
            timeframe=context.timeframe
            or (
                context.timeframes[0]
                if len(context.timeframes) == 1
                else None
            ),
        )
    if require_stage7_review:
        return MarketRegimeSnapshot(
            source="stage7_market_regime_review",
            warnings=[
                "Strict Stage 7 regime review was required; no review "
                "contract was supplied. Filesystem and dataframe fallbacks "
                "were not used."
            ],
        )

    existing = context.metadata.get("regime_context")
    if existing:
        if isinstance(existing, MarketRegimeSnapshot):
            return existing
        if isinstance(existing, dict):
            return MarketRegimeSnapshot(**existing)

    frame = _resolve_market_frame(
        context=context,
        market_data_path=market_data_path,
        latest_processed_prices=latest_processed_prices,
        ticker=ticker,
    )
    if frame is None:
        return MarketRegimeSnapshot(
            source="regime_agent",
            warnings=["No regime context or market price frame was supplied."],
        )

    if engine == "project":
        resolved_close_col = _resolve_column(frame, close_col)
        analyzer_frame = frame.rename(columns={resolved_close_col: "close"}) if resolved_close_col else frame
        snapshot = builder.from_project_analyzer(analyzer_frame)
    else:
        snapshot = builder.from_price_frame(frame, close_col=_resolve_column(frame, close_col), volume_col=_resolve_column(frame, volume_col))
    if market_data_path:
        snapshot.metrics["market_data_path"] = str(market_data_path)
    if ticker:
        snapshot.metrics["ticker"] = str(ticker).upper()
    return snapshot


def _resolve_market_frame(
    context: MarketContext,
    market_data_path: str | Path | None,
    latest_processed_prices: str | None,
    ticker: str | None,
) -> Any | None:
    if market_data_path or latest_processed_prices:
        path = _resolve_market_data_path(market_data_path, latest_processed_prices)
        if path is None or not path.exists():
            return None
        frame = _read_market_frame(path)
        return _filter_ticker(frame, ticker)

    for key in ("market", "prices", "features"):
        frame = context.dataframes.get(key)
        if frame is not None:
            return _filter_ticker(frame, ticker)
    return None


def _snapshot_from_stage7_review(
    *,
    builder: RegimeContextBuilder,
    review: dict[str, Any],
    ticker: str | None,
    timeframe: str | None,
) -> MarketRegimeSnapshot:
    if (
        review.get("schema_version")
        != "dean_stage7_regime_review_v1"
        or review.get("status")
        != "stage7_regime_contexts_recorded"
    ):
        return MarketRegimeSnapshot(
            source="stage7_market_regime_review",
            warnings=[
                "Stage 7 regime review is unavailable or not reviewable."
            ],
        )
    ticker_value = str(ticker).upper() if ticker else None
    timeframe_value = (
        str(timeframe).lower() if timeframe else None
    )
    matches = []
    for item in review.get("contexts", []):
        if not isinstance(item, dict):
            continue
        item_ticker = (
            str(item.get("ticker")).upper()
            if item.get("ticker")
            else None
        )
        item_timeframe = (
            str(item.get("timeframe")).lower()
            if item.get("timeframe")
            else None
        )
        if ticker_value and item_ticker != ticker_value:
            continue
        if timeframe_value and item_timeframe != timeframe_value:
            continue
        if ticker_value is None and item_ticker is not None:
            continue
        if timeframe_value is None and item_timeframe is not None:
            continue
        matches.append(item)
    if len(matches) != 1:
        return MarketRegimeSnapshot(
            source="stage7_market_regime_review",
            warnings=[
                "No unique Stage 7 regime context matches the requested "
                f"ticker/timeframe: {ticker_value}/{timeframe_value}."
            ],
        )
    selected = matches[0]
    metrics = (
        selected.get("metrics")
        if isinstance(selected.get("metrics"), dict)
        else {}
    )
    snapshot = builder.from_analyzer_result(
        {
            "regime": selected.get("regime"),
            "confidence": selected.get("confidence"),
            **metrics,
            "context_key": selected.get("context_key"),
            "ticker": selected.get("ticker"),
            "timeframe": selected.get("timeframe"),
            "evidence_class": review.get("evidence_class"),
            "supporting_review_only": True,
        },
        source="stage7_market_regime_review",
    )
    return snapshot


def _resolve_market_data_path(raw_path: str | Path | None, latest_interval: str | None) -> Path | None:
    if raw_path:
        return Path(raw_path)
    if not latest_interval:
        return None
    candidates = sorted(
        Path("data/processed").glob(f"prices_{latest_interval}_*.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _read_market_frame(path: Path) -> Any:
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported market data file type: {path.suffix}. Use .csv or .parquet.")


def _filter_ticker(frame: Any, ticker: str | None) -> Any:
    if not ticker or not hasattr(frame, "columns"):
        return frame
    ticker_col = _first_existing_column(frame, ["ticker", "symbol", "Ticker", "Symbol"])
    if ticker_col is None:
        return frame
    filtered = frame[frame[ticker_col].astype(str).str.upper() == str(ticker).upper()]
    return filtered if not filtered.empty else frame


def _resolve_column(frame: Any, requested: str | None) -> str | None:
    if requested is None or not hasattr(frame, "columns"):
        return requested
    if requested in frame.columns:
        return requested
    lowered = {str(column).lower(): column for column in frame.columns}
    return lowered.get(requested.lower(), requested)


def _first_existing_column(frame: Any, candidates: list[str]) -> str | None:
    lowered = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _regime_signal(snapshot: MarketRegimeSnapshot) -> float:
    base_by_regime = {
        "BULL_MARKET": 0.25,
        "TRENDING_UP": 0.25,
        "MOMENTUM": 0.2,
        "BREAKOUT": 0.2,
        "NORMAL": 0.05,
        "RANGING": 0.0,
        "SIDEWAYS": 0.0,
        "VOLATILE": -0.15,
        "TRENDING_DOWN": -0.25,
        "BEAR_MARKET": -0.3,
        "CRISIS": -0.45,
    }
    return round(base_by_regime.get(snapshot.regime, 0.0) * snapshot.confidence, 4)


def _regime_reason(
    snapshot: MarketRegimeSnapshot,
    signal_strength: float,
    *,
    shadow_mode: bool = False,
) -> str:
    tags = ", ".join(snapshot.context_tags) or "none"
    influence = (
        "shadow review only; no decision influence"
        if shadow_mode
        else "decision influence enabled"
    )
    return (
        f"Regime context is {snapshot.regime} with confidence {snapshot.confidence:.2f}; "
        f"tags: {tags}; observed signal: {signal_strength:.3f}; "
        f"{influence}."
    )
