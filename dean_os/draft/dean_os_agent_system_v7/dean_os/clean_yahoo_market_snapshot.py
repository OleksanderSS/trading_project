from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from src.data.collectors.yf_collector import YFCollector
from src.pipeline.timeframe_lineage import normalize_timeframe

CLEAN_YAHOO_MARKET_SNAPSHOT_CONTRACT = "dean_clean_yahoo_market_snapshot_v1"


class CleanYahooMarketSnapshot:
    """Collect an identity-validated Yahoo snapshot outside the legacy DB.

    This producer deliberately bypasses the accumulated Stage1 database.  It
    writes a new immutable market artifact only after the collector's source
    ticker, exact-OHLCV identity, cadence, timezone and finite-value gates pass.
    """

    def __init__(
        self,
        artifact_dir: str | Path = "data/dean_os/clean_market_snapshots",
        report_dir: str | Path = "reports/dean_os/clean_market_snapshot_current",
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.report_dir = Path(report_dir)

    async def build(
        self,
        *,
        tickers: list[str],
        config_path: str | Path = "src/config/collectors.yaml",
        timeframes: list[str] | None = None,
        end_date: datetime | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        requested_tickers = _normalize_tickers(tickers)
        if not requested_tickers:
            raise ValueError("At least one ticker is required")

        yahoo_config = _load_yahoo_config(config_path)
        yahoo_config["timeframes"] = _selected_timeframes(
            yahoo_config.get("timeframes", {}),
            timeframes,
        )
        if not yahoo_config["timeframes"]:
            raise ValueError("No Yahoo timeframes were selected")

        resolved_end = end_date or datetime.now(UTC)
        collector = YFCollector(
            configs=yahoo_config,
            http_client_factory=None,
            db_manager=None,
            cache_manager=None,
        )
        records = await collector.run(
            tickers=requested_tickers,
            end_date=resolved_end,
            reference_now=resolved_end,
            persist=False,
        )
        frame = _normalize_collected_frame(records)
        post_normalization_issues = collector._validate_collected_price_data(frame)
        if post_normalization_issues:
            raise RuntimeError(
                "Normalized clean snapshot failed source gate: "
                + "; ".join(post_normalization_issues)
            )

        run_id = _run_id()
        snapshot_sha256 = _frame_sha256(frame)
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "clean_yahoo_market_snapshot",
            "contract": CLEAN_YAHOO_MARKET_SNAPSHOT_CONTRACT,
            "inputs": {
                "config_path": str(config_path),
                "tickers": requested_tickers,
                "requested_timeframes": list(yahoo_config["timeframes"]),
                "end_date": resolved_end.isoformat(),
                "persist_to_legacy_database": False,
                "use_legacy_cache": False,
            },
            "summary": {
                "status": "clean_market_snapshot_validated",
                "row_count": int(len(frame)),
                "ticker_count": int(frame["ticker"].nunique()),
                "timeframe_count": int(frame["interval"].nunique()),
                "snapshot_sha256": snapshot_sha256,
                "source_gate_issues": [],
                "can_feed_stage23": True,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "lineage": {
                "provider": "yahoo_finance",
                "source_intervals": sorted(frame["source_interval"].unique()),
                "canonical_intervals": sorted(frame["interval"].unique()),
                "normalization": "1h_to_60m_label_only",
                "old_stage1_artifact_reused": False,
                "old_database_table_reused": False,
            },
            "lanes": _lane_summaries(frame),
            "safety": {
                "source_ticker_validated_before_relabel": True,
                "cross_identity_exact_ohlcv_gate": True,
                "cadence_gate": True,
                "finite_ohlcv_gate": True,
                "network_access_performed": True,
                "database_write_performed": False,
                "legacy_artifact_write_performed": False,
                "learning_write_performed": False,
                "broker_access_performed": False,
            },
        }

        if save:
            snapshot_path, latest_path = self._save_frame(frame, run_id)
            payload["snapshot"] = {
                "path": str(snapshot_path),
                "latest_path": str(latest_path),
                "format": "parquet",
                "sha256": snapshot_sha256,
            }
            writer = ReviewArtifactWriter(self.report_dir)
            payload["saved_paths"] = writer.write(
                payload=payload,
                markdown=render_clean_snapshot_markdown(payload),
                run_id=run_id,
            )
        return payload

    def _save_frame(self, frame: pd.DataFrame, run_id: str) -> tuple[Path, Path]:
        output_dir = self.artifact_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = output_dir / f"{run_id}.parquet"
        latest_path = output_dir / "latest.parquet"

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{run_id}.",
            suffix=".parquet.tmp",
            dir=str(output_dir),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        latest_tmp = output_dir / f".{run_id}.latest.parquet.tmp"
        try:
            frame.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, snapshot_path)
            shutil.copy2(snapshot_path, latest_tmp)
            os.replace(latest_tmp, latest_path)
        finally:
            for path in (tmp_path, latest_tmp):
                if path.exists():
                    path.unlink()
        return snapshot_path, latest_path


def _load_yahoo_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = dict((payload.get("collectors") or {}).get("yahoo_finance") or {})
    if not config:
        raise ValueError(f"Yahoo collector config not found in {path}")
    return config


def _selected_timeframes(
    configured: dict[str, Any],
    requested: list[str] | None,
) -> dict[str, Any]:
    if not requested:
        return dict(configured)
    normalized_requested = {
        normalize_timeframe(value)
        for value in requested
        if normalize_timeframe(value)
    }
    selected = {}
    for source_interval, params in configured.items():
        if normalize_timeframe(source_interval) in normalized_requested:
            selected[source_interval] = params
    missing = normalized_requested - {
        normalize_timeframe(value) for value in selected
    }
    if missing:
        raise ValueError(
            "Requested timeframe(s) are not configured for Yahoo: "
            + ", ".join(sorted(missing))
        )
    return selected


def _normalize_collected_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("Yahoo returned no rows for the clean snapshot")
    frame = frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise", utc=True)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["source_interval"] = frame["interval"].astype(str).str.lower()
    frame["interval"] = frame["source_interval"].map(normalize_timeframe)
    if frame["interval"].isna().any():
        raise RuntimeError("Yahoo returned an unsupported source interval")
    frame = frame.sort_values(["ticker", "interval", "datetime"]).reset_index(drop=True)
    frame["hash"] = frame.apply(
        lambda row: hashlib.sha256(
            (
                row["datetime"].isoformat()
                + str(row["ticker"])
                + str(row["interval"])
            ).encode("utf-8")
        ).hexdigest(),
        axis=1,
    )
    return frame


def _normalize_tickers(tickers: list[str]) -> list[str]:
    return sorted({str(value).strip().upper() for value in tickers if str(value).strip()})


def _frame_sha256(frame: pd.DataFrame) -> str:
    canonical = frame.sort_values(["ticker", "interval", "datetime"]).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(canonical, index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _lane_summaries(frame: pd.DataFrame) -> list[dict[str, Any]]:
    lanes = []
    for (ticker, interval), lane in frame.groupby(["ticker", "interval"], sort=True):
        lanes.append(
            {
                "ticker": str(ticker),
                "timeframe": str(interval),
                "source_intervals": sorted(lane["source_interval"].unique()),
                "row_count": int(len(lane)),
                "minimum_datetime": lane["datetime"].min().isoformat(),
                "maximum_datetime": lane["datetime"].max().isoformat(),
            }
        )
    return lanes


def _run_id() -> str:
    return "clean_yahoo_market_" + utc_now_iso().replace(":", "").replace("+", "Z")


def render_clean_snapshot_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    snapshot = payload.get("snapshot", {})
    lines = [
        "# Clean Yahoo Market Snapshot",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Status: `{summary['status']}`",
        f"- Rows: `{summary['row_count']}`",
        f"- Tickers: `{summary['ticker_count']}`",
        f"- Timeframes: `{summary['timeframe_count']}`",
        f"- SHA256: `{summary['snapshot_sha256']}`",
        f"- Snapshot: `{snapshot.get('path')}`",
        "- Legacy database reused: `false`",
        "- Can feed Stage23: `true`",
        "- Can trade: `false`",
        "",
        "## Lanes",
        "",
    ]
    for lane in payload["lanes"]:
        lines.append(
            f"- `{lane['ticker']}/{lane['timeframe']}`: "
            f"{lane['row_count']} rows, {lane['minimum_datetime']} -> "
            f"{lane['maximum_datetime']}"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "CLEAN_YAHOO_MARKET_SNAPSHOT_CONTRACT",
    "CleanYahooMarketSnapshot",
    "render_clean_snapshot_markdown",
]
