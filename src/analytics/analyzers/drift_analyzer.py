from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics.interfaces import IAnalyzer
from src.core.logging.logger import ProjectLogger
from src.monitoring.feature_drift_monitor import FeatureDriftMonitor

logger = ProjectLogger.get_logger("DriftAnalyzer")


class DriftAnalyzer(IAnalyzer):
    """Adapter for FeatureDriftMonitor to integrate with UnifiedAnalyticsEngine.

    Three faults kept this from ever running:

    - it called `self.monitor.detect_drift(...)`, and FeatureDriftMonitor has
      no such method -- the real one is `check_drift`. Any invocation would
      have raised AttributeError.
    - `check_drift` raises when no reference data has been set, and nothing
      ever set any, so even the corrected call would fail on first use. The
      first frame seen is now adopted as the baseline and reported as such.
    - the analyzer was not listed in analysis.yaml, so
      UnifiedAnalyticsEngine._register_analyzers_from_config never built it.

    Drift is reported, never raised: a monitoring adapter that throws would
    take down the analytics pass it is supposed to observe.
    """

    def __init__(self, threshold: float = 0.05, config: dict | None = None,
                 baseline_path: str | None = None):
        config = config or {}
        threshold = config.get('threshold', threshold)
        self.monitor = FeatureDriftMonitor(drift_threshold=threshold)
        # The baseline must OUTLIVE the process. A pipeline run builds this
        # analyzer fresh, so an in-memory reference would make every run report
        # "baseline_set" and never actually compare anything -- monitoring that
        # can never fire.
        self.baseline_path = Path(
            baseline_path or config.get('baseline_path')
            or "reports/drift/reference_features.parquet"
        )
        self._load_baseline()

    def _load_baseline(self) -> None:
        self._baseline_set = False
        if not self.baseline_path.exists():
            return
        try:
            self.monitor.set_reference_data(pd.read_parquet(self.baseline_path))
            self._baseline_set = True
            logger.info(f"Drift baseline loaded from {self.baseline_path}")
        except Exception as e:
            logger.warning(f"Could not load drift baseline ({e}); it will be rebuilt.")

    def _store_baseline(self, frame: "pd.DataFrame") -> None:
        self.monitor.set_reference_data(frame)
        self._baseline_set = True
        try:
            self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(self.baseline_path, compression="zstd")
            logger.info(f"Drift baseline stored at {self.baseline_path}")
        except Exception as e:
            logger.warning(
                f"Drift baseline kept in memory only ({e}); the next run will "
                f"have to rebuild it."
            )

    def analyze(self, data: Any, **kwargs) -> dict[str, Any]:
        """Compare the current feature frame against the stored baseline."""
        frame = data.get("features_data") if isinstance(data, dict) else data

        if frame is None or not hasattr(frame, "empty") or frame.empty:
            return {"status": "skipped", "reason": "no_feature_data"}

        if not self._baseline_set:
            self._store_baseline(frame)
            return {
                "status": "baseline_set",
                "reason": "first frame adopted as reference; drift needs a second one",
                "rows": int(len(frame)),
            }

        try:
            result = self.monitor.check_drift(frame)
            return {"status": "checked", **(result or {})}
        except Exception as e:
            logger.warning(f"Drift check unavailable: {e}")
            return {"status": "unavailable", "reason": str(e)}
