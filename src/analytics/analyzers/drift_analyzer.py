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
        self._baseline_signature = ""
        if not self.baseline_path.exists():
            return
        try:
            reference = pd.read_parquet(self.baseline_path)
            self.monitor.set_reference_data(reference)
            self._baseline_set = True
            self._baseline_signature = self._frame_signature(reference)
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

    @staticmethod
    def _frame_signature(frame: "pd.DataFrame") -> str:
        """Content identity for a feature frame, or "" when it cannot be taken.

        Shape and index bounds are NOT enough, and the first version of this
        used only those. Drift's whole subject is the same rows carrying
        different values -- a frame with f1 shifted by 4.0 has an identical
        shape, an identical index and completely different data. That
        signature called it "the same batch" and skipped the comparison,
        which made the monitor blind to precisely what it exists to see.
        Three tests in test_feature_drift_wiring.py said so immediately.

        So the values are hashed. An empty string means "could not tell",
        and the caller must then MEASURE rather than skip: an unknown answer
        has to fall on the side of doing the work, not of assuming it is
        unnecessary.
        """
        try:
            import hashlib

            import pandas as pd

            digest = hashlib.sha256(
                pd.util.hash_pandas_object(frame, index=True).values.tobytes()
            ).hexdigest()
            return f"{frame.shape}|{digest}"
        except Exception:
            # Unhashable column types (a column of lists will do it) reach
            # here. Returning "" makes the equality test below fail, so the
            # comparison runs.
            return ""

    def analyze(self, data: Any, **kwargs) -> dict[str, Any]:
        """Compare the current feature frame against the stored baseline."""
        frame = data.get("features_data") if isinstance(data, dict) else data

        if frame is None or not hasattr(frame, "empty") or frame.empty:
            return {"status": "skipped", "reason": "no_feature_data"}

        if not self._baseline_set:
            self._store_baseline(frame)
            self._baseline_signature = self._frame_signature(frame)
            return {
                "status": "baseline_set",
                "reason": "first frame adopted as reference; drift needs a second one",
                "rows": int(len(frame)),
            }

        # Say when there is nothing to compare, instead of reporting 0.0.
        #
        # The baseline outlives the process, so once it is written the same
        # batch is compared against itself on every later run -- and the
        # honest answer, drift 0.0, is indistinguishable from "checked and
        # found nothing". On 2026-08-10 all 40 completed contexts reported
        # exactly 0.0 for that reason, and it read as a working check.
        #
        # A monitor that cannot fire should say so in the place someone
        # looks, not require them to remember. Reporting it explicitly is
        # what makes "turn this on when a second batch exists" something the
        # run reminds you of rather than something to forget.
        signature = self._frame_signature(frame)
        if signature and signature == getattr(self, "_baseline_signature", None):
            return {
                "status": "not_applicable",
                "reason": (
                    "the baseline was built from this same data, so drift is "
                    "0.0 by construction, not by measurement. It becomes "
                    "meaningful once a second batch exists to compare against."
                ),
                "baseline_path": str(self.baseline_path),
                "rows": int(len(frame)),
            }

        try:
            result = self.monitor.check_drift(frame)
            # The monitor reports its own status ('OK'), and spreading it
            # last means that value wins -- which is right, but the line read
            # as though 'checked' were the answer. It never was. Stating the
            # precedence rather than relying on dict ordering.
            payload = dict(result or {})
            payload.setdefault("status", "checked")
            return payload
        except Exception as e:
            logger.warning(f"Drift check unavailable: {e}")
            return {"status": "unavailable", "reason": str(e)}
