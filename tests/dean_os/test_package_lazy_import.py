from __future__ import annotations

import subprocess
import sys


def test_package_root_does_not_eagerly_load_pipeline_stack():
    code = """
import sys
import dean_os
assert "src.pipeline.hybrid_orchestrator" not in sys.modules
assert "dean_os.pipeline_control_bounded_evidence_run" not in sys.modules
assert "MarketContext" in dean_os.__all__
assert "SavedMacroEvidenceProducer" in dean_os.__all__
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_lazy_public_schema_export_preserves_root_api():
    from dean_os import MarketContext

    context = MarketContext(
        as_of="2026-06-30T12:00:00+00:00",
        tickers=["AMD"],
    )

    assert context.tickers == ["AMD"]
