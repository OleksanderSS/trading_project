"""What the union costs when concatenated, and when written a frame at a time.

Shapes are the real ones from the 110-ticker checkpoints of 2026-08-26 --
15m 150,838 x 466, 1d 704,950 x 480, 60m 377,119 x 473 -- scaled down by
ROW_DIVISOR so the comparison fits on a laptop that is also running a browser.
Column counts are NOT scaled, because the padding is what the argument is
about: every column absent from a frame is what `concat` materialises and the
writer does not.

Run with ROW_DIVISOR=1 on a quiet machine to see the full numbers.
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.pipeline.parquet_union_writer import write_union  # noqa: E402

ROW_DIVISOR = int(os.environ.get("ROW_DIVISOR", "5"))
SHAPES = {"15m": (150_838, 466), "1d": (704_950, 480), "60m": (377_119, 473)}
SHARED = ("datetime", "ticker", "interval", "close")

process = psutil.Process(os.getpid())


def rss() -> float:
    return process.memory_info().rss / 2 ** 30


def build() -> dict[str, pd.DataFrame]:
    """Frames that overlap only on the identity columns, as the real ones do."""
    rng = np.random.default_rng(0)
    frames: dict[str, pd.DataFrame] = {}
    for name, (rows, columns) in SHAPES.items():
        rows = max(rows // ROW_DIVISOR, 1)
        frame = pd.DataFrame({
            "datetime": pd.date_range("2018-01-01", periods=rows, freq="min"),
            "ticker": np.repeat("AAPL", rows),
            "interval": np.repeat(name, rows),
            "close": rng.standard_normal(rows).astype(np.float32),
        })
        for index in range(columns - len(SHARED)):
            frame[f"{name}_f{index}"] = rng.standard_normal(rows).astype(np.float32)
        frames[name] = frame
    return frames


def main() -> None:
    destination = Path("data/temp/union_memory_probe.parquet")
    frames = build()
    gc.collect()
    held = rss()
    rows = sum(len(f) for f in frames.values())
    width = len({c for f in frames.values() for c in f.columns})
    print(f"row divisor {ROW_DIVISOR}: {rows:,} rows, {width:,} union columns")
    print(f"the three frames themselves hold {held:.2f} GiB\n")

    write_union(frames, destination)
    gc.collect()
    after_writer = rss()
    print(f"write_union   peak above the frames: {after_writer - held:+.2f} GiB")
    print(f"              file on disk: {destination.stat().st_size / 2**20:.0f} MiB")

    try:
        combined = pd.concat(frames.values(), ignore_index=True, sort=False)
        after_concat = rss()
        print(f"pd.concat     peak above the frames: {after_concat - held:+.2f} GiB "
              f"({combined.shape[0]:,} x {combined.shape[1]:,})")
        del combined
    except MemoryError as error:
        print(f"pd.concat     FAILED: {error}")

    destination.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
