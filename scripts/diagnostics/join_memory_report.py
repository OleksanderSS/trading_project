"""Measure the concat/sort pattern at the size that killed the run."""
import gc
import os

import numpy as np
import pandas as pd
import psutil

ROWS, COLS, GROUPS = 376_359, 415, 110
proc = psutil.Process(os.getpid())


def rss() -> float:
    return proc.memory_info().rss / 2 ** 30


def make_groups():
    rng = np.random.default_rng(0)
    per = ROWS // GROUPS
    out = []
    order = 0
    for _ in range(GROUPS):
        block = pd.DataFrame(
            rng.standard_normal((per, COLS - 1)).astype(np.float32),
            columns=[f"f{i}" for i in range(COLS - 1)],
        )
        block["__base_order"] = np.arange(order, order + per)
        order += per
        out.append(block)
    return out


def old_way(groups):
    return (
        pd.concat(groups, ignore_index=True, sort=False)
        .sort_values("__base_order", kind="mergesort")
        .reset_index(drop=True)
    )


def new_way(groups):
    result = pd.concat(groups, ignore_index=True, sort=False)
    groups.clear()
    result = result.sort_values("__base_order", kind="mergesort")
    result.reset_index(drop=True, inplace=True)
    return result


for name, fn in (("old", old_way), ("new", new_way)):
    gc.collect()
    groups = make_groups()
    gc.collect()
    before = rss()
    peak = before
    result = fn(groups)
    peak = max(peak, rss())
    print(f"{name}: held {before:.2f} GiB before, {peak:.2f} GiB after "
          f"-> the step itself cost {peak - before:.2f} GiB")
    checksum = float(result["f0"].to_numpy()[:1000].sum())
    ordered = bool((result["__base_order"].to_numpy() ==
                    np.arange(len(result))).all())
    print(f"       rows {len(result):,}  order preserved: {ordered}  "
          f"checksum {checksum:.6f}")
    del result, groups
    gc.collect()
