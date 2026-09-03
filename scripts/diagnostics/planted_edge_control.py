"""Can the gate find an edge that is definitely there, and refuse one that is not?

REGISTER #223. The companion to the money-path calibration (CLAIMS.md R9).
That one proved Stage 7 reports a planted edge at the size it was planted; it
used ready-made predictions and so said nothing about whether the LEARNING
path can find one. This does.

Three targets are planted on the same synthetic panel, and each has a verdict
that is known before the run:

    A. one-column      y = 1{f0 + noise > 0}
       A single feature explains the target completely. The correct verdict is
       a REFUSAL, with the "one column and a straight line" rung binding: a
       model on 20 features that reproduces what f0 already says has added
       nothing. If this promotes, the ladder is not working.

    B. interaction     y = 1{f0 * f1 + noise > 0}
       No single column correlates with the target -- the product does. A tree
       can find it; a straight line on any one column cannot. The correct
       verdict is a CHAMPION. If this is refused, the gate is rejecting real,
       findable edges, and every refusal this project has recorded becomes
       uninterpretable.

    C. nothing         y = a coin flip
       The correct verdict is a REFUSAL. If this promotes, every champion is
       suspect.

B is the one that matters. A and C guard the two ways of being wrong about it:
a gate that promotes anything (C would pass) and a gate that promotes nothing
(A would pass while B fails).

The panel is synthetic on purpose. Real features carry real structure, and a
control has to know its own answer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.models.adapters.data_preparation import (  # noqa: E402
    prepare_data_for_models,
)
from src.pipeline.modeling_context import POOLED_TICKER  # noqa: E402
from src.pipeline.stages.modeling.orchestrator import ModelingStage  # noqa: E402
from src.training.batch_trainer import BatchTrainer  # noqa: E402

N_NAMES = 40
N_BARS = 1500
N_FEATURES = 20
NOISE = 0.9
SEED = 20260901


def _plan(ticker: str, models: list[str] | None) -> dict | None:
    """The plan shape the trainer actually reads, or nothing at all.

    This stand wrote `data["plan"] = {"models": models}` and printed the list
    at the top of every run. `BaseTrainer._prepare_model_training_list` reads

        data['plan']['ticker_plans'][ticker]['models']

    -- one level deeper -- so the key was never found, the list fell back to
    `models.enabled_types`, and every run of this stand trained the full
    configured suite while announcing three models. The gate verdicts it
    produced are unaffected: they are counts of what the gate did, and the
    gate saw whatever was trained. What was wrong is the DESCRIPTION, and a
    control whose header misstates its own setup is one step from a control
    whose result is read against the wrong number of attempts.

    Passing no models omits the key entirely, so the trainer falls back to the
    configured suite ON PURPOSE rather than by accident -- which is also the
    right default here, because a calibration of the real gate should train
    what the real run trains.
    """
    if not models:
        return None
    return {"ticker_plans": {ticker: {"models": list(models)}}}


def _panel(rng: np.random.Generator) -> pd.DataFrame:
    """The synthetic panel, with real timestamps.

    The stamps are not decoration. The gate's fourth opponent asks for a
    `DatetimeIndex` and returns nothing without one
    (`base_trainer._clock_prediction`), so a stand built on a 0..n-1 index
    scores the gate against three opponents while reporting the verdict of a
    gate that has four. The real-null path was fixed on 2026-09-02 by routing
    through `prepare_data_for_models` (REGISTER #230); this panel keeps its own
    split on purpose -- a control for the GATE should not also be a control for
    feature selection -- so it carries the index itself.

    Every name shares one business-day calendar, which is what a pooled
    cross-section looks like: the same forty rows per day, forty names deep.
    """
    rows = N_NAMES * N_BARS
    frame = pd.DataFrame(
        rng.standard_normal((rows, N_FEATURES)).astype("float32"),
        columns=[f"f{i}" for i in range(N_FEATURES)],
    )
    frame["ticker"] = np.repeat([f"T{i:02d}" for i in range(N_NAMES)], N_BARS)
    calendar = pd.bdate_range("2010-01-04", periods=N_BARS, tz="UTC")
    frame["datetime"] = np.tile(calendar.to_numpy(), N_NAMES)
    return frame


def _target(kind: str, frame: pd.DataFrame, rng: np.random.Generator,
            noise_scale: float = NOISE) -> np.ndarray:
    noise = rng.standard_normal(len(frame)) * noise_scale
    if kind == "one-column":
        signal = frame["f0"].to_numpy()
    elif kind == "interaction":
        signal = frame["f0"].to_numpy() * frame["f1"].to_numpy()
    elif kind == "nothing":
        signal = np.zeros(len(frame))
    else:
        raise ValueError(kind)
    return (signal + noise > 0).astype(float)


def _split(frame: pd.DataFrame, y: np.ndarray, features: list[str]) -> dict:
    """Train / val / test / holdout, split by TIME within each name.

    The bars of one name are contiguous, so a positional split per name keeps
    the holdout strictly after everything the model saw -- the same shape the
    real pipeline uses, and the reason a holdout means anything.
    """
    per_name = N_BARS
    train_end, val_end, test_end = int(per_name * 0.6), int(per_name * 0.7), int(per_name * 0.8)
    position = np.tile(np.arange(per_name), N_NAMES)

    def take(lo: int, hi: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        mask = (position >= lo) & (position < hi)
        block = frame.loc[mask, features].copy()
        # `model_datetime` is the index name `prepare_data_for_models` gives
        # every split, and `_purge_gap` looks for exactly that name when it
        # scales the gap for a pooled frame. Matching it here keeps the two
        # paths describing the same thing.
        block.index = pd.DatetimeIndex(
            frame.loc[mask, "datetime"].to_numpy(), name="model_datetime",
        )
        return block, y[mask], frame.loc[mask, "ticker"].to_numpy()

    x_train, y_train, _ = take(0, train_end)
    x_val, y_val, _ = take(train_end, val_end)
    x_test, y_test, _ = take(val_end, test_end)
    x_hold, y_hold, hold_groups = take(test_end, per_name)
    return {
        "X_train": x_train, "y_train": y_train,
        "X_val": x_val, "y_val": y_val,
        "X_test": x_test, "y_test": y_test,
        "X_holdout": x_hold, "y_holdout": y_hold,
        "holdout_groups": hold_groups,
        # y_val is what the clock opponent's scheme is now chosen on
        # (REGISTER #231); without it the choice silently falls back to a
        # fixed order and the stand stops testing what it thinks it tests.
        "feature_names": features,
        "target_type": "classification_binary",
        "timeframe": "1d",
        "preprocessor": None,
    }


def run_case(kind: str, models: list[str], noise_scale: float = NOISE,
             seed: int = SEED, family_size: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    frame = _panel(rng)
    y = _target(kind, frame, rng, noise_scale)
    features = [c for c in frame.columns if c.startswith("f")]
    data = _split(frame, y, features)
    data["target_name"] = f"target_planted_{kind.replace('-', '_')}_1d"
    if family_size is not None:
        data["promotion_family_size"] = family_size
    plan = _plan("SYNTH", models)
    if plan:
        data["plan"] = plan

    trainer = BatchTrainer()
    results = trainer._train_ticker_suite("SYNTH", data)
    gate = results.get("promotion_gate") or {}
    return {
        "case": kind,
        "base_rate": float(np.mean(data["y_holdout"])),
        "winner": results.get("winner"),
        # The holdout dict reports the GOVERNING metric under `score`, with
        # its name in `metric`. Reading 'BalancedAccuracy' returned None on
        # the first run -- a missing key printed as "no score", which reads
        # like a failure to measure and was nothing of the kind.
        "winner_score": (results.get("winner_holdout_metrics") or {}).get("score"),
        "winner_metric": (results.get("winner_holdout_metrics") or {}).get("metric"),
        "holdout_status": (results.get("winner_holdout_metrics") or {}).get("status"),
        # The gate's own key is `passed`; there is no `promoted`. Reading a
        # key that does not exist would have made every case look refused,
        # including the ones that were not -- and the script would have
        # reported a broken gate that was working.
        "promoted": bool(gate.get("passed")),
        # Plural: the gate lists every rung that bound, not one reason.
        "reason": "; ".join(gate.get("reasons") or []) or None,
        "gate": gate,
    }


EXPECTED = {
    "one-column": False,   # the single-feature rung must bind
    "interaction": True,   # a real, findable edge must survive
    "nothing": False,      # noise must be refused
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=(
            "Restrict training to these model types. Omit to train the "
            "configured suite, which is what a real run trains and therefore "
            "the honest default for a calibration. Until 2026-09-02 this "
            "option was written into the wrong level of the plan dictionary "
            "and silently did nothing."
        ),
    )
    parser.add_argument("--cases", nargs="+", default=list(EXPECTED))
    parser.add_argument("--names", type=int, default=N_NAMES)
    parser.add_argument("--bars", type=int, default=N_BARS)
    parser.add_argument(
        "--noise", type=float, nargs="+", default=[NOISE],
        help=(
            "Noise scale(s) on the planted signal. Larger means a smaller "
            "true edge: measured Bayes balanced accuracy for the interaction "
            "target is 0.698 at 0.9, 0.542 at 6, 0.524 at 10, 0.516 at 16."
        ),
    )
    parser.add_argument(
        "--real-null", type=str, default=None, metavar="TARGET",
        help=(
            "Run the null on the REAL batch instead of a synthetic panel: "
            "real daily features, the named target permuted by date. Combine "
            "with --seeds to count how often the gate promotes nothing."
        ),
    )
    parser.add_argument("--real-features", type=int, default=120)
    parser.add_argument(
        "--family-size", type=int, default=None,
        help=(
            "Promotion attempts the run is treated as making. The gate turns "
            "it into the number of standard errors a margin must clear: 1 -> "
            "1.645, 27 -> 2.90, 216 -> 3.50. Omit to use the config value."
        ),
    )
    parser.add_argument(
        "--seeds", type=int, default=1,
        help=(
            "Repeat each case with this many different draws. With "
            "`--cases nothing` this MEASURES the gate's false-positive rate: "
            "the share of pure-noise panels it promotes. The rate cannot be "
            "derived -- the gate's bar is one standard error over the "
            "strongest of four opponents, and the maximum of four is a "
            "stricter test than any one of them by an amount that depends on "
            "how correlated they are. So it is counted instead."
        ),
    )
    args = parser.parse_args()

    globals()["N_NAMES"] = args.names
    globals()["N_BARS"] = args.bars
    print(
        f"panel: {args.names} names x {args.bars} bars = "
        f"{args.names*args.bars:,} rows, models: "
        + (", ".join(args.models) if args.models else "the configured suite"),
        flush=True,
    )

    if args.real_null:
        verdicts = []
        for offset in range(args.seeds):
            outcome = run_real_null(
                args.real_null, args.names, args.bars, args.real_features,
                SEED + offset, args.family_size, args.models,
            )
            verdicts.append(outcome)
            print(
                "\n".join([
                    f"\n=== {outcome['case']}  (seed {SEED + offset}) ===",
                    f"  rows / holdout   {outcome['rows']:,} / {outcome['holdout_rows']:,}",
                    f"  base rate        {outcome['base_rate']:.4f}",
                    f"  winner           {outcome['winner']}",
                    f"  holdout score    {outcome['winner_score']} "
                    f"({outcome['winner_metric']}, {outcome['holdout_status']})",
                    # Printed on every run, because the first version of this
                    # stand skipped this opponent in silence and the number it
                    # produced looked exactly like a number that had been
                    # measured against it.
                    f"  clock opponent   index={outcome['clock_index_present']} "
                    f"scored={outcome['clock_scored']} "
                    f"score={outcome['clock_score']} "
                    f"scheme={outcome['clock_scheme']}",
                    f"  opponents        constant={outcome['opponent_constant']} "
                    f"persistence={outcome['opponent_persistence']} "
                    f"clock={outcome['opponent_clock']} "
                    f"one_feature={outcome['opponent_one_feature']}",
                    f"  promoted         {outcome['promoted']}  (expected False)",
                    f"  reason           {outcome['reason']}",
                ]),
                flush=True,
            )
        promoted = sum(1 for v in verdicts if v["promoted"])
        print(
            f"\n{'=' * 60}\nreal-data null: promoted {promoted} of "
            f"{len(verdicts)} ({promoted / max(len(verdicts), 1):.1%})"
        )
        return 1 if promoted else 0

    verdicts = []
    for kind in args.cases:
      for noise_scale in args.noise:
       for offset in range(args.seeds):
        outcome = run_case(kind, args.models, noise_scale, SEED + offset,
                           args.family_size)
        outcome["noise"] = noise_scale
        outcome["seed"] = SEED + offset
        expected = EXPECTED[kind]
        outcome["expected_promoted"] = expected
        outcome["as_expected"] = outcome["promoted"] == expected
        verdicts.append(outcome)
        print(
            f"\n=== {kind} ===\n"
            f"  base rate        {outcome['base_rate']:.4f}\n"
            f"  winner           {outcome['winner']}\n"
            f"  holdout score    {outcome['winner_score']}\n"
            f"  promoted         {outcome['promoted']}  (expected {expected})\n"
            f"  reason           {outcome['reason']}"
        )

    # A refusal at high noise is CORRECT, not a failure: the planted edge is
    # then below what the sample can resolve, and the expectation baked into
    # EXPECTED only holds at the default noise.
    wrong = [f'{v["case"]}@noise{v["noise"]}' for v in verdicts
             if not v["as_expected"] and v["noise"] == NOISE]
    promoted = sum(1 for v in verdicts if v["promoted"])
    if len(verdicts) > 1:
        print(
            f"\npromoted {promoted} of {len(verdicts)} runs "
            f"({promoted / len(verdicts):.1%})"
        )
    print("\n" + "=" * 60)
    if wrong:
        print(f"THE GATE DISAGREES WITH THE PLANTED ANSWER ON: {', '.join(wrong)}")
        print("Until that is understood, its verdicts on real data cannot be read.")
        return 1
    print("Every planted answer came back as planted.")
    return 0



# --------------------------------------------------------------------------
# The real-data null
# --------------------------------------------------------------------------
#
# Everything above is synthetic: independent features, a clean null. Real
# features are autocorrelated in time and correlated with each other, and a
# model has far more structure to fit by accident. The 15% false-positive rate
# measured on the synthetic null (CLAIMS.md R11) may not be the rate here, and
# the direction of the difference is not obvious -- more structure helps the
# model AND strengthens the "best single column" opponent it must beat.
#
# So the same question is asked of the real batch: shuffle the target, keep
# the features, count how often the gate promotes.

BATCH = PROJECT_ROOT / "data" / "colab" / "accumulated" / "main_database"


def _real_panel(target: str, n_tickers: int, n_bars: int, n_features: int,
                rng: np.random.Generator) -> tuple[pd.DataFrame, str, list[str]]:
    """A daily slice of the real batch, with the target permuted BY DATE.

    Permuting whole dates rather than shuffling within each ticker is the
    point. Shuffling within a ticker would destroy the cross-sectional
    structure of the target as well as its link to the features, and a null
    with no market factor in it is easier than reality. Permuting the date
    labels keeps each day's cross-section intact -- the same names up together
    on the same day -- and breaks only what it must: the correspondence
    between a day's features and that day's outcome.

    The permuted target is returned AS A COLUMN of the frame, not as a
    separate array. It used to be an array, and every later subsetting had to
    re-index it by hand:

        frame = frame.groupby("ticker").tail(per_name).reset_index(drop=True)
        y = y[frame.index.to_numpy()] if len(y) == len(frame) else y[:len(frame)]

    `reset_index(drop=True)` had already thrown away the positions the first
    branch needed, so whenever the tickers had unequal bar counts the second
    branch took the FIRST rows of a target belonging to different rows
    entirely. On a null that direction is harmless -- it scrambles an already
    scrambled target -- but the base rate and the "cross-section preserved"
    claim were then both untrue, and the helper is one edit away from being
    pointed at a target that is not a null. A column cannot come apart from
    its own row.
    """
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(BATCH / "features.parquet").schema_arrow.names
    identity = ["ticker", "datetime", "interval"]
    candidates = [
        name for name in schema
        if name not in identity and not name.startswith("target_")
    ]
    chosen = sorted(rng.choice(candidates, size=min(n_features, len(candidates)),
                               replace=False).tolist())

    features = pd.read_parquet(BATCH / "features.parquet", columns=identity + chosen)
    features = features[features["interval"] == "1d"]
    targets = pd.read_parquet(BATCH / "targets.parquet",
                              columns=identity + [target])
    targets = targets[targets["interval"] == "1d"]

    frame = features.merge(targets[["ticker", "datetime", target]],
                           on=["ticker", "datetime"], how="inner")
    frame = frame.dropna(subset=[target])

    names = sorted(frame["ticker"].unique())[:n_tickers]
    frame = frame[frame["ticker"].isin(names)]
    frame = frame.sort_values(["ticker", "datetime"])
    frame = frame.groupby("ticker", sort=False).tail(n_bars).reset_index(drop=True)

    # Permute the date labels of the target panel.
    dates = frame["datetime"].drop_duplicates().to_numpy()
    permuted = rng.permutation(dates)
    remap = dict(zip(dates, permuted))
    shuffled = frame[["ticker", "datetime", target]].copy()
    shuffled["datetime"] = shuffled["datetime"].map(remap)
    merged = frame[["ticker", "datetime"]].merge(
        shuffled, on=["ticker", "datetime"], how="left", suffixes=("", "_y")
    )

    permuted_name = f"{target}__permuted"
    frame[permuted_name] = merged[target].to_numpy()
    frame = frame[np.isfinite(frame[permuted_name])].reset_index(drop=True)

    # A pooled frame reaches the pipeline in time order. Handing this one over
    # ticker-major would make `prepare_data_for_models` re-sort it and say so,
    # and the split under test would no longer be the split the pipeline
    # builds from its own inputs.
    frame = frame.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    return frame, permuted_name, chosen


def _walk(results: object, key: str, depth: int = 0):
    """First occurrence of `key` anywhere shallow in the result tree."""
    if depth > 4 or not isinstance(results, dict):
        return None
    if key in results:
        return results[key]
    for value in results.values():
        found = _walk(value, key, depth + 1)
        if found is not None:
            return found
    return None


def run_real_null(target: str, n_tickers: int, n_bars: int, n_features: int,
                  seed: int, family_size: int | None,
                  models: list[str] | None = None) -> dict:
    """The null, through the path the pipeline actually uses.

    THIS IS THE FIX THAT MATTERS HERE. The first version of this stand built
    its own train/val/test/holdout split with `_split`, which hands the trainer
    plain frames whose index is 0..n-1. The gate's clock opponent asks for a
    `DatetimeIndex`:

        train_index = getattr(data.get('X_train'), 'index', None)
        if not isinstance(train_index, pd.DatetimeIndex):
            return                      # base_trainer._clock_prediction

    so on this stand that rung returned nothing and the gate was scored against
    three opponents instead of four -- quietly, because a rung that does not
    run and a rung that is passed look identical from outside. A null measured
    against a weaker gate than the real one UNDERSTATES the false-positive
    rate, which is the single direction that would have made the retraction of
    R11's headline look better than it deserved.

    So the frame now goes through `prepare_data_for_models` -- which attaches
    the `model_datetime` index, applies the purge gap, fits the imputer and
    scaler on training rows only, and decides the split -- and then through
    `ModelingStage._build_unified_training_context`, the same adapter Stage 4
    uses to hand that split to the trainer. Nothing about the split is written
    twice here, so the stand cannot drift away from the pipeline it calibrates.
    """
    rng = np.random.default_rng(seed)
    frame, target_col, features = _real_panel(
        target, n_tickers, n_bars, n_features, rng
    )
    empty = {"case": f"real-null:{target}", "promoted": False, "winner": None,
             "winner_score": None, "winner_metric": None, "rows": int(len(frame)),
             "holdout_rows": 0, "clock_index_present": False,
             "clock_scored": False, "clock_score": None, "clock_scheme": None,
             "base_rate": float("nan")}
    if len(frame) < 500:
        return {**empty, "reason": f"only {len(frame)} usable rows",
                "holdout_status": "too_few_rows"}

    prepared = prepare_data_for_models(frame, POOLED_TICKER, "1d", [target_col])
    if not prepared:
        return {**empty, "reason": "prepare_data_for_models returned nothing",
                "holdout_status": "not_prepared"}

    stage = ModelingStage.__new__(ModelingStage)
    stage._promotion_family_size = family_size
    data = stage._build_unified_training_context(
        prepared,
        target_name=f"target_realnull_{target}",
        context_fingerprint=f"realnull::{target}::seed{seed}",
        timeframe="1d",
    )
    plan = _plan(POOLED_TICKER, models)
    if plan:
        data["plan"] = plan

    index = getattr(data.get("X_train"), "index", None)
    clock_possible = isinstance(index, pd.DatetimeIndex)

    trainer = BatchTrainer()
    results = trainer._train_ticker_suite(POOLED_TICKER, data)
    gate = results.get("promotion_gate") or {}
    holdout = results.get("winner_holdout_metrics") or {}
    clock = _walk(results, "baseline_clock_score")
    return {
        "case": f"real-null:{target}",
        "rows": int(len(frame)),
        "holdout_rows": int(len(data.get("X_holdout") if data.get("X_holdout") is not None else [])),
        "base_rate": float(np.mean(np.asarray(data["y_holdout"], dtype=float))),
        "winner": results.get("winner"),
        "winner_score": holdout.get("score"),
        "winner_metric": holdout.get("metric"),
        "holdout_status": holdout.get("status"),
        # The whole reason for this rewrite: say whether the fourth opponent
        # was scored, instead of leaving a silent absence to read as a pass.
        "clock_index_present": clock_possible,
        "clock_scored": clock is not None,
        "clock_score": clock,
        "clock_scheme": _walk(results, "baseline_clock_scheme"),
        # All four opponents, kept per run. "Did it promote?" answers a yes/no
        # that is nearly always no at a 2.90-sigma bar; the informative number
        # is HOW CLOSE the winner came to the strongest opponent, and over
        # several seeds that distribution says what the bar is actually worth.
        "opponent_constant": holdout.get("baseline_constant_score"),
        "opponent_persistence": holdout.get("baseline_persistence_score"),
        "opponent_clock": holdout.get("baseline_clock_score"),
        "opponent_one_feature": holdout.get("single_feature_score"),
        "promoted": bool(gate.get("passed")),
        "reason": "; ".join(gate.get("reasons") or []) or None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
