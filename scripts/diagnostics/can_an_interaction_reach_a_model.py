"""The falsifier for CLAIMS R14: put an interaction in a wide frame and see if it survives.

R14 says the pipeline narrows its features TWICE and both times by the same
marginal statistic -- |Pearson correlation| with the target on training rows:

    _target_correlation_ranking   467 columns -> ceiling 70
    BaseTrainer._select_features_for_model     70    -> model budget 5-35

and that a pure interaction cannot pass either, because the columns carrying it
are marginally indistinguishable from noise. Measured on R10's own panel, the
two columns that FULLY determine `y = 1{f0*f1 + noise > 0}` ranked 7th and 14th
of 20 by |corr|, below a pure-noise column at 0.0098.

That measurement had twenty columns and a ceiling of seventy, so nothing was
ever dropped -- which is exactly why R10 case B promotes and why R10 has never
tested this. The claim about the PIPELINE was therefore an inference from the
ranking, not an observation of the pipeline.

This runs the experiment the claim asked for. Four hundred pure-noise columns
are added to the same panel, the frame goes through the real
`prepare_data_for_models`, and the real `_select_features_for_model` then applies each
model's real budget. Two questions get a number:

    does the pair survive the pre-screen?
    does the pair survive the per-model budget?

R14 IS REFUTED if the interaction reaches a model at a rate that makes the
pipeline's blindness a non-issue. It is CONFIRMED if the pair is dropped at
roughly the rate that treating its rank as random among the noise implies --
about 2% at the ceiling, and about a quarter of that again at a budget of 35.

Nothing is trained here on purpose. Whether the GATE can recognise an
interaction is already known (R10 case B, promoted before and after the clock
fix on 2026-09-02). The open question is whether the pipeline ever hands one
over, and that is decided before a model is fitted.

    python scripts/diagnostics/can_an_interaction_reach_a_model.py
    python scripts/diagnostics/can_an_interaction_reach_a_model.py --seeds 20 --noise-columns 800
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config.feature_budget import (  # noqa: E402
    get_model_max_features,
    get_preselection_ceiling,
)
from src.models.adapters.data_preparation import (  # noqa: E402
    prepare_data_for_models,
)
from src.core.logging.logger import ProjectLogger  # noqa: E402
from src.pipeline.modeling_context import POOLED_TICKER  # noqa: E402
from src.training.batch_trainer import BatchTrainer  # noqa: E402

N_NAMES = 40
N_BARS = 1500
N_REAL = 20
NOISE = 0.9
CARRIERS = ("f0", "f1")
TARGET = "target_planted_interaction_1d"


def _ranker() -> BatchTrainer:
    """The real `_select_features_for_model`, without starting a trainer.

    `BatchTrainer()` opens the DuckDB file for its diary, and DuckDB allows one
    writer, so building one here fails outright whenever a measurement run
    holds the database -- which is exactly when this script is most likely to
    be used. The method under test needs two attributes and touches nothing
    else, so it is given those two.

    This is not a reimplementation: `_select_features_for_model` is the pipeline's own
    code, called unchanged. What is avoided is the constructor's unrelated
    side effect.
    """
    ranker = BatchTrainer.__new__(BatchTrainer)
    ranker.config_manager = None
    ranker.logger = ProjectLogger.get_logger("InteractionReach")
    return ranker


def _panel(rng: np.random.Generator, noise_columns: int) -> pd.DataFrame:
    """R10's case-B panel, widened with pure noise.

    The carriers keep their names and their meaning; everything added is
    independent of the target by construction, so any column that outranks
    them does so by chance and the comparison is honest.
    """
    rows = N_NAMES * N_BARS
    total = N_REAL + noise_columns
    frame = pd.DataFrame(
        rng.standard_normal((rows, total)).astype("float32"),
        columns=[f"f{i}" for i in range(total)],
    )
    signal = frame["f0"].to_numpy() * frame["f1"].to_numpy()
    frame[TARGET] = (signal + rng.standard_normal(rows) * NOISE > 0).astype(float)

    frame["ticker"] = np.repeat([f"T{i:02d}" for i in range(N_NAMES)], N_BARS)
    calendar = pd.bdate_range("2010-01-04", periods=N_BARS, tz="UTC")
    frame["datetime"] = np.tile(calendar.to_numpy(), N_NAMES)
    frame["interval"] = "1d"
    return frame.sort_values(["datetime", "ticker"]).reset_index(drop=True)


def _ranks(frame: pd.DataFrame) -> dict[str, int]:
    features = [c for c in frame.columns if c.startswith("f")]
    corr = frame[features].corrwith(frame[TARGET]).abs()
    order = corr.sort_values(ascending=False, kind="mergesort").index
    position = {name: i + 1 for i, name in enumerate(order)}
    return {name: position[name] for name in CARRIERS}


def run_once(seed: int, noise_columns: int, models: list[str]) -> dict:
    rng = np.random.default_rng(seed)
    frame = _panel(rng, noise_columns)
    total_features = N_REAL + noise_columns
    ranks = _ranks(frame)

    prepared = prepare_data_for_models(frame, POOLED_TICKER, "1d", [TARGET])
    if not prepared:
        return {"seed": seed, "prepared": False}

    light = prepared["light_models"]
    survivors = list(light.get("feature_names") or [])
    after_prescreen = {name: name in survivors for name in CARRIERS}

    trainer = _ranker()
    data = {"X_train": light["X_train"], "y_train": light["y_train"]}
    after_budget = {}
    for model in models:
        chosen = trainer._select_features_for_model(model, data, True) or []
        after_budget[model] = all(name in chosen for name in CARRIERS)

    return {
        "seed": seed,
        "prepared": True,
        "total_features": total_features,
        "ranks": ranks,
        "kept_by_prescreen": len(survivors),
        "pair_survives_prescreen": all(after_prescreen.values()),
        "carrier_prescreen": after_prescreen,
        "pair_survives_budget": after_budget,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--noise-columns", type=int, default=400)
    parser.add_argument("--models", nargs="+",
                        default=["linear", "random_forest", "lightgbm"])
    args = parser.parse_args()

    ceiling = get_preselection_ceiling()
    budgets = {m: get_model_max_features(m, None) for m in args.models}
    total = N_REAL + args.noise_columns
    print(f"panel: {N_NAMES} names x {N_BARS} bars, "
          f"{total} feature columns ({args.noise_columns} pure noise)")
    print(f"pre-screen ceiling: {ceiling}   model budgets: {budgets}")
    print(f"if the pair's rank were random among {total}, both would clear the "
          f"ceiling with probability {(ceiling/total)*((ceiling-1)/(total-1)):.2%}\n")

    outcomes = []
    for offset in range(args.seeds):
        result = run_once(20260901 + offset, args.noise_columns, args.models)
        outcomes.append(result)
        if not result.get("prepared"):
            print(f"seed {result['seed']}: preparation returned nothing")
            continue
        ranks = result["ranks"]
        print(
            f"seed {result['seed']}: "
            f"f0 rank {ranks['f0']:>4}, f1 rank {ranks['f1']:>4} of {total}  |  "
            f"pre-screen kept {result['kept_by_prescreen']:>3}, "
            f"pair survives: {result['pair_survives_prescreen']}  |  "
            f"budget: {result['pair_survives_budget']}",
            flush=True,
        )

    usable = [o for o in outcomes if o.get("prepared")]
    if not usable:
        print("\nnothing measured.")
        return 1

    prescreen = sum(o["pair_survives_prescreen"] for o in usable)
    print(f"\npair survived the pre-screen in {prescreen} of {len(usable)} runs")
    for model in args.models:
        reached = sum(o["pair_survives_budget"].get(model, False) for o in usable)
        print(f"pair reached {model:<15} in {reached} of {len(usable)} runs")

    print("\n" + "=" * 70)
    if prescreen == 0:
        print("R14 CONFIRMED: the interaction never reached a model. The gate's "
              "ability to recognise one (R10 case B) is not the binding "
              "constraint -- the feature ranking is.")
    elif prescreen == len(usable):
        print("R14 REFUTED: the pair survived every time. The ranking argument "
              "does not describe what the pipeline does; rewrite the claim.")
    else:
        chance = (ceiling / total) * ((ceiling - 1) / (total - 1))
        print(f"R14 PARTLY HELD: {prescreen}/{len(usable)} survived. The "
              f"comparison that decides whether that is skill or luck is the "
              f"PAIR's chance of clearing the ceiling, {chance:.2%} -- not the "
              f"per-column {ceiling/total:.1%}, which is the mistake this line "
              f"used to make. P(at least one survivor in {len(usable)} runs by "
              f"chance alone) = {1 - (1 - chance) ** len(usable):.1%}.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
