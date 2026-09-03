"""If the pipeline ranked features some other way, would an interaction survive?

CLAIMS R14, confirmed 2026-09-02: the pipeline narrows features twice and both
times by |Pearson correlation| with the target on training rows. Measured on a
420-column panel whose target is `y = 1{f0*f1 + noise > 0}`, the two columns
that FULLY determine it reached the pre-screen in 1 run of 10 -- the rate
chance alone predicts (2.74%), with their ranks indistinguishable from uniform
(KS D = 0.256, p = 0.121). For models with a five-feature budget: 0 of 10.

That says what the current ranking cannot do. It does not say whether ANY
affordable ranking could, and the difference decides what to do about it:

  * if a joint-effect ranking recovers the pair, R14 is a fixable defect and
    the remaining question is what it costs in memory and time;
  * if nothing affordable recovers it, R14 is a limit of the approach rather
    than a defect, and the honest response is to stop expecting interactions
    rather than to keep looking for them with an instrument that cannot see.

So the same panel is ranked four ways and each is timed. Nothing in the
pipeline changes here -- this measures candidates, it does not adopt one.

    pearson       what the pipeline does now (`_target_correlation_ranking`,
                  `BaseTrainer._select_features_for_model`)
    mutual_info   the usual first suggestion. Included to be REFUTED as much
                  as tested: for a symmetric pure interaction, MI(f0; y) is
                  about zero for the same reason the correlation is, so if it
                  ranks the pair no better than chance that is a result worth
                  having written down rather than asserted from theory.
    forest        RandomForest importances, shallow. A tree splits on f0 and
                  then on f1 inside that branch, so the pair is visible to it
                  in a way no per-column statistic reaches.
    extra_trees   ExtraTrees importances. Randomised split points make it
                  cheaper than a forest and often better at exactly this.

COST IS PART OF THE ANSWER, so every channel is timed. The panel here is
60,000 rows x 420 columns; the real daily frame is 490,799 x 467, about eight
times the rows. A channel that takes a minute here takes roughly eight there,
once per context -- and the budget this would replace exists because the
median imputer died three times in two days on that frame. A ranking that
recovers the pair and costs an hour per context has not solved anything.

    python scripts/diagnostics/which_ranking_can_see_an_interaction.py
    python scripts/diagnostics/which_ranking_can_see_an_interaction.py --seeds 5 --channels pearson forest
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config.feature_budget import get_preselection_ceiling  # noqa: E402

N_NAMES = 40
N_BARS = 1500
N_REAL = 20
NOISE = 0.9
CARRIERS = ("f0", "f1")
TRAIN_SHARE = 0.6

#: Mutual information is estimated with a k-nearest-neighbour method whose cost
#: grows with rows AND columns; on the full panel it is minutes per seed. It is
#: given a subsample and the subsample is reported, because a channel that only
#: works on a tenth of the data is a different proposal from one that does not.
MI_ROWS = 10_000


def _panel(rng: np.random.Generator, noise_columns: int) -> tuple[pd.DataFrame, np.ndarray]:
    rows = N_NAMES * N_BARS
    total = N_REAL + noise_columns
    frame = pd.DataFrame(
        rng.standard_normal((rows, total)).astype("float32"),
        columns=[f"f{i}" for i in range(total)],
    )
    signal = frame["f0"].to_numpy() * frame["f1"].to_numpy()
    y = (signal + rng.standard_normal(rows) * NOISE > 0).astype(float)
    return frame, y


def _rank_pearson(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    return X.corrwith(pd.Series(y, index=X.index)).abs().fillna(-1.0)


def _rank_mutual_info(X: pd.DataFrame, y: np.ndarray, rng) -> pd.Series:
    from sklearn.feature_selection import mutual_info_classif

    take = min(MI_ROWS, len(X))
    rows = rng.choice(len(X), size=take, replace=False)
    scores = mutual_info_classif(
        X.to_numpy()[rows], y[rows], random_state=0, n_neighbors=3,
    )
    return pd.Series(scores, index=X.columns)


def _rank_forest(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=100, max_depth=6, n_jobs=-1, random_state=0,
    ).fit(X.to_numpy(), y)
    return pd.Series(model.feature_importances_, index=X.columns)


def _rank_extra_trees(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    from sklearn.ensemble import ExtraTreesClassifier

    model = ExtraTreesClassifier(
        n_estimators=100, max_depth=6, n_jobs=-1, random_state=0,
    ).fit(X.to_numpy(), y)
    return pd.Series(model.feature_importances_, index=X.columns)


def _rank_forest_deep(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """The forest, configured for the job instead of for the default.

    A shallow forest with the default `max_features=sqrt(p)` is close to the
    worst possible detector for this. To split on f0 and then on f1 inside that
    branch, a tree has to be offered both -- and with 420 columns it sees about
    20 candidates per split and stops at depth 6. Measured with those defaults:
    the pair reached the top 70 in 0 of 2 seeds, though the median carrier rank
    did halve against |corr| (112 against 215), which is the signal that the
    configuration and not the method was the limit.

    So: more trees, deep enough for a two-level path to exist, and a third of
    the columns offered at every split. This is the fair version of the
    question "can a joint-effect ranking see it", and it is also the expensive
    one -- which is the point, because cost is half the answer.
    """
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, max_features=0.3,
        min_samples_leaf=50, n_jobs=-1, random_state=0,
    ).fit(X.to_numpy(), y)
    return pd.Series(model.feature_importances_, index=X.columns)


def _rank_lightgbm(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Boosted trees, gain importance.

    Boosting fits each tree to what the previous ones got wrong, so a pair that
    is useless alone and decisive together tends to surface once the marginal
    structure is exhausted -- and on a target with NO marginal structure that
    is from the first tree. Included because if anything cheap can see this,
    it is this.
    """
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=63, learning_rate=0.05,
        min_child_samples=50, n_jobs=-1, random_state=0, verbose=-1,
    ).fit(X.to_numpy(), y)
    return pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=X.columns,
    )


CHANNELS = {
    "pearson": lambda X, y, rng: _rank_pearson(X, y),
    "mutual_info": lambda X, y, rng: _rank_mutual_info(X, y, rng),
    "forest": lambda X, y, rng: _rank_forest(X, y),
    "extra_trees": lambda X, y, rng: _rank_extra_trees(X, y),
    "forest_deep": lambda X, y, rng: _rank_forest_deep(X, y),
    "lightgbm": lambda X, y, rng: _rank_lightgbm(X, y),
}


# ---------------------------------------------------------------------------
# The falsifier for R15 (REGISTER #233): weakness, and competition.
#
# R15 measured boosting against ONE strong interaction hidden in pure noise:
# Bayes balanced accuracy 0.698, and every other column independent of the
# target. Neither condition holds in this project's data. Real edges are an
# order of magnitude weaker, and real frames are full of columns with genuine
# marginal association -- momentum, volatility, the macro block -- which is
# exactly what a gain ranking spends itself on first.
#
# So the panel below adds both. The target is built from stated VARIANCE
# SHARES so that "weak" is a number rather than an adjective:
#
#     z = sqrt(s_I)*I + sqrt(s_M)*M + sqrt(1 - s_I - s_M)*e ,   y = 1{z > 0}
#
#     I  the standardised interaction f0*f1
#     M  the standardised sum of `competitors` genuinely predictive columns
#     e  independent noise
#
# With s_M = 0.30 spread over 100 columns, each competitor carries 0.3% of the
# variance on its own. Sweeping s_I down through that value asks the question
# that decides #233: does the interaction still rank when it is no stronger
# than any single ordinary column that a marginal statistic CAN see?
#
# R15 SURVIVES if boosting keeps the pair inside the ceiling while the
# interaction is worth having. It is REFUTED, and #233 with it, if the pair
# ranks only when it dominates -- because an interaction that has to dominate
# to be found is one the current |corr| ranking would nearly find anyway.


def _panel_with_competition(
    rng: np.random.Generator, noise_columns: int, competitors: int,
    interaction_share: float, competitor_share: float,
) -> tuple[pd.DataFrame, np.ndarray, list[str], float]:
    """The panel, plus columns that legitimately deserve a high rank.

    The competitors are not decoration and not noise: they are IN the target,
    so a ranking that puts them first is right to. The question is whether the
    two interaction carriers still surface from underneath them.

    Returns the frame, the target, the competitor names, and the best
    balanced accuracy any predictor could reach on this target -- so "weak" is
    reported as a number the rest of the project already uses.
    """
    rows = N_NAMES * N_BARS
    total = N_REAL + noise_columns
    frame = pd.DataFrame(
        rng.standard_normal((rows, total)).astype("float32"),
        columns=[f"f{i}" for i in range(total)],
    )

    def _unit(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return (values - values.mean()) / values.std()

    interaction = _unit(frame["f0"].to_numpy() * frame["f1"].to_numpy())

    # Competitors are drawn from the far end so they never collide with the
    # carriers, and their weights vary so the ranking has a real ordering to
    # find rather than a hundred ties.
    names = [f"f{i}" for i in range(total - competitors, total)]
    weights = rng.normal(0, 1, len(names))
    marginal = _unit(frame[names].to_numpy() @ weights) if names else np.zeros(rows)

    residual = max(0.0, 1.0 - interaction_share - competitor_share)
    z = (
        np.sqrt(interaction_share) * interaction
        + np.sqrt(competitor_share) * marginal
        + np.sqrt(residual) * rng.standard_normal(rows)
    )
    y = (z > 0).astype(float)

    # The best any predictor could do: the signal without the noise.
    ideal = (
        np.sqrt(interaction_share) * interaction
        + np.sqrt(competitor_share) * marginal
    ) > 0
    tp = float(((ideal == 1) & (y == 1)).sum()) / max(float((y == 1).sum()), 1.0)
    tn = float(((ideal == 0) & (y == 0)).sum()) / max(float((y == 0).sum()), 1.0)
    return frame, y, names, (tp + tn) / 2.0


def falsify(args) -> int:
    ceiling = get_preselection_ceiling()
    total = N_REAL + args.noise_columns
    chance = (ceiling / total) * ((ceiling - 1) / (total - 1))
    print(f"panel: {N_NAMES*N_BARS:,} rows, {total} columns, of which "
          f"{args.competitors} carry a genuine MARGINAL signal "
          f"({args.competitor_share:.0%} of the variance between them, "
          f"{args.competitor_share/max(args.competitors,1):.2%} each)")
    print(f"the pair clears the top {ceiling} by chance with probability "
          f"{chance:.2%}\n")

    header = (f"{'interaction share':>18}{'best BalAcc':>13}{'seed':>11}"
              f"{'pearson ranks':>20}{'lightgbm ranks':>20}"
              f"{'lgbm top' + str(ceiling):>14}{'competitors found':>19}")
    print(header)
    print("-" * len(header))

    summary = {}
    for share in args.interaction_share:
        hits, accs = 0, []
        for offset in range(args.seeds):
            rng = np.random.default_rng(20260901 + offset)
            frame, y, competitors, best = _panel_with_competition(
                rng, args.noise_columns, args.competitors, share,
                args.competitor_share,
            )
            accs.append(best)
            cut = int(len(frame) * TRAIN_SHARE)
            X_tr, y_tr = frame.iloc[:cut], y[:cut]

            def _pos(scores):
                order = scores.sort_values(ascending=False, kind="mergesort").index
                return {col: i + 1 for i, col in enumerate(order)}

            p_pos = _pos(_rank_pearson(X_tr, y_tr))
            g_pos = _pos(_rank_lightgbm(X_tr, y_tr))
            p_pair = [p_pos[c] for c in CARRIERS]
            g_pair = [g_pos[c] for c in CARRIERS]
            both = all(r <= ceiling for r in g_pair)
            hits += bool(both)
            found = sum(1 for c in competitors if g_pos[c] <= ceiling)

            print(f"{share:>18.3f}{best:>13.3f}{20260901+offset:>11}"
                  f"{str(p_pair):>20}{str(g_pair):>20}"
                  f"{('YES' if both else 'no'):>14}"
                  f"{f'{found}/{len(competitors)}':>19}", flush=True)
        summary[share] = (hits, float(np.mean(accs)))

    print(f"\n{'interaction share':>18}{'best BalAcc':>13}"
          f"{'pair in top ' + str(ceiling):>22}")
    print("-" * 53)
    for share, (hits, acc) in summary.items():
        print(f"{share:>18.3f}{acc:>13.3f}{f'{hits} of {args.seeds}':>22}")

    print("\n" + "=" * 78)
    weakest_found = [s for s, (h, _) in summary.items() if h == args.seeds]
    if not weakest_found:
        print("R15 REFUTED under competition: boosting did not reliably recover "
              "the pair at any strength tested. Proposal #233 buys nothing and "
              "should not be built.")
    else:
        floor = min(weakest_found)
        acc = summary[floor][1]
        print(f"R15 HOLDS down to an interaction share of {floor:.3f} "
              f"(best achievable balanced accuracy {acc:.3f}), where boosting "
              f"found the pair in every seed. Below that it does not. Whether "
              f"#233 is worth building is now a question about whether edges "
              f"of that size are worth having -- which CLAIMS R8 answers "
              f"separately, per cadence.")
    print("=" * 78)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--noise-columns", type=int, default=400)
    parser.add_argument("--channels", nargs="+", default=list(CHANNELS))
    parser.add_argument(
        "--falsify", action="store_true",
        help=(
            "Run the R15 falsifier instead: a weak interaction competing "
            "against columns that genuinely predict the target."
        ),
    )
    parser.add_argument("--competitors", type=int, default=100)
    parser.add_argument("--competitor-share", type=float, default=0.30)
    parser.add_argument("--interaction-share", type=float, nargs="+",
                        default=[0.20, 0.05, 0.01, 0.003])
    args = parser.parse_args()

    if args.falsify:
        return falsify(args)

    ceiling = get_preselection_ceiling()
    total = N_REAL + args.noise_columns
    chance = (ceiling / total) * ((ceiling - 1) / (total - 1))
    print(f"panel: {N_NAMES} x {N_BARS} = {N_NAMES*N_BARS:,} rows, {total} columns "
          f"({args.noise_columns} pure noise); ranking on the first "
          f"{TRAIN_SHARE:.0%} by time")
    print(f"target: y = 1[f0*f1 + noise > 0]   ceiling: top {ceiling}")
    print(f"the pair clears the ceiling by chance with probability {chance:.2%}\n")

    survived = {name: 0 for name in args.channels}
    seconds = {name: 0.0 for name in args.channels}
    ranks = {name: [] for name in args.channels}

    for offset in range(args.seeds):
        seed = 20260901 + offset
        rng = np.random.default_rng(seed)
        frame, y = _panel(rng, args.noise_columns)
        cut = int(len(frame) * TRAIN_SHARE)
        X_tr, y_tr = frame.iloc[:cut], y[:cut]

        line = [f"seed {seed}:"]
        for name in args.channels:
            started = time.perf_counter()
            scores = CHANNELS[name](X_tr, y_tr, rng)
            elapsed = time.perf_counter() - started
            seconds[name] += elapsed

            order = scores.sort_values(ascending=False, kind="mergesort").index
            position = {col: i + 1 for i, col in enumerate(order)}
            pair = [position[c] for c in CARRIERS]
            ranks[name].extend(pair)
            both = all(r <= ceiling for r in pair)
            survived[name] += bool(both)
            line.append(
                f"  {name:<12} ranks {pair[0]:>4},{pair[1]:>4}"
                f"  top{ceiling}: {'YES' if both else 'no ':<3}  {elapsed:6.1f}s"
            )
        print("\n".join(line), flush=True)

    print(f"\n{'channel':<14}{'pair in top ' + str(ceiling):>18}{'median rank':>14}"
          f"{'mean s/seed':>14}{'est. s on real frame':>22}")
    print("-" * 82)
    for name in args.channels:
        median = float(np.median(ranks[name]))
        per_seed = seconds[name] / max(args.seeds, 1)
        # The real daily frame is 490,799 x 467 against this panel's 60,000 x
        # 420: about 8x the rows. Linear in rows is optimistic for the tree
        # channels and hopeless for mutual information, so this is a FLOOR.
        print(f"{name:<14}{survived[name]:>10} of {args.seeds:<6}{median:>14.1f}"
              f"{per_seed:>14.1f}{per_seed * 8.2:>20.0f}+")

    print("\n" + "=" * 82)
    best = max(args.channels, key=lambda n: survived[n])
    if survived[best] == 0:
        print("No channel recovered the pair. R14 is a limit of the approach, "
              "not a defect to patch: stop expecting interactions from this "
              "machine rather than looking for them with an instrument that "
              "cannot see them.")
    elif survived.get("pearson", -1) >= survived[best]:
        print("Nothing beat the statistic already in use. R14 stands and no "
              "cheap replacement is indicated by this measurement.")
    else:
        print(f"'{best}' recovered the pair {survived[best]} of {args.seeds} "
              f"times against pearson's {survived.get('pearson', 'n/a')}. R14 is fixable; "
              f"what remains is the cost column above, which has to be paid "
              f"once per context on a frame eight times this size.")
    print("=" * 82)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
