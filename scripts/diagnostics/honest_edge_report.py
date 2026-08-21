"""Is there an edge, and how much of it is the threshold peeking at the answer?

Stage 7's equity curve is built from Stage 5, which predicts ONE bar per
context, so no financial figure from it can mean anything. This report takes
the other route, which needs nothing finished: the trainer retains every
champion's holdout predictions -- bars the model never saw and was never
selected on -- and each one can be joined to the return that ACTUALLY followed
the same bar. That join is the whole link between "breakout predicted" and
money.

Two things this exists to stop.

FIRST, THE THRESHOLD LEAK. Every figure this project quoted before 2026-08-17
picked the firing threshold on the same rows it then scored. With ten
candidates and a few hundred trades that alone can manufacture an edge. So the
holdout is split in TIME -- threshold chosen on the earlier half, scored on the
later half, with a purge gap so the target horizon cannot straddle the
boundary. The gap between the two arms IS the optimism, stated rather than
assumed. A third arm fixes the threshold in advance and makes no choice at all;
when it beats the chosen arm, the selection was fitting noise.

SECOND, THE COST ASSUMPTION. The break-even for the hourly models sits at
5-10 bp, and the effective spread is the one term our OHLC bars cannot pin
down: Corwin-Schultz and Abdi-Ranaldo bracket it from ~0 to ~19 bp on the same
rows. So every number is reported under BOTH a pessimistic spread estimated
from the bars and the assumption shipped in targets.yaml. If the two disagree
on the sign, that is the finding, and no amount of modelling settles it -- only
a fill price from a real account does.

    python scripts/diagnostics/honest_edge_report.py
    python scripts/diagnostics/honest_edge_report.py --signal target_hourly_breakout_1h

Commission comes from `cost_profiles` in targets.yaml so this cannot drift away
from what the targets themselves were built with.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.targets.calculators.regression_calculator import RegressionCalculator  # noqa: E402

TARGETS_YAML = Path('src/config/targets.yaml')
TARGETS_PARQUET = Path('data/colab/accumulated/main_database/targets.parquet')
DB = Path('data/trading_data.duckdb')

DEFAULT_SIGNAL = 'target_hourly_breakout_1h'
DEFAULT_PAYOFF = 'target_hourly_return_1h'
DEFAULT_INTERVAL, DEFAULT_DB_INTERVAL = '60m', '1h'

THRESHOLDS = np.round(np.arange(0.50, 0.99, 0.05), 2)
PURGE_BARS = 8
MIN_TRADES = 20
K_CS = 3 - 2 * np.sqrt(2)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def latest_predictions() -> Path | None:
    hits = sorted(glob.glob('data/results/holdout_predictions_*.parquet'))
    return Path(hits[-1]) if hits else None


def cost_profile(name: str = 'ibkr_pro_tiered') -> dict:
    raw = yaml.safe_load(TARGETS_YAML.read_text(encoding='utf-8')) or {}
    profiles = raw.get('cost_profiles') or {}
    if name not in profiles:
        raise SystemExit(f'no cost profile {name!r} in {TARGETS_YAML}')
    return profiles[name]


def naive_utc(values) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors='coerce').dt.tz_localize(None)


def corwin_schultz(high: np.ndarray, low: np.ndarray) -> float:
    """Median two-bar high-low spread estimate, as a fraction of price.

    Known to be biased UPWARD on short intraday bars because volatility leaks
    into it. That is why it is used here as the PESSIMISTIC bound and never as
    the measurement.
    """
    if len(high) < 50:
        return np.nan
    beta = np.log(high[:-1] / low[:-1]) ** 2 + np.log(high[1:] / low[1:]) ** 2
    gamma = np.log(np.maximum(high[:-1], high[1:]) /
                   np.minimum(low[:-1], low[1:])) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / K_CS - np.sqrt(gamma / K_CS)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return float(np.nanmedian(np.clip(spread, 0, None)))


def commission_fraction(price: pd.Series, profile: dict) -> pd.Series:
    """Round-trip commission as a fraction of position value, per row.

    Deliberately NOT a second implementation of the per-share arithmetic. This
    project's most repeated defect is one decision living in two places, where
    a fix lands in one copy and the other goes on being wrong. So the report
    calls the same method the targets themselves are built with, with the
    spread and slippage terms zeroed out to leave commission alone.
    """
    commission_only = {**profile, 'spread_pct': 0.0, 'slippage_pct': 0.0}
    cost = RegressionCalculator._round_trip_cost(price, commission_only)
    if not hasattr(cost, '__len__'):
        return pd.Series(float(cost), index=price.index)
    return cost


def load(signal: str, payoff: str, interval: str, db_interval: str,
         profile: dict, built_with: dict) -> pd.DataFrame:
    pred_path = latest_predictions()
    if pred_path is None:
        raise SystemExit('no data/results/holdout_predictions_*.parquet found')
    print(f'predictions: {pred_path}')

    preds = pd.read_parquet(pred_path)
    preds = preds[preds['target'] == signal].copy()
    if preds.empty:
        raise SystemExit(f'no holdout predictions for {signal}')
    preds['datetime'] = naive_utc(preds['datetime'])

    tgt = pd.read_parquet(TARGETS_PARQUET,
                          columns=['ticker', 'datetime', 'interval', payoff])
    tgt = tgt[tgt['interval'] == interval].copy()
    tgt['datetime'] = naive_utc(tgt['datetime'])

    df = preds.merge(tgt[['ticker', 'datetime', payoff]],
                     on=['ticker', 'datetime'], how='left')
    joined = df[payoff].notna()
    print(f'holdout predictions for {signal}: {len(preds):,}')
    print(f'joined to a realised return:     {int(joined.sum()):,} '
          f'({joined.mean():.1%})')
    df = df[joined].copy()

    con = duckdb.connect(str(DB), read_only=True)
    bars = con.execute(
        'select ticker, datetime, high, low, close from market_data_raw '
        'where interval = ?', [db_interval]).df()
    bars['datetime'] = naive_utc(bars['datetime'])

    spreads = {}
    for ticker, g in bars.sort_values('datetime').groupby('ticker'):
        spreads[ticker] = corwin_schultz(g.high.values, g.low.values)
    spread_series = pd.Series(spreads).dropna()

    df = df.merge(bars[['ticker', 'datetime', 'close']],
                  on=['ticker', 'datetime'], how='left')
    df['close'] = df['close'].fillna(
        df.groupby('ticker')['close'].transform('median'))
    df = df[df['close'].notna()].copy()

    df['spread_pessimistic'] = df['ticker'].map(spread_series).fillna(
        spread_series.median())
    # spread_pct in the config is PER SIDE, so a round trip is twice it
    df['spread_assumed'] = 2 * float(profile.get('spread_pct', 0.0)) \
        + 2 * float(profile.get('slippage_pct', 0.0))
    df['commission'] = commission_fraction(df['close'], profile).values

    # The payoff target already had a round trip subtracted when it was BUILT,
    # and that is not necessarily the profile we are charging now. On the first
    # run of this report the two differed by a factor of ten -- the parquet on
    # disk predates the 2026-08-17 cost change -- and reading the current
    # config for the add-back turned a +0.00010 mean into -0.00442. Hence a
    # separate `--built-with`: an artifact carries the assumptions of the run
    # that produced it, never the ones in the config file today.
    add_back = cost_in_target(built_with, df)
    df['gross'] = df[payoff].astype(float) + add_back
    df['prob'] = df['probability'].astype(float)
    print(f'payoff was built with {2 * float(built_with.get("commission_pct", 0)):.4f}'
          f' flat commission; adding back {float(np.mean(add_back)) * 1e4:.2f} bp')
    return df.sort_values('datetime').reset_index(drop=True)


def cost_in_target(profile: dict, df: pd.DataFrame) -> pd.Series:
    """Whatever regression_calculator took out of the payoff target."""
    if str(profile.get('model', 'flat')).lower() == 'per_share':
        return df['commission'] + 2 * (float(profile.get('spread_pct', 0.0))
                                       + float(profile.get('slippage_pct', 0.0)))
    total = 2 * (float(profile.get('commission_pct', 0.0))
                 + float(profile.get('spread_pct', 0.0))
                 + float(profile.get('slippage_pct', 0.0)))
    return pd.Series(total, index=df.index)


# --------------------------------------------------------------------------
# the three arms
# --------------------------------------------------------------------------

def pick(gross, prob, cost) -> float | None:
    best = None
    for t in THRESHOLDS:
        fired = prob > t
        if int(fired.sum()) < MIN_TRADES:
            continue
        mean = float((gross[fired] - cost[fired]).mean())
        if best is None or mean > best[1]:
            best = (float(t), mean)
    return best[0] if best else None


def score(gross, prob, cost, threshold) -> dict:
    fired = prob > threshold
    n = int(fired.sum())
    if n == 0:
        return dict(threshold=threshold, trades=0)
    net = gross[fired] - cost[fired]
    return dict(threshold=threshold, trades=n, mean=float(net.mean()),
                total=float(net.sum()), win=float((net > 0).mean()))


def arms(df: pd.DataFrame, cost_col: str, fixed: float) -> dict:
    cost = (df[cost_col] + df['commission']).values
    gross, prob = df['gross'].values, df['prob'].values
    cut = len(df) // 2
    early, late = slice(0, cut), slice(cut + PURGE_BARS, len(df))

    t_in = pick(gross, prob, cost)
    t_early = pick(gross[early], prob[early], cost[early])

    print(f'  середня вартість {cost.mean() * 1e4:.2f} bp   '
          f'рання {cut:,} рядків / пізня {len(df) - cut - PURGE_BARS:,}')
    print(f'  {"арм":>30} {"поріг":>6} {"угод":>7} {"частка+":>8} '
          f'{"сер.чистий":>11} {"сума":>8}')

    results = {}
    rows = [('в-вибірці (оптимістично)', t_in, slice(0, len(df)), 'in_sample'),
            ('чесно: обрано на ранній', t_early, late, 'honest'),
            (f'фіксований {fixed:.2f}, без вибору', fixed, late, 'fixed')]
    for label, t, where, key in rows:
        if t is None:
            print(f'  {label:>30}   поріг не обрано (замало угод)')
            continue
        s = score(gross[where], prob[where], cost[where], t)
        results[key] = s
        if not s['trades']:
            print(f'  {label:>30} {t:6.2f}   жодної угоди')
            continue
        print(f'  {label:>30} {t:6.2f} {s["trades"]:7,} {s["win"]:7.1%} '
              f'{s["mean"]:+11.5f} {s["total"]:+8.3f}')
    return results


def transfer_table(df: pd.DataFrame, cost_col: str) -> None:
    """Does the SIGN hold at every threshold, in both halves?

    Worth more than any single arm. If the edge only exists at the one
    threshold that was chosen, it is a selection artifact. If it holds across
    the range in both halves independently, the threshold is not what is
    producing it.
    """
    cost = (df[cost_col] + df['commission']).values
    gross, prob = df['gross'].values, df['prob'].values
    cut = len(df) // 2
    print(f'\n  {"поріг":>6} {"рання половина":>22} {"пізня половина":>22}')
    for t in THRESHOLDS:
        a = score(gross[:cut], prob[:cut], cost[:cut], t)
        b = score(gross[cut + PURGE_BARS:], prob[cut + PURGE_BARS:],
                  cost[cut + PURGE_BARS:], t)
        if a['trades'] < MIN_TRADES and b['trades'] < MIN_TRADES:
            continue
        fa = f'{a["mean"]:+.5f} ({a["trades"]:,})' if a['trades'] else '—'
        fb = f'{b["mean"]:+.5f} ({b["trades"]:,})' if b['trades'] else '—'
        print(f'  {t:6.2f} {fa:>22} {fb:>22}')



# --------------------------------------------------------------------------
# what an account holds
# --------------------------------------------------------------------------

def portfolio_arms(df: pd.DataFrame, cost_col: str, threshold: float,
                   horizon_bars: int) -> None:
    """Turn the selection into an equity curve, because a mean has no denominator.

    Added 2026-08-20 after the same omission fooled this project twice in one
    day. Every arm above reports a MEAN PER TRADE, and a mean cannot show what
    the return cost in risk. Carried to a real portfolio, a cross-sectional
    model that beat passive holding in 20 years of 28 with a median excess of
    +7.15% turned out to be leverage:

        passive equal weight   CAGR +18.06%  vol 22.77%  Sharpe 0.79  maxDD -51%
        the strategy           CAGR +23.62%  vol 30.72%  Sharpe 0.77  maxDD -68%

    Volatility 1.35x against return 1.31x. The same holding levered to the same
    volatility returns +23.14%, so the model added half a percent a year and a
    drawdown seventeen points worse. Nothing per-trade could see that, and I
    only found it by building the portfolio by hand. This makes the instrument
    do it instead.

    NON-OVERLAPPING periods. The payoff column already spans `horizon_bars`, so
    consecutive bars describe overlapping holds and cannot be compounded. Taking
    every horizon-th timestamp gives a portfolio that rebalances with the whole
    capital once per holding period, which is a real strategy and an honest
    curve. An overlapping version holds a fraction of capital per book and needs
    per-bar returns this report does not carry.
    """
    fired_all = df[df['prob'] > threshold]
    if fired_all.empty:
        print()
        print('### ПОРТФЕЛЬ: жодного спрацювання на цьому порозі')
        return

    stamps = sorted(df['datetime'].unique())
    keep = set(stamps[::max(1, horizon_bars)])
    period_rows = df[df['datetime'].isin(keep)]
    if period_rows['datetime'].nunique() < 8:
        print()
        print(f'### ПОРТФЕЛЬ: замало неперекривних періодів '
              f'({period_rows["datetime"].nunique()}) для кривої')
        return

    net = period_rows['gross'] - period_rows[cost_col] - period_rows['commission']
    period_rows = period_rows.assign(net=net)

    def curve(rows: pd.DataFrame, selected: bool) -> pd.Series:
        if selected:
            rows = rows[rows['prob'] > threshold]
        if rows.empty:
            return pd.Series(dtype=float)
        return rows.groupby('datetime')['net'].mean()

    strat = curve(period_rows, True)
    passive = curve(period_rows, False)
    common = strat.index.intersection(passive.index)
    if len(common) < 8:
        print()
        print(f'### ПОРТФЕЛЬ: стратегія торгує лише в {len(common)} періодах — '
              f'замало для кривої')
        return
    strat, passive = strat.reindex(common).fillna(0.0), passive.reindex(common)

    span_days = (pd.Timestamp(max(common)) - pd.Timestamp(min(common))).days or 1
    per_year = len(common) * 365.25 / span_days

    def stats(r: pd.Series) -> tuple:
        eq = (1 + r).cumprod()
        years = span_days / 365.25
        cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else np.nan
        vol = float(r.std(ddof=1) * np.sqrt(per_year))
        dd = float((eq / eq.cummax() - 1).min())
        return cagr, vol, (cagr / vol if vol > 1e-9 else np.nan), dd

    s_cagr, s_vol, s_sharpe, s_dd = stats(strat)
    p_cagr, p_vol, p_sharpe, p_dd = stats(passive)

    print()
    print(f'### ПОРТФЕЛЬ: {len(common)} неперекривних періодів по {horizon_bars} барів')
    print(f'    поріг {threshold:.2f} обрано на ранній половині; крива — на пізній')
    # Annualising is only meaningful over a span long enough to contain the
    # thing being annualised. Thirty-two four-hour periods is three weeks, and
    # a CAGR extrapolated from three weeks is arithmetic, not an estimate.
    if span_days < 365:
        print(f'    УВАГА: вікно {span_days} днів. Річні величини нижче — '
              f'екстраполяція, не оцінка. Читати Sharpe і просадку, не CAGR.')
    print(f'{"":>28} {"CAGR":>9} {"вол-ть":>9} {"Sharpe":>8} {"макс.просадка":>14}')
    print(f'{"пасив (тримати все)":>28} {p_cagr:+9.2%} {p_vol:9.2%} '
          f'{p_sharpe:8.2f} {p_dd:14.2%}')
    print(f'{"стратегія":>28} {s_cagr:+9.2%} {s_vol:9.2%} '
          f'{s_sharpe:8.2f} {s_dd:14.2%}')

    # The comparison that separates skill from leverage: what would owning the
    # same thing return, levered to the risk this selection actually takes?
    # Levering a LOSING benchmark makes it lose more, so any less-bad strategy
    # "beats leverage" by an arbitrarily flattering margin. The comparison only
    # carries meaning when owning the thing pays something to begin with.
    # Caught in this function's own first output: passive -21.38% levered 1.92x
    # gives -41.27%, and a strategy at -14.48% appeared to add +26.79% a year.
    if p_cagr is not None and np.isfinite(p_cagr) and p_cagr <= 0:
        print()
        print('  порівняння за однаковим ризиком тут НЕ ІНФОРМАТИВНЕ: пасив у '
              'мінусі, а плече лише поглиблює збиток, тож будь-яка менш '
              'збиткова стратегія перемагає його механічно.')
    elif p_vol > 1e-9 and np.isfinite(p_cagr):
        lev = s_vol / p_vol
        mu = p_cagr + p_vol ** 2 / 2
        matched = mu * lev - (p_vol * lev) ** 2 / 2
        print(f'{f"пасив x{lev:.2f} (той самий ризик)":>28} {matched:+9.2%} '
              f'{p_vol*lev:9.2%} {"—":>8} {"—":>14}')
        print()
        print(f'внесок моделі понад плече: {s_cagr - matched:+.2%} на рік')
        if s_cagr - matched <= 0:
            print('  ЦЕ ПЛЕЧЕ, А НЕ ВМІННЯ: та сама експозиція з тим самим '
                  'ризиком дає стільки ж або більше.')


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--signal', default=DEFAULT_SIGNAL)
    p.add_argument('--payoff', default=DEFAULT_PAYOFF)
    p.add_argument('--interval', default=DEFAULT_INTERVAL)
    p.add_argument('--db-interval', default=DEFAULT_DB_INTERVAL)
    p.add_argument('--profile', default='ibkr_pro_tiered',
                   help='cost profile to CHARGE in this report')
    p.add_argument('--built-with', default='legacy_flat_50bp',
                   help='cost profile the payoff target on disk was BUILT with; '
                        'switch to the charging profile once the batch has been '
                        'rebuilt after 2026-08-17')
    p.add_argument('--fixed-threshold', type=float, default=0.85)
    p.add_argument('--horizon-bars', type=int, default=4,
                   help='bars the payoff spans; consecutive bars overlap '
                        'and cannot be compounded, so the portfolio '
                        'samples every Nth timestamp')
    args = p.parse_args()

    # Every number below is priced with a broker tariff nobody has chosen yet.
    # Say so above the figures, not in a footnote nobody reads: an edge quoted
    # net of costs is only as decided as the cost schedule behind it.
    try:
        from src.config.pending_decisions import as_report_header
        header = as_report_header()
        if header:
            print(header)
            print()
    except Exception:  # noqa: BLE001 - a notice must not stop a report
        pass

    profile = cost_profile(args.profile)
    built_with = cost_profile(args.built_with)
    df = load(args.signal, args.payoff, args.interval, args.db_interval,
              profile, built_with)
    print(f'{len(df):,} rows, {df.ticker.nunique()} tickers, '
          f'{df.datetime.min():%Y-%m-%d} .. {df.datetime.max():%Y-%m-%d}')
    print(f'gross mean per bar (cost stripped): {df.gross.mean():+.5f}')

    verdicts = {}
    for label, col in (('СПРЕД: оцінка з барів (песимістично)', 'spread_pessimistic'),
                       ('СПРЕД: припущення з targets.yaml', 'spread_assumed')):
        print(f'\n### {label}')
        verdicts[col] = arms(df, col, args.fixed_threshold)
    transfer_table(df, 'spread_assumed')

    # A mean per trade has no denominator. Everything above is blind to
    # what the return cost in risk; this is not.
    # The threshold must come from the EARLY half and the curve from the LATE
    # half, exactly as the honest arm does. The first version of this called
    # pick() on the whole frame and reported a Sharpe of 1.68 while the honest
    # arm was negative -- the same in-sample leak this report exists to
    # prevent, rebuilt one section lower.
    cost_all = (df['spread_assumed'] + df['commission']).values
    cut = len(df) // 2
    early = slice(0, cut)
    late_rows = df.iloc[cut + PURGE_BARS:]
    honest_threshold = pick(df['gross'].values[early], df['prob'].values[early],
                            cost_all[early])
    if honest_threshold is not None:
        portfolio_arms(late_rows, 'spread_assumed', honest_threshold,
                       args.horizon_bars)

    print('\n### ВЕРДИКТ')
    pess = verdicts['spread_pessimistic'].get('honest', {})
    opt = verdicts['spread_assumed'].get('honest', {})
    for name, v in (('песимістичний спред', pess), ('припущений спред', opt)):
        if not v.get('trades'):
            print(f'  {name}: немає угод')
            continue
        sign = 'ПЛЮС' if v['mean'] > 0 else 'МІНУС'
        print(f'  {name}: {sign} {v["mean"]:+.5f} на {v["trades"]:,} угодах')
    if pess.get('mean', 0) * opt.get('mean', 0) < 0:
        print('  РОЗБІЖНІСТЬ У ЗНАКУ: висновок визначає припущення про спред,')
        print('  а не модель. Розв\'язує лише реальна ціна виконання.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
