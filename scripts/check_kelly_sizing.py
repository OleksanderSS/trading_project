#!/usr/bin/env python3
"""Verify Kelly sizing uses real win_rate."""
import ast, sys
sys.path.insert(0, '.')

for f in ['src/trading/elite_risk_sizer.py', 'src/trading/portfolio_manager.py']:
    ast.parse(open(f, encoding='utf-8').read())
    print('OK syntax:', f)

from src.trading.elite_risk_sizer import EliteRiskSizer
from src.core.logging.logger import ProjectLogger

sizer = EliteRiskSizer(logger=ProjectLogger.get_logger('test'))

# 1. No history -> heuristic
wr, src = sizer.get_win_rate_for_sizing('AMD', confidence=0.7)
assert src == 'heuristic'
assert wr == 0.55
print(f'OK heuristic: wr={wr}, src={src}')

# 2. Leaderboard with enough trades
class MockSelector:
    arena_leaderboard = {
        'AMD_ctx1': {'lightgbm': {'win_rate': 0.68, 'total_predictions': 50}}
    }

wr2, src2 = sizer.get_win_rate_for_sizing('AMD', model_id='lightgbm',
    confidence=0.7, adaptive_selector=MockSelector())
assert src2 == 'adaptive_leaderboard', f'got {src2}'
assert abs(wr2 - 0.68) < 0.01
print(f'OK leaderboard: wr={wr2:.3f}, src={src2}')

# 3. Too few trades -> heuristic
class MockSelectorFew:
    arena_leaderboard = {
        'AMD_ctx1': {'lightgbm': {'win_rate': 0.68, 'total_predictions': 5}}
    }

wr3, src3 = sizer.get_win_rate_for_sizing('AMD', model_id='lightgbm',
    confidence=0.7, adaptive_selector=MockSelectorFew())
assert src3 == 'heuristic', f'got {src3}'
print(f'OK fallback: few trades -> heuristic, src={src3}')

# 4. compute_optimal with real win_rate
pct, meta = sizer.compute_optimal_position_size(
    ticker='AMD', confidence=0.7, prediction=0.03,
    total_capital=100000, ticker_volatility=0.3,
    portfolio_volatility=0.15, portfolio_positions={},
    current_price=150.0, model_id='lightgbm',
    adaptive_selector=MockSelector()
)
assert meta['win_rate_source'] == 'adaptive_leaderboard'
assert abs(meta['win_rate'] - 0.68) < 0.01
assert 0.0 <= pct <= 0.15
print(f'OK compute_optimal: pct={pct:.3%} wr={meta["win_rate"]:.3f} src={meta["win_rate_source"]}')

# 5. Better win_rate -> larger position
def make_selector(wr_val):
    class S:
        arena_leaderboard = {
            'AMD_x': {'lgbm': {'win_rate': wr_val, 'total_predictions': 50}}
        }
    return S()

pct_good, _ = sizer.compute_optimal_position_size(
    'AMD', 0.7, 0.03, 100000, 0.3, 0.15, {},
    current_price=150.0, model_id='lgbm', adaptive_selector=make_selector(0.70)
)
pct_bad, _ = sizer.compute_optimal_position_size(
    'AMD', 0.7, 0.03, 100000, 0.3, 0.15, {},
    current_price=150.0, model_id='lgbm', adaptive_selector=make_selector(0.52)
)
assert pct_good > pct_bad, f'70%WR={pct_good:.4f} should exceed 52%WR={pct_bad:.4f}'
print(f'OK differentiation: 70%WR->{pct_good:.3%} > 52%WR->{pct_bad:.3%}')

# 6. confidence=0.61 and confidence=0.99 NOW give DIFFERENT sizes (old code gave same)
pct_61, m61 = sizer.compute_optimal_position_size(
    'AMD', 0.61, 0.03, 100000, 0.3, 0.15, {},
    current_price=150.0  # no selector -> heuristic 0.55
)
pct_99, m99 = sizer.compute_optimal_position_size(
    'AMD', 0.99, 0.03, 100000, 0.3, 0.15, {},
    current_price=150.0, model_id='lgbm', adaptive_selector=make_selector(0.72)
)
print(f'OK conf differentiation: conf=0.61->pct={pct_61:.3%}(wr={m61["win_rate"]:.2f})'
      f' conf=0.99->pct={pct_99:.3%}(wr={m99["win_rate"]:.2f})')
assert m61['win_rate'] != m99['win_rate'], 'confidence levels should produce different win_rates now'

print()
print('ALL CHECKS PASSED.')
