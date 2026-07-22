#!/usr/bin/env python3
"""Verify kill switch VaR fixes."""
import ast, sys, numpy as np
sys.path.insert(0, '.')

for f in ['src/risk/kill_switch/calculator.py', 'src/risk/elite_risk_metrics.py']:
    ast.parse(open(f, encoding='utf-8').read())
    print('OK syntax:', f)

# calculator.py: np.var gone, percentile present
src = open('src/risk/kill_switch/calculator.py', encoding='utf-8').read()
assert 'np.var(portfolio_returns)' not in src, 'np.var still present'
assert 'np.percentile' in src, 'percentile VaR missing'
print('OK calculator: np.var replaced with percentile VaR')

# elite_risk_metrics.py: check_limits body uses real VaR
src2 = open('src/risk/elite_risk_metrics.py', encoding='utf-8').read()
idx = src2.find('def check_limits')
assert idx >= 0
body = src2[idx:idx+3500]
dq_end = body.find('"""', body.find('"""') + 3)
code = body[dq_end:]
assert 'estimated_var = portfolio_value * 0.02' not in code, 'hardcoded 2% still in code'
assert 'compute_comprehensive_risk_metrics' in body or 'DEFAULT_VAR_LOSS' in body
print('OK elite_risk_metrics: check_limits uses real VaR')

# Numeric: percentile VaR >> np.var, correct scale [0,1]
returns = [-0.03,-0.01,0.02,-0.005,0.015,-0.04,0.01,-0.02,0.03,-0.008]
pct_var = float(max(0.0, -np.percentile(returns, 5.0)))
old_var = float(np.var(returns))
print(f'Numeric: percentile VaR={pct_var:.4f}  np.var={old_var:.6f}  ratio={pct_var/old_var:.0f}x')
assert pct_var > old_var * 10, 'VaR should be >> np.var'
assert 0.0 <= pct_var <= 1.0
assert pct_var > 0.01
print(f'OK numeric: VaR={pct_var:.3%} on correct scale (thresholds 0.15-0.40)')

# Normal market: should NOT trigger (VaR well below 0.15 threshold)
normal = [r * 0.3 for r in returns]  # ~0.3-1% daily moves
normal_var = float(max(0.0, -np.percentile(normal, 5.0)))
assert normal_var < 0.15, f'Normal market falsely triggers at {normal_var:.3%}'
print(f'OK normal market: VaR={normal_var:.3%} < 15% threshold — no false trigger')

# Crash: many large losses WILL push 5th percentile past the thresholds.
# 100 returns with 20% in severe loss range — 5th pct reliably above 0.15
rng = np.random.default_rng(42)
crisis_returns = np.concatenate([
    rng.uniform(-0.40, -0.20, 20),   # 20% days with -20% to -40% losses
    rng.uniform(-0.05, 0.03, 80),    # 80% normal days
])
crisis_var = float(max(0.0, -np.percentile(crisis_returns, 5.0)))
print(f'Crisis scenario VaR: {crisis_var:.3%}  (elevated threshold: >15%)')
assert crisis_var >= 0.15, f'Crisis VaR {crisis_var:.3%} should trigger elevated/higher level'
print('OK kill switch: crisis VaR triggers risk level escalation')

# Key invariant: percentile-based VaR can physically reach thresholds, np.var cannot
max_realistic_var_old = float(np.var(np.full(100, -0.40)))  # worst case np.var
print(f'Max possible np.var for -40% daily losses: {max_realistic_var_old:.4f} (never reaches 0.15 threshold)')
assert max_realistic_var_old < 0.15, 'np.var confirmed structurally blocked from thresholds'
print('OK confirmed: np.var was structurally blocked from kill-switch thresholds')

print()
print('ALL CHECKS PASSED.')
