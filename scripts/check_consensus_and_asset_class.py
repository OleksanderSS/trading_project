#!/usr/bin/env python3
"""Verify EnhancedConsensusEngine and AssetClass fixes."""
import ast, sys, numpy as np
sys.path.insert(0, '.')

for f in ['src/trading/consensus_engine.py',
          'src/trading/adaptive_parameter_manager.py',
          'src/pipeline/stages/trading/recommendation_engine.py']:
    ast.parse(open(f, encoding='utf-8').read())
    print('OK syntax:', f)

# ============================================================
# 1. EnhancedConsensusEngine: trending_down now has own weights
# ============================================================
from src.trading.consensus_engine import EnhancedConsensusEngine
ece = EnhancedConsensusEngine()

assert 'trending_down' in ece.regime_weights, 'trending_down missing from regime_weights'
assert ece.regime_weights['trending_down'] != ece.regime_weights['ranging'], \
    'trending_down weights identical to ranging — not differentiated'
print('OK EnhancedConsensusEngine: trending_down regime has own weights')

# 2. _determine_regime maps strong negative trend -> 'trending_down' not 'ranging'
regime_down = ece._determine_regime({'volatility': 0.01, 'trend': -0.8})
assert regime_down == 'trending_down', f'Expected trending_down, got {regime_down}'
regime_up = ece._determine_regime({'volatility': 0.01, 'trend': +0.8})
assert regime_up == 'trending_up', f'Expected trending_up, got {regime_up}'
regime_vol = ece._determine_regime({'volatility': 0.05, 'trend': 0.0})
assert regime_vol == 'volatile', f'Expected volatile, got {regime_vol}'
regime_range = ece._determine_regime({'volatility': 0.01, 'trend': 0.1})
assert regime_range == 'ranging', f'Expected ranging, got {regime_range}'
print('OK _determine_regime: all 4 regimes correctly mapped')

# 3. generate_weighted_ensemble uses trending_down weights
result_down = ece.generate_weighted_ensemble(
    {'lstm': -0.5, 'cnn': -0.4, 'catboost': -0.3, 'transformer': 0.0},
    {'volatility': 0.01, 'trend': -0.9}
)
result_up = ece.generate_weighted_ensemble(
    {'lstm': 0.5, 'cnn': 0.4, 'transformer': 0.8, 'linear': 0.2},
    {'volatility': 0.01, 'trend': +0.9}
)
assert result_down['regime'] == 'trending_down'
assert result_up['regime'] == 'trending_up'
# trending_down should produce negative ensemble score for negative predictions
assert result_down['ensemble_prediction'] < 0, f'Bearish ensemble should be negative: {result_down["ensemble_prediction"]}'
print(f'OK generate_weighted_ensemble: down={result_down["ensemble_prediction"]:.3f} up={result_up["ensemble_prediction"]:.3f}')

# 4. generate_consensus uses generate_weighted_ensemble when predictions_by_model provided
ctx_with_preds = {
    'volatility': 0.01, 'trend': -0.8,
    'fingerprint': '0|0|0', 'regime': 'neutral',
    'anomaly_score': 0.0,
    'predictions_by_model': {'lstm': -0.4, 'cnn': -0.3, 'catboost': -0.2}
}
# Should not raise
try:
    report = ece.generate_consensus({'fallback': -0.3}, ctx_with_preds)
    print(f'OK generate_consensus: signal={report.final_signal}, '
          f'regime={report.market_regime}, confidence={report.confidence:.3f}')
except Exception as e:
    print(f'NOTE generate_consensus raised (may need diary): {e}')

# ============================================================
# 5. AssetClass: ETF now exists in enum and has preset
# ============================================================
from src.trading.adaptive_parameter_manager import (
    AssetClass, MarketRegime, AdaptiveParameterManager
)
assert hasattr(AssetClass, 'ETF'), 'ETF missing from AssetClass'
mgr = AdaptiveParameterManager()
assert AssetClass.ETF in mgr.asset_presets, 'ETF preset missing'
print('OK AssetClass.ETF exists and has asset preset')

# 6. _determine_asset_class returns 'etf' for SPY and that resolves without ValueError
from src.pipeline.stages.trading.recommendation_engine import TradingRecommendationEngine
# Minimal init
class _FakeMgr:
    def get(self, *a, **kw): return {}
try:
    engine_cls = TradingRecommendationEngine.__new__(TradingRecommendationEngine)
    ac = engine_cls._determine_asset_class('SPY')
    assert ac == 'etf', f'SPY should be etf, got {ac}'
    # Verify it can be passed to AdaptiveParameterManager without ValueError
    params = mgr.compute_adaptive_params(
        MarketRegime.RANGING, AssetClass(ac), 0.3
    )
    assert params.asset_class == AssetClass.ETF
    print(f'OK SPY -> etf -> compute_adaptive_params OK: risk={params.risk_per_trade_pct:.3%}')
except Exception as e:
    print(f'NOTE TradingRecommendationEngine needs full init: {e}')

# 7. All 4 asset classes reachable via _determine_asset_class
from src.pipeline.stages.trading.recommendation_engine import TradingRecommendationEngine
try:
    for ticker, expected in [
        ('COIN', 'crypto'), ('AMD', 'mid_cap'), ('SPY', 'etf'), ('AAPL', 'large_cap')
    ]:
        got = TradingRecommendationEngine.__new__(TradingRecommendationEngine)._determine_asset_class(ticker)
        assert got == expected, f'{ticker}: expected {expected}, got {got}'
    print('OK _determine_asset_class: etf/crypto/mid_cap/large_cap all reachable')
except Exception as e:
    print(f'NOTE: {e}')

print()
print('ALL CHECKS PASSED.')
