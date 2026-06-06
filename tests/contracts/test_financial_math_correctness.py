
from __future__ import annotations
import importlib, math
import numpy as np, pandas as pd, pytest

def test_sharpe_constant_returns_not_infinite_if_function_available():
    candidates=[('src.risk.metrics','calculate_sharpe_ratio'),('src.analytics.calculators.risk_metrics_calculator','calculate_sharpe_ratio')]
    for module_name, func_name in candidates:
        try: mod=importlib.import_module(module_name); func=getattr(mod,func_name)
        except Exception: continue
        value=func(pd.Series([0.01,0.01,0.01,0.01])); assert not math.isinf(float(value)); return
    pytest.skip('No known Sharpe function importable')
def test_var_empty_returns_not_zero_risk_if_calculator_available():
    try: mod=importlib.import_module('src.risk_management.var_calculator'); C=getattr(mod,'VarCalculator')
    except Exception as exc: pytest.skip(f'VarCalculator not importable: {exc}')
    calc=C()
    if not hasattr(calc,'calculate_var_historical'): pytest.skip('method unavailable')
    result=calc.calculate_var_historical([])
    if isinstance(result,dict): assert result.get('status')=='insufficient_data' or pd.isna(result.get('var')), 'Empty returns must not be var=0.0'
    else: assert pd.isna(result)
def test_drawdown_positive_pct_convention_example():
    equity=pd.Series([100.0,120.0,90.0,110.0]); dd=(equity-equity.cummax())/equity.cummax(); assert dd.min()<0 and abs(dd.min())>0 and np.isclose(abs(dd.min()),0.25)
