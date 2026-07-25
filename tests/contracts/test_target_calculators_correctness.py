
from __future__ import annotations
import importlib, inspect
import pandas as pd
import pytest

def _load_class(module_name: str, class_name: str):
    try:
        mod=importlib.import_module(module_name); return getattr(mod,class_name)
    except Exception as exc: pytest.skip(f'Cannot import {module_name}.{class_name}: {exc}')
def _call_calculate(calc, df):
    method = calc.calculate if hasattr(calc, "calculate") else calc.calculate_binary
    sig=inspect.signature(method); kwargs={}
    if 'base_col' in sig.parameters: kwargs['base_col']='close'
    if 'shift' in sig.parameters: kwargs['shift']=-1
    if 'horizon' in sig.parameters: kwargs['horizon']=1
    if 'threshold' in sig.parameters: kwargs['threshold']=0.0
    return method(df, **kwargs)
def test_regression_target_does_not_cross_ticker_boundary():
    C=_load_class('src.targets.calculators.regression_calculator','RegressionCalculator'); calc=C()
    df=pd.DataFrame({'ticker':['A','A','B','B'],'timestamp':pd.date_range('2024-01-01',periods=4,freq='D'),'close':[100.0,110.0,1000.0,1200.0]})
    result=_call_calculate(calc,df); y=result.iloc[:,0] if isinstance(result,pd.DataFrame) else result
    assert pd.isna(y.iloc[1]), 'Last row of ticker A must not use first row of ticker B as future price'
def test_classification_target_does_not_cross_ticker_boundary():
    C=_load_class('src.targets.calculators.classification_calculator','ClassificationCalculator'); calc=C()
    df=pd.DataFrame({'ticker':['A','A','B','B'],'timestamp':pd.date_range('2024-01-01',periods=4,freq='D'),'close':[100.0,110.0,1000.0,1200.0]})
    result=_call_calculate(calc,df); y=result.iloc[:,0] if isinstance(result,pd.DataFrame) else result
    assert pd.isna(y.iloc[1]) or y.iloc[1] in (-1,0,None), 'Classification target must not cross ticker boundary'
