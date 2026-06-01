import pandas as pd
import numpy as np
from src.analytics.calculators.advanced_econometrics_calculator import AdvancedEconometricsCalculator

def test_calculator():
    # Створюємо фіктивні дані
    data = {
        'target': np.random.randn(100),
        'predictor': np.random.randn(100)
    }
    df = pd.DataFrame(data)
    
    # Запуск аналізу
    print("Running comprehensive causal analysis...")
    results = AdvancedEconometricsCalculator.run_comprehensive_causal_analysis(df, 'target', ['predictor'])
    print("Success:", results.get('_summary', {}).get('total_tests', 0) == 1)

if __name__ == "__main__":
    test_calculator()
