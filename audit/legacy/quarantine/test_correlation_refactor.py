import pandas as pd
import numpy as np
from src.models.ensemble.model_correlation_analyzer import ModelCorrelationAnalyzer

class MockModel:
    def __init__(self, offset):
        self.offset = offset
    def predict(self, X):
        # Return something correlated with X but with some noise and offset
        return X.iloc[:, 0].values + np.random.randn(len(X)) * 0.1 + self.offset

def test_correlation_analyzer_refactor():
    print("Testing modular ModelCorrelationAnalyzer...")
    
    # Create mock data
    X = pd.DataFrame(np.random.randn(100, 2), columns=['f1', 'f2'])
    y = pd.Series(np.random.randn(100))
    
    models = {
        'm1': MockModel(0.1),
        'm2': MockModel(0.2),
        'm3': MockModel(5.0) # Less correlated due to offset in predictions if we use raw values, 
                             # but Pearson will see them as highly correlated
    }
    
    analyzer = ModelCorrelationAnalyzer(correlation_method='pearson')
    
    print("Running analyze_correlation...")
    results = analyzer.analyze_correlation(models, X, y)
    
    print(f"Correlation Matrix Keys: {list(results.get('correlation_matrix', {}).keys())}")
    assert 'm1' in results['correlation_matrix']
    assert 'm2' in results['correlation_matrix']
    
    print("Testing diverse subset selection...")
    selected = analyzer.select_diverse_subset(models, X, y, max_models=2)
    print(f"Selected models: {selected}")
    assert len(selected) <= 2
    
    print("✅ ModelCorrelationAnalyzer modular tests passed!")

if __name__ == "__main__":
    test_correlation_analyzer_refactor()
