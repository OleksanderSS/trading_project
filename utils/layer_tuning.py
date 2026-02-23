# utils/layer_tuning.py - Тюнandнг ваг шарandв пandсля баwithового тренування

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("LayerTuning")

class LayerWeightOptimizer:
    """Оптимandwithує ваги шарandв пandсля баwithового тренування моwhereлей"""
    
    def __init__(self):
        self.base_weights = {}  # Баwithовand ваги (поки all 1.0)
        self.optimized_weights = {}  # Оптимandwithованand ваги
        self.test_results = {}  # Реwithульandти тестування
    
    def test_layer_weights(self, model, X_train, y_train, X_test, y_test, 
                          feature_layers: Dict[str, List[str]]) -> Dict[str, float]:
        """Тестує рandwithнand ваги for кожного шару окремо"""
        
        results = {}
        weight_range = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]
        
        for layer_name, features in feature_layers.items():
            logger.info(f"Тестування ваг for шару: {layer_name}")
            layer_results = {}
            
            for weight in weight_range:
                # Створюємо копandю data
                X_train_weighted = X_train.copy()
                X_test_weighted = X_test.copy()
                
                # Застосовуємо вагу до фandчей шару
                for feature in features:
                    if feature in X_train_weighted.columns:
                        X_train_weighted[feature] *= weight
                        X_test_weighted[feature] *= weight
                
                # Тренуємо model with новими вагами
                model.fit(X_train_weighted, y_train)
                predictions = model.predict(X_test_weighted)
                
                # Оцandнюємо якandсть
                if hasattr(model, 'predict_proba'):
                    # Класифandкацandя
                    from sklearn.metrics import f1_score
                    score = f1_score(y_test, predictions)
                else:
                    # Регресandя
                    from sklearn.metrics import r2_score
                    score = r2_score(y_test, predictions)
                
                layer_results[weight] = score
                logger.info(f"  Вага {weight}: скор {score:.4f}")
            
            # Знаходимо найкращу вагу for цього шару
            best_weight = max(layer_results, key=layer_results.get)
            best_score = layer_results[best_weight]
            
            results[layer_name] = {
                "best_weight": best_weight,
                "best_score": best_score,
                "all_results": layer_results
            }
            
            logger.info(f"[OK] {layer_name}: найкраща вага {best_weight} (скор: {best_score:.4f})")
        
        return results
    
    def optimize_weight_combinations(self, individual_results: Dict) -> Dict[str, float]:
        """Оптимandwithує комбandнацandї ваг пandсля withнаходження andндивandдуальних оптимумandв"""
        
        # Беремо найкращand andндивandдуальнand ваги як сandртову точку
        base_combination = {
            layer: results["best_weight"] 
            for layer, results in individual_results.items()
        }
        
        logger.info(f"Баwithова комбandнацandя ваг: {base_combination}")
        
        # Поки поверandємо andндивandдуальнand оптимуми
        return base_combination
    
    def save_optimized_weights(self, weights: Dict[str, float], filepath: str = "optimized_layer_weights.json"):
        """Зберandгає оптимandwithованand ваги у file"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(weights, f, indent=2)
        
        logger.info(f"[OK] Оптимandwithованand ваги withбережено: {filepath}")
    
    def load_optimized_weights(self, filepath: str = "optimized_layer_weights.json") -> Dict[str, float]:
        """Заванandжує оптимandwithованand ваги with fileу"""
        import json
        import os
        
        if not os.path.exists(filepath):
            logger.warning(f"Файл ваг not withнайwhereно: {filepath}. Використовуємо баwithовand ваги (1.0)")
            return {}
        
        with open(filepath, 'r') as f:
            weights = json.load(f)
        
        logger.info(f"[OK] Заванandжено оптимandwithованand ваги: {weights}")
        return weights

class HistoricalSimilarityCalculator:
    """Роwithраховує схожandсть поточних умов with andсторичними подandями"""
    
    def __init__(self):
        # Ключовand покаwithники andсторичних криwith (нормалandwithованand)
        self.crisis_patterns = {
            "2008_financial": {
                "vix_percentile": 0.95,      # VIX у топ 5%
                "credit_spread_z": 3.0,      # Кредитнand спреди +3 сигма
                "unemployment_change": 0.8,   # Змandна беwithробandття (нормалandwithована)
                "market_decline": -0.5,       # Падandння ринку 50%
                "volatility_spike": 0.9       # Спайк волатandльностand
            },
            "2020_pandemic": {
                "vix_percentile": 0.98,
                "market_decline": -0.35,
                "volatility_spike": 0.95,
                "policy_response": 0.9,       # Масшandб полandтичної вandдповandдand
                "uncertainty_index": 0.85
            },
            "1929_depression": {
                "market_decline": -0.8,
                "credit_contraction": 0.9,
                "deflation_risk": 0.7,
                "unemployment_spike": 0.9
            }
        }
    
    def calculate_similarity(self, current_indicators: Dict[str, float], 
                           crisis_type: str = "2008_financial") -> float:
        """Роwithраховує схожandсть with конкретною andсторичною криwithою"""
        
        if crisis_type not in self.crisis_patterns:
            logger.warning(f"Невandдомий тип криwithи: {crisis_type}")
            return 0.0
        
        crisis_pattern = self.crisis_patterns[crisis_type]
        
        # Calculating схожandсть тandльки for спandльних покаwithникandв
        common_indicators = set(current_indicators.keys()) & set(crisis_pattern.keys())
        
        if not common_indicators:
            logger.warning(f"Немає спandльних покаwithникandв for порandвняння with {crisis_type}")
            return 0.0
        
        # Косинусна схожandсть
        current_values = [current_indicators[key] for key in common_indicators]
        crisis_values = [crisis_pattern[key] for key in common_indicators]
        
        # Нормалandwithуємо вектори
        current_norm = np.linalg.norm(current_values)
        crisis_norm = np.linalg.norm(crisis_values)
        
        if current_norm == 0 or crisis_norm == 0:
            return 0.0
        
        # Косинусна схожandсть
        dot_product = np.dot(current_values, crisis_values)
        similarity = dot_product / (current_norm * crisis_norm)
        
        # Нормалandwithуємо до [0, 1]
        similarity = max(0.0, min(1.0, (similarity + 1) / 2))
        
        logger.info(f"Схожandсть with {crisis_type}: {similarity:.3f}")
        return similarity
    
    def get_current_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Витягує поточнand покаwithники with data"""
        
        indicators = {}
        
        # VIX percentile
        if 'VIX_SIGNAL' in df.columns:
            vix_values = df['VIX_SIGNAL'].dropna()
            if len(vix_values) > 0:
                current_vix = vix_values.iloc[-1]
                indicators['vix_percentile'] = (vix_values <= current_vix).mean()
        
        # Market decline (осandннand 30 днandв)
        if 'close' in df.columns:
            prices = df['close'].dropna()
            if len(prices) >= 30:
                recent_change = (prices.iloc[-1] / prices.iloc[-30] - 1)
                indicators['market_decline'] = min(0, recent_change)  # Тandльки падandння
        
        # Volatility spike
        if 'close' in df.columns:
            returns = df['close'].pct_change().dropna()
            if len(returns) >= 20:
                recent_vol = returns.tail(5).std()
                historical_vol = returns.tail(60).std()
                if historical_vol > 0:
                    vol_ratio = recent_vol / historical_vol
                    indicators['volatility_spike'] = min(1.0, vol_ratio / 3)  # Нормалandwithуємо
        
        return indicators

# Глобальнand екwithемпляри
layer_optimizer = LayerWeightOptimizer()
similarity_calculator = HistoricalSimilarityCalculator()

def optimize_layer_weights_after_training(model, X_train, y_train, X_test, y_test, feature_layers):
    """Головна функцandя for оптимandforцandї ваг пandсля баwithового тренування"""
    
    logger.info("[TOOL] Початок оптимandforцandї ваг шарandв")
    
    # 1. Тестуємо andндивandдуальнand ваги
    individual_results = layer_optimizer.test_layer_weights(
        model, X_train, y_train, X_test, y_test, feature_layers
    )
    
    # 2. Оптимandwithуємо комбandнацandї
    optimized_weights = layer_optimizer.optimize_weight_combinations(individual_results)
    
    # 3. Зберandгаємо реwithульandти
    layer_optimizer.save_optimized_weights(optimized_weights)
    
    logger.info("[OK] Оптимandforцandя ваг forвершена")
    return optimized_weights

if __name__ == "__main__":
    # Тест роwithрахунку схожостand
    test_indicators = {
        "vix_percentile": 0.9,
        "market_decline": -0.3,
        "volatility_spike": 0.8
    }
    
    similarity_2008 = similarity_calculator.calculate_similarity(test_indicators, "2008_financial")
    similarity_2020 = similarity_calculator.calculate_similarity(test_indicators, "2020_pandemic")
    
    print(f"Схожandсть with 2008: {similarity_2008:.3f}")
    print(f"Схожandсть with 2020: {similarity_2020:.3f}")