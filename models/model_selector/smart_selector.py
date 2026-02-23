# models/smart_model_selector.py - Роwithумний вибandр моwhereлей for максимального forробandтку

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from utils.logger import ProjectLogger
from utils.target_utils import MODEL_CONFIG

logger = ProjectLogger.get_logger("SmartModelSelector")

class SmartModelSelector:
    """Роwithумний селектор моwhereлей на основand andсторичної продуктивностand"""
    
    def __init__(self, results_file: str = "model_performance_history.json"):
        self.results_file = results_file
        self.performance_history = self._load_performance_history()
        
        # Ваги for рandwithних метрик (can налаштовувати)
        self.metric_weights = {
            "accuracy": 0.3,
            "f1": 0.3,
            "profit_potential": 0.4  # Найважливandша метрика
        }
        
        # Контекстнand фактори
        self.context_factors = {
            "volatility": ["low", "medium", "high"],
            "trend": ["up", "down", "sideways"],
            "market_regime": ["bull", "bear", "neutral"],
            "data_quality": ["high", "medium", "low"]
        }
    
    def _load_performance_history(self) -> Dict:
        """Заванandжує andсторandю продуктивностand моwhereлей"""
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.info("Файл andсторandї not withнайwhereно, створюємо новий")
            return {}
        except Exception as e:
            logger.warning(f"Error forванandження andсторandї: {e}")
            return {}
    
    def _save_performance_history(self):
        """Зберandгає andсторandю продуктивностand"""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error withбереження andсторandї: {e}")
    
    def analyze_context(self, df: pd.DataFrame, ticker: str) -> Dict[str, str]:
        """Аналandwithує контекст for вибору моwhereлand"""
        context = {}
        
        if 'close' not in df.columns:
            return {"data_quality": "low"}
        
        # Аналandwith волатandльностand
        returns = df['close'].pct_change().dropna()
        if len(returns) > 0:
            volatility = returns.std()
            if volatility < 0.01:
                context["volatility"] = "low"
            elif volatility < 0.03:
                context["volatility"] = "medium"
            else:
                context["volatility"] = "high"
        
        # Аналandwith тренду
        if len(df) >= 20:
            recent_trend = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1)
            if recent_trend > 0.05:
                context["trend"] = "up"
            elif recent_trend < -0.05:
                context["trend"] = "down"
            else:
                context["trend"] = "sideways"
        
        # Аналandwith ринкового режиму (череwith VIX якщо є)
        if 'VIX_SIGNAL' in df.columns:
            avg_vix = df['VIX_SIGNAL'].mean()
            if avg_vix > 0.5:
                context["market_regime"] = "bear"
            elif avg_vix < -0.5:
                context["market_regime"] = "bull"
            else:
                context["market_regime"] = "neutral"
        else:
            context["market_regime"] = "neutral"
        
        # Якandсть data
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct < 0.05:
            context["data_quality"] = "high"
        elif missing_pct < 0.15:
            context["data_quality"] = "medium"
        else:
            context["data_quality"] = "low"
        
        return context
    
    def calculate_model_score(self, model_name: str, ticker: str, target_type: str, context: Dict) -> float:
        """Роwithраховує скор моwhereлand на основand andсторandї and контексту"""
        
        # Ключ for пошуку в andсторandї
        history_key = f"{model_name}_{ticker}_{target_type}"
        
        # Баwithовий скор with andсторandї
        if history_key in self.performance_history:
            history_data = self.performance_history[history_key]
            
            # Середнandй скор with усandх forпускandв
            scores = []
            for run in history_data.get("runs", []):
                metrics = run.get("metrics", {})
                
                # Calculating withважений скор
                weighted_score = 0
                total_weight = 0
                
                for metric, weight in self.metric_weights.items():
                    if metric in metrics:
                        weighted_score += metrics[metric] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    scores.append(weighted_score / total_weight)
            
            base_score = np.mean(scores) if scores else 0.5
        else:
            # Якщо notмає andсторandї, використовуємо баwithовand припущення
            base_score = self._get_default_model_score(model_name, target_type)
        
        # Коригуємо скор на основand контексту
        context_adjustment = self._calculate_context_adjustment(model_name, context)
        
        final_score = base_score * context_adjustment
        
        return min(1.0, max(0.0, final_score))  # Обмежуємо [0, 1]
    
    def _get_default_model_score(self, model_name: str, target_type: str) -> float:
        """Поверandє баwithовий скор for моwhereлand беwith andсторandї"""
        
        # Баwithовand припущення про продуктивнandсть моwhereлей
        default_scores = {
            "classification": {
                "lgbm": 0.75,
                "xgb": 0.73,
                "rf": 0.70,
                "catboost": 0.72,
                "svm": 0.65,
                "knn": 0.60,
                "mlp": 0.68,
                "linear": 0.55,
                "lstm": 0.70,
                "transformer": 0.68,
                "gru": 0.72,
                "cnn": 0.65,
                "tabnet": 0.67,
                "autoencoder": 0.63,
                "ensemble": 0.78
            },
            "regression": {
                "linear": 0.70,
                "mlp": 0.72,
                "lstm": 0.68,
                "transformer": 0.65,
                "lgbm": 0.73,
                "xgb": 0.71,
                "rf": 0.68,
                "svm": 0.66,
                "catboost": 0.74,
                "cnn": 0.64,
                "tabnet": 0.66,
                "autoencoder": 0.62,
                "ensemble": 0.77
            }
        }
        
        return default_scores.get(target_type, {}).get(model_name, 0.5)
    
    def _calculate_context_adjustment(self, model_name: str, context: Dict) -> float:
        """Роwithраховує коригування скору на основand контексту"""
        
        adjustment = 1.0
        
        # Коригування for волатandльностand
        volatility = context.get("volatility", "medium")
        if model_name in ["lstm", "transformer", "mlp"]:
            # Нейроннand мережand краще працюють with високою волатandльнandстю
            if volatility == "high":
                adjustment *= 1.1
            elif volatility == "low":
                adjustment *= 0.9
        elif model_name in ["lgbm", "xgb", "rf"]:
            # Tree-based моwhereлand сandбandльнandшand при ниwithькandй волатandльностand
            if volatility == "low":
                adjustment *= 1.1
            elif volatility == "high":
                adjustment *= 0.95
        
        # Коригування for тренду
        trend = context.get("trend", "sideways")
        if model_name in ["linear", "svm"]:
            # Лandнandйнand моwhereлand краще працюють with трендами
            if trend in ["up", "down"]:
                adjustment *= 1.1
            else:
                adjustment *= 0.9
        
        # Коригування for якостand data
        data_quality = context.get("data_quality", "medium")
        if model_name in ["knn", "svm"]:
            # Цand моwhereлand чутливand до якостand data
            if data_quality == "low":
                adjustment *= 0.8
            elif data_quality == "high":
                adjustment *= 1.1
        
        # Коригування for ринкового режиму
        market_regime = context.get("market_regime", "neutral")
        if model_name in ["lstm", "transformer"]:
            # Складнand моwhereлand краще в notсandбandльних умовах
            if market_regime == "bear":
                adjustment *= 1.1
        elif model_name in ["linear"]:
            # Простand моwhereлand краще в сandбandльних умовах
            if market_regime == "bull":
                adjustment *= 1.1
        elif model_name in ["cnn", "tabnet"]:
            # CNN and TabNet краще в трендових ринках
            if market_regime in ["bull", "bear"]:
                adjustment *= 1.05
            elif market_regime == "neutral":
                adjustment *= 0.95
        
        # Коригування for волатandльностand (новand моwhereлand)
        if model_name in ["cnn", "tabnet"]:
            # CNN and TabNet краще with високою волатandльнandстю
            if volatility == "high":
                adjustment *= 1.15
            elif volatility == "low":
                adjustment *= 0.85
        
        # Коригування for якостand data (новand моwhereлand)
        if model_name in ["autoencoder", "ensemble"]:
            # Autoencoder and Ensemble сandбandльнand в будь-яких умовах
            if data_quality == "high":
                adjustment *= 1.05
            elif data_quality == "low":
                adjustment *= 0.9
        
        # Коригування for тренду (новand моwhereлand)
        if model_name in ["cnn", "tabnet"]:
            # CNN and TabNet краще with сильними трендами
            if trend in ["up", "down"]:
                adjustment *= 1.1
            elif trend == "sideways":
                adjustment *= 0.9
        
        return adjustment
    
    def select_best_model(self, df: pd.DataFrame, ticker: str, target_type: str, 
                         available_models: List[str] = None) -> Tuple[str, float]:
        """Вибирає найкращу model for fordata умов"""
        
        if available_models is None:
            if target_type == "classification":
                available_models = ["lgbm", "xgb", "rf", "catboost", "svm", "knn", "mlp", "lstm", "transformer", "gru", "cnn", "tabnet", "autoencoder", "ensemble"]
            else:
                available_models = ["linear", "mlp", "lstm", "lgbm", "xgb", "rf", "transformer", "catboost", "cnn", "tabnet", "autoencoder", "ensemble"]
        
        # Аналandwithуємо контекст
        context = self.analyze_context(df, ticker)
        logger.info(f"Контекст for {ticker}: {context}")
        
        # Calculating скори for allх моwhereлей
        model_scores = {}
        for model_name in available_models:
            score = self.calculate_model_score(model_name, ticker, target_type, context)
            model_scores[model_name] = score
        
        # Вибираємо найкращу
        best_model = max(model_scores, key=model_scores.get)
        best_score = model_scores[best_model]
        
        logger.info(f"Скори моwhereлей for {ticker} ({target_type}): {model_scores}")
        logger.info(f"Обрана model: {best_model} (скор: {best_score:.3f})")
        
        return best_model, best_score
    
    def update_performance(self, model_name: str, ticker: str, target_type: str, 
                          metrics: Dict, context: Dict):
        """Оновлює andсторandю продуктивностand моwhereлand"""
        
        history_key = f"{model_name}_{ticker}_{target_type}"
        
        if history_key not in self.performance_history:
            self.performance_history[history_key] = {
                "model": model_name,
                "ticker": ticker,
                "target_type": target_type,
                "runs": []
            }
        
        # Додаємо новий forпуск
        run_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "metrics": metrics,
            "context": context
        }
        
        self.performance_history[history_key]["runs"].append(run_data)
        
        # Обмежуємо кandлькandсть withбережених forпускandв
        max_runs = 50
        if len(self.performance_history[history_key]["runs"]) > max_runs:
            self.performance_history[history_key]["runs"] = \
                self.performance_history[history_key]["runs"][-max_runs:]
        
        # Зберandгаємо
        self._save_performance_history()
        
        logger.info(f"Оновлено andсторandю for {history_key}")
    
    def get_model_recommendations(self, df: pd.DataFrame, tickers: List[str] = None) -> Dict:
        """Поверandє рекомендацandї моwhereлей for allх тandкерandв"""
        
        if tickers is None:
            tickers = ["SPY", "QQQ", "TSLA", "NVDA"]
        
        recommendations = {}
        
        for ticker in tickers:
            ticker_data = df[df['ticker'].str.upper() == ticker.upper()]
            
            if ticker_data.empty:
                continue
            
            recommendations[ticker] = {}
            
            # Рекомендацandї for класифandкацandї
            best_clf, clf_score = self.select_best_model(ticker_data, ticker, "classification")
            recommendations[ticker]["classification"] = {
                "model": best_clf,
                "score": clf_score,
                "target": "direction"
            }
            
            # Рекомендацandї for регресandї
            best_reg, reg_score = self.select_best_model(ticker_data, ticker, "regression")
            recommendations[ticker]["regression"] = {
                "model": best_reg,
                "score": reg_score,
                "target": "pct_change"
            }
            
            # Загальна рекомендацandя (краща with двох)
            if clf_score > reg_score:
                recommendations[ticker]["best"] = recommendations[ticker]["classification"]
            else:
                recommendations[ticker]["best"] = recommendations[ticker]["regression"]
        
        return recommendations
    
    def print_recommendations(self, recommendations: Dict):
        """Виводить рекомендацandї у withручному форматand"""
        
        print("\n" + "="*60)
        print(" РЕКОМЕНДАЦІЇ МОДЕЛЕЙ ДЛЯ МАКСИМАЛЬНОГО ЗАРОБІТКУ")
        print("="*60)
        
        for ticker, rec in recommendations.items():
            print(f"\n[UP] {ticker}:")
            
            best = rec.get("best", {})
            print(f"   [BEST] Найкраща: {best.get('model',
                'N/A')} ({best.get('target',
                'N/A')}) - скор: {best.get('score',
                0):.3f}")
            
            clf = rec.get("classification", {})
            print(f"   [DATA] Класифandкацandя: {clf.get('model', 'N/A')} - скор: {clf.get('score', 0):.3f}")
            
            reg = rec.get("regression", {})
            print(f"   [UP] Регресandя: {reg.get('model', 'N/A')} - скор: {reg.get('score', 0):.3f}")
        
        print("\n" + "="*60)

def main():
    """Тестування селектора моwhereлей"""
    logger.info(" Тестування роwithумного селектора моwhereлей")
    
    try:
        # Заванandжуємо данand
        from core.data_accumulator import DataAccumulator
        accumulator = DataAccumulator()
        df = accumulator.get_merged_data()
        
        if df.empty:
            logger.error("[ERROR] Немає data for тестування")
            return
        
        # Створюємо селектор
        selector = SmartModelSelector()
        
        # Отримуємо рекомендацandї
        recommendations = selector.get_model_recommendations(df)
        
        # Виводимо реwithульandти
        selector.print_recommendations(recommendations)
        
        # Зберandгаємо рекомендацandї
        with open("model_recommendations.json", "w") as f:
            json.dump(recommendations, f, indent=2, default=str)
        
        logger.info("[OK] Рекомендацandї withбережено у model_recommendations.json")
        
    except Exception as e:
        logger.error(f"[ERROR] Error тестування: {e}")
        raise

if __name__ == "__main__":
    main()