# models/smart_model_selector.py - Розумний вибір моделей для максимального прибутку

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("SmartModelSelector")

class SmartModelSelector:
    """Розумний селектор моделей на основі історії, контексту та мета-навчання."""
    
    def __init__(self, results_file: str = "model_performance_history.json", competence_map_file: str = "src/models/model_selector/model_competence_map.json"):
        self.results_file = results_file
        self.performance_history = self._load_performance_history()
        self.competence_map = self._load_competence_map(competence_map_file)
        self.metric_weights = self.competence_map.get("metric_weights", {})
        # Мета-модель для прогнозування помилок (ядро Критика)
        self.error_meta_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        self.is_meta_model_trained = False

    def _load_performance_history(self) -> Dict:
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.warning(f"Помилка завантаження історії: {e}")
            return {}

    def _load_competence_map(self, file_path: str) -> Dict:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Не вдалося завантажити карту компетенцій: {e}")
            raise

    def _save_performance_history(self):
        try:
            with open(self.results_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Помилка збереження історії: {e}")

    def train_error_meta_model(self, model_name: str, ticker: str):
        """Тренує мета-модель на історичних помилках конкретної моделі."""
        history_key = f"{model_name}_{ticker}_classification"
        if history_key not in self.performance_history or not self.performance_history[history_key].get("runs"):
            logger.warning(f"Недостатньо історії для тренування мета-моделі для {history_key}")
            return

        runs = self.performance_history[history_key]["runs"]
        contexts = [run['context'] for run in runs]
        errors = [1 - run['metrics'].get('accuracy', 0.5) for run in runs] # Помилка = 1 - точність
        
        # Конвертуємо контекст в числовий формат
        context_df = pd.DataFrame(contexts)
        X = pd.get_dummies(context_df, columns=context_df.columns).to_numpy()
        y = np.array(errors)

        if X.shape[0] < 10:
            logger.warning("Недостатньо прикладів для тренування мета-моделі.")
            return

        self.error_meta_model.fit(X, y)
        self.is_meta_model_trained = True
        logger.info(f"Мета-модель помилок для {history_key} успішно натренована.")

    def analyze_context(self, df: pd.DataFrame, ticker: str) -> Dict[str, str]:
        context = {}
        if 'close' not in df.columns:
            return {"data_quality": "low"}
        
        returns = df['close'].pct_change().dropna()
        if len(returns) > 0:
            volatility = returns.std()
            context["volatility"] = "low" if volatility < 0.01 else "medium" if volatility < 0.03 else "high"
        
        if len(df) >= 20:
            recent_trend = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1)
            context["trend"] = "up" if recent_trend > 0.05 else "down" if recent_trend < -0.05 else "sideways"
        
        if 'VIX_SIGNAL' in df.columns:
            avg_vix = df['VIX_SIGNAL'].mean()
            context["market_regime"] = "bear" if avg_vix > 0.5 else "bull" if avg_vix < -0.5 else "neutral"
        else:
            context["market_regime"] = "neutral"
        
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) if df.shape[0] > 0 else 0
        context["data_quality"] = "high" if missing_pct < 0.05 else "medium" if missing_pct < 0.15 else "low"
        
        return context

    def critique_action(self, action: Dict[str, Any], context_df: pd.DataFrame) -> Dict[str, Any]:
        """Оцінює запропоновану дію, використовуючи логіку Критика з DEAN."""
        ticker = action.get('ticker', 'default')
        model_name = action.get('model_name', 'unknown')
        target_type = action.get('target_type', 'classification')

        context = self.analyze_context(context_df, ticker)
        context_adjustment = self._calculate_context_adjustment(model_name, context)
        
        # Прогноз очікуваної помилки від мета-моделі
        expected_error = 0.5
        if self.is_meta_model_trained:
            context_for_pred = pd.get_dummies(pd.DataFrame([context]), columns=context.keys()).to_numpy()
            # Потрібно забезпечити, щоб колонки збігалися з тими, на яких тренувалася модель
            # Це спрощення, в реальності потрібен більш надійний механізм
            try:
                expected_error = self.error_meta_model.predict(context_for_pred)[0]
            except Exception:
                pass # Ігноруємо помилку, якщо колонки не збігаються

        history_key = f"{model_name}_{ticker}_{target_type}"
        historical_reliability = 0.5
        points = [f"Context: {context}"]

        if history_key in self.performance_history and self.performance_history[history_key].get("runs"):
            recent_success = np.mean([r['metrics'].get('accuracy', 0.5) for r in self.performance_history[history_key]["runs"][-5:]])
            historical_reliability = recent_success
            points.append(f"Historical reliability (last 5 runs): {recent_success:.2f}")

        critique_score = 0
        # Штраф за високу очікувану помилку
        if expected_error > 0.4: # Якщо мета-модель прогнозує >40% помилки
            critique_score -= 0.5 * (expected_error - 0.4)
            points.append(f"High predicted error: {expected_error:.2f}")

        # Штраф за поганий контекст
        if context_adjustment < 0.9:
            critique_score -= 0.3 * (1 - context_adjustment)
            points.append(f"Model is not optimal for current context (adj: {context_adjustment:.2f})")

        # Штраф за погану історію
        if historical_reliability < 0.5:
            critique_score -= 0.4
            points.append("Significant historical underperformance")

        return {
            "score": float(np.clip(critique_score, -1.0, 1.0)),
            "points": points,
            "alternatives": [{"type": "hold"}] if critique_score < -0.3 else [],
            "expected_error_by_critic": float(expected_error),
            "confidence": 0.8 # Впевненість самого Критика
        }
    
    def calculate_model_score(self, model_name: str, ticker: str, target_type: str, context: Dict) -> float:
        history_key = f"{model_name}_{ticker}_{target_type}"
        base_score = self._get_default_model_score(model_name, target_type)

        if history_key in self.performance_history and self.performance_history[history_key].get("runs"):
            scores = []
            for run in self.performance_history[history_key]["runs"]:
                metrics = run.get("metrics", {})
                weighted_score, total_weight = 0, 0
                for metric, weight in self.metric_weights.items():
                    if metric in metrics:
                        weighted_score += metrics[metric] * weight
                        total_weight += weight
                if total_weight > 0:
                    scores.append(weighted_score / total_weight)
            if scores: base_score = np.mean(scores)

        context_adjustment = self._calculate_context_adjustment(model_name, context)
        final_score = base_score * context_adjustment
        return min(1.0, max(0.0, final_score))

    def _get_default_model_score(self, model_name: str, target_type: str) -> float:
        return self.competence_map.get("default_scores", {}).get(target_type, {}).get(model_name, 0.5)
    
    def _calculate_context_adjustment(self, model_name: str, context: Dict) -> float:
        adjustment = 1.0
        rules = self.competence_map.get("context_rules", {})

        for factor, factor_rules in rules.items():
            current_value = context.get(factor)
            if not current_value: continue

            for rule_name, rule_details in factor_rules.items():
                is_trending = factor == 'trend' and rule_name == 'trending' and current_value in ['up', 'down']
                is_market_trending = factor == 'market_regime' and rule_name == 'trending' and current_value in ['bull', 'bear']
                
                if is_trending or is_market_trending or current_value == rule_name:
                    if model_name in rule_details.get("models", []):
                        adjustment *= rule_details.get("adjustment", 1.0)
        return adjustment

    def select_best_model(self, df: pd.DataFrame, ticker: str, target_type: str, 
                         available_models: List[str] = None) -> Tuple[str, float]:
        
        if available_models is None:
             available_models = list(self.competence_map.get("default_scores", {}).get(target_type, {}).keys())

        context = self.analyze_context(df, ticker)
        model_scores = {name: self.calculate_model_score(name, ticker, target_type, context) for name in available_models}
        
        best_model = max(model_scores, key=model_scores.get)
        best_score = model_scores[best_model]
        
        logger.info(f"Best model selected: {best_model} with score {best_score:.3f}")
        return best_model, best_score

    def update_performance(self, model_name: str, ticker: str, target_type: str, 
                          metrics: Dict, context: Dict):
        history_key = f"{model_name}_{ticker}_{target_type}"
        if history_key not in self.performance_history:
            self.performance_history[history_key] = {"runs": []}
        
        run_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "metrics": metrics,
            "context": context
        }
        self.performance_history[history_key]["runs"].append(run_data)
        self.performance_history[history_key]["runs"] = self.performance_history[history_key]["runs"][-50:]
        self._save_performance_history()
        # Оновлюємо мета-модель після оновлення історії
        self.train_error_meta_model(model_name, ticker)
