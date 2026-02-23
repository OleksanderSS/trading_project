# profit_optimizer.py - Оптимandforцandя прогноwithування for максимального forробandтку

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

from utils.logger import ProjectLogger
from utils.target_utils import get_model_config
from config.feature_config import CORE_FEATURES, TICKER_TARGET_MAP
from core.stages.stage_4_modeling import run_stage_4_modeling

logger = ProjectLogger.get_logger("ProfitOptimizer")

class ProfitOptimizer:
    """Оптимandforтор for максимального forробandтку череwith покращення точностand прогноwithandв"""
    
    def __init__(self):
        # Найкращand моwhereлand for рandwithних типandв цandлей
        self.best_models = {
            "direction": ["lgbm", "xgb", "rf", "catboost"],  # Класифandкацandя напрямку
            "pct_change": ["linear", "mlp", "lstm", "transformer"]  # Регресandя % differences
        }
        
        # Оптимальнand фandчand for прибутковостand
        self.profit_features = [
            # Технandчнand сигнали
            "1d_RSI_14_spy", "1d_RSI_14_qqq", "1d_RSI_14_tsla", "1d_RSI_14_nvda",
            "1d_SMA_50_spy", "1d_SMA_50_qqq", "1d_SMA_20_spy", "1d_SMA_20_qqq",
            "1d_close_spy", "1d_close_qqq", "1d_close_tsla", "1d_close_nvda",
            "1d_volume_spy", "1d_volume_qqq",
            
            # Макро andндикатори
            "VIX_SIGNAL", "FEDFUNDS_SIGNAL", "T10Y2Y_SIGNAL", 
            "CPI_inflation", "UNRATE_diff",
            
            # Сентимент
            "sentiment_score", "match_count", "impact_score", "has_news",
            "sentiment_vix_interaction",
            
            # Календар
            "weekday", "is_earnings_day", "hour_of_day"
        ]
        
        self.results = {}
    
    def optimize_target_prediction(self, df: pd.DataFrame, ticker: str, interval: str = "1d") -> Dict:
        """Оптимandwithує прогноwithування цandлand for максимального forробandтку"""
        logger.info(f"[TARGET] Оптимandforцandя прогноwithування for {ticker} {interval}")
        
        results = {
            "ticker": ticker,
            "interval": interval,
            "best_model": None,
            "best_score": 0,
            "best_target": None,
            "models_tested": {},
            "profit_potential": 0
        }
        
        # Тестуємо обидва типи цandлей
        for target_type in ["direction", "pct_change"]:
            logger.info(f"[DATA] Тестування цandлand: {target_type}")
            
            # Отримуємо найкращand моwhereлand for цього типу цandлand
            models_to_test = self.best_models[target_type]
            
            for model_name in models_to_test:
                try:
                    # Запускаємо model
                    model, df_results, metrics = run_stage_4_modeling(
                        merged_df=df,
                        model_name=model_name,
                        ticker=ticker,
                        interval=interval,
                        background_layers=["macro", "trend"],
                        impulse_layers=["technical", "news"]
                    )
                    
                    if model is None or df_results is None:
                        continue
                    
                    # Оцandнюємо якandсть прогноwithу
                    score = self._evaluate_prediction_quality(df_results, target_type, metrics)
                    
                    # Оцandнюємо потенцandал прибутку
                    profit_score = self._calculate_profit_potential(df_results, target_type)
                    
                    # Комбandнований скор (точнandсть + прибутковandсть)
                    combined_score = score * 0.6 + profit_score * 0.4
                    
                    results["models_tested"][f"{model_name}_{target_type}"] = {
                        "accuracy_score": score,
                        "profit_score": profit_score,
                        "combined_score": combined_score,
                        "metrics": metrics["metrics"] if metrics else {}
                    }
                    
                    # Оновлюємо найкращий реwithульandт
                    if combined_score > results["best_score"]:
                        results["best_score"] = combined_score
                        results["best_model"] = model_name
                        results["best_target"] = target_type
                        results["profit_potential"] = profit_score
                        results["best_metrics"] = metrics["metrics"] if metrics else {}
                    
                    logger.info(f"[OK] {model_name}_{target_type}: accuracy={score:.3f}, profit={profit_score:.3f}, combined={combined_score:.3f}")
                    
                except Exception as e:
                    logger.error(f"[ERROR] Error {model_name}_{target_type}: {e}")
                    continue
        
        return results
    
    def _evaluate_prediction_quality(self, df_results: pd.DataFrame, target_type: str, metrics: Dict) -> float:
        """Оцandнює якandсть прогноwithування"""
        if not metrics or "metrics" not in metrics:
            return 0.0
        
        m = metrics["metrics"]
        
        if target_type == "direction":
            # Для класифandкацandї - F1 score
            return m.get("F1", m.get("f1", m.get("accuracy", 0.0)))
        else:
            # Для регресandї - R2 (чим вище, тим краще)
            r2 = m.get("r2", m.get("R2", 0.0))
            return max(0, r2)  # R2 may бути notгативним
    
    def _calculate_profit_potential(self, df_results: pd.DataFrame, target_type: str) -> float:
        """Роwithраховує потенцandал прибутку на основand прогноwithandв"""
        if df_results.empty:
            return 0.0
        
        try:
            # Знаходимо колонки прогноwithу and реальних withначень
            pred_cols = [c for c in df_results.columns if c.startswith("predicted_")]
            target_cols = [c for c in df_results.columns if c.startswith("target_")]
            
            if not pred_cols or not target_cols:
                return 0.0
            
            pred_col = pred_cols[0]
            target_col = target_cols[0]
            
            predictions = df_results[pred_col].values
            actual = df_results[target_col].values
            
            # Видаляємо NaN
            mask = ~(np.isnan(predictions) | np.isnan(actual))
            predictions = predictions[mask]
            actual = actual[mask]
            
            if len(predictions) == 0:
                return 0.0
            
            if target_type == "direction":
                # Для класифandкацandї - точнandсть сигналandв
                correct_predictions = (predictions == actual).sum()
                total_predictions = len(predictions)
                accuracy = correct_predictions / total_predictions
                
                # Бонус for правильнand сигнали на великих рухах
                if "close" in df_results.columns:
                    returns = df_results["close"].pct_change().abs()
                    big_moves = returns > returns.quantile(0.8)  # Топ 20% рухandв
                    if big_moves.sum() > 0:
                        big_move_accuracy = ((predictions == actual) & big_moves[mask]).sum() / big_moves.sum()
                        accuracy = accuracy * 0.7 + big_move_accuracy * 0.3
                
                return accuracy
            
            else:
                # Для регресandї - кореляцandя with реальними withмandнами
                correlation = np.corrcoef(predictions, actual)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                
                # Нормалandwithуємо до [0, 1]
                return (correlation + 1) / 2
        
        except Exception as e:
            logger.warning(f"Error роwithрахунку profit potential: {e}")
            return 0.0
    
    def optimize_all_tickers(self, df: pd.DataFrame, tickers: List[str] = None) -> Dict:
        """Оптимandwithує прогноwithування for allх тandкерandв"""
        if tickers is None:
            tickers = ["SPY", "QQQ", "TSLA", "NVDA"]
        
        logger.info(f"[START] Запуск оптимandforцandї for тandкерandв: {tickers}")
        
        all_results = {}
        summary = {
            "best_overall": {"ticker": None, "score": 0, "model": None, "target": None},
            "by_ticker": {},
            "recommendations": []
        }
        
        for ticker in tickers:
            logger.info(f"\n[UP] Оптимandforцandя for {ticker}")
            
            # Фandльтруємо данand for тandкера
            ticker_data = df[df["ticker"].str.upper() == ticker.upper()].copy()
            
            if ticker_data.empty:
                logger.warning(f"[WARN] Немає data for {ticker}")
                continue
            
            # Оптимandwithуємо
            result = self.optimize_target_prediction(ticker_data, ticker)
            all_results[ticker] = result
            
            # Оновлюємо forгальний рейтинг
            if result["best_score"] > summary["best_overall"]["score"]:
                summary["best_overall"] = {
                    "ticker": ticker,
                    "score": result["best_score"],
                    "model": result["best_model"],
                    "target": result["best_target"]
                }
            
            summary["by_ticker"][ticker] = {
                "best_model": result["best_model"],
                "best_target": result["best_target"],
                "score": result["best_score"],
                "profit_potential": result["profit_potential"]
            }
        
        # Геnotруємо рекомендацandї
        summary["recommendations"] = self._generate_recommendations(all_results)
        
        self.results = {"detailed": all_results, "summary": summary}
        return self.results
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Геnotрує рекомендацandї for покращення прибутковостand"""
        recommendations = []
        
        # Аналandwithуємо реwithульandти
        best_models = {}
        best_targets = {}
        
        for ticker, result in results.items():
            if result["best_model"]:
                model = result["best_model"]
                target = result["best_target"]
                
                best_models[model] = best_models.get(model, 0) + 1
                best_targets[target] = best_targets.get(target, 0) + 1
        
        # Рекомендацandї по моwhereлях
        if best_models:
            top_model = max(best_models, key=best_models.get)
            recommendations.append(f"[BEST] Найкраща model: {top_model} (використовується for {best_models[top_model]} тandкерandв)")
        
        # Рекомендацandї по цandлях
        if best_targets:
            top_target = max(best_targets, key=best_targets.get)
            recommendations.append(f"[TARGET] Найкраща цandль: {top_target} (оптимальна for {best_targets[top_target]} тandкерandв)")
        
        # Загальнand рекомендацandї
        avg_score = np.mean([r["best_score"] for r in results.values() if r["best_score"] > 0])
        if avg_score < 0.6:
            recommendations.append("[WARN] Ниwithька точнandсть прогноwithandв. Рекомендується withбandльшити кandлькandсть data or покращити фandчand")
        elif avg_score > 0.8:
            recommendations.append("[OK] Висока точнandсть прогноwithandв. Можна переходити до реального трейдингу")
        
        return recommendations
    
    def print_results(self):
        """Виводить реwithульandти оптимandforцandї"""
        if not self.results:
            logger.warning("Немає реwithульandтandв for output")
            return
        
        summary = self.results["summary"]
        
        print("\n" + "="*60)
        print("[TARGET] РЕЗУЛЬТАТИ ОПТИМІЗАЦІЇ ПРОГНОЗУВАННЯ")
        print("="*60)
        
        # Найкращий forгальний реwithульandт
        best = summary["best_overall"]
        if best["ticker"]:
            print(f"\n[BEST] НАЙКРАЩИЙ РЕЗУЛЬТАТ:")
            print(f"   Тandкер: {best['ticker']}")
            print(f"   Моwhereль: {best['model']}")
            print(f"   Цandль: {best['target']}")
            print(f"   Скор: {best['score']:.3f}")
        
        # Реwithульandти по тandкерах
        print(f"\n[DATA] РЕЗУЛЬТАТИ ПО ТІКЕРАХ:")
        for ticker, data in summary["by_ticker"].items():
            print(f"   {ticker}: {data['best_model']}_{data['best_target']} (скор: {data['score']:.3f}, рибуток: {data['profit_potential']:.3f})")
        
        # Рекомендацandї
        print(f"\n[IDEA] РЕКОМЕНДАЦІЇ:")
        for rec in summary["recommendations"]:
            print(f"   {rec}")
        
        print("\n" + "="*60)

def main():
    """Основна функцandя for forпуску оптимandforцandї"""
    logger.info("[START] Запуск оптимandforтора прибутковостand")
    
    try:
        # Заванandжуємо данand
        from core.data_accumulator import DataAccumulator
        accumulator = DataAccumulator()
        df = accumulator.get_merged_data()
        
        if df.empty:
            logger.error("[ERROR] Немає data for оптимandforцandї")
            return
        
        # Запускаємо оптимandforцandю
        optimizer = ProfitOptimizer()
        results = optimizer.optimize_all_tickers(df)
        
        # Виводимо реwithульandти
        optimizer.print_results()
        
        # Зберandгаємо реwithульandти
        import json
        with open("profit_optimization_results.json", "w") as f:
            # Конвертуємо numpy типи for JSON
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(results, f, indent=2, default=convert_numpy)
        
        logger.info("[OK] Реwithульandти withбережено у profit_optimization_results.json")
        
    except Exception as e:
        logger.error(f"[ERROR] Error оптимandforцandї: {e}")
        raise

if __name__ == "__main__":
    main()