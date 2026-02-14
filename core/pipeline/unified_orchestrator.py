# core/pipeline/unified_orchestrator.py - Єдиний оркестратор with порandвнянням моwhereлей

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd

# Import pipeline components
from core.stages.stage_1_collectors_layer import run_stage_1_collect
from core.stages.stage_2_enrichment import run_stage_2_enrich
from core.stages.stage_3_features import prepare_stage3_datasets
from core.stages.stage_4_benchmark import benchmark_all_models
from core.stages.stage_5_pipeline_fixed import run_stage_5_with_models

# Import model comparison
from core.analysis.enhanced_model_comparator import EnhancedModelComparator
from core.models.model_interface import ModelFactory

# Import Colab integration
from utils.colab_manager import ColabManager
from utils.colab_utils import ColabUtils

logger = logging.getLogger(__name__)

class UnifiedOrchestrator:
    """Єдиний оркестратор with порandвнянням моwhereлей"""
    
    def __init__(self, tickers: Dict[str, str], time_frames: List[str], 
                 mode: str = "optimal", debug: bool = False):
        self.tickers = tickers
        self.time_frames = time_frames
        self.mode = mode
        self.debug = debug
        
        # Initialize managers
        self.colab_manager = ColabManager()
        self.colab_utils = ColabUtils()
        self.model_comparator = EnhancedModelComparator()
        
        # Pipeline state
        self.pipeline_state = {
            "stage_1_data": None,
            "stage_2_data": None,
            "stage_3_data": None,
            "stage_4_light_models": None,
            "stage_4_heavy_models": None,
            "stage_5_results": None,
            "model_comparison": None
        }
        
        logger.info(f"UnifiedOrchestrator initialized for {len(tickers)} tickers, {len(time_frames)} timeframes")
    
    def run_complete_pipeline_with_comparison(self, 
                                            compare_models: bool = True,
                                            colab_training: bool = True,
                                            models_to_compare: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Запуск повного пайплайну with порandвнянням моwhereлей
        
        Args:
            compare_models: Чи порandвнювати моwhereлand пandсля тренування
            colab_training: Чи use Colab for важких моwhereлей
            models_to_compare: Список моwhereлей for порandвняння
            
        Returns:
            Dictionary with реwithульandandми пайплайну and порandвняння моwhereлей
        """
        start_time = time.time()
        
        logger.info("="*60)
        logger.info("STARTING UNIFIED PIPELINE WITH MODEL COMPARISON")
        logger.info("="*60)
        
        try:
            # Stage 1: Data Collection
            logger.info("[Stage 1] Running data collection...")
            stage_1_results = self._run_stage_1_local()
            self.pipeline_state["stage_1_data"] = stage_1_results
            
            # Stage 2: Data Enrichment
            logger.info("[Stage 2] Running data enrichment...")
            stage_2_results = self._run_stage_2_local(stage_1_results)
            self.pipeline_state["stage_2_data"] = stage_2_results
            
            # Stage 3: Feature Engineering
            logger.info("[Stage 3] Running feature engineering...")
            stage_3_results = self._run_stage_3_local(stage_2_results)
            self.pipeline_state["stage_3_data"] = stage_3_results
            
            # Stage 4: Model Training & Comparison
            logger.info("[Stage 4] Running model training...")
            stage_4_results = self._run_stage_4_with_comparison(
                stage_3_results, 
                compare_models=compare_models,
                colab_training=colab_training,
                models_to_compare=models_to_compare
            )
            self.pipeline_state.update(stage_4_results)
            
            # Stage 5: Signal Generation
            logger.info("[Stage 5] Running signal generation...")
            stage_5_results = self._run_stage_5_with_models(stage_4_results)
            self.pipeline_state["stage_5_results"] = stage_5_results
            
            # Compile final results
            total_time = time.time() - start_time
            
            final_results = {
                "pipeline_type": f"unified_with_comparison_{self.mode}",
                "total_time": total_time,
                "timestamp": datetime.now().isoformat(),
                "stage_1": stage_1_results,
                "stage_2": stage_2_results,
                "stage_3": stage_3_results,
                "stage_4": stage_4_results,
                "stage_5": stage_5_results,
                "model_comparison": stage_4_results.get("model_comparison"),
                "summary": self._generate_pipeline_summary(),
                "recommendations": self._generate_recommendations(stage_4_results)
            }
            
            logger.info("="*60)
            logger.info(f"UNIFIED PIPELINE COMPLETED IN {total_time:.1f}s")
            logger.info("="*60)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _run_stage_1_local(self) -> Dict[str, Any]:
        """Run Stage 1: Data Collection locally"""
        logger.info("[Stage 1] Running data collection locally...")
        
        results = run_stage_1_collect(debug_no_network=self.debug)
        
        logger.info(f"[Stage 1] Completed: {len(results)} data sources")
        return results
    
    def _run_stage_2_local(self, stage_1_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stage 2: Data Enrichment locally"""
        logger.info("[Stage 2] Running data enrichment locally...")
        
        results = run_stage_2_enrich(
            stage1_data=stage_1_data,
            keyword_dict={},
            tickers=list(self.tickers.keys()),
            time_frames=self.time_frames,
            mode="train"
        )
        
        logger.info(f"[Stage 2] Completed: enriched data available")
        return results
    
    def _run_stage_3_local(self, stage_2_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stage 3: Feature Engineering locally"""
        logger.info("[Stage 3] Running feature engineering locally...")
        
        results = prepare_stage3_datasets(stage_2_data)
        
        logger.info(f"[Stage 3] Completed: features prepared")
        return results
    
    def _run_stage_4_with_comparison(self, stage_3_data: Dict[str, Any],
                                   compare_models: bool = True,
                                   colab_training: bool = True,
                                   models_to_compare: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run Stage 4: Model Training with Comparison"""
        logger.info("[Stage 4] Running model training with comparison...")
        
        stage_4_results = {}
        
        # Light Models Training
        logger.info("[Stage 4] Training light models...")
        light_models_results = self._train_light_models(stage_3_data)
        stage_4_results["stage_4_light_models"] = light_models_results
        
        # Heavy Models Training (if Colab available)
        heavy_models_results = {}
        if colab_training:
            logger.info("[Stage 4] Training heavy models in Colab...")
            heavy_models_results = self._train_heavy_models(stage_3_data)
        
        stage_4_results["stage_4_heavy_models"] = heavy_models_results
        
        # Model Comparison
        model_comparison_results = None
        if compare_models:
            logger.info("[Stage 4] Running model comparison...")
            model_comparison_results = self._run_model_comparison(
                stage_3_data, 
                light_models_results,
                heavy_models_results,
                models_to_compare
            )
        
        stage_4_results["model_comparison"] = model_comparison_results
        
        logger.info(f"[Stage 4] Completed: models trained and compared")
        return stage_4_results
    
    def _train_light_models(self, stage_3_data: Dict[str, Any]) -> Dict[str, Any]:
        """Тренування легких моwhereлей"""
        try:
            from core.stages.stage_4_benchmark import ALL_MODELS
            
            light_models = [m for m in ALL_MODELS if m in [
                "lgbm", "rf", "xgb", "catboost", "linear", "svm", "knn"
            ]]
            
            # Використовуємо andснуючий benchmark
            results = benchmark_all_models(
                merged_df=stage_3_data.get("merged_df", pd.DataFrame()),
                models=light_models,
                tickers=list(self.tickers.keys()),
                time_frames=self.time_frames
            )
            
            return {"benchmark_results": results, "models": light_models}
            
        except Exception as e:
            logger.error(f"Error training light models: {e}")
            return {"error": str(e)}
    
    def _train_heavy_models(self, stage_3_data: Dict[str, Any]) -> Dict[str, Any]:
        """Тренування важких моwhereлей в Colab"""
        try:
            if not self.colab_manager.is_colab_available():
                logger.warning("Colab not available, skipping heavy models")
                return {"error": "Colab not available"}
            
            heavy_models = ["lstm", "cnn", "transformer", "mlp", "autoencoder"]
            
            # Пandдготовка data for Colab
            merged_df = stage_3_data.get("merged_df", pd.DataFrame())
            
            if merged_df.empty:
                return {"error": "No data available for heavy models"}
            
            # Вandдправка в Colab
            colab_results = {}
            for model_type in heavy_models:
                try:
                    result = self.colab_manager.train_model(
                        model_type=model_type,
                        data=merged_df,
                        tickers=list(self.tickers.keys()),
                        time_frames=self.time_frames
                    )
                    colab_results[model_type] = result
                except Exception as e:
                    logger.error(f"Error training {model_type} in Colab: {e}")
                    colab_results[model_type] = {"error": str(e)}
            
            return {"colab_results": colab_results, "models": heavy_models}
            
        except Exception as e:
            logger.error(f"Error training heavy models: {e}")
            return {"error": str(e)}
    
    def _run_model_comparison(self, stage_3_data: Dict[str, Any],
                            light_results: Dict[str, Any],
                            heavy_results: Dict[str, Any],
                            models_to_compare: Optional[List[str]] = None) -> Dict[str, Any]:
        """Порandвняння моwhereлей"""
        try:
            merged_df = stage_3_data.get("merged_df", pd.DataFrame())
            
            if merged_df.empty:
                return {"error": "No data available for comparison"}
            
            # Пandдготовка конфandгурацandї моwhereлей
            models_config = {}
            
            # Легкand моwhereлand
            if light_results.get("benchmark_results") is not None:
                light_models = light_results.get("models", [])
                for model in light_models:
                    if models_to_compare is None or model in models_to_compare:
                        models_config[model] = {
                            "type": model,
                            "category": "light",
                            "task_type": "regression"
                        }
            
            # Важкand моwhereлand
            if heavy_results.get("colab_results"):
                heavy_models = heavy_results.get("models", [])
                for model in heavy_models:
                    if models_to_compare is None or model in models_to_compare:
                        colab_result = heavy_results["colab_results"].get(model, {})
                        if not colab_result.get("error"):
                            models_config[model] = {
                                "type": model,
                                "category": "heavy",
                                "task_type": "regression"
                            }
            
            if not models_config:
                return {"error": "No models available for comparison"}
            
            # Пandдготовка data for порandвняння
            # Використовуємо перший available тandкер and andймфрейм
            sample_ticker = list(self.tickers.keys())[0]
            sample_timeframe = self.time_frames[0]
            
            # Фandльтрацandя data
            ticker_data = merged_df[merged_df['ticker'] == sample_ticker]
            
            if ticker_data.empty:
                return {"error": f"No data for ticker {sample_ticker}"}
            
            # Пandдготовка фandч and andргету
            feature_cols = [col for col in ticker_data.columns 
                          if col not in ['date', 'ticker', 'timeframe', 'target'] 
                          and ticker_data[col].dtype in ['float64', 'int64']]
            
            if not feature_cols:
                return {"error": "No numeric features available"}
            
            X = ticker_data[feature_cols].values
            y = ticker_data['target'].values if 'target' in ticker_data.columns else ticker_data[feature_cols[0]].values
            
            # Роwithдandлення на train/test
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Порandвняння моwhereлей
            comparison_results = self.model_comparator.compare_models(
                models_config=models_config,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                ticker=sample_ticker,
                timeframe=sample_timeframe
            )
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error in model comparison: {e}")
            return {"error": str(e)}
    
    def _run_stage_5_with_models(self, stage_4_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stage 5: Signal Generation with trained models"""
        logger.info("[Stage 5] Running signal generation...")
        
        try:
            # Використовуємо найкращand моwhereлand with порandвняння
            best_models = {}
            
            if stage_4_results.get("model_comparison"):
                comparison = stage_4_results["model_comparison"]
                summary = comparison.get("summary", {})
                best_overall = summary.get("best_models", {}).get("combined")
                
                if best_overall:
                    best_models["best_overall"] = best_overall
            
            # Якщо notмає порandвняння, використовуємо легкand моwhereлand
            if not best_models and stage_4_results.get("stage_4_light_models"):
                light_results = stage_4_results["stage_4_light_models"]
                best_models["default_light"] = "random_forest"
            
            # Запуск Stage 5 with найкращими моwhereлями
            results = run_stage_5_with_models(
                light_models=stage_4_results.get("stage_4_light_models", {}),
                heavy_models=stage_4_results.get("stage_4_heavy_models", {}),
                selected_models=best_models
            )
            
            logger.info(f"[Stage 5] Completed: signals generated")
            return results
            
        except Exception as e:
            logger.error(f"Error in Stage 5: {e}")
            return {"error": str(e)}
    
    def _generate_pipeline_summary(self) -> Dict[str, Any]:
        """Геnotрацandя пandдсумку пайплайну"""
        summary = {
            "pipeline_mode": self.mode,
            "tickers": list(self.tickers.keys()),
            "time_frames": self.time_frames,
            "stages_completed": [],
            "models_trained": {},
            "model_comparison_available": False
        }
        
        # Перевandрка forвершених еandпandв
        for stage_name, stage_data in self.pipeline_state.items():
            if stage_data and not stage_data.get("error"):
                summary["stages_completed"].append(stage_name)
        
        # Пandдрахунок моwhereлей
        if self.pipeline_state.get("stage_4_light_models"):
            light_models = self.pipeline_state["stage_4_light_models"].get("models", [])
            summary["models_trained"]["light"] = len(light_models)
        
        if self.pipeline_state.get("stage_4_heavy_models"):
            heavy_models = self.pipeline_state["stage_4_heavy_models"].get("models", [])
            summary["models_trained"]["heavy"] = len(heavy_models)
        
        # Наявнandсть порandвняння
        if self.pipeline_state.get("model_comparison"):
            summary["model_comparison_available"] = True
            comparison = self.pipeline_state["model_comparison"]
            summary["best_models"] = comparison.get("summary", {}).get("best_models", {})
        
        return summary
    
    def _generate_recommendations(self, stage_4_results: Dict[str, Any]) -> List[str]:
        """Геnotрацandя рекомендацandй на основand реwithульandтandв"""
        recommendations = []
        
        try:
            # Рекомендацandї на основand порandвняння моwhereлей
            if stage_4_results.get("model_comparison"):
                comparison = stage_4_results["model_comparison"]
                summary = comparison.get("summary", {})
                
                if summary.get("successful_models", 0) > 0:
                    best_overall = summary.get("best_models", {}).get("combined")
                    if best_overall:
                        recommendations.append(f"Use {best_overall} as primary model")
                    
                    best_r2 = summary.get("best_models", {}).get("r2")
                    if best_r2 and best_r2 != best_overall:
                        recommendations.append(f"Consider {best_r2} for best accuracy")
                    
                    fastest = summary.get("best_models", {}).get("training_time")
                    if fastest and fastest != best_overall:
                        recommendations.append(f"Use {fastest} for real-time predictions")
                
                # Рекомендацandї на основand сandтистичних тестandв
                significant_diffs = summary.get("significant_differences", [])
                if significant_diffs:
                    recommendations.append("Models show statistically significant differences - consider ensemble methods")
            
            # Рекомендацandї на основand доступних моwhereлей
            light_models = stage_4_results.get("stage_4_light_models", {}).get("models", [])
            heavy_models = stage_4_results.get("stage_4_heavy_models", {}).get("models", [])
            
            if light_models and not heavy_models:
                recommendations.append("Consider enabling Colab for heavy models to improve accuracy")
            
            if not light_models and not heavy_models:
                recommendations.append("No models trained successfully - check data quality and configuration")
            
            # Загальнand рекомендацandї
            if len(recommendations) == 0:
                recommendations.append("Pipeline completed successfully - review detailed results for optimization opportunities")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to errors")
        
        return recommendations
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Отримати сandтус пайплайну"""
        return {
            "pipeline_state": self.pipeline_state,
            "summary": self._generate_pipeline_summary(),
            "timestamp": datetime.now().isoformat()
        }
    
    def save_pipeline_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Збереження реwithульandтandв пайплайну"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_results_{timestamp}.json"
        
        results_path = Path(f"results/pipeline/{filename}")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Конверandцandя for JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        json_results = convert_numpy(results)
        
        import json
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Pipeline results saved to {results_path}")
        return str(results_path)
