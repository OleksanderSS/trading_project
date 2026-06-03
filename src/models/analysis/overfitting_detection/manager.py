import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from .config import OverfittingConfig
from .metrics import OverfittingMetrics
from .analyzer import OverfittingAnalyzer
from .visualizer import OverfittingVisualizer
from src.core.logging.logger import ProjectLogger
from src.core.exceptions import DataProcessingError

logger = ProjectLogger.get_logger("OverfittingDetector")

class OverfittingDetector:
    """Orchestrator for the Overfitting Detector system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.config_manager = OverfittingConfig(config)
        self.metrics_calculator = OverfittingMetrics()
        self.analyzer = OverfittingAnalyzer(self.config_manager, self.metrics_calculator)
        self.visualizer = OverfittingVisualizer(self.config_manager)
        
        self.history = []
        self.logger.info("✅ OverfittingDetector (Modular) initialized")

    async def detect_overfitting(self, 
                               model: Any,
                               X_train: Any,
                               y_train: Any,
                               X_val: Optional[Any] = None,
                               y_val: Optional[Any] = None) -> Dict[str, Any]:
        """Detect overfitting signals in a model."""
        self.logger.info(f"🔍 Analyzing model: {type(model).__name__}")
        
        results = {
            'timestamp': datetime.now(),
            'model_type': type(model).__name__,
            'data_info': self.metrics_calculator.analyze_data_characteristics(X_train, X_val),
            'learning_curve': {},
            'cv_results': {},
            'overfitting_signals': {},
            'recommendations': []
        }
        
        try:
            # 1. Analysis steps
            results['learning_curve'] = await self.analyzer.generate_learning_curve(model, X_train, y_train)
            results['cv_results'] = await self.analyzer.perform_cv_analysis(model, X_train, y_train)
            
            if X_val is not None and y_val is not None:
                results['train_val_gap'] = self.analyzer.analyze_train_val_gap(model, X_train, y_train, X_val, y_val)
            
            # 2. Signal detection
            results['overfitting_signals'] = self.analyzer.detect_signals(
                results['learning_curve'], results['cv_results'], results.get('train_val_gap', {})
            )
            
            # 3. Recommendations
            results['recommendations'] = self.analyzer.generate_recommendations(results['overfitting_signals'])
            
            # 4. Visualizations
            if self.config_manager.enable_visualization:
                await self._create_visualizations(results)
                
            # 5. Storage
            self._store_results(results)
            
            return results
        except Exception as e:
            self.logger.error(f"Error in overfitting detection: {e}", exc_info=True)
            raise DataProcessingError(f"Overfitting detection failed: {e}") from e

    async def _create_visualizations(self, results: Dict[str, Any]):
        timestamp_str = results['timestamp'].strftime('%Y%m%d_%H%M%S')
        lc_path = self.config_manager.storage_path / f"learning_curve_{timestamp_str}.png"
        cv_path = self.config_manager.storage_path / f"cv_dist_{timestamp_str}.png"
        
        self.visualizer.plot_learning_curve(results['learning_curve'], lc_path)
        self.visualizer.plot_cv_distribution(results['cv_results'], cv_path)

    def _store_results(self, results: Dict[str, Any]):
        self.history.append(results)
        try:
            timestamp_str = results['timestamp'].strftime('%Y%m%d_%H%M%S')
            save_path = self.config_manager.storage_path / f"results_{timestamp_str}.json"
            
            # Convert timestamp for JSON
            save_data = results.copy()
            save_data['timestamp'] = save_data['timestamp'].isoformat()
            
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=4, default=str)
        except Exception as e:
            self.logger.error(f"Failed to store results: {e}", exc_info=True)
            raise DataProcessingError(f"Failed to store results: {e}") from e

    def get_overfitting_summary(self) -> Dict[str, Any]:
        """Get summary of recent analyses."""
        return {
            'total_analyses': len(self.history),
            'last_analysis_time': self.history[-1]['timestamp'] if self.history else None,
            'models_analyzed': list(set(r['model_type'] for r in self.history))
        }
