#!/usr/bin/env python3
"""
Main Entry Point - Єдина точка входу для всього пайплайну
Повноцінна реалізація з гнучкими тікерами, відновленням, моніторингом
"""

import yaml
import os
import sys
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import traceback
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed

# [TARGET] ІМПОРТИ МОДУЛІВ
# ВИПРАВЛЕНО: використовуємо StageManager замість видалених функцій
from core.stages.stage_manager import StageManager
from core.stages.stage_2_enrichment import run_stage_2_enrich_ideal
from core.stages.stage_3_features import run_stage_3
from models.stage_4_unified_training import run_stage_4_unified

# [TARGET] ІМПОРТУЄМО ІСНУЮЧУ СИСТЕМУ ВИБОРУ ТІКЕРІВ
from config.enhanced_sector_tickers import get_enhanced_tickers, get_sector_config_for_risk, analyze_sectors, recommend_portfolio

logger = logging.getLogger(__name__)

@dataclass
class PipelineState:
    """Стан пайплайну для відновлення"""
    current_stage: int = 0
    completed_stages: List[int] = field(default_factory=list)
    failed_stages: List[int] = field(default_factory=list)
    stage_results: Dict[int, Dict] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    last_checkpoint: datetime = field(default_factory=datetime.now)
    error_log: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineMetrics:
    """Метрики пайплайну для моніторингу"""
    data_quality_score: float = 0.0
    feature_count: int = 0
    model_performance: float = 0.0
    execution_time: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0
    stages_completed: int = 0
    total_stages: int = 4

class PipelineMonitor:
    """Моніторинг пайплайну в реальному часі"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = PipelineMetrics()
        self.start_time = datetime.now()
        self.stage_times = {}
        self.progress_callbacks = []
        
    def start_stage(self, stage_name: str):
        """Початок етапу"""
        self.stage_times[stage_name] = {'start': datetime.now()}
        logger.info(f"[START] Starting stage: {stage_name}")
        self._notify_progress(f"Stage {stage_name} started", 0)
        
    def complete_stage(self, stage_name: str, result: Dict):
        """Завершення етапу"""
        if stage_name in self.stage_times:
            self.stage_times[stage_name]['end'] = datetime.now()
            self.stage_times[stage_name]['duration'] = (
                self.stage_times[stage_name]['end'] - self.stage_times[stage_name]['start']
            ).total_seconds()
        
        self.metrics.stages_completed += 1
        progress = (self.metrics.stages_completed / self.metrics.total_stages) * 100
        logger.info(f"[OK] Stage {stage_name} completed in {self.stage_times[stage_name].get('duration', 0):.2f}s")
        self._notify_progress(f"Stage {stage_name} completed", progress)
        
    def update_metrics(self, **kwargs):
        """Оновлення метрик"""
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
        
        self._notify_metrics_update()
        
    def _notify_progress(self, message: str, progress: float):
        """Сповіщення про прогрес"""
        for callback in self.progress_callbacks:
            callback(message, progress)
            
    def _notify_metrics_update(self):
        """Сповіщення про оновлення метрик"""
        metrics_dict = {
            'data_quality_score': self.metrics.data_quality_score,
            'feature_count': self.metrics.feature_count,
            'model_performance': self.metrics.model_performance,
            'execution_time': time.time() - self.start_time.timestamp(),
            'memory_usage': self.metrics.memory_usage,
            'error_rate': self.metrics.error_rate,
            'progress': (self.metrics.stages_completed / self.metrics.total_stages) * 100
        }
        
        for callback in self.progress_callbacks:
            callback("metrics_update", metrics_dict)

class PipelineRecovery:
    """Система відновлення після збоїв"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.recovery_enabled = config.get('recovery', {}).get('enabled', True)
        self.checkpoint_dir = Path(config.get('storage', {}).get('base_path', 'results/')) / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, state: PipelineState):
        """Збереження checkpoint"""
        if not self.recovery_enabled:
            return
            
        checkpoint_file = self.checkpoint_dir / f"pipeline_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        checkpoint_data = {
            'current_stage': state.current_stage,
            'completed_stages': state.completed_stages,
            'failed_stages': state.failed_stages,
            'stage_results': state.stage_results,
            'start_time': state.start_time.isoformat(),
            'last_checkpoint': state.last_checkpoint.isoformat(),
            'error_log': state.error_log,
            'metadata': state.metadata
        }
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.info(f"[SAVE] Checkpoint saved: {checkpoint_file}")
            
            # Видаляємо старі checkpoints
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            logger.error(f"[ERROR] Error saving checkpoint: {e}")
    
    def load_latest_checkpoint(self) -> Optional[PipelineState]:
        """Завантаження останнього checkpoint"""
        if not self.recovery_enabled:
            return None
            
        checkpoint_files = list(self.checkpoint_dir.glob("pipeline_checkpoint_*.json"))
        if not checkpoint_files:
            return None
            
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_checkpoint, 'r') as f:
                checkpoint_data = json.load(f)
            
            state = PipelineState()
            state.current_stage = checkpoint_data['current_stage']
            state.completed_stages = checkpoint_data['completed_stages']
            state.failed_stages = checkpoint_data['failed_stages']
            state.stage_results = checkpoint_data['stage_results']
            state.start_time = datetime.fromisoformat(checkpoint_data['start_time'])
            state.last_checkpoint = datetime.fromisoformat(checkpoint_data['last_checkpoint'])
            state.error_log = checkpoint_data['error_log']
            state.metadata = checkpoint_data['metadata']
            
            logger.info(f"[RESTART] Checkpoint loaded: {latest_checkpoint}")
            return state
            
        except Exception as e:
            logger.error(f"[ERROR] Error loading checkpoint: {e}")
            return None
    
    def _cleanup_old_checkpoints(self):
        """Очищення старих checkpoints"""
        checkpoint_files = list(self.checkpoint_dir.glob("pipeline_checkpoint_*.json"))
        if len(checkpoint_files) > 5:  # Зберігаємо останні 5
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for old_checkpoint in checkpoint_files[5:]:
                try:
                    old_checkpoint.unlink()
                    logger.info(f"🗑️ Deleted old checkpoint: {old_checkpoint}")
                except Exception as e:
                    logger.error(f"[ERROR] Error deleting old checkpoint: {e}")

class TickerSelector:
    """Гнучкий вибір тікерів - ІНТЕГРУЄМО ІСНУЮЧУ СИСТЕМУ"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tickers_config = config.get('tickers', {})
        
    def select_tickers(self, selection: Union[str, List[str]] = None, limit: Optional[int] = None) -> List[str]:
        """
        Гнучкий вибір тікерів - ВИКОРИСТОВУЄМО ІСНУЮЧУ СИСТЕМУ
        
        Args:
            selection: 'default', 'all', 'conservative', 'aggressive', 'diversified', 'quick_test', 
                     або стратегії з enhanced_sector_tickers:
                     'extreme_volatility', 'balanced_growth', 'news_driven', 'momentum', 'conservative'
            limit: Обмеження кількості тікерів
        
        Returns:
            List[str]: Вибрані тікери
        """
        if isinstance(selection, list):
            return selection
        
        if selection is None:
            selection = self.tickers_config.get('default_selection', ['large_cap'])
        
        # [TARGET] ВИКОРИСТОВУЄМО ІСНУЮЧУ СИСТЕМУ ДЛЯ СТРАТЕГІЙ
        if selection in ['extreme_volatility', 'balanced_growth', 'news_driven', 'momentum', 'conservative']:
            logger.info(f"[TARGET] Using enhanced sector strategy: {selection}")
            return get_enhanced_tickers(selection, limit)
        
        # [TARGET] ВИКОРИСТОВУЄМО YAML КОНФІГУРАЦІЮ ДЛЯ ПРЕСЕТІВ
        elif selection in self.tickers_config.get('presets', {}):
            preset_tickers = []
            preset_categories = self.tickers_config['presets'][selection]
            for category in preset_categories:
                preset_tickers.extend(self.tickers_config.get(category, []))
            result = list(set(preset_tickers))  # Унікальні тікери
            return result[:limit] if limit else result
        
        elif selection == 'all':
            return self._get_all_tickers()
        
        elif selection in self.tickers_config:
            return self.tickers_config[selection]
        
        else:
            logger.warning(f"Unknown ticker selection: {selection}, using default")
            return self._get_default_tickers()
    
    def get_sector_analysis(self) -> pd.DataFrame:
        """Отримати аналіз секторів - ВИКОРИСТОВУЄМО ІСНУЮЧУ СИСТЕМУ"""
        return analyze_sectors()
    
    def get_portfolio_recommendation(self, capital: float, max_positions: int = 10) -> Dict[str, any]:
        """Отримати рекомендацію портфеля - ВИКОРИСТОВУЄМО ІСНУЮЧУ СИСТЕМУ"""
        return recommend_portfolio(capital, max_positions)
    
    def get_risk_configuration(self, risk_tolerance: str, capital: float) -> Dict[str, any]:
        """Отримати конфігурацію для рівня ризику - ВИКОРИСТОВУЄМО ІСНУЮЧУ СИСТЕМУ"""
        return get_sector_config_for_risk(risk_tolerance, capital)
    
    def _get_all_tickers(self) -> List[str]:
        """Отримати всі тікери"""
        all_tickers = []
        for category, tickers in self.tickers_config.items():
            if isinstance(tickers, list):
                all_tickers.extend(tickers)
        return list(set(all_tickers))
    
    def _get_default_tickers(self) -> List[str]:
        """Отримати тікери за замовчуванням"""
        # [TARGET] ВИКОРИСТОВУЄМО ІСНУЮЧУ СИСТЕМУ
        return get_enhanced_tickers("balanced_growth")

class MainPipeline:
    """
    [START] Основний пайплайн - єдина точка входу
    """
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.state = PipelineState()
        self.monitor = PipelineMonitor(self.config)
        self.recovery = PipelineRecovery(self.config)
        self.ticker_selector = TickerSelector(self.config)
        
        # [TARGET] Налаштування логування
        self._setup_logging()
        
        # [TARGET] Обробники сигналів для graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.shutdown_requested = False
        
    def _load_config(self) -> Dict:
        """Завантаження конфігурації"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"[OK] Config loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"[ERROR] Error loading config: {e}")
            raise
    
    def _setup_logging(self):
        """Налаштування логування"""
        log_config = self.config.get('monitoring', {})
        log_level = getattr(logging, log_config.get('log_level', 'INFO'))
        log_file = log_config.get('log_file', 'logs/pipeline.log')
        
        # Створюємо директорію для логів
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Налаштування логера
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger.info("📝 Logging configured")
    
    def _signal_handler(self, signum, frame):
        """Обробник сигналів для graceful shutdown"""
        logger.info(f"🛑 Signal {signum} received, shutting down gracefully...")
        self.shutdown_requested = True
    
    def add_progress_callback(self, callback):
        """Додати callback для прогресу"""
        self.monitor.progress_callbacks.append(callback)
    
    def run_pipeline(self, 
                    tickers: Union[str, List[str]] = None,
                    timeframes: Union[str, List[str]] = None,
                    config_overrides: Dict = None) -> Dict:
        """
        [START] Запуск повного пайплайну
        
        Args:
            tickers: Вибір тікерів ('default', 'all', 'conservative', 'aggressive', 'diversified', 'quick_test', або список)
            timeframes: Таймфрейми ('default', 'intraday', 'daily', 'all', або список)
            config_overrides: Перевизначення конфігурації
        
        Returns:
            Dict: Результати пайплайну
        """
        # [TARGET] Перевизначення конфігурації
        if config_overrides:
            self._merge_config_overrides(config_overrides)
        
        # [TARGET] Вибір тікерів
        selected_tickers = self.ticker_selector.select_tickers(tickers)
        logger.info(f"[DATA] Selected tickers: {selected_tickers}")
        
        # [TARGET] Вибір таймфреймів
        selected_timeframes = self._select_timeframes(timeframes)
        logger.info(f"[DATA] Selected timeframes: {selected_timeframes}")
        
        # [TARGET] Спроба відновлення з checkpoint
        if self.recovery.recovery_enabled:
            saved_state = self.recovery.load_latest_checkpoint()
            if saved_state:
                logger.info("[RESTART] Recovering from checkpoint")
                self.state = saved_state
                self.monitor.metrics.stages_completed = len(self.state.completed_stages)
        
        # [TARGET] Запуск пайплайну
        try:
            return self._execute_pipeline(selected_tickers, selected_timeframes)
        except Exception as e:
            logger.error(f"[ERROR] Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            self.state.error_log.append(str(e))
            self.recovery.save_checkpoint(self.state)
            raise
    
    def _execute_pipeline(self, tickers: List[str], timeframes: List[str]) -> Dict:
        """Виконання пайплайну"""
        pipeline_start = time.time()
        
        # [TARGET] Етап 1: Збір data
        if not self._is_stage_completed(1):
            self.monitor.start_stage("Stage 1: Data Collection")
            try:
                stage1_result = self._run_stage_1(tickers, timeframes)
                self.state.stage_results[1] = stage1_result
                self.state.completed_stages.append(1)
                self.monitor.complete_stage("Stage 1: Data Collection", stage1_result)
                self.recovery.save_checkpoint(self.state)
            except Exception as e:
                self._handle_stage_failure(1, e)
                raise
        
        # [TARGET] Етап 2: Збагачення data
        if not self._is_stage_completed(2):
            self.monitor.start_stage("Stage 2: Data Enrichment")
            try:
                stage2_result = self._run_stage_2(self.state.stage_results[1], tickers, timeframes)
                self.state.stage_results[2] = stage2_result
                self.state.completed_stages.append(2)
                self.monitor.complete_stage("Stage 2: Data Enrichment", stage2_result)
                self.recovery.save_checkpoint(self.state)
            except Exception as e:
                self._handle_stage_failure(2, e)
                raise
        
        # [TARGET] Етап 3: Feature Engineering
        if not self._is_stage_completed(3):
            self.monitor.start_stage("Stage 3: Feature Engineering")
            try:
                stage3_result = self._run_stage_3(
                    self.state.stage_results[1], 
                    self.state.stage_results[2]
                )
                self.state.stage_results[3] = stage3_result
                self.state.completed_stages.append(3)
                self.monitor.complete_stage("Stage 3: Feature Engineering", stage3_result)
                self.recovery.save_checkpoint(self.state)
            except Exception as e:
                self._handle_stage_failure(3, e)
                raise
        
        # [TARGET] Етап 4: Model Training
        if not self._is_stage_completed(4):
            self.monitor.start_stage("Stage 4: Model Training")
            try:
                stage4_result = self._run_stage_4(self.state.stage_results[3])
                self.state.stage_results[4] = stage4_result
                self.state.completed_stages.append(4)
                self.monitor.complete_stage("Stage 4: Model Training", stage4_result)
                self.recovery.save_checkpoint(self.state)
            except Exception as e:
                self._handle_stage_failure(4, e)
                raise
        
        # [TARGET] Фінальний звіт
        pipeline_time = time.time() - pipeline_start
        final_report = self._create_final_report(pipeline_time)
        
        logger.info("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!")
        return final_report
    
    def _run_stage_1(self, tickers: List[str], timeframes: List[str]) -> Dict:
        """Запуск етапу 1"""
        stage_config = self.config.get('stages', {}).get('stage1', {})
        
        return run_stage_1_collect_intelligent(
            tickers=tickers,
            timeframes=timeframes,
            use_free_data=self.config.get('data_sources', {}).get('yahoo_finance', {}).get('enabled', True),
            enable_cache=stage_config.get('cache_enabled', True),
            cache_ttl_hours=stage_config.get('cache_ttl_hours', 24)
        )
    
    def _run_stage_2(self, stage1_result: Dict, tickers: List[str], timeframes: List[str]) -> Dict:
        """Запуск етапу 2"""
        return run_stage_2_enrich_ideal(
            stage1_data=stage1_result,
            tickers=tickers,
            timeframes=timeframes,
            use_free_data=self.config.get('data_sources', {}).get('news_api', {}).get('enabled', True)
        )
    
    def _run_stage_3(self, stage1_result: Dict, stage2_result: Dict) -> Dict:
        """Запуск етапу 3"""
        stage_config = self.config.get('stages', {}).get('stage3', {})
        
        stage3_config = {
            'use_enhanced': stage_config.get('enabled', True),
            'max_features': stage_config.get('max_features', 100),
            'pattern_aware': stage_config.get('pattern_aware', True)
        }
        
        result = run_stage_3(stage1_result, stage2_result, stage3_config)
        
        # Оновлюємо метрики
        feature_count = len(result.get('features', {}))
        self.monitor.update_metrics(feature_count=feature_count)
        
        return result
    
    def _run_stage_4(self, stage3_result: Dict) -> Dict:
        """Запуск етапу 4"""
        stage_config = self.config.get('stages', {}).get('stage4', {})
        
        stage4_config = {
            'use_enhanced': stage_config.get('enabled', True),
            'max_models': len(stage_config.get('local_models', [])),
            'save_models': True,
            'model_save_path': self.config.get('storage', {}).get('models_path', 'models/trained/'),
            'colab_preparation_path': self.config.get('storage', {}).get('colab_path', 'colab_preparation/')
        }
        
        result = run_stage_4_unified(stage3_result, stage4_config)
        
        # Оновлюємо метрики
        local_models = len(result.get('local_models', {}))
        self.monitor.update_metrics(model_performance=local_models)
        
        return result
    
    def _is_stage_completed(self, stage: int) -> bool:
        """Перевірка чи етап завершено"""
        return stage in self.state.completed_stages
    
    def _handle_stage_failure(self, stage: int, error: Exception):
        """Обробка збою етапу"""
        logger.error(f"[ERROR] Stage {stage} failed: {error}")
        self.state.failed_stages.append(stage)
        self.state.error_log.append(f"Stage {stage}: {str(error)}")
        
        # Спроба відновлення
        if self.config.get('recovery', {}).get('auto_retry', True):
            max_retries = self.config.get('recovery', {}).get('max_retries', 3)
            retry_delay = self.config.get('recovery', {}).get('retry_delay', 60)
            
            for attempt in range(max_retries):
                logger.info(f"[RESTART] Retrying stage {stage}, attempt {attempt + 1}/{max_retries}")
                time.sleep(retry_delay)
                
                try:
                    if stage == 1:
                        result = self._run_stage_1([], [])  # TODO: передати параметри
                    elif stage == 2:
                        result = self._run_stage_2(self.state.stage_results[1], [], [])
                    elif stage == 3:
                        result = self._run_stage_3(self.state.stage_results[1], self.state.stage_results[2])
                    elif stage == 4:
                        result = self._run_stage_4(self.state.stage_results[3])
                    
                    self.state.stage_results[stage] = result
                    self.state.completed_stages.append(stage)
                    self.state.failed_stages.remove(stage)
                    logger.info(f"[OK] Stage {stage} recovered successfully")
                    return
                    
                except Exception as retry_error:
                    logger.error(f"[ERROR] Retry {attempt + 1} failed: {retry_error}")
                    continue
            
            logger.error(f"[ERROR] All retries failed for stage {stage}")
    
    def _select_timeframes(self, timeframes: Union[str, List[str]] = None) -> List[str]:
        """Вибір таймфреймів"""
        if isinstance(timeframes, list):
            return timeframes
        
        if timeframes is None:
            timeframes = 'default'
        
        tf_config = self.config.get('timeframes', {})
        
        if timeframes in tf_config:
            return tf_config[timeframes]
        else:
            logger.warning(f"Unknown timeframes selection: {timeframes}, using default")
            return tf_config.get('default', ['15m', '1h', '1d'])
    
    def _merge_config_overrides(self, overrides: Dict):
        """Злиття перевизначень конфігурації"""
        def merge_dict(base: Dict, override: Dict):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(self.config, overrides)
        logger.info("[FIX] Config overrides applied")
    
    def _create_final_report(self, pipeline_time: float) -> Dict:
        """Створення фінального звіту"""
        return {
            'pipeline_info': {
                'name': self.config.get('pipeline', {}).get('name', 'trading_pipeline'),
                'version': self.config.get('pipeline', {}).get('version', '2.0.0'),
                'execution_time': pipeline_time,
                'timestamp': datetime.now().isoformat(),
                'stages_completed': len(self.state.completed_stages),
                'stages_failed': len(self.state.failed_stages)
            },
            'stage_results': self.state.stage_results,
            'metrics': {
                'data_quality_score': self.monitor.metrics.data_quality_score,
                'feature_count': self.monitor.metrics.feature_count,
                'model_performance': self.monitor.metrics.model_performance,
                'execution_time': pipeline_time,
                'memory_usage': self.monitor.metrics.memory_usage,
                'error_rate': self.monitor.metrics.error_rate
            },
            'stage_times': self.monitor.stage_times,
            'error_log': self.state.error_log,
            'next_actions': self._generate_next_actions(),
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_next_actions(self) -> List[str]:
        """Генерація наступних дій"""
        actions = []
        
        if 4 in self.state.completed_stages:
            stage4_result = self.state.stage_results[4]
            if stage4_result.get('next_step') == 'colab_heavy_training':
                actions.extend([
                    "Transfer colab_preparation/ to Google Colab",
                    "Run colab_heavy_training.py in Colab",
                    "Download trained models back to local",
                    "Update model registry"
                ])
            else:
                actions.extend([
                    "Pipeline completed successfully",
                    "Models ready for deployment",
                    "Check final report for details"
                ])
        else:
            next_stage = max(self.state.completed_stages) + 1 if self.state.completed_stages else 1
            actions.append(f"Resume pipeline from stage {next_stage}")
        
        return actions
    
    def _generate_recommendations(self) -> List[str]:
        """Генерація рекомендацій"""
        recommendations = []
        
        # Рекомендації на основі метрик
        if self.monitor.metrics.data_quality_score < 0.7:
            recommendations.append("Consider improving data quality sources")
        
        if self.monitor.metrics.feature_count < 50:
            recommendations.append("Consider adding more features or enabling enhanced features")
        
        if self.monitor.metrics.error_rate > 0.1:
            recommendations.append("Review error log and fix failing stages")
        
        if len(self.state.failed_stages) > 0:
            recommendations.append(f"Retry failed stages: {self.state.failed_stages}")
        
        return recommendations


# [TARGET] ГОЛОВНА ТОЧКА ВХОДУ - ДЛЯ ВИКОРИСТАННЯ
def run_trading_pipeline(tickers: Union[str, List[str]] = 'balanced_growth',
                        timeframes: Union[str, List[str]] = 'default',
                        config_path: str = 'config/pipeline_config.yaml',
                        config_overrides: Dict = None,
                        progress_callback=None,
                        limit: Optional[int] = None) -> Dict:
    """
    [START] Основна точка входу для запуску пайплайну - ІНТЕГРУЄМО ІСНУЮЧУ СИСТЕМУ ТІКЕРІВ
    
    Args:
        tickers: Вибір тікерів:
                - 'balanced_growth', 'extreme_volatility', 'news_driven', 'momentum', 'conservative' (з enhanced_sector_tickers)
                - 'default', 'all', 'conservative', 'aggressive', 'diversified', 'quick_test' (з YAML)
                - Список тікерів: ['AAPL', 'MSFT', 'GOOGL']
        timeframes: Таймфрейми ('default', 'intraday', 'daily', 'all', або список)
        config_path: Шлях до конфігураційного файлу
        config_overrides: Перевизначення конфігурації
        progress_callback: Callback для прогресу (message, progress)
        limit: Обмеження кількості тікерів
    
    Returns:
        Dict: Результати пайплайну
    """
    pipeline = MainPipeline(config_path)
    
    if progress_callback:
        pipeline.add_progress_callback(progress_callback)
    
    return pipeline.run_pipeline(tickers, timeframes, config_overrides, limit)


# [TARGET] ПРИКЛАДИ ВИКОРИСТАННЯ
if __name__ == "__main__":
    # [TARGET] Приклад 1: Базовий запуск
    print("[START] Running basic pipeline...")
    result = run_trading_pipeline()
    print(f"[OK] Pipeline completed: {result['pipeline_info']['execution_time']:.2f}s")
    
    # [TARGET] Приклад 2: Екстремальна волатильність (з enhanced_sector_tickers)
    print("\n[START] Running extreme volatility pipeline...")
    result = run_trading_pipeline(
        tickers='extreme_volatility',  # ІСНУЮЧА СТРАТЕГІЯ
        timeframes='default',
        progress_callback=lambda msg, prog: print(f"[DATA] {prog:.1f}% - {msg}")
    )
    
    # [TARGET] Приклад 3: Balanced Growth (з enhanced_sector_tickers)
    print("\n[START] Running balanced growth pipeline...")
    result = run_trading_pipeline(
        tickers='balanced_growth',    # ІСНУЮЧА СТРАТЕГІЯ
        timeframes='intraday',
        limit=10,                     # Обмеження кількості тікерів
        progress_callback=lambda msg, prog: print(f"[DATA] {prog:.1f}% - {msg}")
    )
    
    # [TARGET] Приклад 4: News Driven (з enhanced_sector_tickers)
    print("\n[START] Running news driven pipeline...")
    result = run_trading_pipeline(
        tickers='news_driven',        # ІСНУЮЧА СТРАТЕГІЯ
        timeframes='default',
        config_overrides={
            'stages': {
                'stage3': {'max_features': 200}
            },
            'monitoring': {'log_level': 'DEBUG'}
        }
    )
    
    # [TARGET] Приклад 5: Momentum (з enhanced_sector_tickers)
    print("\n[START] Running momentum pipeline...")
    result = run_trading_pipeline(
        tickers='momentum',           # ІСНУЮЧА СТРАТЕГІЯ
        timeframes='intraday',
        limit=15
    )
    
    # [TARGET] Приклад 6: Консервативний (з enhanced_sector_tickers)
    print("\n[START] Running conservative pipeline...")
    result = run_trading_pipeline(
        tickers='conservative',       # ІСНУЮЧА СТРАТЕГІЯ
        timeframes='daily'
    )
    
    # [TARGET] Приклад 7: Кастомні тікери
    print("\n[START] Running custom pipeline...")
    result = run_trading_pipeline(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'],
        timeframes=['15m', '1h', '4h', '1d']
    )
    
    # [TARGET] Приклад 8: Використання додаткових функцій enhanced_sector_tickers
    print("\n[START] Testing enhanced sector features...")
    from config.enhanced_sector_tickers import analyze_sectors, recommend_portfolio
    
    # [TARGET] Аналіз секторів
    sector_analysis = analyze_sectors()
    print(f"[DATA] Sector analysis: {len(sector_analysis)} sectors analyzed")
    
    # [TARGET] Рекомендація портфеля
    portfolio_rec = recommend_portfolio(capital=10000, max_positions=8)
    print(f"[DATA] Portfolio recommendation: {len(portfolio_rec.get('recommended_tickers', []))} tickers")
    
    print("\n[SUCCESS] All examples completed!")
