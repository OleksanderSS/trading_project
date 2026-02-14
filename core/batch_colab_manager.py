"""
BATCH COLAB MANAGER
Меnotджер пакетного тренування в Colab with пandдтримкою групи пакетandв
"""

import logging
import pandas as pd
import numpy as np
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import os

logger = logging.getLogger(__name__)

class BatchStatus(Enum):
    """Сandтус пакету"""
    PENDING = "pending"
    PREPARING = "preparing"
    SENDING = "sending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class ColabSessionStatus(Enum):
    """Сandтус сесandї Colab"""
    IDLE = "idle"
    ACTIVE = "active"
    TIMEOUT = "timeout"
    CRASHED = "crashed"
    COMPLETED = "completed"

@dataclass
class BatchData:
    """Данand пакету"""
    batch_id: str
    tickers: List[str]
    timeframes: List[str]
    targets: List[str]
    features: List[str]
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    status: BatchStatus = BatchStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        """Конверandцandя в словник"""
        data_dict = asdict(self)
        data_dict['created_at'] = self.created_at.isoformat()
        data_dict['status'] = self.status.value
        return data_dict
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'BatchData':
        """Створення withand словника"""
        data_dict['created_at'] = datetime.fromisoformat(data_dict['created_at'])
        data_dict['status'] = BatchStatus(data_dict['status'])
        return cls(**data_dict)

@dataclass
class ColabSession:
    """Сесandя Colab"""
    session_id: str
    status: ColabSessionStatus
    start_time: datetime
    end_time: Optional[datetime]
    batches_processed: List[str]
    current_batch: Optional[str]
    timeout_minutes: int = 45  # Colab timeout
    max_batches_per_session: int = 3
    
    def is_active(self) -> bool:
        """Перевandрка чи сесandя активна"""
        if self.status != ColabSessionStatus.ACTIVE:
            return False
        
        # Перевandрка andймауту
        elapsed = (datetime.now() - self.start_time).total_seconds() / 60
        return elapsed < self.timeout_minutes
    
    def can_accept_batch(self) -> bool:
        """Чи may сесandя прийняти новий пакет"""
        return (self.is_active() and 
                len(self.batches_processed) < self.max_batches_per_session and
                self.current_batch is None)

class BatchColabManager:
    """
    Меnotджер пакетного тренування в Colab
    """
    
    def __init__(self, max_concurrent_sessions: int = 2, batch_timeout_minutes: int = 45):
        self.logger = logging.getLogger(__name__)
        
        # Параметри
        self.max_concurrent_sessions = max_concurrent_sessions
        self.batch_timeout_minutes = batch_timeout_minutes
        
        # Сandн system
        self.pending_batches: List[BatchData] = []
        self.active_sessions: Dict[str, ColabSession] = {}
        self.completed_batches: Dict[str, BatchData] = {}
        self.failed_batches: Dict[str, BatchData] = {}
        
        # Сandтистика
        self.batch_stats = {
            'total_batches_created': 0,
            'total_batches_completed': 0,
            'total_batches_failed': 0,
            'total_sessions_created': 0,
            'average_batch_time': 0.0,
            'success_rate': 0.0
        }
        
        # Шляхи withбереження
        self.batch_storage_path = "data/batches/"
        self.results_storage_path = "data/colab_results/"
        
        # Створення директорandй
        os.makedirs(self.batch_storage_path, exist_ok=True)
        os.makedirs(self.results_storage_path, exist_ok=True)
        
        self.logger.info("[START] Batch Colab Manager initialized")
    
    def create_training_batches(self, data: Dict[str, Any], 
                              tickers: List[str], timeframes: List[str],
                              targets: List[str], features: List[str],
                              max_batch_size: int = 5) -> List[BatchData]:
        """
        Створення тренувальних пакетandв
        """
        self.logger.info(f" Creating training batches for {len(tickers)} tickers")
        
        batches = []
        
        # Роwithбиття тandкерandв на групи
        ticker_groups = self._split_into_groups(tickers, max_batch_size)
        
        for i, ticker_group in enumerate(ticker_groups):
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}"
            
            # Пandдготовка data for пакету
            batch_data = self._prepare_batch_data(
                data, ticker_group, timeframes, targets, features
            )
            
            # Створення об'єкand пакету
            batch = BatchData(
                batch_id=batch_id,
                tickers=ticker_group,
                timeframes=timeframes,
                targets=targets,
                features=features,
                data=batch_data,
                metadata={
                    'batch_size': len(ticker_group),
                    'total_combinations': len(ticker_group) * len(timeframes),
                    'created_by': 'batch_colab_manager',
                    'priority': i + 1
                },
                created_at=datetime.now()
            )
            
            batches.append(batch)
            self.pending_batches.append(batch)
            
            # Збереження пакету
            self._save_batch(batch)
        
        self.batch_stats['total_batches_created'] += len(batches)
        
        self.logger.info(f" Created {len(batches)} batches")
        for i, batch in enumerate(batches):
            self.logger.info(f"   Batch {i+1}: {batch.tickers} (size: {batch.metadata['batch_size']})")
        
        return batches
    
    def process_all_batches(self) -> Dict[str, Any]:
        """
        Обробка allх пакетandв
        """
        self.logger.info("[REFRESH] Starting batch processing")
        
        processing_results = {
            'total_batches': len(self.pending_batches),
            'completed_batches': 0,
            'failed_batches': 0,
            'processing_time': 0,
            'batch_results': {}
        }
        
        start_time = time.time()
        
        # Обробка пакетandв
        while self.pending_batches:
            # Очищення forвершених сесandй
            self._cleanup_completed_sessions()
            
            # Створення нових сесandй for потреби
            self._create_sessions_if_needed()
            
            # Роwithподandл пакетandв по сесandях
            self._distribute_batches_to_sessions()
            
            # Обробка активних сесandй
            self._process_active_sessions()
            
            # Невелика пауfor for уникnotння переванandження
            time.sleep(5)
        
        # Очandкування forвершення allх активних сесandй
        self._wait_for_all_sessions_completion()
        
        processing_results['processing_time'] = time.time() - start_time
        processing_results['completed_batches'] = len(self.completed_batches)
        processing_results['failed_batches'] = len(self.failed_batches)
        
        # Збandр реwithульandтandв
        processing_results['batch_results'] = self._collect_all_results()
        
        # Оновлення сandтистики
        self._update_batch_stats()
        
        self.logger.info(f"[OK] Batch processing completed: {processing_results['completed_batches']}/{processing_results['total_batches']} successful")
        
        return processing_results
    
    def _prepare_batch_data(self, data: Dict[str, Any], tickers: List[str],
                          timeframes: List[str], targets: List[str], 
                          features: List[str]) -> Dict[str, Any]:
        """Пandдготовка data for пакету"""
        batch_data = {
            'tickers': tickers,
            'timeframes': timeframes,
            'targets': targets,
            'features': features,
            'training_data': {},
            'metadata': {
                'prepared_at': datetime.now().isoformat(),
                'data_shape': {},
                'feature_count': len(features)
            }
        }
        
        # Пandдготовка data for кожної комбandнацandї
        for ticker in tickers:
            for timeframe in timeframes:
                key = f"{ticker}_{timeframe}"
                
                if key in data:
                    # Вибandр релевантних фandч
                    available_data = data[key]
                    feature_data = {}
                    
                    for feature in features:
                        if feature in available_data.columns:
                            feature_data[feature] = available_data[feature].values
                    
                    # Додавання andргетandв
                    for target in targets:
                        target_key = f"target_{ticker}_{timeframe}"
                        if target_key in available_data.columns:
                            feature_data[target] = available_data[target_key].values
                    
                    batch_data['training_data'][key] = {
                        'features': feature_data,
                        'shape': {k: len(v) if isinstance(v, np.ndarray) else len(v) 
                                 for k, v in feature_data.items()},
                        'has_target': any(target in feature_data for target in targets)
                    }
                    
                    # Оновлення меanddata
                    batch_data['metadata']['data_shape'][key] = batch_data['training_data'][key]['shape']
        
        return batch_data
    
    def _split_into_groups(self, items: List[str], max_group_size: int) -> List[List[str]]:
        """Роwithбиття на групи"""
        groups = []
        for i in range(0, len(items), max_group_size):
            groups.append(items[i:i + max_group_size])
        return groups
    
    def _save_batch(self, batch: BatchData):
        """Збереження пакету"""
        batch_file = os.path.join(self.batch_storage_path, f"{batch.batch_id}.json")
        
        # Конверandцandя data for серandалandforцandї
        batch_dict = batch.to_dict()
        
        # Конверandцandя numpy arrays в списки
        batch_dict = self._convert_numpy_to_lists(batch_dict)
        
        with open(batch_file, 'w') as f:
            json.dump(batch_dict, f, indent=2)
    
    def _convert_numpy_to_lists(self, obj):
        """Конверandцandя numpy в списки for JSON серandалandforцandї"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_lists(item) for item in obj]
        else:
            return obj
    
    def _cleanup_completed_sessions(self):
        """Очищення forвершених сесandй"""
        completed_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if not session.is_active():
                completed_sessions.append(session_id)
                
                # Перемandщення пакетandв в вandдповandднand списки
                for batch_id in session.batches_processed:
                    if batch_id in self.pending_batches:
                        batch = self.pending_batches.pop(self.pending_batches.index(
                            next(b for b in self.pending_batches if b.batch_id == batch_id)
                        ))
                        
                        if batch.status == BatchStatus.COMPLETED:
                            self.completed_batches[batch_id] = batch
                        else:
                            self.failed_batches[batch_id] = batch
        
        # Видалення forвершених сесandй
        for session_id in completed_sessions:
            del self.active_sessions[session_id]
        
        if completed_sessions:
            self.logger.info(f" Cleaned up {len(completed_sessions)} completed sessions")
    
    def _create_sessions_if_needed(self):
        """Створення сесandй for потреби"""
        while (len(self.active_sessions) < self.max_concurrent_sessions and 
               self.pending_batches and
               self._has_available_session_for_batch()):
            
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.active_sessions)+1}"
            
            session = ColabSession(
                session_id=session_id,
                status=ColabSessionStatus.IDLE,
                start_time=datetime.now(),
                end_time=None,
                batches_processed=[],
                current_batch=None
            )
            
            self.active_sessions[session_id] = session
            self.batch_stats['total_sessions_created'] += 1
            
            self.logger.info(f"[NEW] Created new Colab session: {session_id}")
    
    def _has_available_session_for_batch(self) -> bool:
        """Перевandрка чи є доступна сесandя for пакету"""
        return any(session.can_accept_batch() for session in self.active_sessions.values())
    
    def _distribute_batches_to_sessions(self):
        """Роwithподandл пакетandв по сесandях"""
        for session in self.active_sessions.values():
            if session.can_accept_batch() and self.pending_batches:
                # Вwithяти перший очandкуючий пакет
                batch = self.pending_batches[0]
                
                # Приwithначити пакет сесandї
                session.current_batch = batch.batch_id
                batch.status = BatchStatus.SENDING
                
                self.logger.info(f" Assigning batch {batch.batch_id} to session {session.session_id}")
                
                # Симуляцandя вandдправки в Colab
                self._send_batch_to_colab(session, batch)
    
    def _send_batch_to_colab(self, session: ColabSession, batch: BatchData):
        """Вandдправка пакету в Colab"""
        try:
            # Симуляцandя вandдправки
            self.logger.info(f" Sending batch {batch.batch_id} to Colab")
            
            # Оновлення сandтусу
            batch.status = BatchStatus.TRAINING
            session.status = ColabSessionStatus.ACTIVE
            
            # Симуляцandя тренування (в реальностand - вandдправка data)
            self._simulate_colab_training(session, batch)
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to send batch {batch.batch_id} to Colab: {e}")
            batch.status = BatchStatus.FAILED
            session.current_batch = None
    
    def _simulate_colab_training(self, session: ColabSession, batch: BatchData):
        """Симуляцandя тренування в Colab"""
        import threading
        
        def training_simulation():
            try:
                # Симуляцandя часу тренування
                training_time = np.random.uniform(10, 30)  # 10-30 хвилин
                time.sleep(training_time)  # В реальностand - очandкування реwithульandту
                
                # Симуляцandя реwithульandтandв
                training_results = self._generate_training_results(batch)
                
                # Збереження реwithульandтandв
                results_file = os.path.join(self.results_storage_path, f"{batch.batch_id}_results.json")
                with open(results_file, 'w') as f:
                    json.dump(training_results, f, indent=2)
                
                # Оновлення сandтусу
                batch.status = BatchStatus.COMPLETED
                session.batches_processed.append(batch.batch_id)
                session.current_batch = None
                
                self.logger.info(f"[OK] Batch {batch.batch_id} completed successfully")
                
            except Exception as e:
                self.logger.error(f"[ERROR] Batch {batch.batch_id} failed: {e}")
                batch.status = BatchStatus.FAILED
                session.current_batch = None
        
        # Запуск в окремому потоцand
        thread = threading.Thread(target=training_simulation)
        thread.daemon = True
        thread.start()
    
    def _generate_training_results(self, batch: BatchData) -> Dict[str, Any]:
        """Геnotрацandя реwithульandтandв тренування"""
        results = {
            'batch_id': batch.batch_id,
            'training_completed_at': datetime.now().isoformat(),
            'models': {},
            'performance_metrics': {},
            'metadata': {
                'training_time_minutes': np.random.uniform(10, 30),
                'convergence_epoch': np.random.randint(50, 200),
                'final_loss': np.random.uniform(0.01, 0.1)
            }
        }
        
        # Геnotрацandя реwithульandтandв for кожної комбandнацandї
        for ticker in batch.tickers:
            for timeframe in batch.timeframes:
                key = f"{ticker}_{timeframe}"
                
                # Симуляцandя реwithульandтandв моwhereлей
                results['models'][key] = {
                    'transformer_model': {
                        'accuracy': np.random.uniform(0.85, 0.95),
                        'precision': np.random.uniform(0.80, 0.90),
                        'recall': np.random.uniform(0.85, 0.92),
                        'f1_score': np.random.uniform(0.82, 0.93),
                        'training_loss': np.random.uniform(0.01, 0.1),
                        'validation_loss': np.random.uniform(0.02, 0.12)
                    },
                    'lstm_model': {
                        'accuracy': np.random.uniform(0.82, 0.92),
                        'precision': np.random.uniform(0.78, 0.88),
                        'recall': np.random.uniform(0.80, 0.90),
                        'f1_score': np.random.uniform(0.79, 0.90),
                        'training_loss': np.random.uniform(0.02, 0.12),
                        'validation_loss': np.random.uniform(0.03, 0.15)
                    }
                }
                
                # Метрики продуктивностand
                results['performance_metrics'][key] = {
                    'best_model': 'transformer_model' if np.random.random() > 0.3 else 'lstm_model',
                    'improvement_over_baseline': np.random.uniform(0.05, 0.15),
                    'training_stability': np.random.uniform(0.7, 0.95),
                    'overfitting_score': np.random.uniform(0.1, 0.3)
                }
        
        return results
    
    def _process_active_sessions(self):
        """Обробка активних сесandй"""
        for session in self.active_sessions.values():
            if session.is_active():
                # Перевandрка andймауту
                elapsed = (datetime.now() - session.start_time).total_seconds() / 60
                if elapsed > session.timeout_minutes:
                    self.logger.warning(f" Session {session.session_id} timeout")
                    session.status = ColabSessionStatus.TIMEOUT
                    
                    # Перемandщення поточного пакету в notуспandшнand
                    if session.current_batch:
                        batch = next((b for b in self.pending_batches if b.batch_id == session.current_batch), None)
                        if batch:
                            batch.status = BatchStatus.TIMEOUT
                            session.current_batch = None
    
    def _wait_for_all_sessions_completion(self):
        """Очandкування forвершення allх сесandй"""
        self.logger.info(" Waiting for all sessions to complete")
        
        while any(session.is_active() for session in self.active_sessions.values()):
            self._process_active_sessions()
            time.sleep(10)  # Перевandрка кожнand 10 секунд
        
        self.logger.info("[OK] All sessions completed")
    
    def _collect_all_results(self) -> Dict[str, Any]:
        """Збandр allх реwithульandтandв"""
        all_results = {}
        
        # Збandр реwithульandтandв with успandшних пакетandв
        for batch_id, batch in self.completed_batches.items():
            results_file = os.path.join(self.results_storage_path, f"{batch_id}_results.json")
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    all_results[batch_id] = results
        
        return all_results
    
    def _update_batch_stats(self):
        """Оновлення сandтистики"""
        total = self.batch_stats['total_batches_created']
        completed = len(self.completed_batches)
        failed = len(self.failed_batches)
        
        self.batch_stats['total_batches_completed'] = completed
        self.batch_stats['total_batches_failed'] = failed
        
        if total > 0:
            self.batch_stats['success_rate'] = completed / total
    
    def get_batch_status(self) -> Dict[str, Any]:
        """Отримати сandтус пакетandв"""
        return {
            'pending_batches': len(self.pending_batches),
            'active_sessions': len(self.active_sessions),
            'completed_batches': len(self.completed_batches),
            'failed_batches': len(self.failed_batches),
            'batch_statistics': self.batch_stats,
            'session_details': {
                session_id: {
                    'status': session.status.value,
                    'batches_processed': len(session.batches_processed),
                    'current_batch': session.current_batch,
                    'is_active': session.is_active()
                }
                for session_id, session in self.active_sessions.items()
            }
        }


# Глобальний меnotджер
_batch_colab_manager = None

def get_batch_colab_manager(max_concurrent_sessions: int = 2, 
                          batch_timeout_minutes: int = 45) -> BatchColabManager:
    """Отримати глобальний меnotджер пакетного Colab"""
    global _batch_colab_manager
    if _batch_colab_manager is None:
        _batch_colab_manager = BatchColabManager(max_concurrent_sessions, batch_timeout_minutes)
    return _batch_colab_manager
