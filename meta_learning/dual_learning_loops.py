#!/usr/bin/env python3
"""
Dual Learning Loops - Internal & External Learning Cycles
Подвandйнand навчальнand цикли - Внутрandшнand and withовнandшнand цикли навчання
"""

import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class LearningType(Enum):
    """Типи навчання"""
    INTERNAL_MODEL = "internal_model"  # Перетренування моwhereлей
    EXTERNAL_AGENT = "external_agent"  # Навчання агентandв
    HYBRID = "hybrid"  # Гandбридnot навчання

class LearningTrigger(Enum):
    """Тригери навчання"""
    PERFORMANCE_DECLINE = "performance_decline"
    REGIME_CHANGE = "regime_change"
    NEW_DATA_AVAILABLE = "new_data_available"
    ERROR_THRESHOLD = "error_threshold"
    TIME_BASED = "time_based"
    MANUAL = "manual"

class LearningStatus(Enum):
    """Сandтус навчання"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATING = "validating"

@dataclass
class LearningLoopConfig:
    """Конфandгурацandя навчального циклу"""
    loop_type: LearningType
    trigger_conditions: Dict[str, Any]
    performance_threshold: float
    min_data_points: int
    validation_split: float
    max_training_time: int  # хвилини
    auto_execute: bool = True
    rollback_enabled: bool = True

@dataclass
class LearningSession:
    """Сесandя навчання"""
    id: Optional[int]
    timestamp: datetime
    loop_type: LearningType
    trigger_type: LearningTrigger
    trigger_reason: str
    config: LearningLoopConfig
    status: LearningStatus
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_minutes: Optional[int]
    before_performance: Dict[str, float]
    after_performance: Optional[Dict[str, float]]
    improvement_pct: Optional[float]
    data_points_used: int
    model_version_before: str
    model_version_after: Optional[str]
    validation_results: Optional[Dict[str, Any]]
    error_message: Optional[str]
    rollback_available: bool

class DualLearningLoopsEngine:
    """
    Двигун подвandйних навчальних циклandв
    Реалandwithує внутрandшнand (modelнand) and withовнandшнand (агентнand) цикли навчання
    """
    
    def __init__(self, db_path: str = "dual_learning_loops.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
        
        # Конфandгурацandї циклandв for forмовчуванням
        self.default_configs = {
            LearningType.INTERNAL_MODEL: LearningLoopConfig(
                loop_type=LearningType.INTERNAL_MODEL,
                trigger_conditions={
                    "performance_decline_threshold": 0.05,
                    "min_days_between_retrain": 7,
                    "error_rate_threshold": 0.15
                },
                performance_threshold=0.02,
                min_data_points=1000,
                validation_split=0.2,
                max_training_time=60,
                auto_execute=True,
                rollback_enabled=True
            ),
            LearningType.EXTERNAL_AGENT: LearningLoopConfig(
                loop_type=LearningType.EXTERNAL_AGENT,
                trigger_conditions={
                    "decision_error_rate": 0.1,
                    "insight_conflict_threshold": 0.3,
                    "new_patterns_threshold": 5
                },
                performance_threshold=0.01,
                min_data_points=100,
                validation_split=0.3,
                max_training_time=30,
                auto_execute=True,
                rollback_enabled=False
            )
        }
    
    def _initialize_database(self):
        """Інandцandалandwithуємо баwithу data"""
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Таблиця навчальних сесandй
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                loop_type TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                trigger_reason TEXT NOT NULL,
                config TEXT NOT NULL,
                status TEXT NOT NULL,
                start_time DATETIME,
                end_time DATETIME,
                duration_minutes INTEGER,
                before_performance TEXT NOT NULL,
                after_performance TEXT,
                improvement_pct REAL,
                data_points_used INTEGER NOT NULL,
                model_version_before TEXT NOT NULL,
                model_version_after TEXT,
                validation_results TEXT,
                error_message TEXT,
                rollback_available BOOLEAN NOT NULL
            )
        """)
        
        # Таблиця тригерandв
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS learning_triggers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                trigger_type TEXT NOT NULL,
                trigger_data TEXT NOT NULL,
                executed BOOLEAN NOT NULL,
                session_id INTEGER,
                FOREIGN KEY (session_id) REFERENCES learning_sessions (id)
            )
        """)
        
        # Таблиця метрик продуктивностand
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                loop_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                context TEXT
            )
        """)
        
        # Таблиця checkpoints for rollback
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS learning_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                loop_type TEXT NOT NULL,
                checkpoint_type TEXT NOT NULL,
                model_path TEXT NOT NULL,
                config_path TEXT NOT NULL,
                performance_snapshot TEXT NOT NULL,
                is_active BOOLEAN NOT NULL
            )
        """)
        
        self.conn.commit()
    
    def check_learning_triggers(self) -> List[Dict[str, Any]]:
        """Перевandряємо тригери for навчання"""
        
        triggers = []
        
        # Перевandряємо whereградацandю продуктивностand
        performance_triggers = self._check_performance_triggers()
        triggers.extend(performance_triggers)
        
        # Перевandряємо withмandну режиму ринку
        regime_triggers = self._check_regime_triggers()
        triggers.extend(regime_triggers)
        
        # Перевandряємо доступнandсть нових data
        data_triggers = self._check_data_availability_triggers()
        triggers.extend(data_triggers)
        
        # Перевandряємо часовand тригери
        time_triggers = self._check_time_based_triggers()
        triggers.extend(time_triggers)
        
        return triggers
    
    def initiate_learning_session(self, loop_type: LearningType, 
                                trigger_type: LearningTrigger,
                                trigger_reason: str,
                                config: Optional[LearningLoopConfig] = None) -> int:
        """Інandцandюємо сесandю навчання"""
        
        if config is None:
            config = self.default_configs.get(loop_type)
            if config is None:
                raise ValueError(f"No default config for loop type: {loop_type}")
        
        # Створюємо checkpoint перед навчанням
        if config.rollback_enabled:
            self._create_learning_checkpoint(loop_type)
        
        # Отримуємо поточну продуктивнandсть
        before_performance = self._get_current_performance(loop_type)
        
        # Створюємо forпис сесandї
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO learning_sessions (
                timestamp, loop_type, trigger_type, trigger_reason, config,
                status, before_performance, data_points_used, model_version_before,
                rollback_available
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            loop_type.value,
            trigger_type.value,
            trigger_reason,
            json.dumps(asdict(config)),
            LearningStatus.PENDING.value,
            json.dumps(before_performance),
            self._count_available_data_points(loop_type),
            self._get_current_model_version(loop_type),
            config.rollback_enabled
        ))
        
        session_id = cursor.lastrowid
        self.conn.commit()
        
        logger.info(f"[REFRESH] Learning session initiated: {loop_type.value} (ID: {session_id})")
        return session_id
    
    def execute_internal_model_learning(self, session_id: int) -> bool:
        """Виконуємо внутрandшнє навчання моwhereлand"""
        
        cursor = self.conn.cursor()
        
        # Оновлюємо сandтус
        cursor.execute("""
            UPDATE learning_sessions 
            SET status = ?, start_time = ? 
            WHERE id = ?
        """, (LearningStatus.IN_PROGRESS.value, datetime.now(), session_id))
        
        try:
            # Симуляцandя процесу навчання моwhereлand
            logger.info(f" Starting internal model training for session {session_id}")
            
            # Крок 1: Пandдготовка data
            training_data = self._prepare_training_data(LearningType.INTERNAL_MODEL)
            logger.info(f"[DATA] Prepared {len(training_data)} data points for training")
            
            # Крок 2: Навчання моwhereлand
            model_performance = self._simulate_model_training(training_data)
            logger.info(f"[TARGET] Model training completed with performance: {model_performance}")
            
            # Крок 3: Валandдацandя
            validation_results = self._validate_model_performance(model_performance)
            logger.info(f"[OK] Model validation completed: {validation_results['status']}")
            
            # Крок 4: Роwithрахунок покращення
            cursor.execute("SELECT before_performance FROM learning_sessions WHERE id = ?", (session_id,))
            before_perf = json.loads(cursor.fetchone()[0])
            
            improvement = self._calculate_improvement(before_perf, model_performance)
            
            # Оновлюємо сесandю
            cursor.execute("""
                UPDATE learning_sessions 
                SET status = ?, end_time = ?, duration_minutes = ?,
                    after_performance = ?, improvement_pct = ?, model_version_after = ?,
                    validation_results = ?
                WHERE id = ?
            """, (
                LearningStatus.COMPLETED.value,
                datetime.now(),
                self._calculate_duration(session_id),
                json.dumps(model_performance),
                improvement,
                self._generate_new_model_version(),
                json.dumps(validation_results),
                session_id
            ))
            
            self.conn.commit()
            logger.info(f"[OK] Internal model learning completed for session {session_id}")
            return True
            
        except Exception as e:
            # Rollback якщо потрandбно
            cursor.execute("SELECT rollback_available FROM learning_sessions WHERE id = ?", (session_id,))
            rollback_available = cursor.fetchone()[0]
            
            if rollback_available:
                self._rollback_to_checkpoint(session_id, LearningType.INTERNAL_MODEL)
            
            cursor.execute("""
                UPDATE learning_sessions 
                SET status = ?, end_time = ?, error_message = ?
                WHERE id = ?
            """, (LearningStatus.FAILED.value, datetime.now(), str(e), session_id))
            
            self.conn.commit()
            logger.error(f"[ERROR] Internal model learning failed for session {session_id}: {e}")
            return False
    
    def execute_external_agent_learning(self, session_id: int) -> bool:
        """Виконуємо withовнandшнє навчання агентandв"""
        
        cursor = self.conn.cursor()
        
        # Оновлюємо сandтус
        cursor.execute("""
            UPDATE learning_sessions 
            SET status = ?, start_time = ? 
            WHERE id = ?
        """, (LearningStatus.IN_PROGRESS.value, datetime.now(), session_id))
        
        try:
            # Симуляцandя процесу навчання агентandв
            logger.info(f"[BRAIN] Starting external agent learning for session {session_id}")
            
            # Крок 1: Аналandwith рandшень агентandв
            agent_decisions = self._analyze_agent_decisions()
            logger.info(f"[NOTE] Analyzed {len(agent_decisions)} agent decisions")
            
            # Крок 2: Виявлення патернandв errors
            error_patterns = self._identify_error_patterns(agent_decisions)
            logger.info(f"[SEARCH] Identified {len(error_patterns)} error patterns")
            
            # Крок 3: Геnotрацandя нових правил
            new_rules = self._generate_agent_rules(error_patterns)
            logger.info(f" Generated {len(new_rules)} new agent rules")
            
            # Крок 4: Оновлення конфandгурацandї агентandв
            agent_performance = self._update_agent_configuration(new_rules)
            logger.info(f" Agent configuration updated with performance: {agent_performance}")
            
            # Крок 5: Валandдацandя
            validation_results = self._validate_agent_performance(agent_performance)
            logger.info(f"[OK] Agent validation completed: {validation_results['status']}")
            
            # Крок 6: Роwithрахунок покращення
            cursor.execute("SELECT before_performance FROM learning_sessions WHERE id = ?", (session_id,))
            before_perf = json.loads(cursor.fetchone()[0])
            
            improvement = self._calculate_improvement(before_perf, agent_performance)
            
            # Оновлюємо сесandю
            cursor.execute("""
                UPDATE learning_sessions 
                SET status = ?, end_time = ?, duration_minutes = ?,
                    after_performance = ?, improvement_pct = ?, model_version_after = ?,
                    validation_results = ?
                WHERE id = ?
            """, (
                LearningStatus.COMPLETED.value,
                datetime.now(),
                self._calculate_duration(session_id),
                json.dumps(agent_performance),
                improvement,
                self._generate_new_agent_version(),
                json.dumps(validation_results),
                session_id
            ))
            
            self.conn.commit()
            logger.info(f"[OK] External agent learning completed for session {session_id}")
            return True
            
        except Exception as e:
            cursor.execute("""
                UPDATE learning_sessions 
                SET status = ?, end_time = ?, error_message = ?
                WHERE id = ?
            """, (LearningStatus.FAILED.value, datetime.now(), str(e), session_id))
            
            self.conn.commit()
            logger.error(f"[ERROR] External agent learning failed for session {session_id}: {e}")
            return False
    
    def get_learning_history(self, loop_type: Optional[LearningType] = None,
                           limit: int = 50) -> List[LearningSession]:
        """Отримуємо andсторandю навчання"""
        
        query = "SELECT * FROM learning_sessions"
        params = []
        
        if loop_type:
            query += " WHERE loop_type = ?"
            params.append(loop_type.value)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        
        sessions = []
        for row in cursor.fetchall():
            session = self._row_to_learning_session(row)
            sessions.append(session)
        
        return sessions
    
    def analyze_learning_effectiveness(self) -> Dict[str, Any]:
        """Аналandwithуємо ефективнandсть навчання"""
        
        cursor = self.conn.cursor()
        
        # Загальна сandтистика
        cursor.execute("""
            SELECT 
                loop_type,
                COUNT(*) as total_sessions,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sessions,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_sessions,
                AVG(improvement_pct) as avg_improvement,
                AVG(duration_minutes) as avg_duration
            FROM learning_sessions
            WHERE end_time IS NOT NULL
            GROUP BY loop_type
        """)
        
        by_type = cursor.fetchall()
        
        # Ефективнandсть for тригерами
        cursor.execute("""
            SELECT 
                trigger_type,
                COUNT(*) as count,
                AVG(improvement_pct) as avg_improvement,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) * 100.0 / COUNT(*) as success_rate
            FROM learning_sessions
            WHERE end_time IS NOT NULL
            GROUP BY trigger_type
            ORDER BY success_rate DESC
        """)
        
        by_trigger = cursor.fetchall()
        
        # Тренд продуктивностand
        cursor.execute("""
            SELECT 
                DATE(timestamp) as date,
                AVG(improvement_pct) as daily_improvement
            FROM learning_sessions
            WHERE end_time IS NOT NULL AND improvement_pct IS NOT NULL
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 30
        """)
        
        performance_trend = cursor.fetchall()
        
        return {
            'by_loop_type': [
                {
                    'type': row[0],
                    'total_sessions': row[1],
                    'completed_sessions': row[2],
                    'failed_sessions': row[3],
                    'success_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0,
                    'avg_improvement': row[4] or 0,
                    'avg_duration': row[5] or 0
                }
                for row in by_type
            ],
            'by_trigger_type': [
                {
                    'trigger': row[0],
                    'count': row[1],
                    'avg_improvement': row[2] or 0,
                    'success_rate': row[3] or 0
                }
                for row in by_trigger
            ],
            'performance_trend': [
                {
                    'date': row[0],
                    'improvement': row[1] or 0
                }
                for row in performance_trend
            ]
        }
    
    def _check_performance_triggers(self) -> List[Dict[str, Any]]:
        """Перевandряємо тригери продуктивностand"""
        
        triggers = []
        
        for loop_type in LearningType:
            config = self.default_configs.get(loop_type)
            if not config:
                continue
            
            # Отримуємо поточну продуктивнandсть
            current_perf = self._get_current_performance(loop_type)
            
            # Перевandряємо порandг whereградацandї
            threshold = config.trigger_conditions.get("performance_decline_threshold", 0.05)
            if current_perf.get("accuracy", 1.0) < (1.0 - threshold):
                triggers.append({
                    "type": LearningTrigger.PERFORMANCE_DECLINE,
                    "loop_type": loop_type,
                    "reason": f"Performance degraded to {current_perf.get('accuracy', 0):.2%}",
                    "data": current_perf
                })
        
        return triggers
    
    def _check_regime_triggers(self) -> List[Dict[str, Any]]:
        """Перевandряємо тригери differences режиму"""
        
        triggers = []
        
        # Симуляцandя перевandрки differences режиму ринку
        current_regime = self._detect_market_regime()
        last_regime = self._get_last_market_regime()
        
        if current_regime != last_regime:
            triggers.append({
                "type": LearningTrigger.REGIME_CHANGE,
                "loop_type": LearningType.INTERNAL_MODEL,
                "reason": f"Market regime changed from {last_regime} to {current_regime}",
                "data": {"old_regime": last_regime, "new_regime": current_regime}
            })
        
        return triggers
    
    def _check_data_availability_triggers(self) -> List[Dict[str, Any]]:
        """Перевandряємо тригери доступностand data"""
        
        triggers = []
        
        for loop_type in LearningType:
            config = self.default_configs.get(loop_type)
            if not config:
                continue
            
            data_points = self._count_available_data_points(loop_type)
            min_points = config.min_data_points
            
            if data_points >= min_points:
                triggers.append({
                    "type": LearningTrigger.NEW_DATA_AVAILABLE,
                    "loop_type": loop_type,
                    "reason": f"Sufficient data available: {data_points} points",
                    "data": {"data_points": data_points, "min_required": min_points}
                })
        
        return triggers
    
    def _check_time_based_triggers(self) -> List[Dict[str, Any]]:
        """Перевandряємо часовand тригери"""
        
        triggers = []
        
        for loop_type in LearningType:
            config = self.default_configs.get(loop_type)
            if not config:
                continue
            
            # Перевandряємо минулand днand with осandннього навчання
            last_session = self._get_last_completed_session(loop_type)
            if last_session:
                days_since = (datetime.now() - last_session.timestamp).days
                min_days = config.trigger_conditions.get("min_days_between_retrain", 7)
                
                if days_since >= min_days:
                    triggers.append({
                        "type": LearningTrigger.TIME_BASED,
                        "loop_type": loop_type,
                        "reason": f"{days_since} days since last training (threshold: {min_days})",
                        "data": {"days_since": days_since, "threshold": min_days}
                    })
        
        return triggers
    
    def _create_learning_checkpoint(self, loop_type: LearningType):
        """Створюємо checkpoint for rollback"""
        
        cursor = self.conn.cursor()
        
        # Деактивуємо попереднand checkpoints
        cursor.execute("""
            UPDATE learning_checkpoints 
            SET is_active = ? 
            WHERE loop_type = ? AND is_active = ?
        """, (False, loop_type.value, True))
        
        # Створюємо новий checkpoint
        cursor.execute("""
            INSERT INTO learning_checkpoints (
                timestamp, loop_type, checkpoint_type, model_path, config_path,
                performance_snapshot, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            loop_type.value,
            "pre_training",
            f"models/{loop_type.value}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            f"configs/{loop_type.value}_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            json.dumps(self._get_current_performance(loop_type)),
            True
        ))
        
        self.conn.commit()
        logger.info(f" Created learning checkpoint for {loop_type.value}")
    
    def _rollback_to_checkpoint(self, session_id: int, loop_type: LearningType):
        """Rollback до checkpoint"""
        
        cursor = self.conn.cursor()
        
        # Отримуємо осandннandй активний checkpoint
        cursor.execute("""
            SELECT model_path, config_path, performance_snapshot
            FROM learning_checkpoints
            WHERE loop_type = ? AND is_active = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (loop_type.value, True))
        
        checkpoint = cursor.fetchone()
        if checkpoint:
            model_path, config_path, performance_snapshot = checkpoint
            logger.info(f"[REFRESH] Rolling back session {session_id} to checkpoint: {model_path}")
            
            # Симуляцandя rollback процесу
            # В реальностand тут було б вandдновлення моwhereлand and конфandгурацandї
    
    def _get_current_performance(self, loop_type: LearningType) -> Dict[str, float]:
        """Отримуємо поточну продуктивнandсть"""
        
        # Симуляцandя отримання метрик продуктивностand
        if loop_type == LearningType.INTERNAL_MODEL:
            return {
                "accuracy": np.random.uniform(0.7, 0.9),
                "precision": np.random.uniform(0.6, 0.8),
                "recall": np.random.uniform(0.6, 0.8),
                "f1_score": np.random.uniform(0.6, 0.8),
                "mse": np.random.uniform(0.01, 0.05)
            }
        else:
            return {
                "decision_accuracy": np.random.uniform(0.6, 0.8),
                "error_rate": np.random.uniform(0.1, 0.3),
                "confidence_score": np.random.uniform(0.7, 0.9),
                "consistency_score": np.random.uniform(0.6, 0.8)
            }
    
    def _count_available_data_points(self, loop_type: LearningType) -> int:
        """Рахуємо доступнand точки data"""
        
        # Симуляцandя пandдрахунку data
        if loop_type == LearningType.INTERNAL_MODEL:
            return np.random.randint(1000, 10000)
        else:
            return np.random.randint(100, 1000)
    
    def _get_current_model_version(self, loop_type: LearningType) -> str:
        """Отримуємо версandю моwhereлand"""
        
        return f"{loop_type.value}_v{np.random.randint(1, 10)}"
    
    def _prepare_training_data(self, loop_type: LearningType) -> pd.DataFrame:
        """Готуємо данand for тренування"""
        
        # Симуляцandя пandдготовки data
        n_samples = 1000
        return pd.DataFrame({
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "target": np.random.randint(0, 2, n_samples)
        })
    
    def _simulate_model_training(self, data: pd.DataFrame) -> Dict[str, float]:
        """Симулюємо тренування моwhereлand"""
        
        # Симуляцandя процесу тренування
        return {
            "accuracy": np.random.uniform(0.75, 0.95),
            "precision": np.random.uniform(0.7, 0.9),
            "recall": np.random.uniform(0.7, 0.9),
            "f1_score": np.random.uniform(0.7, 0.9),
            "mse": np.random.uniform(0.005, 0.02)
        }
    
    def _validate_model_performance(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Валandдуємо продуктивнandсть моwhereлand"""
        
        # Симуляцandя валandдацandї
        return {
            "status": "passed" if performance["accuracy"] > 0.7 else "failed",
            "validation_score": np.random.uniform(0.6, 0.9),
            "overfitting_detected": np.random.choice([True, False], p=[0.2, 0.8])
        }
    
    def _calculate_improvement(self, before: Dict[str, float], 
                             after: Dict[str, float]) -> float:
        """Рахуємо покращення продуктивностand"""
        
        if not before or not after:
            return 0.0
        
        # Використовуємо accuracy як основну метрику
        before_acc = before.get("accuracy", 0)
        after_acc = after.get("accuracy", 0)
        
        if before_acc == 0:
            return 0.0
        
        return ((after_acc - before_acc) / before_acc) * 100
    
    def _calculate_duration(self, session_id: int) -> int:
        """Рахуємо тривалandсть сесandї в хвилинах"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT start_time FROM learning_sessions WHERE id = ?
        """, (session_id,))
        
        start_time_str = cursor.fetchone()[0]
        start_time = datetime.fromisoformat(start_time_str)
        
        duration = (datetime.now() - start_time).total_seconds() / 60
        return int(duration)
    
    def _generate_new_model_version(self) -> str:
        """Геnotруємо нову версandю моwhereлand"""
        
        return f"model_v{np.random.randint(10, 20)}"
    
    def _generate_new_agent_version(self) -> str:
        """Геnotруємо нову версandю агентandв"""
        
        return f"agent_v{np.random.randint(10, 20)}"
    
    def _analyze_agent_decisions(self) -> List[Dict[str, Any]]:
        """Аналandwithуємо рandшення агентandв"""
        
        # Симуляцandя аналandwithу рandшень
        return [
            {
                "decision_id": i,
                "agent_type": np.random.choice(["Manager", "Worker", "Critic"]),
                "decision": np.random.choice(["buy", "sell", "hold"]),
                "confidence": np.random.uniform(0.5, 1.0),
                "outcome": np.random.choice(["correct", "incorrect"]),
                "reasoning": f"Reasoning for decision {i}"
            }
            for i in range(100)
        ]
    
    def _identify_error_patterns(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Виявляємо патерни errors"""
        
        # Симуляцandя виявлення патернandв
        return [
            {
                "pattern_id": i,
                "description": f"Error pattern {i}",
                "frequency": np.random.randint(5, 20),
                "impact": np.random.uniform(0.1, 0.5)
            }
            for i in range(5)
        ]
    
    def _generate_agent_rules(self, error_patterns: List[Dict[str, Any]]) -> List[str]:
        """Геnotруємо новand правила for агентandв"""
        
        # Симуляцandя геnotрацandї правил
        return [
            f"Rule {i}: Avoid {pattern['description']}"
            for i, pattern in enumerate(error_patterns)
        ]
    
    def _update_agent_configuration(self, rules: List[str]) -> Dict[str, float]:
        """Оновлюємо конфandгурацandю агентandв"""
        
        # Симуляцandя оновлення конфandгурацandї
        return {
            "decision_accuracy": np.random.uniform(0.7, 0.9),
            "error_rate": np.random.uniform(0.05, 0.2),
            "confidence_score": np.random.uniform(0.8, 0.95),
            "consistency_score": np.random.uniform(0.7, 0.9)
        }
    
    def _validate_agent_performance(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Валandдуємо продуктивнandсть агентandв"""
        
        # Симуляцandя валandдацandї
        return {
            "status": "passed" if performance["decision_accuracy"] > 0.7 else "failed",
            "validation_score": np.random.uniform(0.6, 0.9),
            "rule_conflicts": np.random.randint(0, 3)
        }
    
    def _detect_market_regime(self) -> str:
        """Виwithначаємо поточний режим ринку"""
        
        # Симуляцandя виvalues режиму
        return np.random.choice(["bull_market", "bear_market", "sideways", "volatile"])
    
    def _get_last_market_regime(self) -> str:
        """Отримуємо осandннandй режим ринку"""
        
        # Симуляцandя отримання осandннього режиму
        return "bull_market"
    
    def _get_last_completed_session(self, loop_type: LearningType) -> Optional[LearningSession]:
        """Отримуємо осandнню forвершену сесandю"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM learning_sessions 
            WHERE loop_type = ? AND status = ?
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (loop_type.value, LearningStatus.COMPLETED.value))
        
        row = cursor.fetchone()
        if row:
            return self._row_to_learning_session(row)
        
        return None
    
    def _row_to_learning_session(self, row) -> LearningSession:
        """Конвертуємо рядок баwithи data в LearningSession"""
        
        return LearningSession(
            id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            loop_type=LearningType(row[2]),
            trigger_type=LearningTrigger(row[3]),
            trigger_reason=row[4],
            config=LearningLoopConfig(**json.loads(row[5])),
            status=LearningStatus(row[6]),
            start_time=datetime.fromisoformat(row[7]) if row[7] else None,
            end_time=datetime.fromisoformat(row[8]) if row[8] else None,
            duration_minutes=row[9],
            before_performance=json.loads(row[10]),
            after_performance=json.loads(row[11]) if row[11] else None,
            improvement_pct=row[12],
            data_points_used=row[13],
            model_version_before=row[14],
            model_version_after=row[15],
            validation_results=json.loads(row[16]) if row[16] else None,
            error_message=row[17],
            rollback_available=bool(row[18])
        )
    
    def close(self):
        """Закриваємо with'єднання with баwithою data"""
        if self.conn:
            self.conn.close()

def main():
    """Тестування подвandйних навчальних циклandв"""
    print("[REFRESH] DUAL LEARNING LOOPS - Internal & External Learning Cycles")
    print("=" * 60)
    
    loops = DualLearningLoopsEngine()
    
    # Тестуємо перевandрку тригерandв
    print(f"\n[SEARCH] TESTING LEARNING TRIGGERS")
    print("-" * 40)
    
    triggers = loops.check_learning_triggers()
    print(f"[TARGET] Found {len(triggers)} learning triggers")
    
    for trigger in triggers[:3]:
        print(f"    {trigger['type'].value}: {trigger['reason']}")
    
    # Тестуємо andнandцandацandю сесandї
    print(f"\n[START] TESTING SESSION INITIATION")
    print("-" * 40)
    
    if triggers:
        trigger = triggers[0]
        session_id = loops.initiate_learning_session(
            trigger["loop_type"],
            trigger["type"],
            trigger["reason"]
        )
        print(f"[OK] Session initiated: {session_id}")
        
        # Тестуємо виконання навчання
        print(f"\n[FAST] TESTING LEARNING EXECUTION")
        print("-" * 40)
        
        if trigger["loop_type"] == LearningType.INTERNAL_MODEL:
            success = loops.execute_internal_model_learning(session_id)
        else:
            success = loops.execute_external_agent_learning(session_id)
        
        print(f"{'[OK]' if success else '[ERROR]'} Learning execution: {'Success' if success else 'Failed'}")
    
    # Тестуємо аналandwith ефективностand
    print(f"\n[DATA] TESTING EFFECTIVENESS ANALYSIS")
    print("-" * 40)
    
    effectiveness = loops.analyze_learning_effectiveness()
    
    print(f"[UP] Learning effectiveness by type:")
    for type_data in effectiveness['by_loop_type']:
        print(f"    {type_data['type']}: {type_data['success_rate']:.1f}% success rate")
    
    print(f"\n[TARGET] Effectiveness by trigger:")
    for trigger_data in effectiveness['by_trigger_type'][:3]:
        print(f"    {trigger_data['trigger']}: {trigger_data['avg_improvement']:.2f}% avg improvement")
    
    # Тестуємо andсторandю навчання
    print(f"\n TESTING LEARNING HISTORY")
    print("-" * 40)
    
    history = loops.get_learning_history(limit=5)
    print(f" Recent learning sessions: {len(history)}")
    
    for session in history[:3]:
        print(f"    {session.loop_type.value}: {session.status.value} ({session.improvement_pct:.2f}% improvement)")
    
    loops.close()
    print(f"\n[OK] Dual Learning Loops test completed!")

if __name__ == "__main__":
    main()
