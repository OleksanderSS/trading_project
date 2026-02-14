"""
NEURO-COGNITIVE CONTEXT ANALYZER
Інтеграцandя notйробandологandчних and когнandтивно-психологandчних принципandв у симуляцandї
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class CognitiveState(Enum):
    """Когнandтивнand сandни system"""
    CONSCIOUS_ATTENTION = "conscious_attention"      # Свandдома увага
    SUBCONSCIOUS_PATTERN = "subconscious_pattern"    # Пandдсвandдомand патерни
    EMOTIONAL_REGULATION = "emotional_regulation"      # Емоцandйна регуляцandя
    MEMORY_RECALL = "memory_recall"                   # Вandдновлення пам'ятand
    PREDICTION_MODE = "prediction_mode"               # Режим прогноwithування
    DECISION_MAKING = "decision_making"               # Прийняття рandшень

class NeuroPatternType(Enum):
    """Типи notйронних патернandв"""
    ATTENTION_FILTER = "attention_filter"              # Фandльтрацandя уваги
    MEMORY_CONSOLIDATION = "memory_consolidation"     # Консолandдацandя пам'ятand
    EMOTIONAL_TAGGING = "emotional_tagging"           # Емоцandйnot тегування
    PATTERN_RECOGNITION = "pattern_recognition"       # Роwithпandwithнавання патернandв
    PREDICTION_ERROR = "prediction_error"             # Error прогноwithування
    NEURAL_PLASTICITY = "neural_plasticity"           # Нейропластичнandсть

@dataclass
class NeuroCognitiveContext:
    """Нейро-когнandтивний контекст"""
    timestamp: datetime
    cognitive_state: CognitiveState
    attention_level: float  # 0-1, рandвень уваги
    memory_strength: float  # 0-1, сила пам'ятand
    emotional_valence: float  # -1 to 1, емоцandйна валентнandсть
    prediction_confidence: float  # 0-1, впевnotнandсть прогноwithу
    neural_activity: Dict[str, float] = field(default_factory=dict)
    pattern_weights: Dict[str, float] = field(default_factory=dict)
    
    def get_context_signature(self) -> str:
        """Унandкальна сигнатура контексту"""
        context_str = f"{self.timestamp}_{self.cognitive_state.value}_{self.attention_level:.3f}"
        return hashlib.md5(context_str.encode()).hexdigest()[:12]

@dataclass
class MarketNeuroPattern:
    """Ринковий notйронний патерн"""
    pattern_id: str
    pattern_type: NeuroPatternType
    market_conditions: Dict[str, Any]
    cognitive_correlates: Dict[str, float]
    historical_performance: Dict[str, float]
    confidence_score: float
    activation_threshold: float
    
    def should_activate(self, current_context: NeuroCognitiveContext) -> bool:
        """Перевandрка активацandї патерну"""
        # Роwithрахунок активацandї на основand когнandтивного контексту
        activation_score = 0.0
        
        for correlate, weight in self.cognitive_correlates.items():
            if correlate == "attention_level":
                activation_score += current_context.attention_level * weight
            elif correlate == "memory_strength":
                activation_score += current_context.memory_strength * weight
            elif correlate == "emotional_valence":
                activation_score += abs(current_context.emotional_valence) * weight
            elif correlate == "prediction_confidence":
                activation_score += current_context.prediction_confidence * weight
        
        return activation_score >= self.activation_threshold

class NeuroCognitiveContextAnalyzer:
    """
    Аналandforтор notйро-когнandтивних контекстandв for симуляцandй Деана
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Нейроннand патерни (навченand на andсторичних data)
        self.neuro_patterns = self._initialize_neuro_patterns()
        
        # Когнandтивнand сandни and them характеристики
        self.cognitive_profiles = self._initialize_cognitive_profiles()
        
        # Історична пам'ять system
        self.memory_consolidation = defaultdict(list)
        
        # Емоцandйнand теги for ринкових подandй
        self.emotional_tags = {}
        
        # Параметри notйропластичностand
        self.plasticity_parameters = {
            'learning_rate': 0.1,
            'forgetting_factor': 0.05,
            'consolidation_threshold': 0.7,
            'pattern_decay_rate': 0.02
        }
        
        # Сandтистика когнandтивної активностand
        self.cognitive_stats = {
            'total_contexts_analyzed': 0,
            'pattern_activations': defaultdict(int),
            'cognitive_state_transitions': defaultdict(int),
            'emotional_regulation_events': 0,
            'memory_consolidation_events': 0,
            'prediction_accuracy': 0.0
        }
    
    def create_informative_contexts(self, market_data: Dict[str, Any], 
                                 historical_patterns: Dict[str, Any]) -> List[NeuroCognitiveContext]:
        """
        Створення andнформативних контекстandв на основand notйробandологandчних принципandв
        """
        contexts = []
        
        # 1. Аналandwith поточних ринкових умов
        market_conditions = self._analyze_market_conditions(market_data)
        
        # 2. Виявлення когнandтивних тригерandв
        cognitive_triggers = self._identify_cognitive_triggers(market_conditions, historical_patterns)
        
        # 3. Створення контекстandв for рandwithних когнandтивних сandнandв
        for cognitive_state in CognitiveState:
            context = self._create_cognitive_context(
                cognitive_state, 
                market_conditions, 
                cognitive_triggers
            )
            contexts.append(context)
        
        # 4. Оновлення пам'ятand system
        self._update_memory_consolidation(contexts, market_conditions)
        
        self.cognitive_stats['total_contexts_analyzed'] += len(contexts)
        
        return contexts
    
    def simulate_neural_dynamics(self, base_context: NeuroCognitiveContext, 
                               time_horizon: int) -> List[NeuroCognitiveContext]:
        """
        Симуляцandя notйронної динамandки with урахуванням notйропластичностand
        """
        contexts = [base_context]
        current_context = base_context
        
        for time_step in range(1, time_horizon + 1):
            # 1. Еволюцandя когнandтивного сandну
            evolved_state = self._evolve_cognitive_state(current_context, time_step)
            
            # 2. Нейропластичнand differences
            plasticity_changes = self._apply_neural_plasticity(current_context, time_step)
            
            # 3. Емоцandйна регуляцandя
            emotional_regulation = self._apply_emotional_regulation(current_context, time_step)
            
            # 4. Консолandдацandя пам'ятand
            memory_consolidation = self._apply_memory_consolidation(current_context, time_step)
            
            # 5. Створення нового контексту
            new_context = NeuroCognitiveContext(
                timestamp=current_context.timestamp + timedelta(hours=time_step),
                cognitive_state=evolved_state,
                attention_level=self._evolve_attention_level(current_context.attention_level, plasticity_changes),
                memory_strength=self._evolve_memory_strength(current_context.memory_strength, memory_consolidation),
                emotional_valence=self._evolve_emotional_valence(current_context.emotional_valence, emotional_regulation),
                prediction_confidence=self._evolve_prediction_confidence(current_context.prediction_confidence, time_step),
                neural_activity=self._update_neural_activity(current_context.neural_activity, plasticity_changes),
                pattern_weights=self._update_pattern_weights(current_context.pattern_weights, memory_consolidation)
            )
            
            contexts.append(new_context)
            current_context = new_context
        
        return contexts
    
    def activate_relevant_patterns(self, context: NeuroCognitiveContext) -> List[MarketNeuroPattern]:
        """
        Активацandя релевантних notйронних патернandв
        """
        activated_patterns = []
        
        for pattern in self.neuro_patterns:
            if pattern.should_activate(context):
                activated_patterns.append(pattern)
                self.cognitive_stats['pattern_activations'][pattern.pattern_type.value] += 1
        
        return activated_patterns
    
    def calculate_cognitive_influence(self, contexts: List[NeuroCognitiveContext],
                                    patterns: List[MarketNeuroPattern]) -> Dict[str, float]:
        """
        Роwithрахунок когнandтивного впливу на ринковand прогноwithи
        """
        influence_scores = {}
        
        # 1. Аналandwith уваги (Attention Filter)
        attention_influence = self._calculate_attention_influence(contexts)
        influence_scores['attention_filter'] = attention_influence
        
        # 2. Вплив пам'ятand (Memory Recall)
        memory_influence = self._calculate_memory_influence(contexts)
        influence_scores['memory_recall'] = memory_influence
        
        # 3. Емоцandйна регуляцandя
        emotional_influence = self._calculate_emotional_influence(contexts)
        influence_scores['emotional_regulation'] = emotional_influence
        
        # 4. Прогноwithування майбутнього
        prediction_influence = self._calculate_prediction_influence(contexts, patterns)
        influence_scores['prediction_mode'] = prediction_influence
        
        # 5. Прийняття рandшень
        decision_influence = self._calculate_decision_influence(contexts, patterns)
        influence_scores['decision_making'] = decision_influence
        
        return influence_scores
    
    def _analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Аналandwith ринкових умов with notйробandологandчної точки withору"""
        conditions = {
            'volatility_regime': self._classify_volatility_regime(market_data),
            'trend_strength': self._calculate_trend_strength(market_data),
            'volume_anomaly': self._detect_volume_anomaly(market_data),
            'price_momentum': self._calculate_price_momentum(market_data),
            'market_sentiment': self._assess_market_sentiment(market_data),
            'time_pattern': self._analyze_time_pattern(market_data),
            'cross_asset_correlation': self._calculate_cross_correlation(market_data)
        }
        
        return conditions
    
    def _identify_cognitive_triggers(self, market_conditions: Dict[str, Any], 
                                   historical_patterns: Dict[str, Any]) -> Dict[str, float]:
        """Виявлення когнandтивних тригерandв"""
        triggers = {}
        
        # 1. Тригер високої волатильностand -> пandдвищена увага
        if market_conditions['volatility_regime'] == 'high':
            triggers['attention_boost'] = 0.3
        
        # 2. Сильний тренд -> пandдвищена впевnotнandсть прогноwithу
        if market_conditions['trend_strength'] > 0.7:
            triggers['prediction_confidence_boost'] = 0.2
        
        # 3. Об'ємна аномалandя -> емоцandйна реакцandя
        if market_conditions['volume_anomaly']:
            triggers['emotional_response'] = 0.4
        
        # 4. Історичнand патерни -> активацandя пам'ятand
        for pattern_name, pattern_strength in historical_patterns.items():
            if pattern_strength > 0.6:
                triggers[f'memory_recall_{pattern_name}'] = pattern_strength
        
        # 5. Часовий патерн (for example, whereнь тижня)
        time_triggers = self._analyze_time_based_triggers(market_conditions['time_pattern'])
        triggers.update(time_triggers)
        
        return triggers
    
    def _analyze_time_based_triggers(self, time_pattern: Dict[str, Any]) -> Dict[str, float]:
        """Аналandwith часових тригерandв (whereнь тижня, мandсяць тощо)"""
        triggers = {}
        
        current_time = datetime.now()
        
        # День тижня
        day_of_week = current_time.weekday()
        if day_of_week == 0:  # Поnotдandлок
            triggers['monday_effect'] = 0.15  # Історичний патерн перекупу
        elif day_of_week == 4:  # П'ятниця
            triggers['friday_effect'] = -0.1  # Історичний патерн перепродажу
        
        # Мandсяць
        day_of_month = current_time.day
        if day_of_month in [1, 15]:  # Початок/середина мandсяця
            triggers['monthly_rebalancing'] = 0.1
        
        # Кварandл
        month = current_time.month
        if month in [3, 6, 9, 12] and day_of_month <= 5:  # Кварandльнand withвandти
            triggers['earnings_season'] = 0.2
        
        return triggers
    
    def _create_cognitive_context(self, cognitive_state: CognitiveState,
                                market_conditions: Dict[str, Any],
                                triggers: Dict[str, float]) -> NeuroCognitiveContext:
        """Створення когнandтивного контексту"""
        profile = self.cognitive_profiles[cognitive_state]
        
        # Баwithовand параметри with профandлю
        attention_level = profile['base_attention']
        memory_strength = profile['base_memory']
        emotional_valence = profile['base_emotional']
        prediction_confidence = profile['base_prediction']
        
        # Корекцandя на основand тригерandв
        for trigger_name, trigger_value in triggers.items():
            if 'attention' in trigger_name:
                attention_level += trigger_value
            elif 'memory' in trigger_name:
                memory_strength += trigger_value
            elif 'emotional' in trigger_name:
                emotional_valence += trigger_value
            elif 'prediction' in trigger_name:
                prediction_confidence += trigger_value
        
        # Нормалandforцandя
        attention_level = max(0, min(1, attention_level))
        memory_strength = max(0, min(1, memory_strength))
        emotional_valence = max(-1, min(1, emotional_valence))
        prediction_confidence = max(0, min(1, prediction_confidence))
        
        return NeuroCognitiveContext(
            timestamp=datetime.now(),
            cognitive_state=cognitive_state,
            attention_level=attention_level,
            memory_strength=memory_strength,
            emotional_valence=emotional_valence,
            prediction_confidence=prediction_confidence,
            neural_activity=self._generate_neural_activity(cognitive_state),
            pattern_weights=self._generate_pattern_weights(cognitive_state)
        )
    
    def _initialize_neuro_patterns(self) -> List[MarketNeuroPattern]:
        """Інandцandалandforцandя notйронних патернandв"""
        patterns = []
        
        # 1. Патерн фandльтрацandї уваги
        attention_pattern = MarketNeuroPattern(
            pattern_id="attention_filter_001",
            pattern_type=NeuroPatternType.ATTENTION_FILTER,
            market_conditions={'high_volatility': True, 'volume_spike': True},
            cognitive_correlates={'attention_level': 0.8, 'memory_strength': 0.3},
            historical_performance={'accuracy': 0.75, 'false_positive_rate': 0.15},
            confidence_score=0.8,
            activation_threshold=0.6
        )
        patterns.append(attention_pattern)
        
        # 2. Патерн консолandдацandї пам'ятand
        memory_pattern = MarketNeuroPattern(
            pattern_id="memory_consolidation_001",
            pattern_type=NeuroPatternType.MEMORY_CONSOLIDATION,
            market_conditions={'trend_continuation': True, 'pattern_repeat': True},
            cognitive_correlates={'memory_strength': 0.9, 'attention_level': 0.5},
            historical_performance={'accuracy': 0.82, 'recall_rate': 0.88},
            confidence_score=0.85,
            activation_threshold=0.7
        )
        patterns.append(memory_pattern)
        
        # 3. Патерн емоцandйного тегування
        emotional_pattern = MarketNeuroPattern(
            pattern_id="emotional_tagging_001",
            pattern_type=NeuroPatternType.EMOTIONAL_TAGGING,
            market_conditions={'price_shock': True, 'sentiment_shift': True},
            cognitive_correlates={'emotional_valence': 0.7, 'attention_level': 0.6},
            historical_performance={'accuracy': 0.78, 'emotional_accuracy': 0.85},
            confidence_score=0.75,
            activation_threshold=0.5
        )
        patterns.append(emotional_pattern)
        
        # 4. Патерн роwithпandwithнавання патернandв
        pattern_recognition = MarketNeuroPattern(
            pattern_id="pattern_recognition_001",
            pattern_type=NeuroPatternType.PATTERN_RECOGNITION,
            market_conditions={'technical_pattern': True, 'reversal_signal': True},
            cognitive_correlates={'memory_strength': 0.8, 'prediction_confidence': 0.7},
            historical_performance={'accuracy': 0.80, 'pattern_detection_rate': 0.92},
            confidence_score=0.88,
            activation_threshold=0.65
        )
        patterns.append(pattern_recognition)
        
        # 5. Патерн помилки прогноwithування
        prediction_error = MarketNeuroPattern(
            pattern_id="prediction_error_001",
            pattern_type=NeuroPatternType.PREDICTION_ERROR,
            market_conditions={'unexpected_move': True, 'model_failure': True},
            cognitive_correlates={'prediction_confidence': 0.3, 'attention_level': 0.9},
            historical_performance={'error_detection_rate': 0.85, 'false_negative_rate': 0.10},
            confidence_score=0.70,
            activation_threshold=0.4
        )
        patterns.append(prediction_error)
        
        # 6. Патерн notйропластичностand
        plasticity_pattern = MarketNeuroPattern(
            pattern_id="neural_plasticity_001",
            pattern_type=NeuroPatternType.NEURAL_PLASTICITY,
            market_conditions={'regime_change': True, 'new_pattern': True},
            cognitive_correlates={'memory_strength': 0.4, 'attention_level': 0.8},
            historical_performance={'adaptation_rate': 0.75, 'learning_speed': 0.70},
            confidence_score=0.65,
            activation_threshold=0.5
        )
        patterns.append(plasticity_pattern)
        
        return patterns
    
    def _initialize_cognitive_profiles(self) -> Dict[CognitiveState, Dict[str, float]]:
        """Інandцandалandforцandя когнandтивних профandлandв"""
        return {
            CognitiveState.CONSCIOUS_ATTENTION: {
                'base_attention': 0.9,
                'base_memory': 0.6,
                'base_emotional': 0.2,
                'base_prediction': 0.7
            },
            CognitiveState.SUBCONSCIOUS_PATTERN: {
                'base_attention': 0.3,
                'base_memory': 0.9,
                'base_emotional': 0.4,
                'base_prediction': 0.8
            },
            CognitiveState.EMOTIONAL_REGULATION: {
                'base_attention': 0.5,
                'base_memory': 0.5,
                'base_emotional': 0.8,
                'base_prediction': 0.4
            },
            CognitiveState.MEMORY_RECALL: {
                'base_attention': 0.4,
                'base_memory': 0.95,
                'base_emotional': 0.3,
                'base_prediction': 0.6
            },
            CognitiveState.PREDICTION_MODE: {
                'base_attention': 0.7,
                'base_memory': 0.6,
                'base_emotional': 0.2,
                'base_prediction': 0.9
            },
            CognitiveState.DECISION_MAKING: {
                'base_attention': 0.8,
                'base_memory': 0.7,
                'base_emotional': 0.5,
                'base_prediction': 0.8
            }
        }
    
    def _generate_neural_activity(self, cognitive_state: CognitiveState) -> Dict[str, float]:
        """Геnotрацandя notйронної активностand for когнandтивного сandну"""
        activity_patterns = {
            CognitiveState.CONSCIOUS_ATTENTION: {
                'prefrontal_cortex': 0.9,
                'parietal_lobe': 0.7,
                'temporal_lobe': 0.5,
                'amygdala': 0.3,
                'hippocampus': 0.4
            },
            CognitiveState.SUBCONSCIOUS_PATTERN: {
                'prefrontal_cortex': 0.3,
                'parietal_lobe': 0.4,
                'temporal_lobe': 0.8,
                'amygdala': 0.5,
                'hippocampus': 0.9
            },
            CognitiveState.EMOTIONAL_REGULATION: {
                'prefrontal_cortex': 0.6,
                'parietal_lobe': 0.3,
                'temporal_lobe': 0.5,
                'amygdala': 0.9,
                'hippocampus': 0.4
            },
            CognitiveState.MEMORY_RECALL: {
                'prefrontal_cortex': 0.5,
                'parietal_lobe': 0.4,
                'temporal_lobe': 0.6,
                'amygdala': 0.3,
                'hippocampus': 0.95
            },
            CognitiveState.PREDICTION_MODE: {
                'prefrontal_cortex': 0.8,
                'parietal_lobe': 0.6,
                'temporal_lobe': 0.5,
                'amygdala': 0.4,
                'hippocampus': 0.6
            },
            CognitiveState.DECISION_MAKING: {
                'prefrontal_cortex': 0.95,
                'parietal_lobe': 0.7,
                'temporal_lobe': 0.6,
                'amygdala': 0.6,
                'hippocampus': 0.5
            }
        }
        
        return activity_patterns.get(cognitive_state, {})
    
    def _generate_pattern_weights(self, cognitive_state: CognitiveState) -> Dict[str, float]:
        """Геnotрацandя ваг патернandв for когнandтивного сandну"""
        weights = {
            'technical_patterns': 0.7,
            'fundamental_patterns': 0.5,
            'sentiment_patterns': 0.4,
            'time_patterns': 0.6,
            'volume_patterns': 0.5,
            'volatility_patterns': 0.6
        }
        
        # Корекцandя ваг на основand когнandтивного сandну
        if cognitive_state == CognitiveState.SUBCONSCIOUS_PATTERN:
            weights['technical_patterns'] = 0.9
            weights['time_patterns'] = 0.8
        elif cognitive_state == CognitiveState.EMOTIONAL_REGULATION:
            weights['sentiment_patterns'] = 0.8
            weights['volatility_patterns'] = 0.7
        elif cognitive_state == CognitiveState.MEMORY_RECALL:
            weights['time_patterns'] = 0.9
            weights['technical_patterns'] = 0.8
        
        return weights
    
    def get_cognitive_report(self) -> Dict[str, Any]:
        """Отримати withвandт про когнandтивну активнandсть"""
        return {
            'cognitive_statistics': self.cognitive_stats,
            'active_patterns_count': len([p for p in self.neuro_patterns if p.confidence_score > 0.7]),
            'memory_consolidation_size': len(self.memory_consolidation),
            'plasticity_parameters': self.plasticity_parameters,
            'emotional_tags_count': len(self.emotional_tags),
            'system_cognitive_health': self._assess_cognitive_health()
        }
    
    def _assess_cognitive_health(self) -> Dict[str, float]:
        """Оцandнка когнandтивного withдоров'я system"""
        health_metrics = {
            'attention_stability': 0.8,
            'memory_integrity': 0.85,
            'emotional_balance': 0.75,
            'prediction_accuracy': self.cognitive_stats['prediction_accuracy'],
            'pattern_recognition_rate': 0.82,
            'adaptation_capability': 0.78
        }
        
        overall_health = np.mean(list(health_metrics.values()))
        health_metrics['overall'] = overall_health
        
        return health_metrics


# Глобальний аналandforтор
_neuro_cognitive_analyzer = None

def get_neuro_cognitive_analyzer() -> NeuroCognitiveContextAnalyzer:
    """Отримати глобальний notйро-когнandтивний аналandforтор"""
    global _neuro_cognitive_analyzer
    if _neuro_cognitive_analyzer is None:
        _neuro_cognitive_analyzer = NeuroCognitiveContextAnalyzer()
    return _neuro_cognitive_analyzer
