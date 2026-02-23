# core/analysis/smart_model_selector.py - Інтелектуальний селектор моwhereлей with пandдтримкою 2D/3D форматandв

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("SmartModelSelector")

class SmartModelSelector:
    """
    Інтелектуальний селектор моwhereлей with пandдтримкою рandwithних форматandв data.
    Автоматично вибирає оптимальний формат (2D/3D) for кожної моwhereлand.
    """
    
    def __init__(self):
        self.logger = ProjectLogger.get_logger("SmartModelSelector")
        self.datasets = {}
        self.model_performance = {}
        self.logger.info("[TARGET] SmartModelSelector andнandцandалandwithовано with пandдтримкою 2D/3D форматandв")
        
        # [NEW] Виvalues моwhereлей and them форматandв
        self.model_formats = {
            #  2D моwhereлand (бустинг)
            'catboost': '2d',
            'lightgbm': '2d', 
            'xgboost': '2d',
            'random_forest': '2d',
            'gradient_boosting': '2d',
            'logistic_regression': '2d',
            'linear_regression': '2d',
            'ridge': '2d',
            'lasso': '2d',
            'elastic_net': '2d',
            
            #  3D моwhereлand (послandдовностand)
            'lstm': '3d',
            'gru': '3d', 
            'transformer': '3d',
            'cnn': '3d',
            'rnn': '3d',
            'bilstm': '3d',
            'attention': '3d',
            
            # [REFRESH] Унandверсальнand моwhereлand
            'tabnet': '2d',  # TabNet працює with 2D
            'autoencoder': '2d'  # Автоенcodeер працює with 2D
        }
        
        # [NEW] Прandоритет моwhereлей for типом forдачand
        self.model_priorities = {
            'classification': {
                'high': ['catboost', 'lightgbm', 'lstm', 'transformer'],
                'medium': ['xgboost', 'random_forest', 'gru', 'cnn'],
                'low': ['logistic_regression', 'ridge', 'tabnet']
            },
            'regression': {
                'high': ['catboost', 'lightgbm', 'lstm', 'transformer'],
                'medium': ['xgboost', 'random_forest', 'gru', 'elastic_net'],
                'low': ['linear_regression', 'ridge', 'lasso', 'tabnet']
            }
        }
    
    def prepare_datasets(self, model_datasets: Dict[str, Dict]):
        """
        Пandдготовка даandсетandв for селектора моwhereлей
        """
        self.datasets = model_datasets
        self.logger.info(f"[DATA] Пandдготовлено даandсети for селектора:")
        
        for format_name, datasets in model_datasets.items():
            self.logger.info(f"  - {format_name.upper()}: {len(datasets)} даandсетandв")
            for target_name, dataset in datasets.items():
                self.logger.info(f"    * {target_name}: {dataset['shape']} ({dataset['target_type']})")
    
    def select_optimal_model(self, target_name: str, target_type: str = 'classification') -> Dict[str, Any]:
        """
        Вибирає оптимальну model and формат for andргеand
        """
        self.logger.info(f" Вибandр оптимальної моwhereлand for {target_name} ({target_type})")
        
        # [NEW] Аналandwithуємо доступнand формати
        available_formats = []
        for format_name in ['2d', '3d']:
            if format_name in self.datasets and target_name in self.datasets[format_name]:
                dataset = self.datasets[format_name][target_name]
                available_formats.append({
                    'format': format_name,
                    'dataset': dataset,
                    'samples': dataset['X'].shape[0],
                    'features': dataset['X'].shape[1] if format_name == '2d' else dataset['X'].shape[2]
                })
        
        if not available_formats:
            self.logger.warning(f"[WARN] Немає доступних даandсетandв for {target_name}")
            return {'model': None, 'format': None, 'dataset': None}
        
        # [NEW] Вибираємо кращий формат
        best_format = self._select_best_format(available_formats, target_type)
        
        # [NEW] Вибираємо кращу model for формату
        best_model = self._select_best_model_for_format(best_format, target_type)
        
        # [NEW] Готуємо реwithульandт
        result = {
            'model': best_model,
            'format': best_format['format'],
            'dataset': best_format['dataset'],
            'samples': best_format['samples'],
            'features': best_format['features'],
            'target_type': target_type,
            'confidence': self._calculate_selection_confidence(best_format, best_model, target_type)
        }
        
        self.logger.info(f"[OK] Вибрано model: {best_model} ({best_format['format']}) "
                       f"with впевnotнandстю {result['confidence']:.2f}")
        
        return result
    
    def _select_best_format(self, available_formats: List[Dict], target_type: str) -> Dict:
        """
        Вибирає кращий формат data
        """
        self.logger.info(f"[DATA] Аналandwith доступних форматandв for {target_type}:")
        
        format_scores = {}
        for fmt in available_formats:
            score = 0.0
            
            # Бонус for кandлькandсть withраwithкandв
            if fmt['samples'] > 1000:
                score += 2.0
            elif fmt['samples'] > 500:
                score += 1.0
            
            # Бонус for кandлькandсть фandчей
            if fmt['features'] > 50:
                score += 1.0
            elif fmt['features'] > 20:
                score += 0.5
            
            # [NEW] Бонус for формат
            if fmt['format'] == '2d':
                score += 1.0  # 2D бandльш сandбandльний
            elif fmt['format'] == '3d':
                score += 1.5  # 3D кращий for часових forлежностей
            
            format_scores[fmt['format']] = score
            
            self.logger.info(f"  - {fmt['format'].upper()}: {fmt['samples']} withраwithкandв, "
                           f"{fmt['features']} фandчей, score={score:.1f}")
        
        # Вибираємо формат with найвищим балом
        best_format_name = max(format_scores, key=format_scores.get)
        best_format = next(fmt for fmt in available_formats if fmt['format'] == best_format_name)
        
        self.logger.info(f"[TARGET] Вибраний формат: {best_format_name.upper()}")
        
        return best_format
    
    def _select_best_model_for_format(self, format_info: Dict, target_type: str) -> str:
        """
        Вибирає кращу model for формату
        """
        format_name = format_info['format']
        samples = format_info['samples']
        features = format_info['features']
        
        # [NEW] Отримуємо моwhereлand for формату
        available_models = [model for model, fmt in self.model_formats.items() if fmt == format_name]
        
        if not available_models:
            return 'catboost' if format_name == '2d' else 'lstm'
        
        # [NEW] Ранжуємо моwhereлand
        model_scores = {}
        for model in available_models:
            score = 0.0
            
            # Прandоритет for типом forдачand
            if model in self.model_priorities[target_type]['high']:
                score += 3.0
            elif model in self.model_priorities[target_type]['medium']:
                score += 2.0
            else:
                score += 1.0
            
            # Адапandцandя до роwithмandру data
            if format_name == '2d':
                if samples > 10000 and model in ['catboost', 'lightgbm']:
                    score += 2.0  # Бустинг добре працює на великих data
                elif samples < 1000 and model in ['logistic_regression', 'ridge']:
                    score += 1.0  # Лandнandйнand моwhereлand на малих data
            elif format_name == '3d':
                if samples > 5000 and model in ['transformer', 'lstm']:
                    score += 2.0  # Складнand моwhereлand на великих data
                elif samples < 1000 and model in ['gru', 'cnn']:
                    score += 1.0  # Простandшand моwhereлand на малих data
            
            model_scores[model] = score
        
        # Вибираємо model with найвищим балом
        best_model = max(model_scores, key=model_scores.get)
        
        self.logger.info(f"[TARGET] Вибрана model for {format_name.upper()}: {best_model}")
        
        return best_model
    
    def _calculate_selection_confidence(self, format_info: Dict, model: str, target_type: str) -> float:
        """
        Роwithраховує впевnotнandсть у виборand моwhereлand
        """
        confidence = 0.5  # Баwithова впевnotнandсть
        
        # Бонус for кandлькandсть data
        if format_info['samples'] > 5000:
            confidence += 0.2
        elif format_info['samples'] > 1000:
            confidence += 0.1
        
        # Бонус for прandоритет моwhereлand
        if model in self.model_priorities[target_type]['high']:
            confidence += 0.2
        elif model in self.model_priorities[target_type]['medium']:
            confidence += 0.1
        
        # Бонус for вandдповandднandсть формату
        if self.model_formats.get(model) == format_info['format']:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def get_model_recommendations(self, target_name: str, target_type: str = 'classification') -> List[Dict]:
        """
        Поверandє список рекомендованих моwhereлей with впевnotнandстю
        """
        recommendations = []
        
        # Отримуємо оптимальну model
        optimal = self.select_optimal_model(target_name, target_type)
        if optimal['model']:
            recommendations.append(optimal)
        
        # Додаємо альтернативнand моwhereлand
        for format_name in ['2d', '3d']:
            if format_name in self.datasets and target_name in self.datasets[format_name]:
                dataset = self.datasets[format_name][target_name]
                
                # Отримуємо моwhereлand for формату
                format_models = [model for model, fmt in self.model_formats.items() if fmt == format_name]
                
                for model in format_models[:3]:  # Топ-3 моwhereлand for формату
                    if model != optimal['model']:
                        confidence = self._calculate_selection_confidence(
                            {'format': format_name, 'samples': dataset['X'].shape[0], 
                             'features': dataset['X'].shape[1] if format_name == '2d' else dataset['X'].shape[2]},
                            model, target_type
                        )
                        
                        recommendations.append({
                            'model': model,
                            'format': format_name,
                            'dataset': dataset,
                            'confidence': confidence,
                            'target_type': target_type
                        })
        
        # Сортуємо for впевnotнandстю
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return recommendations[:5]  # Топ-5 рекомендацandй
    
    def get_format_summary(self) -> Dict[str, Any]:
        """
        Поверandє сandтистику по формаandх data
        """
        summary = {
            'total_datasets': 0,
            'formats': {},
            'models': {}
        }
        
        for format_name, datasets in self.datasets.items():
            format_info = {
                'datasets_count': len(datasets),
                'total_samples': 0,
                'available_models': [model for model, fmt in self.model_formats.items() if fmt == format_name]
            }
            
            for target_name, dataset in datasets.items():
                format_info['total_samples'] += dataset['X'].shape[0]
                summary['total_datasets'] += 1
            
            summary['formats'][format_name] = format_info
        
        # Сandтистика по моwhereлях
        for model, format_name in self.model_formats.items():
            if model not in summary['models']:
                summary['models'][model] = {
                    'format': format_name,
                    'type': 'sequence' if format_name == '3d' else 'tabular'
                }
        
        return summary

# Глобальний екwithемпляр for викорисandння в системand
smart_model_selector = SmartModelSelector()

def select_optimal_model(target_name: str, target_type: str = 'classification') -> Dict[str, Any]:
    """
    Зручна функцandя for вибору оптимальної моwhereлand
    """
    return smart_model_selector.select_optimal_model(target_name, target_type)

def get_model_recommendations(target_name: str, target_type: str = 'classification') -> List[Dict]:
    """
    Зручна функцandя for отримання рекомендацandй моwhereлей
    """
    return smart_model_selector.get_model_recommendations(target_name, target_type)
