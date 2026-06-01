import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')
from src.core.logging.logger import ProjectLogger
logger = ProjectLogger.get_logger('SmartModelSelector')


class SmartModelSelector:
    """Smart selector моделей на основі історії, контексту та мета-навчання."""

    def __init__(self, results_file: str='model_performance_history.json',
        competence_map_file: str=
        'src/models/model_selector/model_competence_map.json'):
        self.results_file = results_file
        self.performance_history = self._load_performance_history()
        self.competence_map = self._load_competence_map(competence_map_file)
        self.metric_weights = self.competence_map.get('metric_weights', {})
        self.error_meta_model = RandomForestRegressor(n_estimators=50,
            max_depth=5, min_samples_leaf=2, max_features='sqrt',
            random_state=42)
        self.is_meta_model_trained = False

    def _load_performance_history(self) ->Dict:
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            logger.warning(f'Помилка завантаження історії: {e}')
            return {}

    def _load_competence_map(self, file_path: str) ->Dict:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f'Не вдалося завантажити карту компетенцій: {e}')
            raise

    def _save_performance_history(self):
        try:
            with open(self.results_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f'Помилка збереження історії: {e}')

    def train_error_meta_model(self, model_name: str, ticker: str):
        """Тренує мета-модель на історичних помилках конкретної моделі."""
        history_key = f'{model_name}_{ticker}_classification'
        if (history_key not in self.performance_history or not self.
            performance_history[history_key].get('runs')):
            logger.warning(
                f'Недостатньо історії для Training мета-моделі для {history_key}'
                )
            return
        runs = self.performance_history[history_key]['runs']
        contexts = [run['context'] for run in runs]
        errors = [(1 - run['metrics'].get('accuracy', 0.5)) for run in runs]
        context_df = pd.DataFrame(contexts)
        X = pd.get_dummies(context_df, columns=context_df.columns).to_numpy()
        y = np.array(errors)
        if X.shape[0] < 10:
            logger.warning('Недостатньо прикладів для Training мета-моделі.')
            return
        self.error_meta_model.fit(X, y)
        self.is_meta_model_trained = True
        logger.info(
            f'Мета-модель помилок для {history_key} успішно натренована.')

    def analyze_context(self, df: pd.DataFrame) ->Dict[str, str]:
        context = {}
        if 'close' not in df.columns:
            return {'data_quality': 'low'}
        context['volatility'] = self._determine_volatility_level(df)
        context['trend'] = self._determine_trend_level(df)
        context['market_regime'] = self._determine_market_regime(df)
        context['data_quality'] = self._determine_data_quality(df)
        return context

    def _determine_volatility_level(self, df: pd.DataFrame) ->str:
        returns = df['close'].pct_change(fill_method=None).dropna()
        if len(returns) == 0:
            return 'medium'
        volatility = returns.std()
        if volatility < 0.01:
            return 'low'
        elif volatility < 0.03:
            return 'medium'
        else:
            return 'high'

    def _determine_trend_level(self, df: pd.DataFrame) ->str:
        if len(df) < 20:
            return 'sideways'
        recent_trend = df['close'].iloc[-1] / df['close'].iloc[-20] - 1
        if recent_trend > 0.05:
            return 'up'
        elif recent_trend < -0.05:
            return 'down'
        else:
            return 'sideways'

    def _determine_market_regime(self, df: pd.DataFrame) ->str:
        if 'VIX_SIGNAL' not in df.columns:
            return 'neutral'
        avg_vix = df['VIX_SIGNAL'].mean()
        if avg_vix > 0.5:
            return 'bear'
        elif avg_vix < -0.5:
            return 'bull'
        else:
            return 'neutral'

    def _determine_data_quality(self, df: pd.DataFrame) ->str:
        if df.shape[0] == 0:
            return 'low'
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_pct < 0.05:
            return 'high'
        elif missing_pct < 0.15:
            return 'medium'
        else:
            return 'low'

    def critique_action(self, action: Dict[str, Any], context_df: pd.DataFrame
        ) ->Dict[str, Any]:
        """Оцінює запропоновану дію, using логіку Критика з DEAN."""
        model_name = action.get('model_name', 'unknown')
        target_type = action.get('target_type', 'classification')
        context = self.analyze_context(context_df)
        context_adjustment = self._calculate_context_adjustment(model_name,
            context)
        expected_error = self._get_expected_error(context)
        ticker = action.get('ticker', 'default')
        historical_reliability = self._get_historical_reliability(model_name,
            ticker, target_type)
        critique_score = self._calculate_critique_score(expected_error,
            context_adjustment, historical_reliability)
        points = self._generate_critique_points(context, expected_error,
            context_adjustment, historical_reliability)
        return {'score': float(np.clip(critique_score, -1.0, 1.0)),
            'points': points, 'alternatives': [{'type': 'hold'}] if 
            critique_score < -0.3 else [], 'expected_error_by_critic':
            float(expected_error), 'confidence': 0.8}

    def _get_expected_error(self, context: Dict) ->float:
        """Get expected error from meta-model or default"""
        if not self.is_meta_model_trained:
            return 0.5
        try:
            context_for_pred = pd.get_dummies(pd.DataFrame([context]),
                columns=context.keys()).to_numpy()
            return self.error_meta_model.predict(context_for_pred)[0]
        except Exception as e:
            self.logger.error(f'Error predicting expected error: {e}', exc_info=True)
            return 0.5


    def _get_historical_reliability(self, model_name: str, target_type: str,
        context: Dict) ->float:
        """Get historical reliability for model"""
        for key in self.performance_history:
            if key.startswith(model_name) and key.endswith(target_type):
                recent_success = np.mean([r['metrics'].get('accuracy', 0.5) for
                    r in self.performance_history[key]['runs'][-5:]])
                return recent_success
        return 0.5

    def _calculate_critique_score(self, expected_error: float,
        context_adjustment: float, historical_reliability: float) ->float:
        """Calculate overall critique score"""
        critique_score = 0
        if expected_error > 0.4:
            critique_score -= 0.5 * (expected_error - 0.4)
        if context_adjustment < 0.9:
            critique_score -= 0.3 * (1 - context_adjustment)
        if historical_reliability < 0.5:
            critique_score -= 0.4
        return critique_score

    def _generate_critique_points(self, context: Dict, expected_error:
        float, context_adjustment: float, historical_reliability: float
        ) ->List[str]:
        """Generate critique explanation points"""
        points = [f'Context: {context}']
        if expected_error > 0.4:
            points.append(f'High predicted error: {expected_error:.2f}')
        if context_adjustment < 0.9:
            points.append(
                f'Model is not optimal for current context (adj: {context_adjustment:.2f})'
                )
        if historical_reliability < 0.5:
            points.append('Significant historical underperformance')
        return points

    def calculate_model_score(self, model_name: str, target_type: str,
        context: Dict) ->float:
        """Calculate model score based on history and context"""
        base_score = self._get_default_model_score(model_name, target_type)
        historical_score = self._get_historical_score(model_name, target_type)
        if historical_score is not None:
            base_score = historical_score
        context_adjustment = self._calculate_context_adjustment(model_name,
            context)
        final_score = base_score * context_adjustment
        return min(1.0, max(0.0, final_score))

    def _get_historical_score(self, model_name: str, target_type: str
        ) ->Optional[float]:
        """Calculate historical performance score for model"""
        for key in self.performance_history:
            if key.startswith(model_name) and key.endswith(target_type):
                scores = self._calculate_run_scores(self.
                    performance_history[key].get('runs', []))
                if scores:
                    return np.mean(scores)
        return None

    def _calculate_run_scores(self, runs: List[Dict]) ->List[float]:
        """Calculate scores from individual runs"""
        scores = []
        for run in runs:
            metrics = run.get('metrics', {})
            weighted_score, total_weight = self._calculate_weighted_score(
                metrics)
            if total_weight > 0:
                scores.append(weighted_score / total_weight)
        return scores

    def _calculate_weighted_score(self, metrics: Dict) ->Tuple[float, float]:
        """Calculate weighted score from metrics"""
        weighted_score, total_weight = 0, 0
        for metric, weight in self.metric_weights.items():
            if metric in metrics:
                weighted_score += metrics[metric] * weight
                total_weight += weight
        return weighted_score, total_weight

    def _get_default_model_score(self, model_name: str, target_type: str
        ) ->float:
        return self.competence_map.get('default_scores', {}).get(target_type,
            {}).get(model_name, 0.5)

    def _calculate_context_adjustment(self, model_name: str, context: Dict
        ) ->float:
        adjustment = 1.0
        rules = self.competence_map.get('context_rules', {})
        for factor, factor_rules in rules.items():
            current_value = context.get(factor)
            if not current_value:
                continue
            for rule_name, rule_details in factor_rules.items():
                is_trending = (factor == 'trend' and rule_name ==
                    'trending' and current_value in ['up', 'down'])
                is_market_trending = (factor == 'market_regime' and 
                    rule_name == 'trending' and current_value in ['bull',
                    'bear'])
                if (is_trending or is_market_trending or current_value ==
                    rule_name):
                    if model_name in rule_details.get('models', []):
                        adjustment *= rule_details.get('adjustment', 1.0)
        return adjustment

    def select_best_model(self, df: pd.DataFrame, target_type: str,
        available_models: List[str]=None) ->Tuple[str, float]:
        if available_models is None:
            available_models = list(self.competence_map.get(
                'default_scores', {}).get(target_type, {}).keys())
        context = self.analyze_context(df)
        model_scores = {name: self.calculate_model_score(name, target_type,
            context) for name in available_models}
        best_model = max(model_scores, key=model_scores.get)
        best_score = model_scores[best_model]
        logger.info(
            f'Best model selected: {best_model} with score {best_score:.3f}')
        return best_model, best_score

    def update_performance(self, model_name: str, ticker: str, target_type:
        str, metrics: Dict, context: Dict):
        history_key = f'{model_name}_{ticker}_{target_type}'
        if history_key not in self.performance_history:
            self.performance_history[history_key] = {'runs': []}
        run_data = {'timestamp': pd.Timestamp.now().isoformat(), 'metrics':
            metrics, 'context': context}
        self.performance_history[history_key]['runs'].append(run_data)
        self.performance_history[history_key]['runs'
            ] = self.performance_history[history_key]['runs'][-50:]
        self._save_performance_history()
        self.train_error_meta_model(model_name, ticker)
