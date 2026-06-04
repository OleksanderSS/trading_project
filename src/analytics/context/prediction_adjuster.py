import logging
from typing import Any

from ..interfaces import IAnalyzer

logger = logging.getLogger(__name__)


class PredictionAdjuster(IAnalyzer):
    """
    Adjusts model predictions based on a generic, declarative rule engine that uses
    market context.
    """

    def __init__(self, config: dict[str, Any] | None=None):
        """
        Initializes the PredictionAdjuster with a list of adjustment rules.

        Args:
            config (Dict[str, Any]): A dictionary containing a 'rules' list.
        """
        self.rules: list[dict[str, Any]] = (config or {}).get('rules', [])
        if not self.rules:
            logger.warning(
                'PredictionAdjuster initialized with no rules. No adjustments will be made.'
                )
        else:
            logger.info(
                f'PredictionAdjuster initialized with {len(self.rules)} adjustment rules.'
                )

    def analyze(self, data: dict[str, Any], **kwargs) ->dict[str, Any]:
        """
        Enhances model predictions by applying the configured rules to the context data.

        Args:
            data (Dict[str, Any]): A dictionary containing:
                - 'predictions' (Dict[str, float]): Model names and their raw predictions.
                - Other context features (e.g., 'market_phase', 'avg_sentiment') used by the rules.

        Returns:
            Dict[str, Any]: A dictionary containing the 'enhanced_predictions'.
        """
        model_predictions = data.get('predictions', {})
        if not model_predictions:
            return self._create_empty_predictions_result()
        context_info = self._extract_context_info(data)
        enhanced_predictions = self._process_all_predictions(model_predictions,
            context_info)
        logger.info(
            f'Completed prediction adjustments for {len(model_predictions)} model(s).'
            )
        return {'enhanced_predictions': enhanced_predictions}

    def _create_empty_predictions_result(self) ->dict[str, Any]:
        """Create result for empty predictions."""
        logger.warning("No 'predictions' found in data to adjust.")
        return {'enhanced_predictions': {}}

    def _extract_context_info(self, data: dict[str, Any]) ->dict[str, Any]:
        """Extract context information from data."""
        return {key: val for key, val in data.items() if key != 'predictions'}

    def _process_all_predictions(self, model_predictions: dict[str, float],
        context_info: dict[str, Any]) ->dict[str, float]:
        """Process all model predictions with adjustments."""
        enhanced_predictions = {}
        for model_name, prediction_value in model_predictions.items():
            adjusted_value = self._adjust_single_prediction(model_name,
                prediction_value, context_info)
            enhanced_predictions[model_name] = adjusted_value
        return enhanced_predictions

    def _adjust_single_prediction(self, model_name: str, prediction_value:
        float, context_info: dict[str, Any]) ->float:
        """Adjust a single prediction value."""
        adjustment_factor = self._calculate_adjustment_factor(context_info)
        enhanced_value = prediction_value * adjustment_factor
        self._log_adjustment(model_name, prediction_value,
            adjustment_factor, enhanced_value)
        return enhanced_value

    def _calculate_adjustment_factor(self, context_info: dict[str, Any]
        ) ->float:
        """Calculate adjustment factor based on rules."""
        adjustment_factor = 1.0
        for rule in self.rules:
            try:
                if self._evaluate_rule_conditions(rule.get('if', {}),
                    context_info):
                    adjustment_factor = self._apply_rule_action(rule.get(
                        'then', {}), adjustment_factor)
            except Exception as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                self._log_rule_error(rule, e)
                raise
        return adjustment_factor

    def _log_rule_error(self, rule: dict[str, Any], error: Exception):
        """Log rule processing error."""
        logger.error(
            f"Error processing rule '{rule.get('name', 'Unnamed')}': {error}",
            exc_info=True)

    def _log_adjustment(self, model_name: str, original: float, factor:
        float, enhanced: float):
        """Log prediction adjustment details."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Adjusting '{model_name}': Original={original:.4f}, Factor={factor:.4f}, Enhanced={enhanced:.4f}"
                )

    def _evaluate_rule_conditions(self, conditions: dict[str, Any], context:
        dict[str, Any]) ->bool:
        """
        Evaluates the 'if' block of a rule. Currently supports an 'all' (AND) block.
        """
        if not self._has_valid_conditions(conditions):
            return False
        return all(self._evaluate_single_condition(condition, context) for
            condition in conditions['all'])

    def _has_valid_conditions(self, conditions: dict[str, Any]) ->bool:
        """Check if conditions are valid."""
        return 'all' in conditions and isinstance(conditions['all'], list)

    def _evaluate_single_condition(self, condition: dict[str, Any], context:
        dict[str, Any]) ->bool:
        """Evaluate a single condition."""
        feature = condition.get('context_feature')
        if not self._is_feature_available(feature, context):
            return False
        context_value = context[feature]
        return self._evaluate_all_condition_types(condition, context_value)

    def _is_feature_available(self, feature: str, context: dict[str, Any]
        ) ->bool:
        """Check if feature is available in context."""
        return feature in context

    def _evaluate_all_condition_types(self, condition: dict[str, Any],
        context_value: Any) ->bool:
        """Evaluate all condition types for a feature."""
        categorical_ok = self._check_categorical_condition(condition,
            context_value)
        numerical_ok = self._check_numerical_conditions(condition,
            context_value)
        return categorical_ok and numerical_ok

    def _check_categorical_condition(self, condition: dict[str, Any],
        context_value: Any) ->bool:
        """Check categorical condition (e.g., market_phase == 'Growth')."""
        if 'is' in condition:
            return context_value == condition['is']
        return True

    def _check_numerical_conditions(self, condition: dict[str, Any],
        context_value: Any) ->bool:
        """Check numerical conditions (greater_than, less_than)."""
        if 'greater_than' in condition and context_value <= condition[
            'greater_than']:
            return False
        if 'less_than' in condition and context_value >= condition['less_than'
            ]:
            return False
        return True

    def _apply_rule_action(self, action: dict[str, Any], current_factor: float
        ) ->float:
        """
        Applies the action from a rule's 'then' block.
        This version modifies a single adjustment factor.
        """
        if self._is_multiplier_action(action):
            return self._apply_multiplier_action(action, current_factor)
        return current_factor

    def _is_multiplier_action(self, action: dict[str, Any]) ->bool:
        """Check if action is a multiplier action."""
        return action.get('action') == 'apply_multiplier'

    def _apply_multiplier_action(self, action: dict[str, Any],
        current_factor: float) ->float:
        """Apply multiplier action to current factor."""
        multiplier = action.get('multiplier', 1.0)
        weight = action.get('weight', 1.0)
        return current_factor * (1 + (multiplier - 1) * weight)
