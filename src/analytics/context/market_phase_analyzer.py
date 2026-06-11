import logging
import operator
import re
from typing import Any

import pandas as pd

from ..interfaces import IAnalyzer

logger = logging.getLogger(__name__)


class MarketPhaseAnalyzer(IAnalyzer):
    """
    Determines the current market phase based on a declarative set of rules
    from a configuration dictionary.
    """

    def __init__(self, config: dict[str, Any] | None=None):
        """
        Initializes the analyzer with market phase definition rules.

        Args:
            config (Dict[str, Any]): A dictionary containing the market phase definition,
                                     including 'indicators' and 'rules'.
        """
        phase_config = config or {}
        self.logger = logger
        self.indicators = phase_config.get('indicators', {})
        self.rules = phase_config.get('rules', [])
        self._CONDITION_ALLOWED = re.compile('^[\\w\\s<>=!.&|()+-]+$')
        # Order matters: match >= before >, <= before <, != before =, etc.
        self._OPS = {'>=': operator.ge, '<=': operator.le, '==': operator.eq,
            '!=': operator.ne, '>': operator.gt, '<': operator.lt}
        self.logger.info(
            f'MarketPhaseAnalyzer initialized with {len(self.rules)} rules.')

    def analyze(self, data: dict[str, pd.DataFrame], **kwargs) ->dict[str, Any
        ]:
        """
        Analyzes the latest market data to determine the current market phase.

        Args:
            data (Dict[str, pd.DataFrame]): Expects a key 'market_data' with a DataFrame
                                             containing the required indicator columns.

        Returns:
            Dict[str, Any]: A dictionary containing the determined 'market_phase'.
        """
        validation_result = self._validate_market_data(data)
        if not validation_result['valid']:
            result: dict[str, Any] = validation_result['result']
            return result
        try:
            market_phase = self._determine_market_phase(validation_result[
                'market_data'])
            self.logger.info(f'Determined market phase: {market_phase}')
            return {'market_phase': market_phase}
        except KeyError as e:
            self.logger.error(
                f'A required indicator column was not found in the market data: {e}'
                , exc_info=True)
            return {'market_phase': 'error'}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Error evaluating market phase: {e}',
                exc_info=True)
            return {'market_phase': 'error'}

    def _validate_market_data(self, data: dict[str, pd.DataFrame]) ->dict[
        str, Any]:
        """Validate input market data."""
        market_data = data.get('market_data')
        if not self._is_valid_dataframe(market_data):
            return self._create_validation_error('Invalid input data')
        missing_columns = self._check_missing_columns(market_data)
        if missing_columns:
            return self._create_missing_columns_error(missing_columns,
                market_data)
        return {'valid': True, 'market_data': market_data}

    def _is_valid_dataframe(self, market_data: pd.DataFrame) ->bool:
        """Check if market data is a valid DataFrame."""
        return isinstance(market_data, pd.DataFrame) and not market_data.empty

    def _create_validation_error(self, reason: str) ->dict[str, Any]:
        """Create validation error result."""
        return {'valid': False, 'result': {'market_phase': 'error',
            'reason': reason}}

    def _create_missing_columns_error(self, missing_columns: set,
        market_data: pd.DataFrame) ->dict[str, Any]:
        """Create missing columns error result."""
        logger.warning(
            f'Missing required columns for market phase analysis: {missing_columns}'
            )
        logger.warning(f'Available columns: {list(set(market_data.columns))}')
        return {'valid': False, 'result': {'market_phase': 'neutral',
            'reason': f'Missing indicators: {missing_columns}'}}

    def _check_missing_columns(self, market_data: pd.DataFrame) ->set:
        """Check for missing required columns."""
        required_columns = set(self.indicators.values())
        available_columns = set(market_data.columns)
        return required_columns - available_columns

    def _determine_market_phase(self, market_data: pd.DataFrame) ->str:
        """Determine market phase from rules."""
        latest_data_point = market_data.iloc[-1]
        latest_values = self._extract_latest_values(latest_data_point)
        return self._evaluate_rules(latest_values)

    def _extract_latest_values(self, latest_data_point: pd.Series) ->dict[
        str, float]:
        """Extract latest indicator values."""
        latest_values = self._build_indicator_values(latest_data_point)
        self._validate_extracted_values(latest_values)
        return latest_values

    def _build_indicator_values(self, latest_data_point: pd.Series) ->dict[
        str, float]:
        """Build indicator values from latest data point."""
        return {key: latest_data_point[val] for key, val in self.indicators
            .items() if val in latest_data_point}

    def _validate_extracted_values(self, latest_values: dict[str, float]):
        """Validate that all required indicators were extracted."""
        if len(latest_values) != len(self.indicators):
            missing = set(self.indicators.keys()) - set(latest_values.keys())
            logger.warning(
                f'Missing indicators in data: {missing}. Phase evaluation might be incorrect.'
                )

    def _evaluate_rules(self, latest_values: dict[str, float]) ->str:
        """Evaluate phase rules against latest values."""
        for rule in self.rules:
            if self._evaluate_condition(rule.get('condition', 'False'),
                latest_values):
                phase: str = rule.get('phase', 'unknown')
                return phase
        return 'unknown'

    def _evaluate_condition(self, condition_str: str, values: dict[str, float]
        ) ->bool:
        """
        Safely evaluates a condition string using the provided values.
        Example: "gdp > 0.5 and vix < 20"
        """
        if not self._is_valid_condition_string(condition_str):
            return False
        try:
            return self._safe_eval_condition(condition_str, values)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self._log_evaluation_error(condition_str, values, e)
            return False

    def _is_valid_condition_string(self, condition_str: str) ->bool:
        """Validates that condition string contains only safe tokens (whitelist)."""
        if not condition_str:
            return False
        return bool(self._CONDITION_ALLOWED.match(condition_str))

    def _eval_factor(self, factor: str, values: dict[str, float]) -> bool:
        """Evaluate a single factor in the condition."""
        factor = factor.strip()
        if factor.lower() == 'true':
            return True
        if factor.lower() == 'false':
            return False
        for op_str, op_fn in self._OPS.items():
            if op_str not in factor:
                continue
            parts = factor.split(op_str)
            if len(parts) != 2:
                continue
            lhs, rhs = parts[0].strip(), parts[1].strip()
            lhs_val = values.get(lhs)
            if lhs_val is None:
                return False
            try:
                rhs_val = float(rhs)
            except ValueError:
                return False
            return bool(op_fn(lhs_val, rhs_val))
        self.logger.warning(f"Unrecognized sub-condition: '{factor}'")
        return False

    def _safe_eval_condition(self, condition_str: str, values: dict[str, float]
        ) ->bool:
        """
        Safely evaluate a simple comparison condition without using eval().
        Supports: 'key op value' and 'key op value and/or key op value' patterns.
        SEC-1: pd.eval() replaced by explicit operator dispatch to prevent code injection.
        """
        # Split by OR first (lower precedence), then AND.
        or_terms = re.split('\\bor\\b', condition_str, flags=re.IGNORECASE)
        for term in or_terms:
            and_factors = re.split('\\band\\b', term, flags=re.IGNORECASE)
            if all(self._eval_factor(f, values) for f in and_factors if f.strip()):
                return True
        return False

    def _log_evaluation_error(self, condition_str: str, values: dict[str,
        float], error: Exception):
        """Log condition evaluation error."""
        self.logger.error(
            f"Could not safely evaluate condition '{condition_str}' with values {values}: {error}"
            , exc_info=True)
