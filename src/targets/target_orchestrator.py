import logging
from typing import Any

import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.targets.calculators.classification_calculator import ClassificationCalculator
from src.targets.calculators.cross_sectional_calculator import CrossSectionalCalculator
from src.targets.calculators.indicator_prediction_calculator import IndicatorPredictionCalculator
from src.targets.calculators.regression_calculator import RegressionCalculator
from src.targets.timeframe_contract import (
    mask_targets_across_time_boundaries,
    resolve_column_for_frame,
    resolve_target_timeframe_contract,
    target_applies_to_timeframe,
)

logger = ProjectLogger.get_logger('TargetOrchestrator')


class TargetOrchestrator:
    """
    Orchestrates the generation of target variables based on a YAML configuration.
    It dynamically loads and applies the required calculators.
    """

    def __init__(self, targets_list, timeframe=None):
        """
        Initialize with targets in either dict or list format.

        Args:
            targets_list: Either a dict {target_name: config} or list [{name: ..., type: ..., params: ...}]
            timeframe: Optional timeframe to filter targets (e.g., '15m', '60m', '1d')
        """
        self.CALCULATOR_MAPPING = {'regression': RegressionCalculator,
            'classification_binary': ClassificationCalculator,
            'classification_multiclass': ClassificationCalculator,
            'indicator_prediction': IndicatorPredictionCalculator,
            'cross_sectional': CrossSectionalCalculator}
        self.METHOD_MAPPING = {'classification_binary': 'calculate_binary',
            'classification_multiclass': 'calculate_multiclass'}
        self.timeframe = timeframe
        if isinstance(targets_list, dict):
            self.targets = [{'name': name, **config} for name, config in
                targets_list.items()]
        else:
            self.targets = targets_list
        if timeframe:
            original_count = len(self.targets)
            self.targets = self._filter_targets_by_timeframe(timeframe)
            if self.targets:
                logger.info(
                    f'🎯 TIMEFRAME FILTERING: {timeframe} (was {original_count}, remaining {len(self.targets)})'
                    )
            else:
                logger.warning(
                    f"⚠️ No targets found for timeframe '{timeframe}'. Using all targets."
                    )
        import json
        runtime_params = {}
        config_manager = get_current_config()
        params_path = config_manager.get_runtime_params_path()
        if params_path.exists():
            try:
                with open(params_path) as f:
                    runtime_params = json.load(f)
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.error(f'Виникла помилка: {e}', exc_info=True)
                logger.warning(f'Could not load runtime_params.json: {e}')
                raise
        test_mode = runtime_params.get('test_mode', {})
        test_target = test_mode.get('test_target') or runtime_params.get(
            'test_target')
        if test_target:
            original_count = len(self.targets)
            self.targets = [t for t in self.targets if t['name'] == test_target
                ]
            if self.targets:
                logger.info(
                    f'🎯 TARGET FILTERING: {test_target} (was {original_count}, remaining {len(self.targets)})'
                    )
            else:
                logger.warning(
                    f"⚠️ test_target '{test_target}' not found in configuration! Using all targets."
                    )
                if isinstance(targets_list, dict):
                    self.targets = [{'name': name, **config} for name,
                        config in targets_list.items()]
                else:
                    self.targets = targets_list
        logger.info(
            f'TargetOrchestrator initialized with {len(self.targets)} target configurations.'
            )
        self._warn_about_unread_params()

    # Params consumed by the orchestration/contract layer rather than by a
    # calculator. `source_timeframe` and `indicator_col` are read by
    # timeframe_contract._target_source_timeframe.
    _FRAMEWORK_PARAMS = frozenset({
        'description', 'horizon', 'source_timeframe', 'indicator_col',
    })

    def _warn_about_unread_params(self) -> None:
        """Flag configured target params that no calculator actually reads.

        Calculators take `**kwargs`, so a param nobody implements is swallowed
        in silence and the target quietly computes something other than what
        its description promises. Real examples found in targets.yaml:
        `method: slope_strength` / `window: 20` on a regression target, which
        RegressionCalculator ignores entirely -- so
        `target_daily_trend_strength_1d` is a plain next-bar return, identical
        to `target_daily_momentum_score_1d`; and `compare_to: "average"` on
        `target_hourly_volume_spike_1h`, which is compared to the current bar
        instead. This makes that class of mistake loud instead of silent.
        """
        for target in self.targets:
            t_type = target.get('type')
            calculator_cls = self.CALCULATOR_MAPPING.get(t_type)
            if calculator_cls is None:
                continue

            supported = getattr(calculator_cls, 'SUPPORTED_PARAMS', None)
            if supported is None:
                logger.debug(
                    f"{calculator_cls.__name__} declares no SUPPORTED_PARAMS; "
                    f"cannot check target '{target.get('name')}' for dead params."
                )
                continue

            known = set(supported) | self._FRAMEWORK_PARAMS
            unread = sorted(set((target.get('params') or {}).keys()) - known)
            if unread:
                logger.warning(
                    f"⚠️ Target '{target.get('name')}' ({t_type}) declares "
                    f"param(s) {unread} that {calculator_cls.__name__} never "
                    f"reads — they are silently ignored, so this target may "
                    f"not compute what its description says."
                )

    def _is_target_for_timeframe(self, target: dict[str, Any], timeframe: str) -> bool:
        """Check if a target is applicable for the given timeframe."""
        return target_applies_to_timeframe(target, timeframe)

    def _filter_targets_by_timeframe(self, timeframe: str) ->list:
        """
        Filter targets based on timeframe.

        Rules:
        - 15m timeframe: Only intraday_15m targets
        - 60m timeframe: Only hourly_1h targets + general targets that make sense
        - 1d timeframe: Only daily/weekly targets + general targets

        Args:
            timeframe: The timeframe to filter for

        Returns:
            Filtered list of target configurations
        """
        return [target for target in self.targets if self._is_target_for_timeframe(target, timeframe)]

    def generate_targets(self, df: pd.DataFrame, **kwargs) ->pd.DataFrame:
        """
        Generates all configured targets for the given DataFrame.

        Returns ONLY target columns + minimal metadata (datetime, ticker, interval).
        This prevents data leakage and keeps targets DataFrame clean.
        """
        self._validate_input_dataframe(df)
        targets_dict = self._prepare_metadata_container(df)
        logger.info(
            f'🎯 Generating {len(self.targets)} targets (clean mode - no feature leakage)'
            )
        for target_config in self.targets:
            self._generate_single_target(df, target_config, targets_dict,
                **kwargs)
        return self._create_targets_dataframe(targets_dict)

    def _validate_input_dataframe(self, df: pd.DataFrame) ->None:
        """Validate that DataFrame has required columns."""
        if 'ticker' not in df.columns:
            logger.error(
                "DataFrame must contain a 'ticker' column for target generation."
                )
            raise ValueError("Missing 'ticker' column.")

    def _prepare_metadata_container(self, df: pd.DataFrame) ->dict:
        """Prepare metadata container with only essential columns."""
        metadata_columns = ['datetime', 'ticker', 'interval']
        available_metadata = [col for col in metadata_columns if col in df.
            columns]
        return {col: df[col].copy() for col in available_metadata}

    def _generate_single_target(self, df: pd.DataFrame, target_config: dict,
        targets_dict: dict, **kwargs) ->None:
        """Generate a single target and add it to the targets dictionary."""
        name = target_config['name']
        target_type = target_config['type']
        params = target_config.get('params', {})
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Generating target: {name} (Type: {target_type})')
        calculator_class = self.CALCULATOR_MAPPING.get(target_type)
        if not calculator_class:
            logger.warning(
                f"No calculator found for target type '{target_type}'. Skipping target '{name}'."
                )
            return
        try:
            self._handle_standard_target(df, name, target_type, params,
                targets_dict)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Failed to generate target '{name}'. Error: {e}")

    def _handle_standard_target(self, df: pd.DataFrame, name: str,
        target_type: str, params: dict, targets_dict: dict) ->None:
        """Handle standard target generation."""
        calculator_class = self.CALCULATOR_MAPPING[target_type]
        calculator_instance = calculator_class()
        method_name = self.METHOD_MAPPING.get(target_type, 'calculate')
        calculation_method = getattr(calculator_instance, method_name)
        # Some targets are a property of the CROSS-SECTION, not of one name.
        # "Did AAPL beat the average of everything we hold" needs every ticker
        # at the same instant, and computing it per ticker group would make the
        # cross-sectional mean of one ticker equal to that ticker -- a column of
        # exact zeros, emitted silently with no error and no missing values.
        # See CrossSectionalCalculator.
        # A target names one indicator and then runs wherever its horizon is
        # valid. The horizon already resolves per frame; the column has to as
        # well, or the target dies on frames it was meant to run on.
        params = self._resolve_column_params(params, df, self.timeframe, name)

        needs_full_frame = getattr(calculator_class, 'REQUIRES_FULL_FRAME', False)
        if needs_full_frame:
            if 'ticker' not in df.columns or df['ticker'].nunique() < 2:
                logger.warning(
                    "Target '%s' is cross-sectional but the frame carries %d "
                    "ticker(s); there is no cross-section to measure against, "
                    "so it is skipped rather than emitted as zeros.",
                    name, df['ticker'].nunique() if 'ticker' in df.columns else 0,
                )
                return
            target_series = calculation_method(df, **params)
        elif 'ticker' in df.columns:
            target_series = self._process_by_ticker_groups(df,
                calculation_method, params)
        else:
            target_series = calculation_method(df, **params)
        targets_dict[name] = target_series
        logger.info(f"Successfully generated target '{name}'.")

    def _process_by_ticker_groups(self, df: pd.DataFrame,
        calculation_method, params: dict) ->pd.Series:
        """Process target calculation by ticker groups."""
        target_series_list = []
        group_columns = ['ticker']
        if 'interval' in df.columns:
            group_columns.append('interval')
        for _identity, group in df.groupby(group_columns, dropna=False):
            sorted_group = self._sort_group_for_targets(group)
            resolved_params, contract = resolve_target_timeframe_contract(
                params,
                sorted_group,
                default_timeframe=self.timeframe,
            )
            group_target = calculation_method(sorted_group, **resolved_params)
            group_target = mask_targets_across_time_boundaries(
                sorted_group,
                group_target,
                contract,
            )
            group_target.index = sorted_group.index
            target_series_list.append(group_target)
        if not target_series_list:
            return pd.Series(index=df.index, dtype=float)
        return pd.concat(target_series_list).reindex(df.index)

    @staticmethod
    def _resolve_column_params(
        params: dict[str, Any],
        frame: pd.DataFrame,
        timeframe: str | None,
        target_name: str,
    ) -> dict[str, Any]:
        """Point a target's column names at the frame it is running on.

        The horizon already resolves per frame: a "1h" horizon becomes four
        bars on 15-minute data and one bar on hourly data, which is why the
        same target legitimately runs on both. Its columns did not follow.
        The feature stage suffixes indicators with the frame's own `interval`,
        so the band is `BB_Upper_15m` on one frame and `BB_Upper_1h` on the
        other, while the config named `BB_Upper_60m` -- which exists on
        neither.

        `target_hourly_breakout_1h` and `target_volatility_spike_1h` therefore
        failed on every frame of every run, and simply did not exist in any
        batch. The log said so twice per run, under ERROR, for long enough
        that the lines had become part of the scenery.

        An exact match wins, so a target that deliberately reaches for another
        timeframe's indicator keeps it.
        """
        adjusted = dict(params)
        for key in ('base_col', 'indicator_col'):
            configured = adjusted.get(key)
            if not configured:
                continue
            resolved = resolve_column_for_frame(str(configured), frame, timeframe)
            if resolved is None or resolved == configured:
                continue
            logger.info(
                "Target '%s': %s '%s' resolved to '%s' for the %s frame.",
                target_name, key, configured, resolved, timeframe or 'unknown',
            )
            adjusted[key] = resolved
        return adjusted

    def _sort_group_for_targets(self, group: pd.DataFrame) ->pd.DataFrame:
        """Sort each ticker group chronologically before future-shift target generation."""
        for col in ('datetime', 'timestamp', 'date'):
            if col in group.columns:
                return group.sort_values(col, kind='mergesort').copy()
        return group.sort_index(kind='mergesort').copy()

    def _create_targets_dataframe(self, targets_dict: dict) ->pd.DataFrame:
        """Create final targets DataFrame and log summary."""
        targets_df = pd.DataFrame(targets_dict)
        target_cols = [col for col in targets_df.columns if col.startswith(
            'target_')]
        logger.info(
            f'✅ Generated {len(target_cols)} target columns (total {len(targets_df.columns)} with metadata)'
            )
        return targets_df
