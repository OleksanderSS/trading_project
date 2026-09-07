# audit-ignore: ARCHITECTURAL_USAGE
# src/pipeline/stages/stage_3_improvements.py
"""
Improvements for Stage 3: Feature Engineering
- Validation of alignment between features and targets
- Data quality metrics
"""

from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("Stage3Improvements")


def validate_and_align_features_targets(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    timeframe: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate and align features and targets DataFrames.

    Ensures:
    1. Same number of rows
    2. Same datetime index
    3. No NaT values in datetime

    Args:
        features_df: Features DataFrame
        targets_df: Targets DataFrame
        timeframe: Timeframe identifier for logging

    Returns:
        Tuple of (aligned_features_df, aligned_targets_df)
    """
    try:
        # Check if both DataFrames have datetime column
        if 'datetime' not in features_df.columns or 'datetime' not in targets_df.columns:
            logger.warning(f"⚠️ Missing datetime column for {timeframe}. Cannot validate alignment.")
            return features_df, targets_df

        # Check row count alignment
        if len(features_df) != len(targets_df):
            logger.warning(
                f"⚠️ Alignment mismatch for {timeframe}: "
                f"features={len(features_df)}, targets={len(targets_df)}"
            )

            # Align by datetime (inner join)
            features_aligned = features_df.merge(
                targets_df[['datetime']],
                on='datetime',
                how='inner'
            )
            targets_aligned = targets_df.merge(
                features_df[['datetime']],
                on='datetime',
                how='inner'
            )

            logger.info(
                f"✅ Aligned {timeframe}: "
                f"{len(features_df)} features → {len(features_aligned)}, "
                f"{len(targets_df)} targets → {len(targets_aligned)}"
            )

            return features_aligned, targets_aligned

        # Check datetime alignment (same values)
        if not features_df['datetime'].equals(targets_df['datetime']):
            logger.warning(f"⚠️ Datetime values mismatch for {timeframe}. Aligning...")

            # Sort both by datetime
            features_sorted = features_df.sort_values('datetime').reset_index(drop=True)
            targets_sorted = targets_df.sort_values('datetime').reset_index(drop=True)

            # Inner join on datetime
            features_aligned = features_sorted.merge(
                targets_sorted[['datetime']],
                on='datetime',
                how='inner'
            )
            targets_aligned = targets_sorted.merge(
                features_sorted[['datetime']],
                on='datetime',
                how='inner'
            )

            logger.info(f"✅ Datetime aligned for {timeframe}: {len(features_aligned)} rows")
            return features_aligned, targets_aligned

        # All checks passed
        logger.info(f"✅ Alignment validated for {timeframe}: {len(features_df)} rows")
        return features_df, targets_df

    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.error(f"❌ Error validating alignment for {timeframe}: {e}")
        return features_df, targets_df


# Status values for a quality report. NO_EVIDENCE means "nothing was measured
# here"; it is deliberately not a low score, because a low score is a claim
# about data that exists.
NO_EVIDENCE = 'no_evidence'
MEASURED = 'measured'


def calculate_data_quality_metrics(
    enriched_prices: dict[str, pd.DataFrame],
    all_targets: dict[str, pd.DataFrame],
    enrichers_count: int
) -> dict[str, Any]:
    """
    Calculate data quality metrics for enriched data.

    Args:
        enriched_prices: Dictionary of enriched price DataFrames by timeframe
        all_targets: Dictionary of target DataFrames by timeframe
        enrichers_count: Number of enrichers used

    Returns:
        Dictionary with quality metrics
    """
    try:
        metrics = {
            'timeframes': {},
            'overall': {}
        }

        # Calculate metrics for each timeframe
        total_features = 0
        total_targets = 0
        total_rows = 0
        total_nat_count = 0

        timeframes_with_evidence = 0

        for tf in enriched_prices.keys():
            features_df = enriched_prices.get(tf, pd.DataFrame())
            targets_df = all_targets.get(tf, pd.DataFrame())

            if features_df.empty:
                # No rows is not a quality of zero — it is the absence of a
                # measurement. Reporting 0 features / 0% nulls here would read
                # as "measured, and bad", hiding the absence behind a number.
                metrics['timeframes'][tf] = {
                    'status': NO_EVIDENCE,
                    'reason': 'no enriched rows for this timeframe',
                    'features_shape': features_df.shape,
                    'targets_shape': targets_df.shape,
                    'features_count': None,
                    'targets_count': None,
                    'nat_count': None,
                    'null_percentage': None,
                    'alignment_valid': None,
                }
                continue

            timeframes_with_evidence += 1

            # Calculate timeframe-specific metrics
            tf_metrics = {
                'status': MEASURED,
                'features_shape': features_df.shape,
                'targets_shape': targets_df.shape,
                'features_count': len([col for col in features_df.columns if not col.startswith('target_')]),
                'targets_count': len([col for col in targets_df.columns if col.startswith('target_')]),
                'nat_count': features_df['datetime'].isna().sum() if 'datetime' in features_df.columns else 0,
                'null_percentage': (features_df.isnull().sum().sum() / (features_df.shape[0] * features_df.shape[1]) * 100) if features_df.shape[0] > 0 else None,
                # Tri-state on purpose. No targets means there is nothing to
                # align, which is not the same claim as "aligned incorrectly".
                'alignment_valid': (len(features_df) == len(targets_df)) if not targets_df.empty else None,
            }

            metrics['timeframes'][tf] = tf_metrics

            # Accumulate for overall metrics
            total_features += tf_metrics['features_count']
            total_targets += tf_metrics['targets_count']
            total_rows += features_df.shape[0]
            total_nat_count += tf_metrics['nat_count']

        # Calculate overall metrics. Averages are taken over the timeframes
        # that actually had rows: dividing by every requested timeframe let an
        # empty one drag the average down as though it had been measured and
        # found to contain nothing.
        has_evidence = timeframes_with_evidence > 0
        metrics['overall'] = {
            'status': MEASURED if has_evidence else NO_EVIDENCE,
            'total_timeframes': len(enriched_prices),
            'timeframes_with_evidence': timeframes_with_evidence,
            'total_features': total_features if has_evidence else None,
            'total_targets': total_targets if has_evidence else None,
            'total_rows': total_rows,
            'total_nat_count': total_nat_count if has_evidence else None,
            'enrichers_count': enrichers_count,
            'avg_features_per_timeframe': (total_features / timeframes_with_evidence) if has_evidence else None,
            'avg_targets_per_timeframe': (total_targets / timeframes_with_evidence) if has_evidence else None,
        }

        # Add context fingerprint metrics if available
        for tf, features_df in enriched_prices.items():
            if features_df.empty:
                continue
            if 'context_fingerprint' in features_df.columns:
                unique_contexts = features_df['context_fingerprint'].nunique()
                metrics['timeframes'][tf]['unique_contexts'] = unique_contexts
                metrics['overall']['total_unique_contexts'] = metrics['overall'].get('total_unique_contexts', 0) + unique_contexts

        return metrics

    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.error(f"❌ Error calculating data quality metrics: {e}")
        return {'error': str(e)}


def _render(value: Any) -> str:
    """Render a metric, saying so when there was nothing to measure."""
    return "no evidence" if value is None else str(value)


def _render_alignment(value: bool | None) -> str:
    """
    Alignment is tri-state.

    ``None`` means there were no targets to align against, which used to be
    printed as ❌ — a verdict of "misaligned" against data that does not exist.
    """
    if value is None:
        return "no evidence (no targets to align against)"
    return "✅" if value else "❌"


def log_data_quality_report(metrics: dict[str, Any]) -> None:
    """
    Log data quality report in a readable format.

    Args:
        metrics: Data quality metrics dictionary
    """
    try:
        logger.info("=" * 80)
        logger.info("📊 DATA QUALITY REPORT")
        logger.info("=" * 80)

        # Overall metrics
        overall = metrics.get('overall', {})
        logger.info("📈 Overall Metrics:")
        logger.info(f"   Timeframes: {overall.get('total_timeframes', 0)}")
        logger.info(f"   Timeframes with data: {overall.get('timeframes_with_evidence', 0)}")
        if overall.get('status') == NO_EVIDENCE:
            logger.warning(
                "   ⚠️  NO EVIDENCE: no timeframe produced any enriched rows. "
                "Nothing below was measured — this is missing data, not poor "
                "quality data."
            )
        logger.info(f"   Total Features: {_render(overall.get('total_features'))}")
        logger.info(f"   Total Targets: {_render(overall.get('total_targets'))}")
        logger.info(f"   Total Rows: {overall.get('total_rows', 0)}")
        logger.info(f"   NaT Count: {_render(overall.get('total_nat_count'))}")
        logger.info(f"   Enrichers Used: {overall.get('enrichers_count', 0)}")

        if 'total_unique_contexts' in overall:
            logger.info(f"   Unique Contexts: {overall.get('total_unique_contexts', 0)}")

        # Timeframe-specific metrics
        timeframes = metrics.get('timeframes', {})
        if timeframes:
            logger.info("\n📋 Timeframe-Specific Metrics:")
            for tf, tf_metrics in timeframes.items():
                logger.info(f"\n   {tf}:")
                if tf_metrics.get('status') == NO_EVIDENCE:
                    logger.info(
                        f"      ⚠️  no evidence — {tf_metrics.get('reason', 'no rows')}"
                    )
                    continue
                logger.info(f"      Features: {tf_metrics.get('features_shape', (0, 0))}")
                logger.info(f"      Targets: {tf_metrics.get('targets_shape', (0, 0))}")
                logger.info(f"      NaT Count: {_render(tf_metrics.get('nat_count'))}")
                null_pct = tf_metrics.get('null_percentage')
                logger.info(
                    f"      Null %: {f'{null_pct:.2f}%' if null_pct is not None else 'no evidence'}")
                logger.info(f"      Alignment Valid: {_render_alignment(tf_metrics.get('alignment_valid'))}")

                if 'unique_contexts' in tf_metrics:
                    logger.info(f"      Unique Contexts: {tf_metrics.get('unique_contexts', 0)}")

        logger.info("=" * 80)

    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.exception(f"❌ Error logging data quality report: {e}")
