# Sector Market Coverage Bridge Field Audit

## Overview
This is a mechanical, read-only field-level audit of the data transition bridge: 
`CleanYahooMarketSnapshot` → `PipelineControlSavedDataCoverage` → `PipelineControlSavedPriceRepair`.

## Transition 1: CleanYahooMarketSnapshot → PipelineControlSavedDataCoverage

**Files & Lines:**
- Output: `CleanYahooMarketSnapshot._normalize_collected_frame` (`dean_os/clean_yahoo_market_snapshot.py:215`)
- Input: `PipelineControlSavedDataCoverage._analyze_price_source` (`dean_os/pipeline_control/pipeline_control_saved_data_coverage.py:234`)

### 1. Compatible Fields
- `datetime`, `ticker`, and `close` are fully compatible.
- The `datetime` column from the parquet is successfully loaded and converted to timezone-aware UTC via `pd.to_datetime(..., utc=True)` (`pipeline_control_saved_data_coverage.py:264`), perfectly matching the clean snapshot's output constraints.

### 2. Missing or Incompatible Fields for `eligible_contexts`
- **Filename Timeframe Assumption**: 
  - `PipelineControlSavedDataCoverage` derives the timeframe from the file path via `_timeframe_from_path` (`pipeline_control_saved_data_coverage.py:245`), looking for substrings like `_15m_`.
  - The clean snapshot outputs files named `clean_yahoo_market_...parquet` and `latest.parquet`. Since these do not contain `15m`, the timeframe resolves to `None`, resulting in immediate rejection: `"blocked_missing_or_unknown_timeframe"` (`pipeline_control_saved_data_coverage.py:322`).
- **Mixed Interval Contamination**: 
  - The clean snapshot consolidates 15m, 60m, and 1d data into a single Parquet file. 
  - `PipelineControlSavedDataCoverage._analyze_context` evaluates cadences via `diff().dt.total_seconds()` (`pipeline_control_saved_data_coverage.py:334`) without filtering by the `interval` column first. 
  - If a file contains mixed intervals, the cadence calculation will be heavily skewed, triggering `"timeframe_cadence_mismatch"`.

## Transition 2: PipelineControlSavedDataCoverage → PipelineControlSavedPriceRepair

**Files & Lines:**
- Output: `PipelineControlSavedDataCoverage.build` (JSON Manifest) (`dean_os/pipeline_control/pipeline_control_saved_data_coverage.py:115`)
- Input: `PipelineControlSavedPriceRepair.build` (`dean_os/pipeline_control/pipeline_control_saved_price_repair.py:24`)

### 3. Determination of `effective_start`
- Calculated individually for each ticker in `PipelineControlSavedDataCoverage._analyze_context` (`pipeline_control_saved_data_coverage.py:327-341`).
- **Mechanism**: Calculates absolute percentage returns (`pct_change().abs()`). Identifies rows where returns exceed `max_abs_return` (default 0.25). 
- It drops all rows prior to and including the last identified extreme anomaly: `clean = frame.iloc[clean_start_position:]`.
- The `effective_start` is strictly set to the earliest UTC timestamp in this "clean suffix": `clean["_coverage_datetime"].min()`.

## 4. Missing Root CLI Entrypoints
The following CLI scripts are explicitly required by the pipeline control runbook but do **not** exist in the project root or the `scripts/` folder (they are currently isolated in `.archive_temp/agent_scripts/`):
1. `run_agent_pipeline_control_saved_data_coverage.py`
2. `run_agent_pipeline_control_saved_price_repair.py`

## 5. Required Offline Fixtures & Malformed-Input Tests
To robustly test this bridge without side-effects, the following offline fixtures are required:
1. **Filename Fallback Fixture**: A `.parquet` file lacking `_15m_` in its name but containing a valid `interval="15m"` column, to ensure `PipelineControlSavedDataCoverage` correctly parses the column instead of falling back to `None`.
2. **Mixed Cadence Fixture**: A clean snapshot parquet containing 15m, 60m, and 1d rows for the same ticker. Validates that coverage strips non-15m rows before evaluating the 15m cadence ratio.
3. **NaT `effective_start` Test**: A mock coverage JSON where `effective_start: null`. When `PipelineControlSavedPriceRepair` parses this via `pd.to_datetime` (`pipeline_control_saved_price_repair.py:213`), it yields `NaT`. Testing must confirm that `.ge(start)` handles `NaT` cleanly without dropping all legitimate rows.
4. **Cross-Ticker Contamination Test**: A fixture with duplicated `datetime` + `OHLCV` across different tickers to ensure `_cross_ticker_identity_groups` (`pipeline_control_saved_price_repair.py:327`) properly aborts execution with a `ValueError`.
