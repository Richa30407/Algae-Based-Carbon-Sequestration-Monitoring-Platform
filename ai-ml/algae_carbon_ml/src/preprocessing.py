"""
Data validation and cleaning for the Algae Carbon 360 ML pipeline.

Responsibilities:
  - Load raw CSV
  - Validate column presence and data types
  - Detect and log missing values, duplicates, invalid numerics
  - Impute missing values (interpolation per farm, or median fallback)
  - Return a clean DataFrame ready for feature engineering
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    ID_COLUMN,
    RAW_CSV,
    TIMESTAMP_COLUMN,
    MISSING_VALUE_STRATEGY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    ID_COLUMN,
    TIMESTAMP_COLUMN,
    "temperature_c",
    "sunlight_hours",
    "color_intensity",
    "ph",
    "nutrients_mg_l",
    "co2_ppm",
    "dissolved_oxygen_mg_l",
    "water_quality_index",
    "current_biomass_kg_m2",
    "future_biomass_kg_m2",
]

NUMERIC_COLUMNS = [
    "temperature_c",
    "sunlight_hours",
    "color_intensity",
    "ph",
    "nutrients_mg_l",
    "co2_ppm",
    "dissolved_oxygen_mg_l",
    "water_quality_index",
    "current_biomass_kg_m2",
    "future_biomass_kg_m2",
]


def _log_issue(message: str) -> None:
    """Log a data-quality issue at WARNING level."""
    logger.warning("DATA ISSUE: %s", message)


# ---------------------------------------------------------------------------
# Main loading & validation
# ---------------------------------------------------------------------------

def load_and_validate(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw CSV, validate structure, and return a cleaned DataFrame.

    Steps performed:
      1. Read CSV
      2. Check required columns exist
      3. Convert timestamp to datetime
      4. Sort by (farm_id, timestamp) chronologically
      5. Report duplicate timestamps per farm
      6. Report missing values
      7. Check for impossible negatives in non-negative fields
      8. Check for infinite values
      9. Impute missing numeric values
     10. Validate chronological ordering after sort
    """
    csv_path = path or RAW_CSV
    logger.info("Loading dataset from %s", csv_path)

    # --- 1. Read ----------------------------------------------------------
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # --- 2. Required columns ----------------------------------------------
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # --- 3. Timestamp conversion ------------------------------------------
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")
    nat_count = df[TIMESTAMP_COLUMN].isna().sum()
    if nat_count > 0:
        _log_issue(f"{nat_count} rows have unparseable timestamps – dropping them")
        df = df.dropna(subset=[TIMESTAMP_COLUMN]).reset_index(drop=True)

    # --- 4. Sort chronologically per farm ---------------------------------
    df = df.sort_values([ID_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)

    # --- 5. Duplicate timestamps per farm ---------------------------------
    dup_mask = df.duplicated(subset=[ID_COLUMN, TIMESTAMP_COLUMN], keep=False)
    dup_count = dup_mask.sum()
    if dup_count > 0:
        _log_issue(
            f"{dup_count} rows share a (farm_id, timestamp) – "
            "keeping first occurrence per group"
        )
        df = (
            df.drop_duplicates(subset=[ID_COLUMN, TIMESTAMP_COLUMN], keep="first")
            .reset_index(drop=True)
        )

    # --- 6. Missing values ------------------------------------------------
    null_counts = df[NUMERIC_COLUMNS].isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        _log_issue(
            f"Total missing values in numeric columns: {total_nulls}\n"
            f"{null_counts[null_counts > 0].to_string()}"
        )

    # --- 7. Impossible negatives ------------------------------------------
    non_negative_cols = [
        "sunlight_hours",
        "color_intensity",
        "nutrients_mg_l",
        "co2_ppm",
        "dissolved_oxygen_mg_l",
        "water_quality_index",
        "current_biomass_kg_m2",
        "future_biomass_kg_m2",
    ]
    for col in non_negative_cols:
        neg_count = (df[col].dropna() < 0).sum()
        if neg_count > 0:
            _log_issue(f"{col}: {neg_count} negative values found (impossible)")

    # --- 8. Infinite values ------------------------------------------------
    for col in NUMERIC_COLUMNS:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            _log_issue(f"{col}: {inf_count} infinite values – replacing with NaN")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # --- 9. Imputation ----------------------------------------------------
    df = _impute_missing(df, strategy=MISSING_VALUE_STRATEGY)

    # --- 10. Chronological ordering verification --------------------------
    for fid, group in df.groupby(ID_COLUMN):
        timestamps = group[TIMESTAMP_COLUMN].values
        if not np.all(timestamps[1:] >= timestamps[:-1]):
            _log_issue(f"Farm {fid}: timestamps are NOT in chronological order after sort")

    logger.info("Preprocessing complete – %d rows ready", len(df))
    return df


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def _impute_missing(df: pd.DataFrame, strategy: str = "interpolation") -> pd.DataFrame:
    """Impute missing numeric values per farm.

    Strategy 'interpolation':
      - Uses time-aware interpolation within each farm (order-aware).
      - Falls back to farm-level median for any remaining NaN at group edges.

    Strategy 'median':
      - Simple median imputation across the entire dataset.

    Assumption:
      - Each farm's time series is independent.
      - Interpolation is biologically reasonable for short gaps (1-3 days)
        in environmental sensor data.
    """
    df = df.copy()

    if strategy == "interpolation":
        numeric_in_df = [c for c in NUMERIC_COLUMNS if c in df.columns]
        for fid, group_idx in df.groupby(ID_COLUMN).groups.items():
            grp = df.loc[group_idx].sort_values(TIMESTAMP_COLUMN).copy()
            # Time-based interpolation requires a DatetimeIndex
            grp_indexed = grp.set_index(TIMESTAMP_COLUMN)[numeric_in_df]
            grp_indexed = grp_indexed.interpolate(
                method="time", limit_direction="both"
            )
            # Fallback median for any remaining NaN at edges
            grp_indexed = grp_indexed.fillna(grp_indexed.median())
            df.loc[group_idx, numeric_in_df] = grp_indexed[numeric_in_df].values

    elif strategy == "median":
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                median_val = df[col].median()
                filled = df[col].fillna(median_val)
                df[col] = filled

    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")

    # Final check – there should be no NaN left in numeric columns
    remaining_nulls = df[NUMERIC_COLUMNS].isnull().sum().sum()
    if remaining_nulls > 0:
        logger.warning(
            "After imputation, %d NaN values remain – "
            "applying global median fallback",
            remaining_nulls,
        )
        for col in NUMERIC_COLUMNS:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    return df
