"""
Feature engineering for the Algae Carbon 360 ML pipeline.

Responsibilities:
  - Create previous_biomass (lag-1 biomass per farm)
  - Create future_biomass target (shifted biomass per farm)
  - Extract time-based features from the timestamp
  - Drop rows that cannot be used for training (first/last row per farm)
  - Return a feature-ready DataFrame
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CURRENT_BIOMASS_COLUMN,
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core feature engineering
# ---------------------------------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from the preprocessed DataFrame.

    Steps:
      1. Sort by (farm_id, timestamp) to guarantee chronological order.
      2. Create previous_biomass by shifting current_biomass backward per farm.
      3. Verify that future_biomass_kg_m2 already equals the next-step target.
         (The raw dataset pre-computes this; we just rename for clarity.)
      4. Extract day_of_year from the timestamp.
      5. Drop rows where previous_biomass is NaN (first row per farm).
      6. Drop the last row per farm (no future_biomass target).

    Returns
    -------
    pd.DataFrame with columns: FEATURE_COLUMNS + [TARGET_COLUMN] + extras
    """
    df = df.copy()

    # 1. Ensure chronological sort per farm
    df = df.sort_values([ID_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)

    # 2. Create previous_biomass (lag-1 per farm)
    df["previous_biomass_kg_m2"] = (
        df.groupby(ID_COLUMN)[CURRENT_BIOMASS_COLUMN].shift(1)
    )

    # 3. The raw dataset already has future_biomass_kg_m2 as the next-step
    #    biomass.  Verify that it is consistent: future_biomass at row i
    #    should equal current_biomass at row i+1 for the same farm.
    _verify_target_consistency(df)

    # 4. Extract time feature: day_of_year
    df["day_of_year"] = df[TIMESTAMP_COLUMN].dt.dayofyear.astype(float)

    # 5. Drop rows where previous_biomass is NaN (first observation per farm)
    before = len(df)
    df = df.dropna(subset=["previous_biomass_kg_m2"]).reset_index(drop=True)
    logger.info(
        "Dropped %d rows with NaN previous_biomass (first row per farm)",
        before - len(df),
    )

    # 6. Drop the last row per farm (no future target available)
    last_idx = df.groupby(ID_COLUMN).tail(1).index
    df = df.drop(index=last_idx).reset_index(drop=True)
    logger.info("Dropped %d last-row-per-farm rows (no target)", len(last_idx))

    # 7. Log final shape
    logger.info(
        "Feature engineering complete – %d rows, %d columns",
        len(df),
        len(df.columns),
    )
    return df


def _verify_target_consistency(df: pd.DataFrame) -> None:
    """Warn if future_biomass_kg_m2 does not match the next row's current_biomass."""
    # For each farm, check: future_biomass[i] should ≈ current_biomass[i+1]
    max_diffs = []
    for fid, grp in df.groupby(ID_COLUMN):
        grp = grp.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
        if len(grp) < 2:
            continue
        diffs = (
            grp[CURRENT_BIOMASS_COLUMN].iloc[1:].values
            - grp[TARGET_COLUMN].iloc[:-1].values
        )
        max_diffs.append(np.max(np.abs(diffs)))

    if max_diffs:
        worst = max(max_diffs)
        if worst > 0.01:
            logger.warning(
                "Target consistency check: max |current[i+1] - future[i]| = %.6f "
                "(expected ≈ 0). The pre-computed target may have rounding drift.",
                worst,
            )
        else:
            logger.info("Target consistency check passed (max drift = %.6f)", worst)


# ---------------------------------------------------------------------------
# Helpers for splitting
# ---------------------------------------------------------------------------

def get_feature_target_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) using the canonical feature and target columns."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def chronological_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split each farm's data chronologically into train / test.

    Returns the concatenated train set and test set across all farms.

    Rationale:
      Random splitting would allow the model to "peek" into future data
      during training, causing data leakage and overly optimistic metrics.
      Chronological splitting ensures the model only learns from the past
      and is evaluated on future observations it has never seen.
    """
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for fid, grp in df.groupby(ID_COLUMN):
        grp = grp.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
        split_idx = int(len(grp) * (1 - test_size))
        train_parts.append(grp.iloc[:split_idx])
        test_parts.append(grp.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    logger.info(
        "Chronological split: train=%d rows, test=%d rows (%.0f%% / %.0f%%)",
        len(train_df),
        len(test_df),
        (1 - test_size) * 100,
        test_size * 100,
    )
    return train_df, test_df
