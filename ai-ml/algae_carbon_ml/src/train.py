"""
Model training for the Algae Carbon 360 ML pipeline.

Responsibilities:
  - Build sklearn Pipelines for each candidate model
  - Train on chronological train set
  - Return fitted pipelines and metadata
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS, MODEL_PARAMETERS, RANDOM_STATE

logger = logging.getLogger(__name__)


def _build_preprocessor() -> ColumnTransformer:
    """Build a column transformer that handles remaining missing values.

    Tree-based models do not require feature scaling, so we keep it simple:
    - Median imputation for any residual NaN (safety net after preprocessing).
    - No scaling (tree models are scale-invariant).

    For Linear Regression we optionally add StandardScaler via a separate path.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, FEATURE_COLUMNS),
    ])


def _build_linear_preprocessor() -> ColumnTransformer:
    """Preprocessor for Linear Regression (includes scaling)."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, FEATURE_COLUMNS),
    ])


def build_models() -> Dict[str, Pipeline]:
    """Build sklearn Pipelines for all candidate models.

    Returns a dict mapping model name -> fitted-ready Pipeline.
    """
    models: Dict[str, Pipeline] = {}

    # --- Linear Regression (with scaling) ---
    lr_params = MODEL_PARAMETERS.get("LinearRegression", {})
    models["LinearRegression"] = Pipeline([
        ("preprocessor", _build_linear_preprocessor()),
        ("model", LinearRegression(**lr_params)),
    ])

    # --- Random Forest (no scaling needed) ---
    rf_params = MODEL_PARAMETERS.get("RandomForestRegressor", {})
    models["RandomForestRegressor"] = Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("model", RandomForestRegressor(**rf_params)),
    ])

    # --- Gradient Boosting (no scaling needed) ---
    gb_params = MODEL_PARAMETERS.get("GradientBoostingRegressor", {})
    models["GradientBoostingRegressor"] = Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("model", GradientBoostingRegressor(**gb_params)),
    ])

    logger.info("Built %d candidate models: %s", len(models), list(models.keys()))
    return models


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Pipeline]:
    """Train all candidate models and return fitted pipelines.

    Parameters
    ----------
    X_train : training features
    y_train : training target

    Returns
    -------
    Dict of model_name -> fitted Pipeline
    """
    models = build_models()
    fitted: Dict[str, Pipeline] = {}

    for name, pipeline in models.items():
        logger.info("Training %s ...", name)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        logger.info("  %s trained successfully", name)

    return fitted
