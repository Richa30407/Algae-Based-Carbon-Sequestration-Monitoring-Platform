"""
Prediction module for the Algae Carbon 360 ML pipeline.

Responsibilities:
  - Load the saved trained pipeline
  - Accept new farm conditions as a dict or DataFrame
  - Apply the same preprocessing used during training
  - Return predicted biomass with uncertainty estimate
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BEST_MODEL_FILE,
    FEATURE_COLUMNS,
    METRICS_DIR,
)

logger = logging.getLogger(__name__)


def load_model(path: Optional[Path] = None) -> Any:
    """Load the saved sklearn Pipeline from disk.

    Parameters
    ----------
    path : path to the .pkl file (defaults to config.BEST_MODEL_FILE)

    Returns
    -------
    Fitted sklearn Pipeline
    """
    model_path = path or BEST_MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    logger.info("Loaded model from %s", model_path)
    return model


def predict_biomass(
    input_data: Union[Dict[str, Any], pd.DataFrame],
    model: Optional[Any] = None,
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Predict future biomass from the latest farm conditions.

    Parameters
    ----------
    input_data : dict or DataFrame with columns matching FEATURE_COLUMNS
    model : optional pre-loaded model (avoids re-loading from disk)
    model_path : optional path override for model loading

    Returns
    -------
    dict with:
      - predicted_biomass : float (kg/m2)
      - prediction_interval : dict with 'lower' and 'upper' bounds
      - model_name : str
    """
    # Ensure we have a model
    if model is None:
        model = load_model(model_path)

    # Convert dict to DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # Validate required features are present
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Select only the feature columns in the correct order
    X = df[FEATURE_COLUMNS]

    # Predict
    y_pred = model.predict(X)

    # Extract model name from pipeline
    model_name = type(model.named_steps.get("model", model)).__name__

    return {
        "predicted_biomass": float(y_pred[0]),
        "prediction_interval": _compute_prediction_interval(y_pred, model),
        "model_name": model_name,
    }


def _compute_prediction_interval(
    y_pred: np.ndarray,
    model: Any,
) -> Dict[str, float]:
    """Compute a prediction interval using model residual error.

    Method:
      We use the test-set RMSE as an estimate of typical prediction error.
      The 95% prediction interval is approximated as:
        predicted ± 1.96 * RMSE

      This is a standard residual-based approach that assumes errors are
      approximately normally distributed. It is an approximation, not an
      exact statistical guarantee.

    For a single prediction, the interval is:
      lower = predicted - 1.96 * rmse
      upper = predicted + 1.96 * rmse
    """
    # Load the saved RMSE from training evaluation
    rmse = _load_training_rmse()
    if rmse is None:
        # Fallback: use 5% of predicted value as uncertainty estimate
        rmse = abs(float(y_pred[0])) * 0.05

    margin = 1.96 * rmse
    pred = float(y_pred[0])

    return {
        "lower": round(pred - margin, 6),
        "upper": round(pred + margin, 6),
    }


def _load_training_rmse() -> Optional[float]:
    """Load the RMSE from the best model's evaluation metrics."""
    metrics_file = METRICS_DIR / "model_comparison.csv"
    if not metrics_file.exists():
        return None
    try:
        df = pd.read_csv(metrics_file)
        # Get the best model's RMSE (first row after sorting)
        return float(df.sort_values("RMSE").iloc[0]["RMSE"])
    except Exception:
        return None
