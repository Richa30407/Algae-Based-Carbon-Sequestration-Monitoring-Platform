"""
Model evaluation for the Algae Carbon 360 ML pipeline.

Responsibilities:
  - Evaluate each trained model on the test set
  - Compute MAE, RMSE, R²
  - Produce a comparison DataFrame
  - Generate actual-vs-predicted plots
  - Generate feature importance plots (for tree-based models)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    FEATURE_COLUMNS,
    METRICS_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    METRICS_FILE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Compute MAE, RMSE, R² for a single model on the test set.

    Metrics explained:
      MAE  — Mean Absolute Error: average magnitude of prediction errors.
      RMSE — Root Mean Squared Error: penalises large errors more heavily.
      R²   — Coefficient of determination: proportion of variance explained.
             1.0 = perfect, 0.0 = predicts the mean, < 0 = worse than mean.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {"mae": mae, "rmse": rmse, "r2": r2}


def compare_models(
    fitted_models: Dict[str, any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate all models and return a comparison DataFrame.

    Columns: Model, MAE, RMSE, R²
    Sorted by RMSE (ascending = best first).
    """
    rows = []
    for name, model in fitted_models.items():
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({"Model": name, **metrics})

    comparison = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    comparison.columns = ["Model", "MAE", "RMSE", "R²"]
    comparison[["MAE", "RMSE"]] = comparison[["MAE", "RMSE"]].round(6)
    comparison["R²"] = comparison["R²"].round(6)

    logger.info("Model comparison:\n%s", comparison.to_string(index=False))
    return comparison


# ---------------------------------------------------------------------------
# Best model selection
# ---------------------------------------------------------------------------

def select_best_model(
    comparison: pd.DataFrame,
    fitted_models: Dict[str, any],
) -> Tuple[str, any]:
    """Select the best model based on lowest RMSE.

    Returns (model_name, fitted_pipeline).
    """
    best_row = comparison.iloc[0]  # already sorted by RMSE ascending
    best_name = best_row["Model"]
    best_model = fitted_models[best_name]
    logger.info(
        "Best model: %s (RMSE=%.6f, MAE=%.6f, R²=%.6f)",
        best_name, best_row["RMSE"], best_row["MAE"], best_row["R²"],
    )
    return best_name, best_model


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    save_path: Path | None = None,
) -> None:
    """Plot actual vs predicted biomass on the test set."""
    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.4, s=15)

    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect prediction')

    ax.set_xlabel('Actual Biomass (kg/m2)', fontsize=12)
    ax.set_ylabel('Predicted Biomass (kg/m2)', fontsize=12)
    ax.set_title(f'Actual vs Predicted — {model_name}', fontsize=14)
    ax.legend()
    plt.tight_layout()

    path = save_path or (PLOTS_DIR / f'actual_vs_predicted_{model_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved actual-vs-predicted plot: %s", path)


def plot_feature_importance(
    model,
    feature_names: list,
    model_name: str,
    save_path: Path | None = None,
) -> pd.DataFrame | None:
    """Plot feature importance for tree-based models.

    Returns the importance DataFrame, or None for non-tree models.

    Note: Feature importance indicates how useful the feature was for the
    model's predictions. It does NOT prove biological causation.
    """
    # Extract the tree-based model from the pipeline
    regressor = model.named_steps.get("model", None)
    if regressor is None or not hasattr(regressor, "feature_importances_"):
        logger.info("Model %s does not expose feature_importances_ — skipping", model_name)
        return None

    importances = regressor.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp_df["Feature"], imp_df["Importance"], color='steelblue')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Feature Importance — {model_name}', fontsize=14)
    plt.tight_layout()

    path = save_path or (PLOTS_DIR / f'feature_importance_{model_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved feature importance plot: %s", path)

    return imp_df


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_comparison(comparison: pd.DataFrame) -> None:
    """Save the comparison table to CSV."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(METRICS_FILE, index=False)
    logger.info("Saved model comparison to %s", METRICS_FILE)
