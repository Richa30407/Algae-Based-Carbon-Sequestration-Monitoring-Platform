"""
Central configuration for the Algae Carbon 360 ML pipeline.

All constants and hyperparameters live here so that source files
stay clean and changes are easy to track.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

RAW_CSV = RAW_DATA_DIR / "algae_synthetic_dataset.csv"

# ---------------------------------------------------------------------------
# Feature / target definitions
# ---------------------------------------------------------------------------
# Columns the model receives as input
FEATURE_COLUMNS = [
    "temperature_c",
    "sunlight_hours",
    "color_intensity",
    "ph",
    "nutrients_mg_l",
    "co2_ppm",
    "dissolved_oxygen_mg_l",
    "water_quality_index",
    "previous_biomass_kg_m2",
]

# The column we are trying to predict (next-step biomass)
TARGET_COLUMN = "future_biomass_kg_m2"

# The current biomass column (used for feature engineering)
CURRENT_BIOMASS_COLUMN = "current_biomass_kg_m2"

# Identifier and timestamp columns (not features)
ID_COLUMN = "farm_id"
TIMESTAMP_COLUMN = "timestamp"

# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2  # last 20 % of each farm's timeline
RANDOM_STATE = 42  # used only for models that accept a random seed

# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------
MODEL_PARAMETERS = {
    "LinearRegression": {},
    "RandomForestRegressor": {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "GradientBoostingRegressor": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
    },
}

# ---------------------------------------------------------------------------
# Carbon / CO2 estimation (configurable biological assumptions)
# ---------------------------------------------------------------------------
# Fraction of dry biomass that is carbon.
# Typical algae values range from 0.45 – 0.55 depending on species.
CARBON_FRACTION = 0.50  # configurable – clearly an assumption

# Molar masses:  CO2 = 44 g/mol,  C = 12 g/mol
CO2_CONVERSION_FACTOR = 44.0 / 12.0  # ≈ 3.667

# ---------------------------------------------------------------------------
# Prediction uncertainty
# ---------------------------------------------------------------------------
# Reliability score thresholds (training-data distribution check)
RELIABILITY_FULL_WEIGHT = 1.0
RELIABILITY_DECAY_FACTOR = 0.85  # multiplier for each std outside training range

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
MISSING_VALUE_STRATEGY = "interpolation"  # 'interpolation' | 'median'

# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------
BEST_MODEL_FILE = MODEL_DIR / "best_biomass_model.pkl"
METRICS_FILE = METRICS_DIR / "model_comparison.csv"
PREDICTIONS_FILE = PREDICTIONS_DIR / "test_predictions.csv"
