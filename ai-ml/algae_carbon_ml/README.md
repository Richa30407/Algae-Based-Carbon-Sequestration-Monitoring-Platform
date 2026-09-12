# Algae Carbon 360 — ML Module

AI/ML pipeline for predicting algae biomass growth and estimating CO2 uptake.

## Setup

### Python version
Python 3.13+

### Install dependencies
```bash
pip install -r requirements.txt
```

### Virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Project structure

```
algae_carbon_ml/
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned data (generated)
├── models/                   # Saved trained models
├── notebooks/
│   ├── eda_and_experiments.ipynb
│   ├── run_eda.py
│   └── test_pipeline.py
├── src/
│   ├── __init__.py
│   ├── config.py             # All constants and hyperparameters
│   ├── preprocessing.py      # Data validation and cleaning
│   ├── feature_engineering.py
│   ├── train.py              # Model training
│   ├── evaluate.py           # Metrics and comparison
│   ├── predict.py            # Prediction from saved model
│   ├── carbon.py             # CO2 uptake estimation
│   └── utils.py              # Logging setup
├── outputs/
│   ├── plots/                # EDA and evaluation plots
│   ├── metrics/              # model_comparison.csv
│   └── predictions/          # Saved test predictions
├── requirements.txt
└── README.md
```

## Usage

### 1. Run EDA (optional)
```bash
cd algae_carbon_ml
python notebooks/run_eda.py
```
Plots saved to `outputs/plots/`.

### 2. Train models
```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.utils import setup_logging; setup_logging()
from src.preprocessing import load_and_validate
from src.feature_engineering import create_features, get_feature_target_split, chronological_train_test_split
from src.train import train_models
from src.evaluate import compare_models, select_best_model, save_comparison
from src.config import BEST_MODEL_FILE
import joblib

df_raw = load_and_validate()
df = create_features(df_raw)
train_df, test_df = chronological_train_test_split(df)
X_train, y_train = get_feature_target_split(train_df)
X_test, y_test = get_feature_target_split(test_df)
fitted = train_models(X_train, y_train)
comparison = compare_models(fitted, X_test, y_test)
best_name, best_model = select_best_model(comparison, fitted)
save_comparison(comparison)
joblib.dump(best_model, BEST_MODEL_FILE)
print('Model saved to', BEST_MODEL_FILE)
"
```

### 3. Run prediction
```bash
python notebooks/test_pipeline.py
```

### 4. Use from Python code
```python
import sys; sys.path.insert(0, '.')
from src.predict import predict_biomass
from src.carbon import estimate_co2_uptake

input_data = {
    "temperature_c": 29.5,
    "sunlight_hours": 8.0,
    "color_intensity": 245.0,
    "ph": 7.5,
    "nutrients_mg_l": 75.0,
    "co2_ppm": 420.0,
    "dissolved_oxygen_mg_l": 10.5,
    "water_quality_index": 87.0,
    "previous_biomass_kg_m2": 5.5,
}

result = predict_biomass(input_data)
print(result)
# {
#   "predicted_biomass": 5.5447,
#   "prediction_interval": {"lower": 5.4666, "upper": 5.6229},
#   "model_name": "LinearRegression"
# }

co2 = estimate_co2_uptake(
    current_biomass_kg_m2=5.5,
    predicted_biomass_kg_m2=result["predicted_biomass"],
    farm_area_m2=1000.0,
)
print(co2["estimated_co2_uptake_kg_m2"])
```

## Model performance

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **LinearRegression** | 0.0316 | 0.0399 | 0.9976 |
| RandomForestRegressor | 0.0588 | 0.0873 | 0.9887 |
| GradientBoostingRegressor | 0.0614 | 0.0941 | 0.9868 |

## Teammate integration

The prediction API returns a Python dictionary:

```python
{
    "current_biomass": 5.5,
    "predicted_biomass": 5.5447,
    "biomass_gain": 0.0447,
    "growth_rate_percent": 0.81,
    "estimated_co2_uptake": 0.0820,
    "prediction_interval": {"lower": 5.4666, "upper": 5.6229},
    "model_name": "LinearRegression",
    "model_metrics": {"mae": 0.0316, "rmse": 0.0399, "r2": 0.9976}
}
```

## Scientific limitations

- This system is a **prediction prototype**, not a verified CO2 removal measurement.
- CO2 uptake is **estimated** from predicted biomass increase using configurable conversion assumptions.
- All biological parameters (carbon fraction, dry biomass fraction) are configurable in `config.py` and `carbon.py`.
- Prediction intervals are based on training-set RMSE (residual-based approximation).

## Configuration

All constants live in `src/config.py`:
- `FEATURE_COLUMNS` — model input features
- `TARGET_COLUMN` — what we predict
- `MODEL_PARAMETERS` — hyperparameters for each model
- `CARBON_FRACTION` — fraction of dry biomass that is carbon (default: 0.50)
- `CO2_CONVERSION_FACTOR` — molar mass ratio CO2/C (44/12)
- `TEST_SIZE` — chronological test split fraction (default: 0.20)
