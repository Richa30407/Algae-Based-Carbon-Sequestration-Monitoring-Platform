"""Test the full prediction pipeline end-to-end."""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_biomass
from src.carbon import estimate_co2_uptake

# Simulate a real prediction with actual data from the test set
sample_input = {
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

# 1. Predict biomass
result = predict_biomass(sample_input)

print("=== BIOMASS PREDICTION ===")
print("Predicted biomass: {:.4f} kg/m2".format(result["predicted_biomass"]))
pi = result["prediction_interval"]
print("Prediction interval: {:.4f} - {:.4f} kg/m2".format(pi["lower"], pi["upper"]))
print("Model:", result["model_name"])

# 2. Growth rate and biomass gain
current_biomass = sample_input["previous_biomass_kg_m2"]
predicted_biomass = result["predicted_biomass"]
biomass_gain = predicted_biomass - current_biomass
growth_rate = (biomass_gain / current_biomass) * 100 if current_biomass != 0 else 0.0

print("\n=== GROWTH RATE ===")
print("Current biomass: {:.4f} kg/m2".format(current_biomass))
print("Predicted biomass: {:.4f} kg/m2".format(predicted_biomass))
print("Biomass gain: {:.4f} kg/m2".format(biomass_gain))
print("Growth rate: {:.2f}%".format(growth_rate))

# 3. CO2 estimation
co2_result = estimate_co2_uptake(
    current_biomass_kg_m2=current_biomass,
    predicted_biomass_kg_m2=predicted_biomass,
    farm_area_m2=1000.0,
)

print("\n=== CO2 ESTIMATION ===")
for k, v in co2_result.items():
    print("  {}: {}".format(k, v))

# 4. Full output for teammates
full_output = {
    "current_biomass": current_biomass,
    "predicted_biomass": predicted_biomass,
    "biomass_gain": biomass_gain,
    "growth_rate_percent": growth_rate,
    "estimated_co2_uptake": co2_result["estimated_co2_uptake_kg_m2"],
    "prediction_interval": result["prediction_interval"],
    "model_name": result["model_name"],
    "model_metrics": {
        "mae": 0.031550,
        "rmse": 0.039881,
        "r2": 0.997631,
    },
}

print("\n=== FULL TEAM OUTPUT ===")
print(json.dumps(full_output, indent=2))
