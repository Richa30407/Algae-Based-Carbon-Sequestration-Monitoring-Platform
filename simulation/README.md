# 🧪 Simulink Simulation Module

MATLAB/Simulink model for the Algae-Based Carbon Sequestration Monitoring Platform.

> **Note:** The actual `.slx` Simulink model file is developed separately in MATLAB. These images document the model architecture and subsystems.

## Model Architecture

The Simulink model consists of the following subsystems:

| # | File | Description |
|---|------|-------------|
| 1 | `01_simulink_main_model.jpeg` | Complete Simulink model overview — full pipeline from IoT Sensors → Dashboard |
| 2 | `02_iot_sensor_subsystem.jpeg` | IoT Sensor subsystem — Temperature, Humidity, pH, Light, Turbidity, Water Level inputs |
| 3 | `03_data_processing_subsystem.jpeg` | Data Processing subsystem — Sensor Data → Processed Data |
| 4 | `04_ai_ml_engine_subsystem.jpeg` | AI/ML Engine subsystem — Processed Data + Remote Health → Algae Health score |
| 5 | `05_health_assessment_matlab_code.jpeg` | MATLAB function for health assessment (weighted scoring of sensor parameters) |
| 6 | `06_carbon_sequestration_subsystem.jpeg` | Carbon Sequestration subsystem — Algae Health → CO₂ Captured |
| 7 | `07_co2_estimation_matlab_code.jpeg` | MATLAB function for CO₂ estimation (`co2 = biomass * 1.83`) |
| 8 | `08_decision_control_subsystem.jpeg` | Decision Control subsystem — Algae Health → Decision output |
| 9 | `09_decision_control_matlab_code.jpeg` | MATLAB function for decision logic (health < 40 → alert) |
| 10 | `10_simulink_dashboard.jpeg` | Simulink Dashboard — CO₂ captured display, Health gauge, Decision indicator |

## Data Flow

```
IoT Sensors → Data Processing → AI/ML Engine → Carbon Sequestration → Decision Control → Dashboard
                                     ↑
                              Remote Sensing
```

## Key MATLAB Functions

### Health Assessment
- Combines 6 sensor scores with weighted average
- Weights: Temp (0.20), Humidity (0.20), pH (0.20), Light (0.15), Turbidity (0.15), Water (0.10)
- Final health = 0.8 × IoT Health + 0.2 × Remote Sensing Health

### CO₂ Estimation
- `biomass = 0.5 × (health / 100)`
- `co2 = biomass × 1.83`

### Decision Control
- If health < 40 → Decision = 1 (alert/action needed)
- Else → Decision = 0 (system healthy)
