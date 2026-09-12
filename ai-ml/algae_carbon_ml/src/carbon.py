"""
Carbon / CO2 estimation for the Algae Carbon 360 ML pipeline.

Responsibilities:
  - Convert predicted biomass gain into estimated CO2 uptake
  - Use configurable biological parameters
  - Clearly label all results as estimates, not verified measurements

Scientific notes:
  - Algae biomass is mostly water (wet weight). The carbon fraction applies
    to DRY biomass.  If your data is in wet biomass, you need an additional
    dry-matter fraction (typically 0.10 - 0.20 for microalgae).
  - The formula below assumes the input biomass is already expressed as
    DRY biomass.  If it is wet biomass, set DRY_BIOMASS_FRACTION accordingly.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from src.config import CARBON_FRACTION, CO2_CONVERSION_FACTOR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable biological assumption
# ---------------------------------------------------------------------------
# Fraction of wet biomass that is dry matter.
# For microalgae, typical values: 0.10 – 0.20 (10–20% solids).
# Set to 1.0 if your biomass data is already in dry-weight units.
DRY_BIOMASS_FRACTION = 1.0  # Assumption: input is dry biomass (kg/m2)


def estimate_co2_uptake(
    current_biomass_kg_m2: float,
    predicted_biomass_kg_m2: float,
    carbon_fraction: float = CARBON_FRACTION,
    co2_conversion: float = CO2_CONVERSION_FACTOR,
    dry_biomass_fraction: float = DRY_BIOMASS_FRACTION,
    farm_area_m2: Optional[float] = None,
) -> Dict[str, float]:
    """Estimate CO2 uptake from predicted biomass gain.

    Parameters
    ----------
    current_biomass_kg_m2 : current (measured) biomass in kg/m2
    predicted_biomass_kg_m2 : predicted (future) biomass in kg/m2
    carbon_fraction : fraction of dry biomass that is carbon (configurable)
    co2_conversion : molar mass ratio CO2/C = 44/12
    dry_biomass_fraction : fraction of wet biomass that is dry matter
    farm_area_m2 : optional total farm area in m2 (for total estimates)

    Returns
    -------
    dict with:
      - biomass_gain_kg_m2 : float
      - carbon_mass_kg_m2 : float
      - estimated_co2_uptake_kg_m2 : float (per m2)
      - total_biomass_gain_kg : float (if farm_area provided)
      - total_carbon_mass_kg : float (if farm_area provided)
      - total_estimated_co2_uptake_kg : float (if farm_area provided)
      - units : str (describing units used)

    Important disclaimers:
      - This is an ESTIMATE based on configurable conversion assumptions.
      - It does NOT represent verified permanent CO2 removal.
      - Actual CO2 uptake depends on many biological and environmental factors.
    """
    # Biomass gain (per m2)
    biomass_gain = predicted_biomass_kg_m2 - current_biomass_kg_m2

    # Convert wet to dry biomass if needed
    dry_gain = biomass_gain * dry_biomass_fraction

    # Carbon mass
    carbon_mass = dry_gain * carbon_fraction

    # CO2 equivalent
    co2_uptake = carbon_mass * co2_conversion

    result: Dict[str, float] = {
        "biomass_gain_kg_m2": round(biomass_gain, 6),
        "carbon_mass_kg_m2": round(carbon_mass, 6),
        "estimated_co2_uptake_kg_m2": round(co2_uptake, 6),
        "units": "kg/m2",
        "carbon_fraction_assumed": carbon_fraction,
        "dry_biomass_fraction_assumed": dry_biomass_fraction,
    }

    # Optional: scale to total farm area
    if farm_area_m2 is not None and farm_area_m2 > 0:
        result["total_biomass_gain_kg"] = round(biomass_gain * farm_area_m2, 4)
        result["total_carbon_mass_kg"] = round(carbon_mass * farm_area_m2, 4)
        result["total_estimated_co2_uptake_kg"] = round(co2_uptake * farm_area_m2, 4)
        result["farm_area_m2"] = farm_area_m2
        result["units"] = f"kg/m2 (per m2) and kg (total for {farm_area_m2} m2)"

    logger.info(
        "CO2 estimation: gain=%.4f kg/m2, carbon=%.4f kg/m2, CO2=%.4f kg/m2",
        biomass_gain, carbon_mass, co2_uptake,
    )
    return result
