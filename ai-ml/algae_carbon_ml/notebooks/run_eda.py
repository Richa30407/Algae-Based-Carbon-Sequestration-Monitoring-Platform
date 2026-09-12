"""Standalone EDA script — generates all plots and saves them to outputs/plots/."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import load_and_validate
from src.feature_engineering import create_features
from src.config import PLOTS_DIR

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')

df_raw = load_and_validate()
df = create_features(df_raw)
print(f'Dataset: {df.shape[0]} rows, {df.shape[1]} columns')

# ---- 1. Correlation heatmap ----
feature_cols = [
    'temperature_c', 'sunlight_hours', 'color_intensity', 'ph',
    'nutrients_mg_l', 'co2_ppm', 'dissolved_oxygen_mg_l',
    'water_quality_index', 'previous_biomass_kg_m2', 'future_biomass_kg_m2'
]
corr = df[feature_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'correlation_heatmap.png', dpi=150)
plt.close()
print('Saved: correlation_heatmap.png')

# ---- 2. Feature vs biomass scatter plots ----
env_features = [
    'temperature_c', 'sunlight_hours', 'color_intensity', 'ph',
    'nutrients_mg_l', 'co2_ppm', 'dissolved_oxygen_mg_l', 'water_quality_index'
]
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, col in enumerate(env_features):
    ax = axes[i // 4, i % 4]
    ax.scatter(df[col], df['future_biomass_kg_m2'], alpha=0.3, s=10)
    ax.set_xlabel(col)
    ax.set_ylabel('future_biomass (kg/m2)')
    ax.set_title(f'{col} vs biomass')
fig.suptitle('Environmental Features vs Future Biomass', fontsize=16)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'feature_vs_biomass.png', dpi=150)
plt.close()
print('Saved: feature_vs_biomass.png')

# ---- 3. Biomass over time ----
fig, ax = plt.subplots(figsize=(14, 5))
for fid, grp in df.groupby('farm_id'):
    grp = grp.sort_values('timestamp')
    ax.plot(grp['timestamp'], grp['current_biomass_kg_m2'], alpha=0.7, label=f'Farm {fid}')
ax.set_xlabel('Date')
ax.set_ylabel('Biomass (kg/m2)')
ax.set_title('Algae Biomass Over Time by Farm')
ax.legend(ncol=5, fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'biomass_over_time.png', dpi=150)
plt.close()
print('Saved: biomass_over_time.png')

# ---- 4. Variable distributions ----
dist_cols = ['temperature_c', 'ph', 'nutrients_mg_l', 'co2_ppm',
             'current_biomass_kg_m2', 'previous_biomass_kg_m2']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, col in enumerate(dist_cols):
    ax = axes[i // 3, i % 3]
    df[col].hist(bins=40, ax=ax, edgecolor='white')
    ax.set_title(f'Distribution: {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'variable_distributions.png', dpi=150)
plt.close()
print('Saved: variable_distributions.png')

# ---- 5. Previous vs future biomass ----
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(df['previous_biomass_kg_m2'], df['future_biomass_kg_m2'], alpha=0.3, s=10)
lo = min(df['previous_biomass_kg_m2'].min(), df['future_biomass_kg_m2'].min())
hi = max(df['previous_biomass_kg_m2'].max(), df['future_biomass_kg_m2'].max())
ax.plot([lo, hi], [lo, hi], 'r--', label='Perfect prediction')
ax.set_xlabel('Previous Biomass (kg/m2)')
ax.set_ylabel('Future Biomass (kg/m2)')
ax.set_title('Previous Biomass vs Future Biomass')
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'prev_vs_future_biomass.png', dpi=150)
plt.close()
print('Saved: prev_vs_future_biomass.png')

print('\nAll EDA plots saved to outputs/plots/')
