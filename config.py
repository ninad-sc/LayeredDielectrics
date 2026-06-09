"""
Centralized configuration for LayeredDielectrics.

Edit this file to customize paths, plotting options, and simulation parameters.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_TABLES_DIR = OUTPUT_DIR / "tables"

# Create output directories if they don't exist
OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MATPLOTLIB SETTINGS
# ============================================================================

# Backend: 'Agg' for non-interactive, 'Qt5Agg' for interactive display
MATPLOTLIB_BACKEND = "Agg"

# Figure settings
FIGURE_DPI = 300
FIGURE_SIZE_DEFAULT = (16, 9)
FONT_SIZE = 24

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Default frequency resolution (Hz)
FREQ_RESOLUTION = 0.25e9  # 250 MHz

# Spatial angle sampling for APD calculations
KXN_LINSPACE = (0, 2, 200)  # (start, stop, num_points)
KXN_ARANGE = (0, 2.01, 0.01)  # (start, stop, step) - more detailed

# Temperature-related parameters
DELTA_T_DEFAULT = 2  # Maximum temperature change [°C]

# ============================================================================
# MATERIAL PARAMETERS
# ============================================================================

# Uncertainty/deviation percentages for sensitivity analysis
UNCERTAINTY_PERCENTAGES = {
    "epsr": 5.0,           # ε_r uncertainty [%]
    "sigma": 5.0,          # conductivity uncertainty [%]
    "thickness": 20.0,     # thickness uncertainty [%]
    "measurement": 3.0,    # measurement uncertainty [%]
    "ssl_epsr": 10.0,      # SSL material ε_r uncertainty [%]
    "ssl_sigma": 10.0,     # SSL material σ uncertainty [%]
    "ssl_measurement_epsr": 3.2,  # SSL measurement ε_r [%]
    "ssl_measurement_sigma": 5.2,  # SSL measurement σ [%]
}

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Plot file formats
PLOT_FORMATS = ["svg", "pdf", "png"]

# Plot transparency
PLOT_TRANSPARENT = True

# Table format
TABLE_FORMAT = "xlsx"  # or 'csv'

# ============================================================================
# LOGGING
# ============================================================================

VERBOSE = False  # Set to True for detailed console output
