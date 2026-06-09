# Layered Dielectrics

Transfer matrix based layered dielectric propagation model for reflection coefficient and APD (Absorbed Power Density) calculation.

## Overview

This project implements electromagnetic wave propagation through layered dielectric structures using the transfer matrix method. It's designed for applications including:

- Reflection coefficient calculation for multi-layer materials
- Absorbed Power Density (APD) analysis
- Skin phantom modeling for millimeter-wave applications
- Frequency-dependent material property analysis

## Features

- **Transfer Matrix Method**: Efficient computation for arbitrary number of layers
- **Cole-Cole Dispersion Model**: Support for frequency-dependent material properties
- **TE/TM Polarization**: Separate analysis for both polarization modes
- **Material Libraries**: Pre-defined models for skin, silicone, foam, epoxy, and shell materials
- **Phantom Configurations**: Multiple phantom definitions across frequency bands (10-110 GHz)

## Installation

```bash
# Clone the repository
git clone https://github.com/ninad-sc/LayeredDielectrics.git
cd LayeredDielectrics

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Example: Calculate Reflection Coefficient

```python
import numpy as np
from src.dielectrics import helpers, phantoms

# Define frequency range
freq = np.linspace(24e9, 30e9, 100)  # 24-30 GHz
omega = 2 * np.pi * freq

# Create phantom model
pha = phantoms.PHA24_30G_V2()

# Initialize material
phantom = helpers.Phantom_4layer(freq, *phantom_params)
phantom.calc_k0(omega)

# Calculate reflection coefficient
R, T = helpers.get_R_T(transfer_matrix)
```

### Sensitivity Analysis

Run the sensitivity analysis script:

```bash
python scripts/sensitivity_analysis.py
```

### Broadband Analysis

Run the broadband phantom analysis:

```bash
python scripts/broadband_analysis.py
```

## Project Structure

```
LayeredDielectrics/
├── src/
│   └── dielectrics/
│       ├── __init__.py
│       ├── helpers.py           # Core transfer matrix functions
│       ├── phantoms.py          # Phantom and material definitions
│       └── models.py            # Material model classes
├── scripts/
│   ├── broadband_analysis.py    # Broadband phantom analysis
│   └── sensitivity_analysis.py  # Sensitivity/uncertainty analysis
├── tests/                       # Unit tests
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project metadata
├── README.md                    # This file
└── LICENSE                      # MIT License
```

## Configuration

Edit `config.py` to customize:

- Output directories for plots and tables
- Matplotlib backend and figure settings
- Frequency ranges and resolution
- Material parameters

```python
# config.py
OUTPUT_PLOTS_DIR = 'output/plots'
OUTPUT_TABLES_DIR = 'output/tables'
MATPLOTLIB_BACKEND = 'Agg'  # Change for interactive display
```

## Physics Background

### Transfer Matrix Method

For a layered structure, the overall transfer matrix is computed as:

```
T_total = T_n @ T_(n-1) @ ... @ T_1 @ T_0
```

where each T_i represents the transmission through interface i and propagation in layer i.

### Cole-Cole Dispersion Model

Frequency-dependent permittivity:

```
ε(ω) = ε∞ + Σ[Δε_i / (1 + (jωτ_i)^(1-α_i))] - j·σ_0/ω
```

## References

- Transfer matrix method for electromagnetic wave propagation
- Cole-Cole relaxation model for dielectric materials
- Skin phantom models for millimeter-wave applications

## Contributing

Contributions are welcome! Please:

1. Create a feature branch from `develop`
2. Make your changes with descriptive commits
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Author

Chitnis - Created January 2025

## Citation

If you use this code in your research, please cite:

```bibtex
@software{layered_dielectrics_2025,
  author = {Chitnis},
  title = {Layered Dielectrics: Transfer Matrix Based Propagation Model},
  year = {2025},
  url = {https://github.com/ninad-sc/LayeredDielectrics}
}
```
