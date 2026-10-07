# Layered Dielectrics

LayeredDielectrics calculates reflection, transmission, and absorbed power density (APD) for layered dielectric structures using transfer matrices. It supports TE and TM polarization, frequency-dependent Cole-Cole material models, skin models, and phantom stacks.

The model is described in Chitnis, N., Karimi, F., Kühn, S., Fallahi, A., Christ, A. and Kuster, N. (2025), *Traceable Assessment of the Absorbed Power Density of Body Mounted Devices at Frequencies Above 10 GHz*, Bioelectromagnetics, 46: e70018. [DOI: 10.1002/bem.70018](https://doi.org/10.1002/bem.70018).

## Installation

Run the scripts from the repository root. Modules and scripts are stored directly in this directory.

```sh
git clone https://github.com/ninad-sc/LayeredDielectrics.git
cd LayeredDielectrics
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```sh
# Linux/macOS
source .venv/bin/activate
```

Install the numerical and file-export dependencies:

```sh
python -m pip install -r requirements.txt
```

The current configuration and several plotting scripts select `Qt5Agg`. Interactive plotting requires a Qt binding, such as PyQt5, which is not included in `requirements.txt`:

```sh
python -m pip install PyQt5
```

For headless use, select `Agg` in `config.py` and in scripts that explicitly call `matplotlib.use('Qt5Agg')`, before importing `matplotlib.pyplot`. Some scripts set their backend independently of `config.py`.

## APD decay through all layers

```sh
python plot_apd_vs_z.py
```

This produces one figure per frequency at **10, 15, 20, 30, and 45 GHz**. Each figure combines the selected phantom profiles, a skin profile, and a material-layer diagram over z = 0–10 mm.

| Frequency/GHz | Phantom classes plotted |
| --- | --- |
| 10 | `PHA10_18G` |
| 15 | `PHA10_18G` |
| 20 | `PHA18_24G` |
| 30 | `PHA24_30G_V2`, `PHA_TH30G` |
| 45 | `PHA30_45G` |

The script uses normal incidence and the `Skin_2std` reference model, labelled “Skin” in the figures. Skin curves are solid black. Both epoxy layers use the same red colour in the layer diagrams. The foam and final SSL layers of `PHA_TH30G` are air in its class definition; both are labelled “Air” and use the same grey as skin's air layer.

Material properties and thicknesses come from the classes without overrides. Array-valued properties are sampled on the class frequency grid; scalar properties remain fixed. The current `PHA_TH30G` shell has relative permittivity 7.75961, conductivity 5.24061 S/m, and thickness 1 mm. Change the class definition if different shell properties are required.

The depth coordinate starts at the entrance to class layer 1. For phantoms, this is the first epoxy layer, including zero-thickness layers in the stack definition. For skin, the fixed 2 mm air spacer precedes the SC and dermis. The final layer is treated as semi-infinite.

The ordinate is **APD/IPD**, the net inward Poynting flux at depth z divided by incident power density. It represents power remaining at that plane, rather than cumulative absorption from 0 to z. Power absorbed between two planes is the difference between their fluxes. In a lossless final air layer, the remaining flux is transmitted power and does not decay further. The calculation checks interface power continuity, surface energy balance, TE/TM agreement at normal incidence, and passive decay.

Outputs:

- `output/plots/apd_vs_z/10_15_20_30_45GHz/apd_layers_<frequency>GHz.svg`
- Matching PDF and PNG files.
- `output/tables/apd_vs_z_10_15_20_30_45GHz.csv`, containing the frequency, depth, layer indices, and normalized phantom/skin fluxes.

## Broadband reflection and APD summaries

```sh
python run_phantom_summary.py --list
python run_phantom_summary.py PHA24_30G_V2
python run_phantom_summary.py --all
```

The runner discovers active `PHA*` classes from `phantoms.py`; without arguments, it runs all classes. It calls `broadband_phantom_summary.py` for each selection and uses a noninteractive backend. The summary evaluates phantom APD at the final layer's entrance and skin APD at the SC entrance, across each class frequency grid and normalized transverse wave numbers.

## Sensitivity analysis

```sh
python run_sensitivity.py --list
python run_sensitivity.py PHA10_18G
python run_sensitivity.py --all
```

Without arguments, the runner selects `PHA10_18G`. Supported classes are `PHA10_18G`, `PHA18_24G`, `PHA24_30G`, `PHA24_30G_V2`, and `PHA30_45G`. Results include uncertainty workbooks under `output/tables/`.

`standard_compliant_sensitivity.py` is an additional exporter for grouping uncertainty components into workbooks under `output/standard_compliant_tables/`. Its `STANDARD_COMPONENTS` mapping currently expects legacy names (`epsr_measurement`, `sigma_measurement`, and `composite_shell_thickness`). The current `sensitivity.py` exports separate SSL/solid measurement and lamination/foam-shell thickness contributors; reconcile that mapping before using the exporter. Thermal-property entries are left empty.

## Material-property and layer exports

```sh
python plot_ssl24_comparison.py
python calculate_ssl_penetration_depth.py --list
python calculate_ssl_penetration_depth.py --no-show
python export_phantom_layers.py
```

- `plot_ssl24_comparison.py` compares the SSL24 and SSL24 V2 permittivity/conductivity models over 24–30 GHz and plots signed percentage deviations. It saves SVG, PDF, and PNG files.
- `calculate_ssl_penetration_depth.py` plots the 1/e electric-field amplitude penetration depth for SSL-backed phantom classes. It accepts class names and an `--output` path. The current default output is `output/plots/ssl_penetration_depth.pdf`; `--no-show` suppresses `plt.show()` but retains the script's Qt backend selection.
- `export_phantom_layers.py` exports five-layer material properties as CSV. Edit its `PHANTOM`, `FREQUENCY_GHZ`, and output settings before running. It preserves zero-thickness layers and rejects frequencies outside the stored class range. Its SSL thickness label is a display setting, not a finite layer boundary in the propagation model.

## Basic reflection calculation

This standalone example evaluates the full `PHA24_30G_V2` stack at normal incidence using its stored frequency grid:

```python
import config
config.MATPLOTLIB_BACKEND = 'Agg'

import numpy as np
from scipy.constants import epsilon_0
import helpers
import phantoms

pha = phantoms.PHA24_30G_V2()
frequency = pha.freq
omega = 2 * np.pi * frequency
material = helpers.Phantom_4layer(
    frequency,
    pha.epsr_epoxy, pha.epsr_foam, pha.epsr_epoxy,
    pha.epsr_shell, pha.epsr_ssl,
    pha.sigma_epoxy, pha.sigma_foam, pha.sigma_epoxy,
    pha.sigma_shell, pha.sigma_ssl,
    pha.epoxy_thickness, pha.foam_thickness,
    pha.epoxy_thickness, pha.shell_thickness,
)
kxn = np.array([0.0])
k0_air = helpers.get_k0(omega, 1.0, 0.0)
kx = k0_air[:, None] * kxn[None, :]
kz_air = helpers.get_kz(k0_air, kx)
eps_air = np.full(frequency.shape, epsilon_0)
material.calc_k0(omega)
material.calc_kz(kxn, k0_air)
transfer_te, transfer_tm = helpers.get_overall_T_4layer(
    kxn, kz_air, eps_air, material)
reflection_te, transmission_te = helpers.get_R_T(transfer_te)
print(frequency / 1e9, np.abs(reflection_te[:, 0]))
```

The transfer matrices propagate forward/backward field coefficients through interfaces and finite layers. `get_R_T` returns field coefficients, not power ratios. The APD helpers use the same field-amplitude convention for incident and transmitted flux, so normalization cancels the common scale factor.

## Repository files and configuration

| File | Purpose |
| --- | --- |
| `helpers.py` | Transfer matrices, field/APD calculations, skin and material models |
| `phantoms.py` | SSL dispersion models and phantom layer parameters |
| `config.py` | Shared output paths, backend, figure settings, and defaults |
| `broadband_phantom_summary.py`, `run_phantom_summary.py` | Broadband analysis and class runner |
| `sensitivity.py`, `run_sensitivity.py` | Sensitivity calculations and class runner |
| `standard_compliant_sensitivity.py` | Grouped uncertainty workbook exporter |
| `plot_apd_vs_z.py` | Selected-frequency APD decay and layer diagrams |
| `plot_ssl24_comparison.py` | SSL24 material comparison |
| `calculate_ssl_penetration_depth.py` | SSL penetration-depth plots |
| `export_phantom_layers.py` | Layer-property CSV export |

Shared settings live in `config.py`; scripts may also define their own frequencies, backend, styling, or output paths. Phantom frequency grids and layer properties live in `phantoms.py`, and skin dispersion parameters live in `helpers.py`.

Generated files under `output/` are ignored by Git. Run the scripts locally to regenerate plots and tables.

## Contributing and license

Create a branch from `master`, include checks appropriate to the change, and submit a pull request. The project is licensed under the MIT License; see [LICENSE](LICENSE).

## Citation

```bibtex
@software{layered_dielectrics_2025,
  author = {Chitnis},
  title = {Layered Dielectrics: Transfer Matrix Based Propagation Model},
  year = {2025},
  url = {https://github.com/ninad-sc/LayeredDielectrics}
}
```
