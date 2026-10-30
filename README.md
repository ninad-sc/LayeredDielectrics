# Layered Dielectrics

LayeredDielectrics calculates reflection, transmission, and absorbed power density (APD) for layered dielectric structures using transfer matrices. It supports TE and TM polarization, frequency-dependent Cole-Cole material models, skin models, and phantom stacks.

The model is described in Chitnis, N., Karimi, F., Kühn, S., Fallahi, A., Christ, A. and Kuster, N. (2025), *Traceable Assessment of the Absorbed Power Density of Body Mounted Devices at Frequencies Above 10 GHz*, Bioelectromagnetics, 46: e70018. [DOI: 10.1002/bem.70018](https://doi.org/10.1002/bem.70018).

## Installation

Install directly from GitHub:

```sh
python -m pip install git+https://github.com/ninad-sc/LayeredDielectrics.git
```

Alternatively, clone the repository and create a virtual environment:

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

Install the repository and its numerical and file-export dependencies:

```sh
python -m pip install .
```

For development, use `python -m pip install -e .` so changes to the source modules take effect immediately.

The installed package is imported as `layered_dielectrics`. Numerical imports
do not initialize Matplotlib or create files. Analysis commands work from any
directory and save results under `output/` in that directory.

```sh
layered-dielectrics-summary --list
layered-dielectrics-sensitivity --list
layered-dielectrics-summary PHA24_30G_V2
```

The equivalent module invocation is:

```sh
python -m layered_dielectrics.run_phantom_summary --list
```

Plotting defaults to the noninteractive `Agg` backend and saves SVG and PDF
figures. An application that has already initialized pyplot keeps its backend.
To open interactive windows, install `python -m pip install ".[gui]"` and set
`MPLBACKEND=Qt5Agg` before starting Python (PowerShell: `$env:MPLBACKEND = "Qt5Agg"`).

Set `LAYERED_DIELECTRICS_OUTPUT_DIR` before starting Python to change the output
root. In Python, use `from layered_dielectrics import config` followed by
`config.set_output_directory("my-results")` before running an analysis. Output
directories are created only when an operation saves a result. Plot settings are
scoped to each plotting operation and restored afterward: font size 24, figure
size 16 × 9, constrained layout, and red/blue/cyan/orange/purple curve colours.

Source-checkout launchers such as `python run_phantom_summary.py --list` remain
available after installing the package. Existing callers should replace
`import helpers`, `import phantoms`, or `import config` with
`from layered_dielectrics import helpers, phantoms, config`.

## APD decay through all layers

```sh
layered-dielectrics-apd
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

## Skin reflection versus frequency

```sh
python plot_skin_reflection.py
# Optionally override the incidence-angle variable for a single run:
python plot_skin_reflection.py --angles 0 30 60 80
```

Edit `ANGLES_DEG` in `src/layered_dielectrics/plot_skin_reflection.py` to set
the default incidence angles in degrees from the surface normal. The script
converts them to `kxn = sin(theta)`; angles must satisfy 0 <= theta < 90 degrees.
Edit `FREQUENCIES_GHZ` to change the default 10–45 GHz frequency grid.

The plot shows the field reflection magnitude `abs(Gamma)` for TE (solid) and TM
(dashed), with angle-colored patches and black polarization lines in the legend,
using the existing `Skin_2std` model and its default layer thicknesses (air
spacer, skin layer, semi-infinite dermis). Power reflectance is `abs(Gamma)**2`.
Figures are saved as `output/plots/skin_reflection_vs_frequency.svg` and `.pdf`.
The script also runs as `python -m layered_dielectrics.plot_skin_reflection`.

## Broadband reflection and APD summaries

```sh
layered-dielectrics-summary --list
layered-dielectrics-summary PHA24_30G_V2
layered-dielectrics-summary --all
```

The runner discovers active `PHA*` classes from `phantoms.py`; without arguments, it runs all classes. It calls `layered_dielectrics.broadband_phantom_summary.run(phantom)` for each selection and uses a noninteractive backend. The summary evaluates phantom APD at the final layer's entrance and skin APD at the SC entrance, across each class frequency grid and normalized transverse wave numbers.

## Sensitivity analysis

```sh
layered-dielectrics-sensitivity --list
layered-dielectrics-sensitivity PHA10_18G
layered-dielectrics-sensitivity --all
```

Without arguments, the runner selects `PHA10_18G`. Supported classes are `PHA10_18G`, `PHA18_24G`, `PHA24_30G`, `PHA24_30G_V2`, and `PHA30_45G`. Results include uncertainty workbooks under `output/tables/`.

## Material-property and layer exports

```sh
layered-dielectrics-ssl24
layered-dielectrics-penetration-depth --list
layered-dielectrics-penetration-depth --no-show
layered-dielectrics-export-layers
```

- `plot_ssl24_comparison.py` compares the SSL24 and SSL24 V2 permittivity/conductivity models over 24–30 GHz and plots signed percentage deviations. It saves SVG, PDF, and PNG files.
- `calculate_ssl_penetration_depth.py` plots the 1/e electric-field amplitude penetration depth for SSL-backed phantom classes. It accepts class names and an `--output` path. By default it saves both `output/plots/ssl_penetration_depth.svg` and `.pdf`; an explicit `--output` path saves only that format. `--no-show` suppresses `plt.show()`.
- `export_phantom_layers.py` exports five-layer material properties as CSV. Select a phantom and frequency with `layered-dielectrics-export-layers PHA24_30G_V2 --frequency 27 --output layers.csv`. It preserves zero-thickness layers and rejects frequencies outside the stored class range. Its SSL thickness label is a display setting, not a finite layer boundary in the propagation model.

## Basic reflection calculation

This example evaluates the full `PHA24_30G_V2` stack at normal incidence using
its stored frequency grid. Each coefficient array has shape `(frequency, kxn)`:

```python
import numpy as np
from layered_dielectrics import calculate_reflection, phantoms

result = calculate_reflection(phantoms.PHA24_30G_V2())
print(result.frequency / 1e9, np.abs(result.reflection_te[:, 0]))

# Multiple transverse wave numbers; kxn is normalized to the air wave number.
result = calculate_reflection(phantoms.PHA24_30G_V2(), kxn=[0.0, 0.5, 0.9])
```

The returned `ReflectionResult` contains `frequency` (Hz), `kxn`, and TE/TM
reflection and transmission field coefficients. Exact grazing incidence
(`kxn=1`) is singular in the transfer matrix formulation and is rejected by
this convenience API. Low-level calculations and material classes remain
available through `from layered_dielectrics import helpers, phantoms`.

Analyses also have explicit callable APIs:

```python
from layered_dielectrics import config, phantoms
from layered_dielectrics.broadband_phantom_summary import run as summary
from layered_dielectrics.sensitivity import run as sensitivity

config.set_output_directory("results")
fields = summary(phantoms.PHA24_30G_V2())  # arrays plus saved plots
table = sensitivity(phantoms.PHA10_18G())  # DataFrame plus saved workbook
```

The legacy sensitivity grid includes exact grazing incidence; intermediate
undefined values at that point can emit numerical warnings. Its existing
NaN-aware aggregation and scientific calculation are retained.

The transfer matrices propagate forward/backward field coefficients through interfaces and finite layers. `get_R_T` returns field coefficients, not power ratios. The APD helpers use the same field-amplitude convention for incident and transmitted flux, so normalization cancels the common scale factor.

## Repository files and configuration

| File | Purpose |
| --- | --- |
| `helpers.py` | Transfer matrices, field/APD calculations, skin and material models |
| `phantoms.py` | SSL dispersion models and phantom layer parameters |
| `config.py` | Shared output paths, backend, figure settings, and defaults |
| `broadband_phantom_summary.py`, `run_phantom_summary.py` | Broadband analysis and class runner |
| `sensitivity.py`, `run_sensitivity.py` | Sensitivity calculations and class runner |
| `plot_apd_vs_z.py` | Selected-frequency APD decay and layer diagrams |
| `plot_ssl24_comparison.py` | SSL24 material comparison |
| `calculate_ssl_penetration_depth.py` | SSL penetration-depth plots |
| `export_phantom_layers.py` | Layer-property CSV export |

Package sources live in `src/layered_dielectrics/`. Shared settings live in `layered_dielectrics.config`; analysis modules may also define their own frequencies and styling. Phantom frequency grids and layer properties live in `phantoms.py`, and skin dispersion parameters live in `helpers.py`.

Generated files under `output/` are ignored by Git. Run the scripts locally to regenerate plots and tables.

## Development and verification

```sh
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

Tests cover import side effects, installed module discovery from another working
directory, matched-medium and Fresnel limits, lossless energy conservation,
stored phantom reflection, sensitivity regression, and saved summary formats.
CI runs on Python 3.8, 3.11, and 3.13 and builds the source and wheel distributions.

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
