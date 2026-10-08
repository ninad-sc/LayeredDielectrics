"""Build a phantom layer DataFrame and save it as CSV.

Edit the settings below, then run ``python export_phantom_layers.py``.
Properties come from phantoms.py; sampled properties are linearly interpolated
at the requested frequency. Frequencies outside the stored range are rejected.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import phantoms


# User settings
PHANTOM = phantoms.PHA30_45G()
FREQUENCY_GHZ = 45.0
# This display value comes from the example, not from the phantom model.
SSL_THICKNESS_MM = "> 20"
# Set a material name if known; otherwise use WP156 for PHA10_18G as in
# the reference, and a generic shell label elsewhere.
SHELL_MATERIAL = None
OUTPUT_DIR = Path.cwd() / "output" / "tables" / "phantom_layers"


def create_layer_dataframe(phantom, frequency_ghz, ssl_thickness_mm="> 20",
                           shell_material=None):
    """Return the five layers in the reference order (thicknesses in mm).

    Zero-thickness layers are retained so layer numbering stays consistent.
    Scalar properties are used directly; arrays are interpolated on phantom.freq.
    The SSL thickness is a user-supplied label, since no depth is stored in the
    phantom definitions. No rounding is applied to the material properties.
    """
    frequency_hz = float(frequency_ghz) * 1e9
    frequencies = np.asarray(phantom.freq, dtype=float)
    if not np.isfinite(frequency_hz) or not (
        frequencies[0] <= frequency_hz <= frequencies[-1]
    ):
        raise ValueError(
            f"Frequency must be between {frequencies[0] / 1e9:g} and "
            f"{frequencies[-1] / 1e9:g} GHz for {phantom.name}."
        )

    def at_frequency(property_name):
        values = np.asarray(getattr(phantom, property_name), dtype=float)
        if values.ndim == 0:
            return float(values)
        return float(np.interp(frequency_hz, frequencies, values))

    if shell_material is None:
        shell_material = (
            "WP156" if isinstance(phantom, phantoms.PHA10_18G)
            else "Shell"
        )

    layers = [
        ("Glass fiber", phantom.epoxy_thickness * 1e3, "epoxy"),
        ("ROHACELL", phantom.foam_thickness * 1e3, "foam"),
        ("Glass fiber", phantom.epoxy_thickness * 1e3, "epoxy"),
        (shell_material, phantom.shell_thickness * 1e3, "shell"),
        (f"SSL @ {frequency_ghz:g} GHz", ssl_thickness_mm, "ssl"),
    ]
    return pd.DataFrame(
        [
            (layer, material, thickness, at_frequency(f"epsr_{key}"),
             at_frequency(f"sigma_{key}"))
            for layer, (material, thickness, key) in enumerate(layers)
        ],
        columns=["layer", "material", "thickness/mm", "ε_r", "σ/(S/m)"],
    )


def main():
    dataframe = create_layer_dataframe(
        PHANTOM, FREQUENCY_GHZ, SSL_THICKNESS_MM, SHELL_MATERIAL
    )
    output_path = OUTPUT_DIR / f"{PHANTOM.name}_{FREQUENCY_GHZ:g}GHz_layers.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_dataframe = dataframe.copy()
    for column in ["ε_r", "σ/(S/m)"]:
        csv_dataframe[column] = csv_dataframe[column].map("{:.2f}".format)
    csv_dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
    # ASCII console headings also work in Windows terminals using cp1252.
    print(dataframe.rename(columns={"ε_r": "eps_r", "σ/(S/m)": "sigma/(S/m)"})
          .to_string(index=False))
    print(f"\nSaved: {output_path}")
    
    return dataframe


if __name__ == "__main__":
    df = main()
