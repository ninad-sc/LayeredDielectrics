"""Plot skin reflection magnitude versus frequency for selectable incidence angles.

Angles are measured in degrees from the surface normal in incident air.
The Skin_2std layer properties and thicknesses use their class defaults.
"""

import argparse

import numpy as np
import matplotlib.patches as mpatch
from matplotlib.lines import Line2D
from scipy.constants import epsilon_0

from . import config, helpers
from .plotting import plotting_context

# Edit these variables to choose the curves and frequency range.
ANGLES_DEG = [0, 30, 60]
FREQUENCIES_GHZ = np.arange(6, 110.1, 0.1)
COLORS = ['r', 'b', 'c', 'tab:orange', 'tab:purple',
          'tab:green', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']


def get_legend_handles(angles_deg):
    """Show angle colors as patches and polarization styles as black lines."""
    return [mpatch.Patch(color=COLORS[i % len(COLORS)],
                         label=rf'${angle:g}^\circ$')
            for i, angle in enumerate(angles_deg)] + [
                Line2D([], [], color='k', linestyle='-', linewidth=2.5, label='TE'),
                Line2D([], [], color='k', linestyle='--', linewidth=2.5, label='TM')]


def calculate_skin_reflection(frequency_hz, angles_deg):
    """Return complex TE/TM coefficients shaped (frequency, angle).

Coefficients follow the existing transfer-matrix field convention. Their
absolute values give reflection magnitudes; squared magnitudes give power
reflectance in incident air.
"""
    frequency = np.asarray(frequency_hz, dtype=float)
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    if (frequency.ndim != 1 or not frequency.size
            or not np.isfinite(frequency).all() or np.any(frequency <= 0)):
        raise ValueError('Frequencies must be a nonempty 1D array of positive finite Hz')
    if (angles.ndim != 1 or not angles.size or not np.isfinite(angles).all()
            or np.any((angles < 0) | (angles >= 90))):
        raise ValueError('Incidence angles must be finite and satisfy 0 <= angle < 90 degrees')

    omega = 2 * np.pi * frequency
    kxn = np.sin(np.deg2rad(angles))
    k_air = helpers.get_k0(omega, 1., 0.)
    kz_air = helpers.get_kz(k_air, k_air[:, None] * kxn[None, :])
    skin = helpers.Skin_2std(frequency)
    skin.calc_k0(omega)
    skin.calc_kz(kxn, k_air)
    transfer_te, transfer_tm = helpers.get_overall_T(
        kxn, kz_air, np.full(frequency.shape, epsilon_0), skin)
    reflection_te, _ = helpers.get_R_T(transfer_te)
    reflection_tm, _ = helpers.get_R_T(transfer_tm)
    return reflection_te, reflection_tm


@plotting_context()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--angles', nargs='+', type=float, default=ANGLES_DEG,
                        help='Incidence angles in degrees (default: ANGLES_DEG variable)')
    args = parser.parse_args(argv)
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 24})
    frequency = np.asarray(FREQUENCIES_GHZ) * 1e9
    reflection_te, reflection_tm = calculate_skin_reflection(frequency, args.angles)
    handles = get_legend_handles(args.angles)
    fig, ax = plt.subplots(figsize=(16, 9), constrained_layout=True)
    for linestyle, reflection in zip(['-', '--'], [reflection_te, reflection_tm]):
        for i, angle in enumerate(args.angles):
            ax.plot(frequency / 1e9, 20*np.log10(np.abs(reflection[:, i])),
                    color=COLORS[i % len(COLORS)], linestyle=linestyle,
                    linewidth=2)
    ax.set(xlabel='Frequency /GHz', ylabel=r'$|\Gamma|$ |dB',
           xlim=(frequency.min() / 1e9, frequency.max() / 1e9), ylim=(-12., 0.),
           xscale='log',
           )
    ax.grid(True, alpha=0.25)
    ax.legend(handles=handles)
    fig.suptitle('Plane wave reflection coefficient')
    output = config.OUTPUT_PLOTS_DIR
    output.mkdir(parents=True, exist_ok=True)
    for extension in ('svg', 'pdf'):
        path = output / f'skin_reflection_vs_frequency.{extension}'
        fig.savefig(path, transparent=True)
        print(path)
    #plt.close(fig)


if __name__ == '__main__':
    main()
