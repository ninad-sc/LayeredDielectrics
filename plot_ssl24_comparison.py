"""Plot SSL24 material properties and signed V2 deviations over 24–30 GHz."""
from pathlib import Path

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import helpers
import phantoms

plt.rcParams.update({'font.size': 24})
COLORS = ['r', 'b', 'c', 'tab:orange', 'tab:purple']


def get_legend_handles(labels):
    return [Line2D([], [], color=COLORS[i], linewidth=3, label=label)
            for i, label in enumerate(labels)]


def properties(material, frequency):
    omega = 2 * np.pi * frequency
    parameters = [getattr(material, f'{prefix}_{i}')
                  for prefix in ('depsr', 'tau', 'alpha') for i in range(1, 5)]
    epsilon = helpers.getColeCole_5term(
        material.epsr_inf, material.sigma0, omega, *parameters)
    return helpers.getEpsrSigma(epsilon, omega)


def save_figure(fig, output, name):
    for extension in ('svg', 'pdf', 'png'):
        fig.savefig(output / f'{name}.{extension}', dpi=150,
                    transparent=False, facecolor='white')
    plt.close(fig)


def main():
    output = Path(__file__).resolve().parent / 'output' / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    materials = [phantoms.SSL24_30G, phantoms.SSL24_30GV2]
    frequency = np.linspace(max(m.fmin for m in materials),
                            min(m.fmax for m in materials), 601)
    values = np.asarray([properties(m, frequency) for m in materials])
    deviations = 100 * (values[1] - values[0]) / values[0]
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(deviations))
    assert np.all(values > 0)
    labels = [m.__name__ for m in materials]
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    for j, (ax, ylabel) in enumerate(zip(axes, [
            r'Relative permittivity $\epsilon_r^{\prime}$/1',
            r'Conductivity $\sigma$/(S/m)'])):
        for i in range(2):
            ax.plot(frequency / 1e9, values[i, j], color=COLORS[i], linewidth=3)
        ax.set(xlabel='Frequency/GHz', ylabel=ylabel, xlim=(24, 30))
        ax.grid(True, alpha=0.25)
        ax.legend(handles=get_legend_handles(labels), fontsize=19)
    fig.suptitle('SSL24 dielectric properties')
    save_figure(fig, output, 'ssl24_dielectric_properties')

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    for j, (ax, label) in enumerate(zip(axes, ['Relative permittivity', 'Conductivity'])):
        ax.plot(frequency / 1e9, deviations[j], color=COLORS[j], linewidth=3)
        ax.axhline(0, color='0.5', linewidth=1, linestyle='--')
        ax.set(xlabel='Frequency/GHz', ylabel='Deviation/%', title=label, xlim=(24, 30))
        ax.grid(True, alpha=0.25)
    fig.suptitle('SSL24_30GV2 relative to SSL24_30G\n' + r'$100\,(V2-V1)/V1$')
    save_figure(fig, output, 'ssl24_percentage_deviation')
    for j, label in enumerate(['Relative permittivity', 'Conductivity']):
        print(f'{label}: deviation {deviations[j].min():.3f}% to {deviations[j].max():.3f}%')
    print(output)


if __name__ == '__main__':
    main()
