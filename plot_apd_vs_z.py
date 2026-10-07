"""APD decay through all layers, with class parameters held fixed.

Only 10, 15, 20, 30 and 45 GHz, normal incidence, and Skin_2std.
Each phantom is evaluated only inside its stored frequency range.
PHA24_30G and mmW classes are excluded; PHA30_45G is used only at 45 GHz.
PHA_TH30G is used only at 30 GHz. One figure per frequency includes layer layouts.
z = 0 is the entrance of class layer 1. For skin, this includes the default
2 mm air spacer; SC and dermis follow it. For phantoms, layer 1 is epoxy.
APD/IPD is the net inward Poynting flux remaining at z, not cumulative
absorbed power. The final class layer is semi-infinite.
"""
import inspect
from pathlib import Path
import warnings
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.constants import epsilon_0, mu_0
import helpers
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='divide by zero encountered in divide')
    import phantoms
plt.rcParams.update({'font.size': 24})
COLORS = ['r', 'b', 'c', 'tab:orange', 'tab:purple', 'tab:green', 'tab:brown', 'tab:pink']
ROOT = Path(__file__).resolve().parent
FREQUENCIES_GHZ = (10., 15., 20., 30., 45.)
AIR_COLOR = '0.85'


def get_legend_handles(labels, colors=COLORS, styles=None):
    styles = ['-'] * len(labels) if styles is None else styles
    return [Line2D([], [], color='k' if 'skin' in l.lower() else c,
                   linestyle='-' if 'skin' in l.lower() else s, linewidth=3, label=l)
            for l, c, s in zip(labels, colors, styles)]


def phantom_material(pha, f):
    if np.any((f < pha.freq[0]) | (f > pha.freq[-1])):
        raise ValueError(f'Frequency outside the stored band for {pha.name}')
    def sample(key):
        value = np.asarray(getattr(pha, key))
        return np.full(f.shape, value) if value.ndim == 0 else np.interp(f, pha.freq, value)
    names = ['epoxy', 'foam', 'epoxy', 'shell', 'ssl']
    return helpers.Phantom_4layer(f, *[sample('epsr_' + n) for n in names],
        *[sample('sigma_' + n) for n in names], pha.epoxy_thickness,
        pha.foam_thickness, pha.epoxy_thickness, pha.shell_thickness)


def depth_profile(material, f, z, skin=False):
    """Evaluate forward and reflected fields inside every fixed layer.

    Solve backwards from the final-layer radiation condition to avoid
    cancellation errors in the backward wave at large depths.
    """
    omega = 2 * np.pi * f
    kxn = np.array([0.])
    k_air = helpers.get_k0(omega, 1., 0.)
    kz_air = k_air[:, None]
    eps_air = np.full(f.shape, epsilon_0)
    z_air = np.sqrt(mu_0 / epsilon_0)
    material.calc_k0(omega)
    material.calc_kz(kxn, k_air)
    overall = helpers.get_overall_T if skin else helpers.get_overall_T_4layer
    te, tm = overall(kxn, kz_air, eps_air, material)
    count = 3 if skin else 5
    thicknesses = [getattr(material, f'd_{i}') for i in range(1, count)]
    starts = np.r_[0., np.cumsum(thicknesses)]
    profiles = np.empty((2, len(z)))
    for p, (total, calc, incident) in enumerate([
            (te, helpers.get_APD_vector_te, 1.),
            (tm, helpers.get_APD_vector_tm, 1. / z_air)]):
        r, t = helpers.get_R_T(total)
        fields = {count: np.array([t[0, 0] * incident, 0.j])}
        for i in range(count, 1, -1):
            before_kz = getattr(material, f'kz_{i-1}')
            after_kz = getattr(material, f'kz_{i}')
            d = getattr(material, f'd_{i-1}')
            if p == 0:
                transfer = helpers.get_transfer_matrix_TE(before_kz, after_kz, d)
            else:
                transfer = helpers.get_transfer_matrix_TM(before_kz, after_kz, d,
                    getattr(material, f'eps_c{i-1}'), getattr(material, f'eps_c{i}'))
            fields[i - 1] = np.linalg.solve(transfer[:, :, 0, 0], fields[i])
        entrances, exits = [], []
        for i in range(1, count + 1):
            kz = getattr(material, f'kz_{i}')
            impedance = getattr(material, f'Z{i}')
            start = starts[i - 1]
            end = starts[i] if i < count else np.inf
            mask = (z >= start) & (z < end)

            def power(offset):
                offset = np.atleast_1d(offset)
                a = fields[i][0] * np.exp(-1j * kz * offset)
                b = fields[i][1] * np.exp(1j * kz * offset)
                return calc(a, b, np.zeros_like(a), np.broadcast_to(kz, a.shape),
                            impedance)[0] * z_air

            if mask.any():
                profiles[p, mask] = power(z[mask] - start)
            entrances.append(power(0.)[0])
            if i < count:
                exits.append(power(end - start)[0])
        np.testing.assert_allclose(entrances[0] + abs(r[0, 0])**2, 1., atol=1e-10)
        np.testing.assert_allclose(exits, entrances[1:], atol=1e-10)
    np.testing.assert_allclose(profiles[0], profiles[1], atol=1e-12, rtol=1e-10)
    if not np.all(np.isfinite(profiles)) or np.min(profiles) < -1e-10 or np.max(profiles) > 1 + 1e-10:
        raise ValueError('Invalid passive APD profile')
    if np.max(np.diff(profiles, axis=1)) > 1e-9:
        raise ValueError('Power increases with depth in a passive layer')
    return profiles[0], starts


def save_figure(fig, out, name):
    for extension in ('svg', 'pdf', 'png'):
        fig.savefig(out / f'{name}.{extension}', dpi=120, transparent=False, facecolor='white')
    #plt.close(fig)


def main():
    out = ROOT / 'output' / 'plots' / 'apd_vs_z' / '10_15_20_30_45GHz'
    out.mkdir(parents=True, exist_ok=True)
    classes = [cls for name, cls in inspect.getmembers(phantoms, inspect.isclass)
               if name.startswith('PHA') and cls.__module__ == phantoms.__name__
               and name != 'PHA24_30G' and not name.startswith('PHAmmW')]
    classes.sort(key=lambda cls: inspect.getsourcelines(cls)[1])
    rows = []
    for frequency in FREQUENCIES_GHZ:
        selected = [cls for cls in classes
                    if cls.freq[0] <= frequency * 1e9 <= cls.freq[-1]
                    and (cls.__name__ != 'PHA30_45G' or frequency == 45.)
                    and (cls.__name__ != 'PHA_TH30G' or frequency == 30.)]
        f = np.array([frequency * 1e9])
        skin = helpers.Skin_2std(f)
        entries = [(cls.__name__, phantom_material(cls(), f)) for cls in selected]
        boundaries = [np.cumsum([m.d_1, m.d_2, m.d_3, m.d_4]) for _, m in entries]
        z = np.unique(np.r_[np.arange(0., 10.01e-3, 0.01e-3),
                            np.concatenate(boundaries), skin.d_1, skin.d_1 + skin.d_2])
        skin_apd, skin_starts = depth_profile(skin, f, z, skin=True)
        fig, (ax, layers) = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True,
            gridspec_kw={'height_ratios': [5, 0.8 * (len(entries) + 1)]}, sharex=True)
        layer_rows, thickness_labels = [], []
        for i, (name, material) in enumerate(entries):
            apd, starts = depth_profile(material, f, z)
            ax.plot(z * 1e3, apd, color=COLORS[i], linewidth=3)
            layer_names = ['Epoxy', 'Foam', 'Epoxy', 'Shell', 'SSL']
            layer_colors = ['r', 'b', 'r', 'c', 'tab:orange']
            if name == 'PHA_TH30G':
                for j in (1, 4):
                    layer_names[j] = 'Air'
                    layer_colors[j] = AIR_COLOR
            layer_rows.append((name, starts, layer_names, layer_colors))
            thickness_labels.append(name + ': ' + ' | '.join(
                f'{label} {d * 1e3:g}' for label, d in zip(
                    layer_names[:-1],
                    [material.d_1, material.d_2, material.d_3, material.d_4])))
            rows.extend((name, 'Skin', frequency, zz * 1e3,
                         np.searchsorted(starts[1:], zz, side='right') + 1,
                         np.searchsorted(skin_starts[1:], zz, side='right') + 1, pp, ss)
                        for zz, pp, ss in zip(z, apd, skin_apd))
            print(f'{name}: {frequency:g} GHz; APD/IPD(10 mm)={apd[-1]:.6g}')
        ax.plot(z * 1e3, skin_apd, color='k', linestyle='-', linewidth=3)
        ax.set(ylabel='APD/IPD /1', xlim=(0, 10), ylim=(0, 1),
               title=f'APD decay through all layers at {frequency:g} GHz\nNormal incidence; fixed class parameters')
        ax.grid(alpha=0.25)
        ax.legend(handles=get_legend_handles([name for name, _ in entries] + ['Skin']))
        layer_rows.append(('Skin', skin_starts, ['Air', 'SC', 'Dermis'],
                           [AIR_COLOR, 'tab:olive', 'tab:gray']))
        for y, (_, starts, names, colors) in enumerate(reversed(layer_rows)):
            for j, start in enumerate(starts):
                end = starts[j + 1] if j + 1 < len(starts) else 10e-3
                if end <= start:
                    continue
                layers.barh(y, (end - start) * 1e3, left=start * 1e3,
                            height=0.6, color=colors[j], alpha=0.65)
                if (end - start) * 1e3 > 0.25:
                    layers.text((start + end) * 500, y, names[j], ha='center',
                                va='center', fontsize=16)
            for boundary in np.unique(starts[1:]):
                ax.axvline(boundary * 1e3, color='0.6', linestyle=':', linewidth=1)
        layers.set(yticks=range(len(layer_rows)),
                   yticklabels=[row[0] for row in reversed(layer_rows)],
                   xlabel='Depth z/mm', ylim=(-0.5, len(layer_rows) - 0.5))
        layers.tick_params(axis='y', labelsize=18)
        layers.set_title('Thicknesses/mm (final layers semi-infinite; epoxy = red)\n'
                         + '\n'.join(thickness_labels), fontsize=16)
        save_figure(fig, out, f'apd_layers_{frequency:g}GHz')
    tables = ROOT / 'output' / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=['phantom', 'skin_model', 'frequency/GHz', 'z/mm',
                               'phantom_layer', 'skin_layer', 'phantom_APD/IPD',
                               'skin_APD/IPD']).to_csv(tables / 'apd_vs_z_10_15_20_30_45GHz.csv', index=False)
    print(f'Saved plots: {out}')


if __name__ == '__main__':
    main()
