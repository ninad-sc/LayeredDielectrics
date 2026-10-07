"""Calculate and plot SSL penetration depth for every SSL-backed phantom.

The penetration depth is the 1/e electric-field amplitude depth. Frequencies
are taken from each phantom class, so every curve is shown over the frequency
band for which that phantom was designed.
"""

import argparse
import inspect
from pathlib import Path
import warnings

import numpy as np


COLORS = ['r', 'b', 'c', 'tab:orange', 'tab:brown']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot SSL penetration depth over frequency for all phantoms."
    )
    parser.add_argument(
        "phantoms",
        nargs="*",
        help=(
            "Phantom class names to plot, for example PHA10_18G PHA24_30G. "
            "Defaults to all SSL-backed phantom classes."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path. The default is "
            "output/plots/ssl_penetration_depth.svg."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot without opening an interactive window.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the discovered SSL-backed phantom classes and exit.",
    )
    return parser.parse_args()


def discover_ssl_phantoms(phantoms_module):
    """Return phantom classes that define an SSL material, in source order."""
    discovered = []
    for class_name, phantom_class in inspect.getmembers(
        phantoms_module, inspect.isclass
    ):
        if not class_name.startswith("PHA"):
            continue
        if phantom_class.__module__ != phantoms_module.__name__:
            continue
        if not hasattr(phantom_class, "ssl"):
            continue
        source_line = inspect.getsourcelines(phantom_class)[1]
        discovered.append((source_line, class_name, phantom_class))

    return [(name, cls) for _, name, cls in sorted(discovered)]


def calculate_penetration_depth(phantom_class, helpers):
    """Calculate the SSL properties and penetration depth for one phantom."""
    frequency = np.asarray(phantom_class.freq, dtype=float)
    ssl = phantom_class.ssl
    omega = 2 * np.pi * frequency

    complex_permittivity = helpers.getColeCole_5term(
        ssl.epsr_inf,
        ssl.sigma0,
        omega,
        ssl.depsr_1,
        ssl.depsr_2,
        ssl.depsr_3,
        ssl.depsr_4,
        ssl.tau_1,
        ssl.tau_2,
        ssl.tau_3,
        ssl.tau_4,
        ssl.alpha_1,
        ssl.alpha_2,
        ssl.alpha_3,
        ssl.alpha_4,
    )
    epsilon_r, conductivity = helpers.getEpsrSigma(
        complex_permittivity, omega
    )
    depth_mm = helpers.penetration_depth(
        frequency, epsilon_r, conductivity
    )
    return frequency, depth_mm


def main():
    args = parse_args()

    # Select a non-interactive backend before helpers imports pyplot.
    import config
    if args.no_show:
        config.MATPLOTLIB_BACKEND = "Qt5Agg"

    import matplotlib
    matplotlib.use(config.MATPLOTLIB_BACKEND)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 28})

    import helpers
    # The module also defines lossless placeholder phantoms whose infinite
    # penetration depths emit a divide-by-zero warning during import.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        import phantoms

    phantom_classes = discover_ssl_phantoms(phantoms)
    if args.list:
        print("\n".join(name for name, _ in phantom_classes))
        return
    if not phantom_classes:
        raise SystemExit("No SSL-backed phantom classes were found.")

    available_phantoms = dict(phantom_classes)
    if args.phantoms:
        unknown = [name for name in args.phantoms if name not in available_phantoms]
        if unknown:
            supported = ", ".join(available_phantoms)
            raise SystemExit(
                f"Unknown phantom class(es): {', '.join(unknown)}. "
                f"Available classes: {supported}."
            )
        phantom_classes = [
            (name, available_phantoms[name]) for name in args.phantoms
        ]

    fig, axis = plt.subplots(figsize=(16, 9))
    for index, (class_name, phantom_class) in enumerate(phantom_classes):
        frequency, depth_mm = calculate_penetration_depth(
            phantom_class, helpers
        )
        label = getattr(phantom_class, "name", class_name)
        axis.plot(
            frequency * 1e-9,
            depth_mm,
            color=COLORS[index % len(COLORS)],
            linewidth=2,
            label=label,
        )
        print(
            f"{class_name}: {frequency[0] * 1e-9:.2f}-"
            f"{frequency[-1] * 1e-9:.2f} GHz, "
            f"penetration depth {depth_mm.min():.3f}-"
            f"{depth_mm.max():.3f} mm"
        )

    axis.set_xlabel("Frequency /GHz")
    axis.set_ylabel("SSL penetration depth /mm")
    axis.set_title("SSL penetration depth by phantom class")
    axis.grid(True)
    axis.legend()
    fig.tight_layout()

    output_path = args.output or (
        config.OUTPUT_PLOTS_DIR / "ssl_penetration_depth.pdf"
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    if not args.no_show:
        plt.show()
    #plt.close(fig)


if __name__ == "__main__":
    main()
