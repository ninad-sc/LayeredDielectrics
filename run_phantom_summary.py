"""Run broadband_phantom_summary.py for phantom classes in phantoms.py."""

import argparse
import inspect
from pathlib import Path
import runpy
import sys
import warnings

import config
import matplotlib

config.MATPLOTLIB_BACKEND = "Agg"
pyplot = sys.modules.get("matplotlib.pyplot")
if pyplot is None:
    matplotlib.use(config.MATPLOTLIB_BACKEND)
else:
    pyplot.switch_backend(config.MATPLOTLIB_BACKEND)
    pyplot.ioff()
matplotlib.interactive(False)

matplotlib.rcParams.update({
    'font.size': config.FONT_SIZE,
    'savefig.transparent': getattr(config, 'PLOT_TRANSPARENT', True),
})

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    import phantoms


SUMMARY_SCRIPT = Path(__file__).with_name("broadband_phantom_summary.py")


def get_phantom_classes():
    phantom_classes = []
    for class_name, phantom_class in inspect.getmembers(phantoms, inspect.isclass):
        if not class_name.startswith("PHA"):
            continue
        if phantom_class.__module__ != phantoms.__name__:
            continue
        source_line = inspect.getsourcelines(phantom_class)[1]
        phantom_classes.append((source_line, class_name, phantom_class))

    return {
        class_name: phantom_class
        for _, class_name, phantom_class in sorted(phantom_classes)
    }


PHANTOM_CLASSES = get_phantom_classes()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run broadband_phantom_summary.py for phantom classes."
    )
    parser.add_argument(
        "phantoms",
        nargs="*",
        help="Phantom class names to run, or 'all'. Defaults to all phantom classes.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all discovered phantom classes.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered phantom class names and exit.",
    )
    return parser.parse_args()


def resolve_phantom_names(args):
    if args.list:
        return []

    if args.all and args.phantoms:
        raise ValueError("Use either --all or explicit phantom names, not both.")

    if args.all:
        return list(PHANTOM_CLASSES)

    if not args.phantoms:
        return list(PHANTOM_CLASSES)

    if "all" in args.phantoms:
        if len(args.phantoms) > 1:
            raise ValueError("Use 'all' by itself, without other phantom names.")
        return list(PHANTOM_CLASSES)

    unknown = sorted(set(args.phantoms) - set(PHANTOM_CLASSES))
    if unknown:
        supported = ", ".join(PHANTOM_CLASSES)
        raise ValueError(
            f"Unknown phantom name(s): {', '.join(unknown)}. "
            f"Supported names: {supported}."
        )

    return args.phantoms


def run_phantom_summary(pha):
    print(f"Running broadband_phantom_summary.py for {pha.name}...")
    runpy.run_path(
        str(SUMMARY_SCRIPT),
        init_globals={"pha": pha},
        run_name="__main__",
    )
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")
    print(f"Finished {pha.name}: {config.OUTPUT_PLOTS_DIR}")


def main():
    args = parse_args()
    if args.list:
        print("\n".join(PHANTOM_CLASSES))
        return

    try:
        phantom_names = resolve_phantom_names(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")

    for phantom_name in phantom_names:
        run_phantom_summary(PHANTOM_CLASSES[phantom_name]())


if __name__ == "__main__":
    main()
