"""Run sensitivity.py for one or more phantom instances."""

import argparse
from pathlib import Path
import runpy
import warnings

import config

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    from phantoms import PHA10_18G, PHA18_24G, PHA24_30G, PHA24_30G_V2, PHA30_45G


SENSITIVITY_SCRIPT = Path(__file__).with_name("sensitivity.py")

PHANTOM_CLASSES = {
    "PHA10_18G": PHA10_18G,
    "PHA18_24G": PHA18_24G,
    "PHA24_30G": PHA24_30G,
    "PHA24_30G_V2": PHA24_30G_V2,
    "PHA30_45G": PHA30_45G,
}

DEFAULT_PHANTOMS = ("PHA10_18G",)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sensitivity.py with one or more phantom instances."
    )
    parser.add_argument(
        "phantoms",
        nargs="*",
        help="Phantom class names to run, or 'all'.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all supported phantom classes.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List supported phantom class names and exit.",
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
        return list(DEFAULT_PHANTOMS)

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


def run_sensitivity(pha):
    print(f"Running sensitivity.py for {pha.name}...")
    runpy.run_path(
        str(SENSITIVITY_SCRIPT),
        init_globals={"pha": pha},
        run_name="__main__",
    )
    output_path = config.OUTPUT_TABLES_DIR / f"{pha.name}.xlsx"
    print(f"Finished {pha.name}: {output_path}")


def main():
    args = parse_args()
    if args.list:
        print("\n".join(PHANTOM_CLASSES))
    else:
        try:
            phantom_names = resolve_phantom_names(args)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}")

        for phantom_name in phantom_names:
            run_sensitivity(PHANTOM_CLASSES[phantom_name]())


if __name__ == "__main__":
    main()
