"""Run sensitivity.py and save standard-compliant sensitivity workbooks."""

import argparse
from pathlib import Path
import runpy
import warnings

import numpy as np
import pandas as pd

import config

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    from phantoms import PHA10_18G, PHA18_24G, PHA24_30G, PHA24_30G_V2, PHA30_45G


SENSITIVITY_SCRIPT = Path(__file__).with_name("sensitivity.py")
STANDARD_COMPLIANT_TABLES_DIR = config.OUTPUT_DIR / "standard_compliant_tables"

PHANTOM_CLASSES = {
    "PHA10_18G": PHA10_18G,
    "PHA18_24G": PHA18_24G,
    "PHA24_30G": PHA24_30G,
    "PHA24_30G_V2": PHA24_30G_V2,
    "PHA30_45G": PHA30_45G,
}

DEFAULT_PHANTOMS = ("PHA10_18G",)

STANDARD_COMPONENTS = (
    (
        "Deviation of the phantom from the reference skin tissue model",
        ("skin_emulation",),
    ),
    (
        "Deviation of the dielectric parameters of the phantom materials",
        (
            "ssl_epsr",
            "ssl_sigma",
            "composite_shell_epsr",
            "composite_shell_sigma",
        ),
    ),
    (
        "Measurement of the dielectric parameters",
        ("epsr_measurement", "sigma_measurement"),
    ),
    (
        "Temperature dependence",
        ("ssl_epsr_temperature", "ssl_sigma_temperature"),
    ),
    (
        "Thickness of the shell and the spacer",
        ("composite_shell_thickness",),
    ),
    (
        "Thermal properties of the phantom materials",
        (),
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run sensitivity.py with one or more phantom instances and save "
            "standard-compliant RSS-combined Excel workbooks."
        )
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


def rss_components(source_table, component_names):
    values = source_table.loc[list(component_names), ["TE", "TM"]].astype(float)
    return np.sqrt(np.square(values.to_numpy()).sum(axis=0))


def save_standard_compliant_workbook(source_path, output_path):
    source_df = pd.read_excel(source_path)
    if "Contributor" not in source_df.columns:
        raise ValueError(f"Missing Contributor column in {source_path}.")

    source_table = source_df.set_index("Contributor")
    rows = []

    for component_label, source_components in STANDARD_COMPONENTS:
        if not source_components:
            rows.append(
                {
                    "Contributor": component_label,
                    "TE": None,
                    "TM": None,
                    "Unc. |dB": None,
                }
            )
            continue

        missing = [
            component
            for component in source_components
            if component not in source_table.index
        ]
        if missing:
            raise ValueError(
                f"Missing component(s) in {source_path}: {', '.join(missing)}."
            )

        te, tm = rss_components(source_table, source_components)
        rows.append(
            {
                "Contributor": component_label,
                "TE": te,
                "TM": tm,
                "Unc. |dB": np.round(max(te, tm), 3),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(output_path, index=False)


def run_sensitivity(pha):
    print(f"Running sensitivity.py for {pha.name}...")
    runpy.run_path(
        str(SENSITIVITY_SCRIPT),
        init_globals={"pha": pha},
        run_name="__main__",
    )
    source_path = config.OUTPUT_TABLES_DIR / f"{pha.name}.xlsx"
    output_path = STANDARD_COMPLIANT_TABLES_DIR / f"{pha.name}.xlsx"
    save_standard_compliant_workbook(source_path, output_path)
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
