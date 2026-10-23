"""Plot setup scoped to an explicit plotting operation."""

from contextlib import contextmanager
import sys

from . import config


def get_legend_handles(axis):
    """Return labelled artists in drawing order for an explicit legend."""
    return axis.get_legend_handles_labels()[0]


@contextmanager
def plotting_context():
    """Apply the project plot defaults and restore the caller's settings.

    Use the configured backend only before pyplot has been initialized;
    an application's existing backend always takes precedence.
    """
    import matplotlib
    from cycler import cycler

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use(config.MATPLOTLIB_BACKEND)
    with matplotlib.rc_context({
        "font.size": config.FONT_SIZE,
        "figure.figsize": config.FIGURE_SIZE_DEFAULT,
        "figure.constrained_layout.use": True,
        "axes.prop_cycle": cycler(color=config.COLORS),
        "savefig.transparent": config.PLOT_TRANSPARENT,
    }):
        yield
