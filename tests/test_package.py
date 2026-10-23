"""Import safety, analytical limits, and installed command contracts."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from layered_dielectrics import calculate_reflection, helpers, phantoms


def test_imports_do_not_plot_or_write(tmp_path):
    script = """
import importlib, pkgutil, sys, warnings
import matplotlib
# Bypass RcParams.__getitem__: reading its backend key resolves the backend
# and can itself import pyplot before the package under test is imported.
before = {key: dict.__getitem__(matplotlib.rcParams, key) for key in matplotlib.rcParams}
with warnings.catch_warnings():
    warnings.simplefilter('error', RuntimeWarning)
    import layered_dielectrics
    for module in pkgutil.iter_modules(layered_dielectrics.__path__):
        importlib.import_module('layered_dielectrics.' + module.name)
assert 'matplotlib.pyplot' not in sys.modules
assert before == {key: dict.__getitem__(matplotlib.rcParams, key) for key in matplotlib.rcParams}
"""
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)
    assert not list(tmp_path.iterdir())


def uniform_stack(final_epsilon):
    return SimpleNamespace(
        freq=np.array([10e9, 20e9]),
        epsr_epoxy=1., epsr_foam=1., epsr_shell=1., epsr_ssl=final_epsilon,
        sigma_epoxy=0., sigma_foam=0., sigma_shell=0., sigma_ssl=0.,
        epoxy_thickness=0., foam_thickness=0., shell_thickness=0.,
    )


def test_matched_air_stack():
    result = calculate_reflection(uniform_stack(1.), [0., 0.5, 0.9])
    for reflection in [result.reflection_te, result.reflection_tm]:
        np.testing.assert_allclose(reflection, 0., atol=1e-14)
    for transmission in [result.transmission_te, result.transmission_tm]:
        np.testing.assert_allclose(transmission, 1., atol=1e-14)


def test_lossless_interface_matches_fresnel_and_conserves_power():
    result = calculate_reflection(uniform_stack(4.))
    np.testing.assert_allclose(np.abs(result.reflection_te), 1 / 3)
    np.testing.assert_allclose(np.abs(result.reflection_tm), 1 / 3)
    np.testing.assert_allclose(
        np.abs(result.reflection_te)**2 + 2 * np.abs(result.transmission_te)**2, 1.)
    np.testing.assert_allclose(
        np.abs(result.reflection_tm)**2 + 0.5 * np.abs(result.transmission_tm)**2, 1.)


def test_stored_phantom_normal_incidence_regression():
    result = calculate_reflection(phantoms.PHA24_30G_V2())
    assert result.reflection_te.shape == (25, 1)
    assert np.isfinite(result.reflection_te).all()
    np.testing.assert_allclose(np.abs(result.reflection_te), np.abs(result.reflection_tm))
    np.testing.assert_allclose(
        [np.abs(result.reflection_te).min(), np.abs(result.reflection_te).max()],
        [0.6360936127543734, 0.661316810755356], rtol=1e-10)


@pytest.mark.parametrize("kxn", [[], [[0.]], [np.nan], [np.inf], [1.]])
def test_invalid_wave_numbers(kxn):
    with pytest.raises(ValueError):
        calculate_reflection(uniform_stack(1.), kxn)


def test_lossless_penetration_depth_is_infinite():
    with np.errstate(all="raise"):
        assert np.isinf(helpers.penetration_depth(10e9, 1., 0.))


@pytest.mark.parametrize("module", ["run_phantom_summary", "run_sensitivity",
                                   "calculate_ssl_penetration_depth"])
def test_module_discovery_from_another_directory(module, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "layered_dielectrics." + module, "--list"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
        env=dict(os.environ, MPLBACKEND="Agg"),
    )
    assert "PHA24_30G_V2" in result.stdout
    assert not list(tmp_path.iterdir())


def test_export_rejects_out_of_range_frequency():
    from layered_dielectrics.export_phantom_layers import create_layer_dataframe
    with pytest.raises(ValueError, match="Frequency must be between"):
        create_layer_dataframe(phantoms.PHA24_30G_V2(), 10.)


def test_plot_defaults_are_scoped():
    import matplotlib
    from layered_dielectrics.plotting import plotting_context
    before = matplotlib.rcParams['font.size']
    with plotting_context():
        assert matplotlib.rcParams['font.size'] == 24
        assert matplotlib.rcParams['figure.figsize'] == [16, 9]
        assert matplotlib.rcParams['figure.constrained_layout.use']
        assert matplotlib.rcParams['axes.prop_cycle'].by_key()['color'] == [
            'r', 'b', 'c', 'tab:orange', 'tab:purple']
    assert matplotlib.rcParams['font.size'] == before
