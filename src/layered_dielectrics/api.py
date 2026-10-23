"""Convenience functions for the numerical propagation model."""

from dataclasses import dataclass

import numpy as np
from scipy.constants import epsilon_0

from . import helpers


@dataclass(frozen=True)
class ReflectionResult:
    """Field coefficients shaped (frequency, transverse wave number).

    Frequencies are in Hz; ``kxn`` is normalized to the incident air wave
    number. Transmission coefficients are field amplitudes, not power ratios.
    """

    frequency: np.ndarray
    kxn: np.ndarray
    reflection_te: np.ndarray
    transmission_te: np.ndarray
    reflection_tm: np.ndarray
    transmission_tm: np.ndarray


def calculate_reflection(phantom, kxn=None):
    """Evaluate a five-medium phantom stack on its stored frequency grid.

    ``phantom`` is an instance of a ``phantoms.PHA*`` class. The final medium
    is semi-infinite. ``kxn`` is a scalar or nonempty one-dimensional array;
    the default is normal incidence (zero). Exact grazing incidence (one)
    is singular in the underlying transfer matrices and is rejected.
    """
    kxn = np.atleast_1d(np.asarray(0.0 if kxn is None else kxn, dtype=float))
    if kxn.ndim != 1 or not kxn.size or not np.isfinite(kxn).all():
        raise ValueError("kxn must be a finite scalar or nonempty 1D array")
    if np.any(kxn == 1):
        raise ValueError("Exact grazing incidence (kxn=1) is singular; use a nearby value")
    frequency = np.asarray(phantom.freq, dtype=float).copy()
    omega = 2 * np.pi * frequency
    material = helpers.Phantom_4layer(
        frequency,
        phantom.epsr_epoxy, phantom.epsr_foam, phantom.epsr_epoxy,
        phantom.epsr_shell, phantom.epsr_ssl,
        phantom.sigma_epoxy, phantom.sigma_foam, phantom.sigma_epoxy,
        phantom.sigma_shell, phantom.sigma_ssl,
        phantom.epoxy_thickness, phantom.foam_thickness,
        phantom.epoxy_thickness, phantom.shell_thickness,
    )
    k0_air = helpers.get_k0(omega, 1.0, 0.0)
    kz_air = helpers.get_kz(k0_air, k0_air[:, None] * kxn[None, :])
    material.calc_k0(omega)
    material.calc_kz(kxn, k0_air)
    transfer_te, transfer_tm = helpers.get_overall_T_4layer(
        kxn, kz_air, np.full(frequency.shape, epsilon_0), material
    )
    reflection_te, transmission_te = helpers.get_R_T(transfer_te)
    reflection_tm, transmission_tm = helpers.get_R_T(transfer_tm)
    return ReflectionResult(frequency, kxn.copy(), reflection_te, transmission_te,
                            reflection_tm, transmission_tm)
