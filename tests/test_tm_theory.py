"""TM checks against Fresnel theory and independent input-impedance recursion."""

import numpy as np
import pytest
from scipy.constants import epsilon_0, mu_0

from layered_dielectrics import helpers
from layered_dielectrics.plot_skin_reflection import calculate_skin_reflection


@pytest.mark.parametrize('eps_r', [4., 12. - 8.j])
def test_tm_interface_matches_complex_fresnel(eps_r):
    frequency = np.array([10e9, 30e9, 45e9])
    omega = 2 * np.pi * frequency
    angles = np.deg2rad([0., 15., 30., 45., 60., 80., 89.9])
    k_air = omega * np.sqrt(mu_0 * epsilon_0)
    kz_air = k_air[:, None] * np.cos(angles)
    kz_medium = k_air[:, None] * np.sqrt(eps_r - np.sin(angles)**2 + 0j)
    eps_air = np.full(frequency.shape, epsilon_0)
    eps_medium = np.full(frequency.shape, eps_r * epsilon_0)
    matrix = helpers.get_transfer_matrix_TM(
        kz_air, kz_medium, 0., eps_air, eps_medium)
    reflection, transmission = helpers.get_R_T(matrix)
    # TM amplitudes in the package are Hy. Tangential Ex has the opposite
    # reflected sign: Ex,r/Ex,i = -Hy,r/Hy,i.
    expected = (eps_r * kz_air - kz_medium) / (eps_r * kz_air + kz_medium)
    np.testing.assert_allclose(reflection, expected, atol=1e-14)
    np.testing.assert_allclose(transmission, 1 + expected, atol=1e-14)
    if np.isreal(eps_r):
        power_t = np.real(kz_medium / eps_r) / kz_air * np.abs(transmission)**2
        np.testing.assert_allclose(np.abs(reflection)**2 + power_t, 1., atol=1e-14)


def test_tm_brewster_angle():
    # Air to n=2: theta_B = atan(n2/n1), with zero TM reflection.
    angle = np.arctan(2.)
    kz_air = np.array([[np.cos(angle)]])
    kz_medium = np.array([[np.sqrt(4. - np.sin(angle)**2)]])
    matrix = helpers.get_transfer_matrix_TM(
        kz_air, kz_medium, 0., np.array([epsilon_0]), np.array([4 * epsilon_0]))
    reflection, _ = helpers.get_R_T(matrix)
    np.testing.assert_allclose(reflection, 0., atol=1e-14)


def test_skin_tm_matches_input_impedance_theory():
    frequency = np.linspace(10e9, 45e9, 701)
    angles = np.array([0., 15., 30., 45., 60., 80., 89.9])
    omega = 2 * np.pi * frequency[:, None]
    skin = helpers.Skin_2std(frequency)
    # Reuse material dispersion only, not the propagation/transfer helpers.
    eps = [np.full(frequency.shape, epsilon_0),
           skin.eps_c1, skin.eps_c2, skin.eps_c3]
    kx = omega * np.sqrt(mu_0 * epsilon_0) * np.sin(np.deg2rad(angles))
    kz = [np.sqrt(omega**2 * mu_0 * e[:, None] - kx**2 + 0j) for e in eps]
    # Ex/Hy for a forward TM wave is kz/(omega*epsilon).
    impedance = [k / (omega * e[:, None]) for k, e in zip(kz, eps)]
    z_in = impedance[3]
    for layer, thickness in [(2, skin.d_2), (1, skin.d_1)]:
        z = impedance[layer]
        tangent = np.tan(kz[layer] * thickness)
        z_in = z * (z_in + 1j * z * tangent) / (z + 1j * z_in * tangent)
    expected = (impedance[0] - z_in) / (impedance[0] + z_in)
    _, actual = calculate_skin_reflection(frequency, angles)
    error = np.max(np.abs(actual - expected))
    print(f'Maximum complex skin TM error: {error:.3e}')
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
