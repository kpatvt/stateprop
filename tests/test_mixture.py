"""Test suite for stateprop.mixture.

Covers:
  - Reducing functions (KW mixing rules)
  - Composition derivatives vs finite differences
  - Fugacity coefficient formula
  - Stability analysis (Michelsen TPD)
  - Rachford-Rice
  - PT flash + fugacity equality
  - State-function flash round-trips (Tbeta, Pbeta, PH, PS, TH, TS)
  - Bubble / dew point solvers
"""
import numpy as np
import pytest

from stateprop.mixture import (
    load_mixture, flash_pt, flash_tbeta, flash_pbeta,
    flash_ph, flash_ps, flash_th, flash_ts,
    bubble_point_p, bubble_point_T, dew_point_p, dew_point_T,
    stability_test_TPD, rachford_rice, wilson_K, ln_phi,
    density_from_pressure,
)
from stateprop.mixture.properties import alpha_r_mix_derivs


# ----------------------------------------------------------------------
# Reducing function tests
# ----------------------------------------------------------------------

def test_reducing_pure_limit_co2():
    """At pure CO2 composition, reducing T and rho must equal CO2's Tc, rhoc."""
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[1.0, 0.0])
    Tr, rho_r = mx.reduce([1.0, 0.0])
    fl_co2 = mx.components[0].fluid
    assert abs(Tr - fl_co2.T_c) < 1e-10
    assert abs(rho_r - fl_co2.rho_c) < 1e-10


def test_reducing_pure_limit_n2():
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.0, 1.0])
    Tr, rho_r = mx.reduce([0.0, 1.0])
    fl_n2 = mx.components[1].fluid
    assert abs(Tr - fl_n2.T_c) < 1e-10
    assert abs(rho_r - fl_n2.rho_c) < 1e-10


def test_reducing_composition_derivatives_fd():
    """Composition derivatives of reducing T and rho must match finite differences
    at multiple asymmetric compositions.
    """
    mx = load_mixture(['carbondioxide', 'nitrogen'])
    for x0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
        x = np.array([x0, 1.0 - x0])
        Tr, rho_r, dTr, d_invrho = mx.reducing.derivatives(x)
        # Compare to finite differences of evaluate()
        eps = 1e-6
        for i in range(2):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            Tp, rp = mx.reduce(xp)
            Tm, rm = mx.reduce(xm)
            dTr_fd = (Tp - Tm) / (2 * eps)
            d_invrho_fd = (1.0 / rp - 1.0 / rm) / (2 * eps)
            assert abs(dTr[i] - dTr_fd) / abs(dTr_fd) < 1e-6, (
                f"dTr/dx_{i} mismatch at x={x}: "
                f"analytic={dTr[i]}, fd={dTr_fd}"
            )
            assert abs(d_invrho[i] - d_invrho_fd) / abs(d_invrho_fd) < 1e-6


# ----------------------------------------------------------------------
# Fugacity coefficient tests (key building block)
# ----------------------------------------------------------------------

def _n_alpha_r_at(n_vec, V_total, mixture, T):
    """Total reduced residual Helmholtz at (T, V, n) via alpha^r."""
    n = n_vec.sum()
    xx = n_vec / n
    rho_v = n / V_total
    return n * alpha_r_mix_derivs(rho_v, T, xx, mixture)['a_r']


def test_ln_phi_matches_fd_binary():
    """ln_phi + ln Z = n * d alpha^r / d n_i validated by FD at multiple compositions."""
    mx = load_mixture(['carbondioxide', 'nitrogen'])
    for x0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
        mx.set_composition([x0, 1 - x0])
        z = mx.x.copy()
        T, p = 300.0, 3e6
        rho = density_from_pressure(p, T, z, mx, phase_hint='vapor')
        V_total = 1.0 / rho

        lnphi = ln_phi(rho, T, z, mx)
        res = alpha_r_mix_derivs(rho, T, z, mx)
        Z = 1.0 + res['delta'] * res['a_r_d']
        analytic = lnphi + np.log(Z)

        fd = np.zeros(2)
        eps = 1e-7
        for i in range(2):
            np_p = z.copy(); np_p[i] += eps
            np_m = z.copy(); np_m[i] -= eps
            fd[i] = (_n_alpha_r_at(np_p, V_total, mx, T) -
                     _n_alpha_r_at(np_m, V_total, mx, T)) / (2 * eps)

        max_err = np.max(np.abs(analytic - fd) / np.maximum(np.abs(fd), 1e-10))
        assert max_err < 1e-5, f"x={z}: max_err={max_err}"


def test_ln_phi_matches_fd_ternary():
    """FD check on 3-component mixture."""
    mx = load_mixture(['carbondioxide', 'nitrogen', 'water'])
    T, p = 400.0, 5e6
    for z_list in [[0.3, 0.3, 0.4], [0.5, 0.2, 0.3], [0.2, 0.6, 0.2]]:
        mx.set_composition(z_list)
        z = mx.x.copy()
        rho = density_from_pressure(p, T, z, mx, phase_hint='vapor')
        V_total = 1.0 / rho
        lnphi = ln_phi(rho, T, z, mx)
        res = alpha_r_mix_derivs(rho, T, z, mx)
        Z = 1.0 + res['delta'] * res['a_r_d']
        analytic = lnphi + np.log(Z)
        fd = np.zeros(3)
        eps = 1e-7
        for i in range(3):
            np_p = z.copy(); np_p[i] += eps
            np_m = z.copy(); np_m[i] -= eps
            fd[i] = (_n_alpha_r_at(np_p, V_total, mx, T) -
                     _n_alpha_r_at(np_m, V_total, mx, T)) / (2 * eps)
        max_err = np.max(np.abs(analytic - fd) / np.maximum(np.abs(fd), 1e-10))
        assert max_err < 1e-5


# ----------------------------------------------------------------------
# Rachford-Rice
# ----------------------------------------------------------------------

def test_rachford_rice_simple():
    """RR on a trivial analytic case: K=[2, 0.5], z=[0.5, 0.5] -> beta=0.5."""
    z = np.array([0.5, 0.5])
    K = np.array([2.0, 0.5])
    beta, x, y = rachford_rice(z, K)
    assert abs(beta - 0.5) < 1e-10
    # At beta=0.5: x_1 = z_1/(1 + 0.5 * (K_1-1)) = 0.5/1.5 = 1/3
    assert abs(x[0] - 1.0/3.0) < 1e-10
    assert abs(y[0] - 2.0/3.0) < 1e-10


def test_rachford_rice_single_phase_all_vapor():
    """All K > 1, expected beta = 1 (all vapor)."""
    z = np.array([0.3, 0.7])
    K = np.array([1.5, 2.0])
    beta, x, y = rachford_rice(z, K)
    assert beta >= 0.999  # may slightly exceed 1 or be clamped


def test_rachford_rice_single_phase_all_liquid():
    """All K < 1, expected beta = 0."""
    z = np.array([0.3, 0.7])
    K = np.array([0.5, 0.7])
    beta, x, y = rachford_rice(z, K)
    assert beta <= 0.001


# ----------------------------------------------------------------------
# Stability tests (Michelsen TPD)
# ----------------------------------------------------------------------

def test_stability_unstable_2phase():
    """CO2+N2 at T=220K, p=2MPa, z=[0.5, 0.5] is known 2-phase -> unstable."""
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.5, 0.5])
    stable, K, Sm1 = stability_test_TPD(mx.x, 220.0, 2e6, mx)
    assert not stable
    assert Sm1 > 1e-6


def test_stability_stable_supercritical():
    """At high T above mixture critical, should be stable."""
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.5, 0.5])
    stable, K, Sm1 = stability_test_TPD(mx.x, 500.0, 5e6, mx)
    assert stable


def test_stability_stable_low_pressure_vapor():
    """At low p, pure vapor is stable."""
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.5, 0.5])
    stable, K, Sm1 = stability_test_TPD(mx.x, 220.0, 0.1e6, mx)
    assert stable


# ----------------------------------------------------------------------
# PT flash
# ----------------------------------------------------------------------

def test_pt_flash_fugacity_equality():
    """Converged 2-phase PT flash has equal fugacities in liquid and vapor."""
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.5, 0.5])
    T, p = 220.0, 3e6
    r = flash_pt(p, T, mx.x, mx)
    assert r.phase == 'two_phase'
    f_L = r.x * np.exp(ln_phi(r.rho_L, T, r.x, mx)) * p
    f_V = r.y * np.exp(ln_phi(r.rho_V, T, r.y, mx)) * p
    assert np.max(np.abs(f_L / f_V - 1.0)) < 1e-6


def test_pt_flash_supercritical():
    """At T >> Tc, should be supercritical."""
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.5, 0.5])
    r = flash_pt(5e6, 500.0, mx.x, mx)
    assert r.phase == 'supercritical'


def test_pt_flash_physical_partitioning():
    """At 2-phase CO2+N2, liquid must be CO2-rich, vapor N2-rich."""
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.5, 0.5])
    r = flash_pt(3e6, 220.0, mx.x, mx)
    assert r.phase == 'two_phase'
    assert r.x[0] > 0.9   # liquid CO2-rich
    assert r.y[1] > 0.6   # vapor N2-rich


# ----------------------------------------------------------------------
# State-function flash round-trips
# ----------------------------------------------------------------------

@pytest.mark.parametrize("T,p", [(220.0, 3e6), (200.0, 2e6)])
def test_flash_tbeta_roundtrip(T, p):
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.5, 0.5])
    r0 = flash_pt(p, T, mx.x, mx)
    assert r0.phase == 'two_phase'
    r = flash_tbeta(T, r0.beta, mx.x, mx)
    assert abs(r.p - p) / p < 1e-5


@pytest.mark.parametrize("T,p", [(220.0, 3e6), (200.0, 2e6)])
def test_flash_pbeta_roundtrip(T, p):
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.5, 0.5])
    r0 = flash_pt(p, T, mx.x, mx)
    assert r0.phase == 'two_phase'
    r = flash_pbeta(p, r0.beta, mx.x, mx)
    assert abs(r.T - T) / T < 1e-5


@pytest.mark.parametrize("T,p,phase_expect", [
    (220.0, 3e6, 'two_phase'),
    (400.0, 5e6, 'supercritical'),
    (200.0, 2e6, 'two_phase'),
])
def test_flash_ph_roundtrip(T, p, phase_expect):
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.5, 0.5])
    r0 = flash_pt(p, T, mx.x, mx)
    assert r0.phase == phase_expect
    r = flash_ph(p, r0.h, mx.x, mx)
    assert abs(r.T - T) / T < 1e-4


@pytest.mark.parametrize("T,p", [(220.0, 3e6), (400.0, 5e6), (200.0, 2e6)])
def test_flash_ps_roundtrip(T, p):
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.5, 0.5])
    r0 = flash_pt(p, T, mx.x, mx)
    r = flash_ps(p, r0.s, mx.x, mx)
    assert abs(r.T - T) / T < 1e-4


@pytest.mark.parametrize("T,p", [(220.0, 3e6), (400.0, 5e6), (200.0, 2e6)])
def test_flash_th_roundtrip(T, p):
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.5, 0.5])
    r0 = flash_pt(p, T, mx.x, mx)
    r = flash_th(T, r0.h, mx.x, mx)
    assert abs(r.p - p) / p < 1e-4


@pytest.mark.parametrize("T,p", [(220.0, 3e6), (400.0, 5e6), (200.0, 2e6)])
def test_flash_ts_roundtrip(T, p):
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.5, 0.5])
    r0 = flash_pt(p, T, mx.x, mx)
    r = flash_ts(T, r0.s, mx.x, mx)
    assert abs(r.p - p) / p < 1e-4


# ----------------------------------------------------------------------
# Bubble and dew point solvers
# ----------------------------------------------------------------------

@pytest.mark.parametrize("z_list,T", [
    ([0.99, 0.01], 240),
    ([0.99, 0.01], 260),
    ([0.995, 0.005], 290),
])
def test_bubble_point_p(z_list, T):
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=z_list)
    z = np.array(z_list)
    r = bubble_point_p(T, z, mx)
    # At bubble, flash should give beta ~ 0
    r_check = flash_pt(r.p, T, z, mx, check_stability=False)
    beta = r_check.beta if r_check.beta is not None else 0.0
    assert beta < 1e-3, f"Expected near 0, got {beta}"


@pytest.mark.parametrize("z_list,p", [
    ([0.99, 0.01], 3e6),
    ([0.995, 0.005], 5e6),
])
def test_bubble_point_T(z_list, p):
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=z_list)
    z = np.array(z_list)
    r = bubble_point_T(p, z, mx)
    r_check = flash_pt(p, r.T, z, mx, check_stability=False)
    beta = r_check.beta if r_check.beta is not None else 0.0
    assert beta < 1e-3


@pytest.mark.parametrize("z_list,T", [
    ([0.5, 0.5], 220),
    ([0.3, 0.7], 240),
])
def test_dew_point_p(z_list, T):
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=z_list)
    z = np.array(z_list)
    r = dew_point_p(T, z, mx)
    r_check = flash_pt(r.p, T, z, mx, check_stability=False)
    beta = r_check.beta if r_check.beta is not None else 1.0
    assert beta > 0.999


@pytest.mark.parametrize("z_list,p", [
    ([0.5, 0.5], 3e6),
    ([0.3, 0.7], 5e6),
])
def test_dew_point_T(z_list, p):
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=z_list)
    z = np.array(z_list)
    r = dew_point_T(p, z, mx)
    r_check = flash_pt(p, r.T, z, mx, check_stability=False)
    beta = r_check.beta if r_check.beta is not None else 1.0
    assert beta > 0.999


# ----------------------------------------------------------------------
# Error handling: physically unreachable bubble/dew points
# ----------------------------------------------------------------------

def test_bubble_point_T_no_solution():
    """At z=[0.98, 0.02], p=3MPa: S(T) never reaches 1 -> solver should raise."""
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.98, 0.02])
    z = mx.x.copy()
    with pytest.raises(RuntimeError):
        bubble_point_T(3e6, z, mx)


def test_dew_point_p_above_critical():
    """At T >> mixture Tc, no dew line exists."""
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.1, 0.9])
    z = mx.x.copy()
    with pytest.raises(RuntimeError):
        dew_point_p(260, z, mx)
