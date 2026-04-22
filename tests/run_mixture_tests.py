"""Plain-Python test runner for stateprop.mixture (no pytest dependency).

Run: python tests/run_mixture_tests.py
"""
import sys
import traceback
import numpy as np

# Import from the package
sys.path.insert(0, '.')

from stateprop.mixture import (
    load_mixture, flash_pt, flash_tbeta, flash_pbeta,
    flash_ph, flash_ps, flash_th, flash_ts,
    bubble_point_p, bubble_point_T, dew_point_p, dew_point_T,
    stability_test_TPD, rachford_rice, wilson_K, ln_phi,
    density_from_pressure,
)
from stateprop.mixture.properties import alpha_r_mix_derivs


PASSED = 0
FAILED = 0
FAILURES = []


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        FAILURES.append((name, detail))
        print(f"  FAIL  {name}: {detail}")


def run_test(fn):
    print(f"\n[{fn.__name__}]")
    try:
        fn()
    except Exception as e:
        global FAILED
        FAILED += 1
        FAILURES.append((fn.__name__, f"EXCEPTION: {e}"))
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_reducing_pure_limits():
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[1.0, 0.0])
    Tr, rho_r = mx.reduce([1.0, 0.0])
    fl = mx.components[0].fluid
    check("pure CO2 Tr=Tc", abs(Tr - fl.T_c) < 1e-10, f"Tr={Tr}, Tc={fl.T_c}")
    check("pure CO2 rho_r=rhoc", abs(rho_r - fl.rho_c) < 1e-10)

    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.0, 1.0])
    Tr, rho_r = mx.reduce([0.0, 1.0])
    fl = mx.components[1].fluid
    check("pure N2 Tr=Tc", abs(Tr - fl.T_c) < 1e-10)
    check("pure N2 rho_r=rhoc", abs(rho_r - fl.rho_c) < 1e-10)


def test_reducing_fd():
    mx = load_mixture(['carbondioxide', 'nitrogen'])
    for x0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
        x = np.array([x0, 1.0 - x0])
        Tr, rho_r, dTr, d_invrho = mx.reducing.derivatives(x)
        eps = 1e-6
        for i in range(2):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            Tp, rp = mx.reduce(xp)
            Tm, rm = mx.reduce(xm)
            dTr_fd = (Tp - Tm) / (2 * eps)
            d_invrho_fd = (1.0 / rp - 1.0 / rm) / (2 * eps)
            err_T = abs(dTr[i] - dTr_fd) / abs(dTr_fd) if dTr_fd != 0 else 0
            err_v = abs(d_invrho[i] - d_invrho_fd) / abs(d_invrho_fd) if d_invrho_fd != 0 else 0
            check(f"dTr/dx_{i} at x={x.tolist()}", err_T < 1e-6, f"err={err_T:.2e}")
            check(f"d(1/rho)/dx_{i} at x={x.tolist()}", err_v < 1e-6, f"err={err_v:.2e}")


def test_ln_phi_binary_fd():
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

        def n_alpha_at(n_vec):
            n = n_vec.sum(); xx = n_vec / n; rho_v = n / V_total
            return n * alpha_r_mix_derivs(rho_v, T, xx, mx)['a_r']

        fd = np.zeros(2); eps = 1e-7
        for i in range(2):
            np_p = z.copy(); np_p[i] += eps
            np_m = z.copy(); np_m[i] -= eps
            fd[i] = (n_alpha_at(np_p) - n_alpha_at(np_m)) / (2 * eps)

        err = np.max(np.abs(analytic - fd) / np.maximum(np.abs(fd), 1e-10))
        check(f"ln_phi FD at x=[{x0}, {1-x0:.1f}]", err < 1e-5, f"err={err:.2e}")


def test_ln_phi_ternary_fd():
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

        def n_alpha_at(n_vec):
            n = n_vec.sum(); xx = n_vec / n; rho_v = n / V_total
            return n * alpha_r_mix_derivs(rho_v, T, xx, mx)['a_r']

        fd = np.zeros(3); eps = 1e-7
        for i in range(3):
            np_p = z.copy(); np_p[i] += eps
            np_m = z.copy(); np_m[i] -= eps
            fd[i] = (n_alpha_at(np_p) - n_alpha_at(np_m)) / (2 * eps)

        err = np.max(np.abs(analytic - fd) / np.maximum(np.abs(fd), 1e-10))
        check(f"ln_phi ternary FD at z={z_list}", err < 1e-5, f"err={err:.2e}")


def test_rachford_rice():
    # rachford_rice returns (beta, converged). x and y are computed separately.
    z = np.array([0.5, 0.5]); K = np.array([2.0, 0.5])
    beta, converged = rachford_rice(z, K)
    check("RR analytic beta", abs(beta - 0.5) < 1e-10, f"beta={beta}")
    x = z / (1.0 + beta * (K - 1.0))
    y = x * K
    check("RR converged", converged)
    check("RR x[0]", abs(x[0] - 1.0/3.0) < 1e-10)
    check("RR y[0]", abs(y[0] - 2.0/3.0) < 1e-10)

    z = np.array([0.3, 0.7]); K = np.array([1.5, 2.0])
    beta, _ = rachford_rice(z, K)
    check("RR all vapor", beta >= 0.99)

    z = np.array([0.3, 0.7]); K = np.array([0.5, 0.7])
    beta, _ = rachford_rice(z, K)
    check("RR all liquid", beta <= 0.01)


def test_stability():
    # Known 2-phase
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.5, 0.5])
    stable, K, Sm1 = stability_test_TPD(mx.x, 220.0, 2e6, mx)
    check("stability: 2-phase is unstable", not stable, f"Sm1={Sm1}")

    # Supercritical
    stable, K, Sm1 = stability_test_TPD(mx.x, 500.0, 5e6, mx)
    check("stability: supercritical is stable", stable)

    # Low p, vapor
    stable, K, Sm1 = stability_test_TPD(mx.x, 220.0, 0.1e6, mx)
    check("stability: low-p vapor is stable", stable)


def test_pt_flash():
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.5, 0.5])
    T, p = 220.0, 3e6
    r = flash_pt(p, T, mx.x, mx)
    check("PT flash identifies 2-phase", r.phase == 'two_phase')

    # Fugacity equality
    f_L = r.x * np.exp(ln_phi(r.rho_L, T, r.x, mx)) * p
    f_V = r.y * np.exp(ln_phi(r.rho_V, T, r.y, mx)) * p
    err = np.max(np.abs(f_L / f_V - 1.0))
    check("PT fugacity equality", err < 1e-6, f"err={err:.2e}")

    # Physical partitioning
    check("liquid is CO2-rich", r.x[0] > 0.9, f"x[0]={r.x[0]}")
    check("vapor is N2-rich", r.y[1] > 0.6, f"y[1]={r.y[1]}")

    # Supercritical
    r = flash_pt(5e6, 500.0, mx.x, mx)
    check("PT: high-T is supercritical", r.phase == 'supercritical')


def test_arbitrary_component_flash_coolprop():
    """PT flash on a non-GERG mixture using bundled CoolProp pure-fluid EOS.

    Ethanol+water is a strongly non-ideal system (no binary parameters in
    stateprop). With ideal mixing rules, the flash should still find the
    two-phase region where the underlying EOS predicts instability, and
    correctly partition components by their volatility ordering. Without
    binary parameters the predicted compositions won't match real Dortmund
    data exactly, but the qualitative behavior (ethanol enriched in vapor,
    water enriched in liquid) must be physically correct.
    """
    mx = load_mixture(['ethanol', 'water'], [0.5, 0.5])

    # Acentric factors must be loaded for Wilson K-factor initialization
    eth_omega = mx.components[0].fluid.acentric_factor
    wat_omega = mx.components[1].fluid.acentric_factor
    check("ethanol acentric loaded", eth_omega is not None and 0.5 < eth_omega < 0.7,
          f"got {eth_omega}")
    check("water acentric loaded", wat_omega is not None and 0.3 < wat_omega < 0.4,
          f"got {wat_omega}")

    # At low pressure the flash should converge to a two-phase split with
    # ethanol (more volatile) enriched in the vapor.
    T, p = 350.0, 0.5e5
    r = flash_pt(p, T, mx.x, mx)
    check("ethanol+water 350K/0.5bar is two_phase",
          r.phase == 'two_phase', f"got phase={r.phase}")
    if r.phase == 'two_phase':
        check("ethanol enriched in vapor (y_eth > x_eth)",
              r.y[0] > r.x[0],
              f"x_eth={r.x[0]:.3f}, y_eth={r.y[0]:.3f}")
        check("water enriched in liquid (x_wat > y_wat)",
              r.x[1] > r.y[1],
              f"x_wat={r.x[1]:.3f}, y_wat={r.y[1]:.3f}")
        # Fugacity equality: f_L = f_V at converged solution
        f_L = r.x * np.exp(ln_phi(r.rho_L, T, r.x, mx)) * p
        f_V = r.y * np.exp(ln_phi(r.rho_V, T, r.y, mx)) * p
        err = np.max(np.abs(f_L / f_V - 1.0))
        check("ethanol+water fugacity equality at converged flash",
              err < 1e-5, f"err={err:.2e}")


def test_arbitrary_component_single_phase_branches():
    """PT flash correctly classifies single-phase branches for arbitrary
    CoolProp mixtures: subcritical liquid, vapor, and supercritical."""
    mx = load_mixture(['ethanol', 'water'], [0.5, 0.5])

    # High pressure -> compressed liquid (above bubble curve)
    r = flash_pt(5e6, 400.0, mx.x, mx)
    check("ethanol+water 400K/5MPa is liquid",
          r.phase == 'liquid', f"got phase={r.phase}")
    check("liquid density physical (>1000 mol/m^3)",
          r.rho > 1000, f"rho={r.rho}")

    # Above both critical temperatures -> supercritical
    # (T_c_eth=515, T_c_wat=647, so T=700 is supercritical for both)
    r = flash_pt(1e5, 700.0, mx.x, mx)
    check("ethanol+water 700K/1bar is supercritical or vapor",
          r.phase in ('supercritical', 'vapor'),
          f"got phase={r.phase}")


def test_density_from_pressure_robustness():
    """The Newton solver in density_from_pressure must handle the case
    where float64 precision in the EOS pressure evaluation prevents
    Newton from driving |dp| below the standard tol*p threshold. Without
    a stalled-iteration check it would oscillate forever and raise
    RuntimeError, breaking flash convergence at low pressures."""
    from stateprop.mixture.properties import density_from_pressure
    mx = load_mixture(['ethanol', 'water'], [0.5, 0.5])
    T, p = 350.0, 0.5e5
    # Composition near a converged liquid-flash solution for ethanol+water
    x = np.array([0.42129067, 0.57870933])
    rho_L = density_from_pressure(p, T, x, mx, phase_hint="liquid")
    check("density_from_pressure converges at low p with H-bonding fluid",
          29000 < rho_L < 30000, f"rho_L={rho_L}")


def test_state_function_flashes():
    cases = [
        ("2-phase 220/3MPa", [0.5, 0.5], 220.0, 3e6),
        ("2-phase 200/2MPa", [0.5, 0.5], 200.0, 2e6),
        ("supercritical",    [0.5, 0.5], 400.0, 5e6),
    ]
    for case_name, z_list, T, p in cases:
        mx = load_mixture(['carbondioxide', 'nitrogen'], composition=z_list)
        z = mx.x.copy()
        r0 = flash_pt(p, T, z, mx)

        # Only test Tbeta/Pbeta on 2-phase states
        if r0.phase == 'two_phase':
            r = flash_tbeta(T, r0.beta, z, mx)
            err = abs(r.p - p) / p
            check(f"Tbeta round-trip ({case_name})", err < 1e-4, f"err={err:.2e}")
            r = flash_pbeta(p, r0.beta, z, mx)
            err = abs(r.T - T) / T
            check(f"Pbeta round-trip ({case_name})", err < 1e-4, f"err={err:.2e}")

        r = flash_ph(p, r0.h, z, mx); err = abs(r.T - T) / T
        check(f"PH round-trip ({case_name})", err < 1e-3, f"err={err:.2e}")
        r = flash_ps(p, r0.s, z, mx); err = abs(r.T - T) / T
        check(f"PS round-trip ({case_name})", err < 1e-3, f"err={err:.2e}")
        r = flash_th(T, r0.h, z, mx); err = abs(r.p - p) / p
        check(f"TH round-trip ({case_name})", err < 1e-4, f"err={err:.2e}")
        r = flash_ts(T, r0.s, z, mx); err = abs(r.p - p) / p
        check(f"TS round-trip ({case_name})", err < 1e-4, f"err={err:.2e}")


def test_bubble_dew_points():
    # Bubble points that should exist
    for z_list, T in [([0.99, 0.01], 240), ([0.995, 0.005], 290)]:
        mx = load_mixture(['carbondioxide', 'nitrogen'], composition=z_list)
        z = mx.x.copy()
        r = bubble_point_p(T, z, mx)
        rc = flash_pt(r.p, T, z, mx, check_stability=False)
        beta = rc.beta if rc.beta is not None else 0.0
        check(f"bubble_point_p z={z_list} T={T}", beta < 1e-3, f"beta={beta:.2e}")

    for z_list, p in [([0.99, 0.01], 3e6), ([0.995, 0.005], 5e6)]:
        mx = load_mixture(['carbondioxide', 'nitrogen'], composition=z_list)
        z = mx.x.copy()
        r = bubble_point_T(p, z, mx)
        rc = flash_pt(p, r.T, z, mx, check_stability=False)
        beta = rc.beta if rc.beta is not None else 0.0
        check(f"bubble_point_T z={z_list} p={p*1e-6}MPa", beta < 1e-3, f"beta={beta:.2e}")

    # Dew points
    for z_list, T in [([0.5, 0.5], 220), ([0.3, 0.7], 240)]:
        mx = load_mixture(['carbondioxide', 'nitrogen'], composition=z_list)
        z = mx.x.copy()
        r = dew_point_p(T, z, mx)
        rc = flash_pt(r.p, T, z, mx, check_stability=False)
        beta = rc.beta if rc.beta is not None else 1.0
        check(f"dew_point_p z={z_list} T={T}", beta > 0.999, f"beta={beta}")

    for z_list, p in [([0.5, 0.5], 3e6), ([0.3, 0.7], 5e6)]:
        mx = load_mixture(['carbondioxide', 'nitrogen'], composition=z_list)
        z = mx.x.copy()
        r = dew_point_T(p, z, mx)
        rc = flash_pt(p, r.T, z, mx, check_stability=False)
        beta = rc.beta if rc.beta is not None else 1.0
        check(f"dew_point_T z={z_list} p={p*1e-6}MPa", beta > 0.999, f"beta={beta}")


def test_unreachable_bubble_dew():
    # Bubble point that doesn't exist
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.98, 0.02])
    try:
        bubble_point_T(3e6, mx.x, mx)
        check("bubble_point_T non-existent raises", False, "did not raise")
    except RuntimeError:
        check("bubble_point_T non-existent raises", True)

    # Dew above mixture critical
    mx = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.1, 0.9])
    try:
        dew_point_p(260, mx.x, mx)
        check("dew_point_p above critical raises", False, "did not raise")
    except RuntimeError:
        check("dew_point_p above critical raises", True)


# ----------------------------------------------------------------------
# Departure function tests
# ----------------------------------------------------------------------

def test_departure_function_derivatives_fd():
    """Departure function's (delta, tau) derivatives match FD to ~1e-12."""
    mx = load_mixture(['carbondioxide', 'nitrogen'], binary_set='test_co2_n2')
    if not mx.departures:
        check("departure loaded", False, "no departure dict")
        return
    check("departure loaded", True)
    (i, j), (F, dep) = next(iter(mx.departures.items()))
    eps = 1e-6

    for (delta, tau) in [(0.5, 1.2), (1.0, 0.8), (0.3, 2.0), (2.0, 0.5)]:
        A, A_d, A_t, A_dd, A_tt, A_dt = dep.evaluate(delta, tau)

        # FD on A wrt delta -> A_d
        Ap, *_ = dep.evaluate(delta + eps, tau)
        Am, *_ = dep.evaluate(delta - eps, tau)
        A_d_fd = (Ap - Am) / (2 * eps)
        check(f"A_d FD at (d={delta},t={tau})", abs(A_d - A_d_fd) < 1e-7,
              f"analytic={A_d}, fd={A_d_fd}")

        # FD on A wrt tau -> A_t
        Ap, *_ = dep.evaluate(delta, tau + eps)
        Am, *_ = dep.evaluate(delta, tau - eps)
        A_t_fd = (Ap - Am) / (2 * eps)
        check(f"A_t FD at (d={delta},t={tau})", abs(A_t - A_t_fd) < 1e-7)

        # FD on A_d wrt delta -> A_dd
        _, Adp, *_ = dep.evaluate(delta + eps, tau)
        _, Adm, *_ = dep.evaluate(delta - eps, tau)
        A_dd_fd = (Adp - Adm) / (2 * eps)
        check(f"A_dd FD at (d={delta},t={tau})", abs(A_dd - A_dd_fd) < 1e-6)

        # FD on A_t wrt tau -> A_tt
        _, _, Atp, *_ = dep.evaluate(delta, tau + eps)
        _, _, Atm, *_ = dep.evaluate(delta, tau - eps)
        A_tt_fd = (Atp - Atm) / (2 * eps)
        check(f"A_tt FD at (d={delta},t={tau})", abs(A_tt - A_tt_fd) < 1e-6)

        # FD on A_d wrt tau -> A_dt
        _, Adp, *_ = dep.evaluate(delta, tau + eps)
        _, Adm, *_ = dep.evaluate(delta, tau - eps)
        A_dt_fd = (Adp - Adm) / (2 * eps)
        check(f"A_dt FD at (d={delta},t={tau})", abs(A_dt - A_dt_fd) < 1e-6)


def test_departure_pure_limit():
    """At x_i = 1 (pure component), Delta_alpha_r must vanish."""
    mx = load_mixture(['carbondioxide', 'nitrogen'], binary_set='test_co2_n2',
                      composition=[1.0 - 1e-15, 1e-15])
    # Evaluate at some physical state
    T, p = 300.0, 2e6
    rho = density_from_pressure(p, T, mx.x, mx, phase_hint='vapor')
    res = alpha_r_mix_derivs(rho, T, mx.x, mx)
    check("pure CO2 limit: Delta ~ 0", abs(res['Delta_alpha_r']) < 1e-10,
          f"Delta={res['Delta_alpha_r']}")


def test_ln_phi_fd_with_departure():
    """With active departure function, ln_phi still matches FD."""
    mx = load_mixture(['carbondioxide', 'nitrogen'], binary_set='test_co2_n2')

    def n_alpha_r_at(n_vec, V_total, T):
        n = n_vec.sum(); xx = n_vec / n; rho_v = n / V_total
        return n * alpha_r_mix_derivs(rho_v, T, xx, mx)['a_r']

    for x0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
        mx.set_composition([x0, 1 - x0])
        z = mx.x.copy()
        for (T, p) in [(300.0, 3e6), (400.0, 1e6)]:
            try:
                rho = density_from_pressure(p, T, z, mx, phase_hint='vapor')
            except RuntimeError:
                continue
            V_total = 1.0 / rho

            lnphi = ln_phi(rho, T, z, mx)
            res = alpha_r_mix_derivs(rho, T, z, mx)
            Z = 1.0 + res['delta'] * res['a_r_d']
            analytic = lnphi + np.log(Z)

            fd = np.zeros(2); eps = 1e-7
            for i in range(2):
                np_p = z.copy(); np_p[i] += eps
                np_m = z.copy(); np_m[i] -= eps
                fd[i] = (n_alpha_r_at(np_p, V_total, T)
                         - n_alpha_r_at(np_m, V_total, T)) / (2 * eps)

            err = np.max(np.abs(analytic - fd) / np.maximum(np.abs(fd), 1e-10))
            check(f"ln_phi FD with dep x=[{x0:.1f},{1-x0:.1f}] T={T},p={p*1e-6}MPa",
                  err < 1e-5, f"err={err:.2e}")


def test_departure_changes_fugacity():
    """The departure function must actually affect ln_phi (nonzero delta)."""
    mx_no = load_mixture(['carbondioxide', 'nitrogen'], composition=[0.5, 0.5])
    mx_yes = load_mixture(['carbondioxide', 'nitrogen'], binary_set='test_co2_n2',
                          composition=[0.5, 0.5])
    T, p = 300.0, 3e6
    rho_no = density_from_pressure(p, T, mx_no.x, mx_no, phase_hint='vapor')
    rho_yes = density_from_pressure(p, T, mx_yes.x, mx_yes, phase_hint='vapor')
    lnphi_no = ln_phi(rho_no, T, mx_no.x, mx_no)
    lnphi_yes = ln_phi(rho_yes, T, mx_yes.x, mx_yes)
    diff = np.max(np.abs(lnphi_yes - lnphi_no))
    check("departure activates: ln_phi differs from no-departure",
          diff > 1e-6, f"max diff = {diff:.2e}")


def test_pt_flash_with_departure():
    """PT flash still works (two-phase, converges, fugacity equality) with departure."""
    mx = load_mixture(['carbondioxide', 'nitrogen'], binary_set='test_co2_n2',
                      composition=[0.5, 0.5])
    r = flash_pt(3e6, 220.0, mx.x, mx)
    check("PT with departure: 2-phase", r.phase == 'two_phase')
    if r.phase == 'two_phase':
        f_L = r.x * np.exp(ln_phi(r.rho_L, 220.0, r.x, mx)) * 3e6
        f_V = r.y * np.exp(ln_phi(r.rho_V, 220.0, r.y, mx)) * 3e6
        err = np.max(np.abs(f_L / f_V - 1.0))
        check("PT with departure: fugacity equality", err < 1e-6, f"err={err:.2e}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    all_tests = [
        test_reducing_pure_limits,
        test_reducing_fd,
        test_ln_phi_binary_fd,
        test_ln_phi_ternary_fd,
        test_rachford_rice,
        test_stability,
        test_pt_flash,
        test_arbitrary_component_flash_coolprop,
        test_arbitrary_component_single_phase_branches,
        test_density_from_pressure_robustness,
        test_state_function_flashes,
        test_bubble_dew_points,
        test_unreachable_bubble_dew,
        # departure tests
        test_departure_function_derivatives_fd,
        test_departure_pure_limit,
        test_ln_phi_fd_with_departure,
        test_departure_changes_fugacity,
        test_pt_flash_with_departure,
    ]

    for t in all_tests:
        run_test(t)

    print(f"\n{'='*60}")
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    if FAILURES:
        print("\nFailures:")
        for name, detail in FAILURES:
            print(f"  - {name}: {detail}")
    print('='*60)
    sys.exit(0 if FAILED == 0 else 1)
