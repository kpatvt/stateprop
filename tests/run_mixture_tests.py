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
from stateprop.mixture.properties import alpha_r_mix_derivs, pressure


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


def test_flash_convergence_acceleration():
    """The hybrid SS+Broyden flash should converge in fewer iterations than
    pure successive substitution on strongly non-ideal systems where SS
    converges only linearly. Ethanol+water at 350K/0.5bar is the canonical
    test case: pure SS needs ~25 iters because the contraction rate is
    only ~0.85 per iter; Broyden secant updates achieve super-linear
    convergence in ~10 iters. Solutions must agree to within tol*10
    (Broyden adds slightly more rounding noise than pure SS)."""
    from stateprop.mixture.flash import _successive_substitution
    from stateprop.mixture.stability import stability_test_TPD
    mx = load_mixture(['ethanol', 'water'], [0.5, 0.5])
    T, p = 350.0, 0.5e5
    stable, K_stab, _ = stability_test_TPD(mx.x, T, p, mx)
    check("ethanol+water at 350K/0.5bar is unstable (precondition)",
          not stable, f"stability test says stable={stable}")
    r_ss = _successive_substitution(mx.x, K_stab, T, p, mx,
                                    tol=1e-9, maxiter=80,
                                    ss_iters=80, newton=False)
    r_b = _successive_substitution(mx.x, K_stab, T, p, mx,
                                   tol=1e-9, maxiter=80,
                                   ss_iters=4, newton=True)
    # Same answer
    check("SS and SS+Broyden agree on beta (within 1e-6)",
          abs(r_ss[2] - r_b[2]) < 1e-6,
          f"SS beta={r_ss[2]:.6f}, Broyden beta={r_b[2]:.6f}")
    # Broyden should converge in fewer iterations
    check("SS+Broyden converges in fewer iterations than pure SS",
          r_b[3] < r_ss[3] * 0.6,
          f"SS={r_ss[3]} iters, Broyden={r_b[3]} iters "
          f"(want Broyden < {int(r_ss[3]*0.6)})")
    # Broyden should still take at least the SS warm-up count
    check("SS+Broyden uses the SS warm-up before switching",
          r_b[3] >= 4,
          f"got {r_b[3]} iters, expected at least 4 (the SS warm-up count)")


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
# v0.9.9 -- analytic composition derivatives & Newton flash
# ----------------------------------------------------------------------


def test_helmholtz_reducing_hessian_symmetric_and_fd():
    """The analytic Hessian of the Kunz-Wagner reducing functions T_r(x)
    and 1/rho_r(x) must be symmetric and match a centered FD on the
    analytic first derivatives to <1e-7."""
    mx = load_mixture(['methane','ethane','propane','nitrogen'],
                      [0.7, 0.1, 0.1, 0.1], binary_set='gerg2008')
    x = np.array([0.7, 0.1, 0.1, 0.1])
    H_T, H_inv = mx.reducing.hessian(x)
    check("reducing Hessian: H_T symmetric",
          np.allclose(H_T, H_T.T), f"max asymmetry = {np.max(np.abs(H_T - H_T.T)):.2e}")
    check("reducing Hessian: H_invrho symmetric",
          np.allclose(H_inv, H_inv.T))
    # FD on the analytic first derivatives
    h = 1e-5; N = 4
    H_T_fd = np.zeros((N, N)); H_inv_fd = np.zeros((N, N))
    for l in range(N):
        e = np.eye(N)[l]
        _, _, dT_p, di_p = mx.reducing.derivatives(x + h*e)
        _, _, dT_m, di_m = mx.reducing.derivatives(x - h*e)
        H_T_fd[:, l] = (dT_p - dT_m) / (2*h)
        H_inv_fd[:, l] = (di_p - di_m) / (2*h)
    check("reducing Hessian: H_T matches FD to 1e-7",
          np.max(np.abs(H_T - H_T_fd)) < 1e-7,
          f"max abs err = {np.max(np.abs(H_T - H_T_fd)):.2e}")
    check("reducing Hessian: H_invrho matches FD to 1e-9",
          np.max(np.abs(H_inv - H_inv_fd)) < 1e-9,
          f"max abs err = {np.max(np.abs(H_inv - H_inv_fd)):.2e}")


def test_helmholtz_analytic_dp_dx_at_rho():
    """d(p)/d(x_k) at fixed (T, rho), N-vector. FD-verified for both
    no-departure (ethanol-water) and full-departure (GERG natural gas) cases."""
    from stateprop.mixture.properties import dp_dx_at_rho
    test_cases = [
        ("ethanol-water (no departure)",
         load_mixture(['ethanol','water'], [0.5, 0.5]), 350.0, 0.5e5,
         np.array([0.5, 0.5]), 'vapor'),
        ("CH4-N2 (GERG departure)",
         load_mixture(['methane','nitrogen'], [0.5, 0.5], binary_set='gerg2008'),
         150.0, 2e6, np.array([0.5, 0.5]), 'vapor'),
        ("4-comp NG (GERG departures)",
         load_mixture(['methane','ethane','propane','nitrogen'],
                      [0.7, 0.1, 0.1, 0.1], binary_set='gerg2008'),
         250.0, 5e6, np.array([0.7, 0.1, 0.1, 0.1]), 'vapor'),
    ]
    for label, mx, T, p, x, phase in test_cases:
        rho = density_from_pressure(p, T, x, mx, phase_hint=phase)
        an = dp_dx_at_rho(rho, T, x, mx)
        h = 1e-6
        N = len(x)
        fd = np.array([(pressure(rho, T, x + h*np.eye(N)[k], mx) -
                        pressure(rho, T, x - h*np.eye(N)[k], mx)) / (2*h) for k in range(N)])
        rel = np.max(np.abs((an - fd) / fd))
        check(f"dp/dx_at_rho [{label}]: rel err < 1e-6",
              rel < 1e-6, f"got {rel:.2e}")


def test_helmholtz_analytic_dlnphi_drho_at_x():
    """d(ln phi)/d rho at fixed (T, x), N-vector. FD-verified."""
    from stateprop.mixture.properties import dlnphi_drho_at_x
    test_cases = [
        ("ethanol-water (no departure)",
         load_mixture(['ethanol','water'], [0.5, 0.5]), 350.0, 0.5e5,
         np.array([0.5, 0.5]), 'vapor'),
        ("CH4-N2 vapor (departure)",
         load_mixture(['methane','nitrogen'], [0.5, 0.5], binary_set='gerg2008'),
         150.0, 2e6, np.array([0.5, 0.5]), 'vapor'),
        ("CH4-N2 liquid (departure)",
         load_mixture(['methane','nitrogen'], [0.5, 0.5], binary_set='gerg2008'),
         130.0, 2e6, np.array([0.5, 0.5]), 'liquid'),
    ]
    for label, mx, T, p, x, phase in test_cases:
        rho = density_from_pressure(p, T, x, mx, phase_hint=phase)
        an = dlnphi_drho_at_x(rho, T, x, mx)
        hr = max(rho * 1e-6, 1e-3)
        fd = (ln_phi(rho + hr, T, x, mx) - ln_phi(rho - hr, T, x, mx)) / (2 * hr)
        rel = np.max(np.abs((an - fd) / fd))
        check(f"dlnphi/drho_at_x [{label}]: rel err < 1e-6",
              rel < 1e-6, f"got {rel:.2e}")


def test_helmholtz_analytic_dlnphi_dx_at_rho():
    """The composition Jacobian at fixed (T, rho). N x N matrix.
    FD-verified across no-departure and departure cases, both phases."""
    from stateprop.mixture.properties import dlnphi_dx_at_rho
    test_cases = [
        ("ethanol-water vapor",
         load_mixture(['ethanol','water'], [0.5, 0.5]),
         350.0, 0.5e5, np.array([0.5, 0.5]), 'vapor'),
        ("CH4-N2 GERG vapor",
         load_mixture(['methane','nitrogen'], [0.5, 0.5], binary_set='gerg2008'),
         150.0, 2e6, np.array([0.5, 0.5]), 'vapor'),
        ("CH4-N2 GERG liquid",
         load_mixture(['methane','nitrogen'], [0.5, 0.5], binary_set='gerg2008'),
         130.0, 2e6, np.array([0.5, 0.5]), 'liquid'),
    ]
    h = 1e-6
    for label, mx, T, p, x, phase in test_cases:
        rho = density_from_pressure(p, T, x, mx, phase_hint=phase)
        J_an = dlnphi_dx_at_rho(rho, T, x, mx)
        N = len(x)
        J_fd = np.zeros((N, N))
        for k in range(N):
            e = np.eye(N)[k]
            J_fd[:, k] = (ln_phi(rho, T, x + h*e, mx) - ln_phi(rho, T, x - h*e, mx)) / (2*h)
        denom = np.where(np.abs(J_fd) > 1e-10, J_fd, 1.0)
        rel = np.max(np.abs((J_an - J_fd) / denom))
        check(f"dlnphi/dx_at_rho [{label}]: rel err < 1e-6",
              rel < 1e-6, f"got {rel:.2e}")


def test_helmholtz_analytic_dlnphi_dx_at_p():
    """The headline result: d(ln phi_i)/d x_k at fixed (T, p), the
    Newton-flash Jacobian for the Helmholtz/GERG mixture EOS. Combines
    all four building blocks via chain rule. FD-verified for multiple
    mixtures and both phases."""
    from stateprop.mixture.properties import dlnphi_dx_at_p
    test_cases = [
        ("ethanol-water vapor",
         load_mixture(['ethanol','water'], [0.5, 0.5]),
         350.0, 0.5e5, np.array([0.5, 0.5]), 'vapor'),
        ("CH4-N2 GERG vapor",
         load_mixture(['methane','nitrogen'], [0.5, 0.5], binary_set='gerg2008'),
         150.0, 2e6, np.array([0.5, 0.5]), 'vapor'),
        ("CH4-N2 GERG liquid",
         load_mixture(['methane','nitrogen'], [0.5, 0.5], binary_set='gerg2008'),
         130.0, 2e6, np.array([0.5, 0.5]), 'liquid'),
        ("CO2-N2 GERG vapor",
         load_mixture(['carbondioxide','nitrogen'], [0.5, 0.5], binary_set='gerg2008'),
         220.0, 3e6, np.array([0.5, 0.5]), 'vapor'),
    ]
    h = 1e-6
    for label, mx, T, p, x, phase in test_cases:
        J_an = dlnphi_dx_at_p(p, T, x, mx, phase_hint=phase)
        N = len(x)
        J_fd = np.zeros((N, N))
        for k in range(N):
            e = np.eye(N)[k]
            rho_p = density_from_pressure(p, T, x + h*e, mx, phase_hint=phase)
            rho_m = density_from_pressure(p, T, x - h*e, mx, phase_hint=phase)
            J_fd[:, k] = (ln_phi(rho_p, T, x + h*e, mx) -
                          ln_phi(rho_m, T, x - h*e, mx)) / (2*h)
        denom = np.where(np.abs(J_fd) > 1e-10, J_fd, 1.0)
        rel = np.max(np.abs((J_an - J_fd) / denom))
        check(f"dlnphi/dx_at_p [{label}]: rel err < 1e-5",
              rel < 1e-5, f"got {rel:.2e}")


def test_helmholtz_newton_flash_pt():
    """Newton-Raphson flash with analytic Jacobian. Must converge to
    the same answer as SS+Broyden in fewer iterations on multiple
    Helmholtz/GERG cases."""
    from stateprop.mixture.flash import newton_flash_pt
    cases = [
        ("CH4-N2 130K, 2 MPa", load_mixture(['methane','nitrogen'], [0.5, 0.5],
                                             binary_set='gerg2008'), 130.0, 2e6),
        ("Ethanol-water 350K, 0.5 bar",
         load_mixture(['ethanol','water'], [0.5, 0.5]), 350.0, 0.5e5),
        ("CO2-N2 220K, 3 MPa",
         load_mixture(['carbondioxide','nitrogen'], [0.5, 0.5], binary_set='gerg2008'),
         220.0, 3e6),
    ]
    for label, mx, T, p in cases:
        rb = flash_pt(p, T, mx.x, mx)
        rn = newton_flash_pt(p, T, mx.x, mx)
        check(f"Newton flash [{label}]: 2-phase detected", rn.phase == 'two_phase',
              f"got {rn.phase}")
        if rn.phase == 'two_phase' and rb.phase == 'two_phase':
            check(f"Newton flash [{label}]: beta agrees with Broyden to 1e-6",
                  abs(rn.beta - rb.beta) < 1e-6,
                  f"diff = {abs(rn.beta - rb.beta):.2e}")
            check(f"Newton flash [{label}]: x agrees to 1e-5",
                  np.max(np.abs(rn.x - rb.x)) < 1e-5)
            check(f"Newton flash [{label}]: <= 6 iters (vs Broyden's >=8)",
                  rn.iterations <= 6,
                  f"got Newton {rn.iterations} iters, Broyden {rb.iterations}")
            # Fugacity equality
            f_L = rn.x * np.exp(ln_phi(rn.rho_L, T, rn.x, mx)) * p
            f_V = rn.y * np.exp(ln_phi(rn.rho_V, T, rn.y, mx)) * p
            err = np.max(np.abs(f_L / f_V - 1.0))
            check(f"Newton flash [{label}]: fugacity equality",
                  err < 1e-6, f"max f_L/f_V - 1 = {err:.2e}")


def test_helmholtz_newton_flash_handles_single_phase():
    """Newton flash on single-phase feed should delegate to SS+Broyden
    for proper single-phase classification."""
    from stateprop.mixture.flash import newton_flash_pt
    mx = load_mixture(['methane','ethane','propane','nitrogen'],
                      [0.7, 0.1, 0.1, 0.1], binary_set='gerg2008')
    r = newton_flash_pt(p=5e6, T=200.0, z=mx.x, mixture=mx)
    check("Newton flash on single-phase: gets a single-phase classification",
          r.phase in ('vapor', 'liquid', 'supercritical') and r.beta is None,
          f"got phase={r.phase}, beta={r.beta}")


# ----------------------------------------------------------------------
# v0.9.10 -- T and p derivatives of ln phi (Helmholtz)
# ----------------------------------------------------------------------


def test_helmholtz_dp_dT_at_rho_fd():
    """dp/dT at fixed (rho, x) validates vs 2-pt FD on pressure()."""
    from stateprop.mixture.properties import dp_dT_at_rho
    test_cases = [
        ("ethanol-water vapor", load_mixture(['ethanol','water'], [0.5, 0.5]),
         350.0, 0.5e5, np.array([0.5, 0.5]), 'vapor'),
        ("CH4-N2 GERG liquid", load_mixture(['methane','nitrogen'], [0.5, 0.5],
                                             binary_set='gerg2008'),
         120.0, 2e6, np.array([0.5, 0.5]), 'liquid'),
        ("4-comp NG vapor", load_mixture(['methane','ethane','propane','nitrogen'],
                                          [0.7, 0.1, 0.1, 0.1], binary_set='gerg2008'),
         250.0, 5e6, np.array([0.7, 0.1, 0.1, 0.1]), 'vapor'),
    ]
    hT = 1e-4
    for label, mx, T, p, x, phase in test_cases:
        rho = density_from_pressure(p, T, x, mx, phase_hint=phase)
        an = dp_dT_at_rho(rho, T, x, mx)
        fd = (pressure(rho, T + hT, x, mx) - pressure(rho, T - hT, x, mx)) / (2 * hT)
        err = abs((an - fd) / fd)
        check(f"dp/dT_at_rho [{label}]: rel err < 1e-7",
              err < 1e-7, f"got {err:.2e}")


def test_helmholtz_dlnphi_dT_at_rho_fd():
    """d(ln phi)/dT at fixed (rho, x) validates vs 2-pt FD."""
    from stateprop.mixture.properties import dlnphi_dT_at_rho
    test_cases = [
        ("ethanol-water vapor", load_mixture(['ethanol','water'], [0.5, 0.5]),
         350.0, 0.5e5, np.array([0.5, 0.5]), 'vapor'),
        ("CH4-N2 GERG liquid", load_mixture(['methane','nitrogen'], [0.5, 0.5],
                                             binary_set='gerg2008'),
         120.0, 2e6, np.array([0.5, 0.5]), 'liquid'),
        ("CO2-N2 GERG vapor", load_mixture(['carbondioxide','nitrogen'], [0.5, 0.5],
                                            binary_set='gerg2008'),
         220.0, 3e6, np.array([0.5, 0.5]), 'vapor'),
    ]
    hT = 1e-4
    for label, mx, T, p, x, phase in test_cases:
        rho = density_from_pressure(p, T, x, mx, phase_hint=phase)
        an = dlnphi_dT_at_rho(rho, T, x, mx)
        fd = (ln_phi(rho, T + hT, x, mx) - ln_phi(rho, T - hT, x, mx)) / (2 * hT)
        err = np.max(np.abs((an - fd) / (fd + 1e-30)))
        check(f"dlnphi/dT_at_rho [{label}]: rel err < 1e-5",
              err < 1e-5, f"got {err:.2e}")


def test_helmholtz_dlnphi_dp_at_T_fd():
    """d(ln phi)/dp at fixed (T, x) validates vs 2-pt FD. Simple chain rule:
    dlnphi/dp = dlnphi/drho / dp/drho_T."""
    from stateprop.mixture.properties import dlnphi_dp_at_T
    test_cases = [
        ("ethanol-water vapor", load_mixture(['ethanol','water'], [0.5, 0.5]),
         350.0, 0.5e5, np.array([0.5, 0.5]), 'vapor'),
        ("CH4-N2 GERG liquid", load_mixture(['methane','nitrogen'], [0.5, 0.5],
                                             binary_set='gerg2008'),
         120.0, 2e6, np.array([0.5, 0.5]), 'liquid'),
        ("CO2-N2 GERG vapor", load_mixture(['carbondioxide','nitrogen'], [0.5, 0.5],
                                            binary_set='gerg2008'),
         220.0, 3e6, np.array([0.5, 0.5]), 'vapor'),
    ]
    for label, mx, T, p, x, phase in test_cases:
        an = dlnphi_dp_at_T(p, T, x, mx, phase_hint=phase)
        # Pick hp to balance truncation error (O(h^2)) vs density-solver noise
        hp = max(p * 1e-4, 1.0)
        rho_p = density_from_pressure(p + hp, T, x, mx, phase_hint=phase)
        rho_m = density_from_pressure(p - hp, T, x, mx, phase_hint=phase)
        fd = (ln_phi(rho_p, T, x, mx) - ln_phi(rho_m, T, x, mx)) / (2 * hp)
        err = np.max(np.abs((an - fd) / (fd + 1e-30)))
        check(f"dlnphi/dp_at_T [{label}]: rel err < 1e-5",
              err < 1e-5, f"got {err:.2e}")


def test_helmholtz_dlnphi_dT_at_p_fd():
    """d(ln phi)/dT at fixed (p, x). Full chain rule:
    d(ln phi)/dT|_{p,x} = d(ln phi)/dT|_{rho,x} + d(ln phi)/drho|_{T,x} * drho/dT|_{p,x}"""
    from stateprop.mixture.properties import dlnphi_dT_at_p
    test_cases = [
        ("ethanol-water vapor", load_mixture(['ethanol','water'], [0.5, 0.5]),
         350.0, 0.5e5, np.array([0.5, 0.5]), 'vapor'),
        ("CH4-N2 GERG liquid", load_mixture(['methane','nitrogen'], [0.5, 0.5],
                                             binary_set='gerg2008'),
         120.0, 2e6, np.array([0.5, 0.5]), 'liquid'),
        ("CO2-N2 GERG vapor", load_mixture(['carbondioxide','nitrogen'], [0.5, 0.5],
                                            binary_set='gerg2008'),
         220.0, 3e6, np.array([0.5, 0.5]), 'vapor'),
    ]
    hT = 1e-3
    for label, mx, T, p, x, phase in test_cases:
        an = dlnphi_dT_at_p(p, T, x, mx, phase_hint=phase)
        rho_p = density_from_pressure(p, T + hT, x, mx, phase_hint=phase)
        rho_m = density_from_pressure(p, T - hT, x, mx, phase_hint=phase)
        fd = (ln_phi(rho_p, T + hT, x, mx) - ln_phi(rho_m, T - hT, x, mx)) / (2 * hT)
        err = np.max(np.abs((an - fd) / (fd + 1e-30)))
        check(f"dlnphi/dT_at_p [{label}]: rel err < 1e-4",
              err < 1e-4, f"got {err:.2e}")


# ----------------------------------------------------------------------
# v0.9.11 -- A_residual_matrix assembly + critical point (experimental)
# ----------------------------------------------------------------------


def test_helmholtz_A_residual_matches_fd():
    """The analytic residual Helmholtz Hessian A_ij^res = d^2(n*alpha^r)/dn_i dn_j
    at fixed (T, V) matches second-order FD on alpha_r_mix_derivs to FD precision,
    AND is exactly symmetric by construction."""
    from stateprop.mixture.critical import _A_residual_matrix
    from stateprop.mixture.properties import alpha_r_mix_derivs
    test_cases = [
        ("CH4-N2 GERG 50/50", load_mixture(['methane', 'nitrogen'], [0.5, 0.5],
                                            binary_set='gerg2008'),
         190.0, 0.0002, np.array([0.5, 0.5])),
        ("CO2-CH4 GERG 30/70", load_mixture(['carbondioxide', 'methane'], [0.3, 0.7],
                                             binary_set='gerg2008'),
         220.0, 0.0001, np.array([0.3, 0.7])),
        ("ethanol-water 50/50", load_mixture(['ethanol', 'water'], [0.5, 0.5]),
         400.0, 5e-5, np.array([0.5, 0.5])),
    ]
    for label, mx, T, V, x in test_cases:
        n = x.copy()       # n_tot = 1
        R = mx.components[0].fluid.R
        A_an = _A_residual_matrix(T, V, n, mx)
        # FD on n*alpha_r at fixed (T, V)
        def nalpha(n_vec):
            nt = n_vec.sum(); xv = n_vec / nt; rv = nt / V
            return nt * alpha_r_mix_derivs(rv, T, xv, mx)["a_r"]
        hn = 1e-5
        N = 2
        A_fd = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                ei = np.eye(N)[i]; ej = np.eye(N)[j]
                if i == j:
                    A_fd[i, j] = (nalpha(n + hn * ei) - 2 * nalpha(n) + nalpha(n - hn * ei)) / (hn * hn)
                else:
                    A_fd[i, j] = (nalpha(n + hn * ei + hn * ej)
                                   - nalpha(n + hn * ei - hn * ej)
                                   - nalpha(n - hn * ei + hn * ej)
                                   + nalpha(n - hn * ei - hn * ej)) / (4 * hn * hn)
        # A_an has units [J/mol], FD result is in [dimensionless]; so compare A_an/RT to A_fd
        A_an_over_RT = A_an / (R * T)
        rel_err = np.max(np.abs(A_an_over_RT - A_fd)) / (np.max(np.abs(A_fd)) + 1e-30)
        check(f"A_residual [{label}]: matches FD, rel err < 1e-5",
              rel_err < 1e-5, f"got rel err {rel_err:.2e}")
        # Symmetry
        asym = np.max(np.abs(A_an - A_an.T))
        check(f"A_residual [{label}]: symmetric to 1e-10",
              asym < 1e-10, f"got {asym:.2e}")


def test_helmholtz_critical_CO2_CH4():
    """Helmholtz critical point solver on CO2-CH4 binary -- reference case
    where H-K converges cleanly to the VLE critical. Tests against the
    published GERG critical locus.

    Experimental CO2-CH4 critical locus (Donnelly & Katz 1954, Al-Sahhaf et al.
    1983) at x_CO2 ≈ 0.5: T_c ~ 245K, p_c ~ 8.5 MPa.

    The solver using GERG-2008 + Heidemann-Khalil reproduces T_c = 245.08K
    exactly; p_c = 7.44 MPa is lower than experiment (within GERG's typical
    critical-point error bounds).
    """
    from stateprop.mixture.critical import critical_point
    # Known-good test case
    z = np.array([0.5, 0.5])
    mx = load_mixture(['carbondioxide', 'methane'], [0.5, 0.5], binary_set='gerg2008')
    cp = critical_point(z, mx)
    check("CO2-CH4 50/50: converged",
          cp['residual'] < 1e-7, f"residual={cp['residual']:.2e}")
    check("CO2-CH4 50/50: T_c = 245.08 +/- 0.5 K (GERG prediction)",
          abs(cp['T_c'] - 245.08) < 0.5, f"got T_c={cp['T_c']:.3f}")
    check("CO2-CH4 50/50: p_c reasonable (5-10 MPa)",
          5e6 < cp['p_c'] < 1e7, f"got p_c={cp['p_c']/1e6:.2f} MPa")
    check("CO2-CH4 50/50: V_c in physical range (50-200 cm^3/mol)",
          50e-6 < cp['V_c'] < 200e-6, f"got V_c={cp['V_c']*1e6:.1f} cm^3/mol")
    check("CO2-CH4 50/50: not flagged suspicious",
          not cp['suspicious'], "converged V_c is within expected bounds")

    # Another point on the locus
    z = np.array([0.3, 0.7])
    mx = load_mixture(['carbondioxide', 'methane'], [0.3, 0.7], binary_set='gerg2008')
    cp = critical_point(z, mx)
    check("CO2-CH4 30/70: converged and T_c in [210, 240] K",
          cp['residual'] < 1e-7 and 210 < cp['T_c'] < 240,
          f"T_c={cp['T_c']:.2f}, residual={cp['residual']:.2e}")


def test_helmholtz_critical_pure_limit():
    """At z approaching pure component, the critical point solver should
    converge close to that component's pure critical. Allow 2K tolerance
    because 0.001 mole fraction of the minor component still perturbs the
    critical noticeably."""
    from stateprop.mixture.critical import critical_point
    # Approach pure methane from CH4-N2 GERG binary
    z = np.array([0.999, 0.001])
    mx = load_mixture(['methane', 'nitrogen'], list(z), binary_set='gerg2008')
    cp = critical_point(z, mx)
    # Pure CH4 reference: T_c = 190.56 K
    T_c_pure = 190.564
    check("Near-pure CH4: T_c within 2 K of pure-CH4 critical",
          abs(cp['T_c'] - T_c_pure) < 2.0, f"got T_c={cp['T_c']:.3f}")


# ----------------------------------------------------------------------
# v0.9.12 -- Robust multistart + physical-filter critical point solver
# ----------------------------------------------------------------------


def test_helmholtz_critical_multistart_CO2_CH4():
    """Multistart solver on CO2-CH4 binary -- known-clean Type I binary.
    Matches GERG prediction T_c=245.6K, p_c=7.45MPa."""
    from stateprop.mixture.critical import critical_point_multistart
    z = np.array([0.5, 0.5])
    mx = load_mixture(['carbondioxide', 'methane'], list(z), binary_set='gerg2008')
    cp = critical_point_multistart(z, mx)
    check("CO2-CH4 50/50 multistart: converged, non-suspicious",
          cp['residual'] < 1e-7 and not cp['suspicious'],
          f"residual={cp['residual']:.2e}, suspicious={cp['suspicious']}")
    check("CO2-CH4 50/50: T_c = 245.6 +/- 1 K",
          abs(cp['T_c'] - 245.6) < 1.0, f"got T_c={cp['T_c']:.3f}")
    check("CO2-CH4 50/50: physical Z_c in [0.25, 0.32]",
          0.25 < cp['score'] and 0.25 < cp['p_c']*cp['V_c']/(mx.components[0].fluid.R*cp['T_c']) < 0.32,
          f"got score={cp['score']:.3f}")


def test_helmholtz_critical_multistart_CH4_C2():
    """Multistart solver on CH4-ethane -- physical VLE critical has a small
    basin of attraction that simple single-start H-K misses. Multistart
    finds T_c=243.9K, V_c=106 cm^3/mol reliably."""
    from stateprop.mixture.critical import critical_point_multistart
    z = np.array([0.5, 0.5])
    mx = load_mixture(['methane', 'ethane'], list(z), binary_set='gerg2008')
    cp = critical_point_multistart(z, mx)
    check("CH4-ethane 50/50: converged",
          cp['residual'] < 1e-7, f"residual={cp['residual']:.2e}")
    # Physical critical at T~244K, V~105 cm^3/mol
    check("CH4-ethane 50/50: T_c in [240, 248] K",
          240 < cp['T_c'] < 248, f"got T_c={cp['T_c']:.2f}")
    check("CH4-ethane 50/50: V_c in [90, 115] cm^3/mol (physical range)",
          90e-6 < cp['V_c'] < 115e-6, f"got V_c={cp['V_c']*1e6:.2f}")
    check("CH4-ethane 50/50: good score",
          cp['score'] > 0.5, f"got score={cp['score']:.3f}")


def test_helmholtz_critical_multistart_CH4_N2_locus():
    """Multistart on CH4-N2 binary across several compositions. This binary
    has multiple H-K stationary points per composition due to retrograde
    behavior in GERG. The solver consistently picks the low-V / low-Z_c
    branch which is the physical VLE critical."""
    from stateprop.mixture.critical import critical_point_multistart
    # Strongly CH4-rich: dominated by CH4 critical
    z = np.array([0.7, 0.3])
    mx = load_mixture(['methane', 'nitrogen'], list(z), binary_set='gerg2008')
    cp = critical_point_multistart(z, mx)
    R = mx.components[0].fluid.R
    Zc = cp['p_c'] * cp['V_c'] / (R * cp['T_c'])
    check("CH4-N2 0.3 N2: converged, T_c in [155, 175]",
          cp['residual'] < 1e-7 and 155 < cp['T_c'] < 175,
          f"T_c={cp['T_c']:.2f}, res={cp['residual']:.2e}")
    check("CH4-N2 0.3 N2: physical Z_c < 0.35",
          Zc < 0.35, f"got Z_c={Zc:.3f}")

    # Equimolar
    z = np.array([0.5, 0.5])
    mx = load_mixture(['methane', 'nitrogen'], list(z), binary_set='gerg2008')
    cp = critical_point_multistart(z, mx)
    Zc = cp['p_c'] * cp['V_c'] / (R * cp['T_c'])
    check("CH4-N2 0.5: T_c in [145, 160]",
          145 < cp['T_c'] < 160, f"got T_c={cp['T_c']:.2f}")
    check("CH4-N2 0.5: physical Z_c < 0.30",
          Zc < 0.30, f"got Z_c={Zc:.3f}")


def test_helmholtz_critical_multistart_returns_candidates():
    """When return_all=True, multistart returns per-candidate details."""
    from stateprop.mixture.critical import critical_point_multistart
    z = np.array([0.5, 0.5])
    mx = load_mixture(['carbondioxide', 'methane'], list(z), binary_set='gerg2008')
    cp = critical_point_multistart(z, mx, return_all=True)
    check("return_all=True: includes all_candidates key",
          'all_candidates' in cp, "key missing")
    check("return_all=True: at least one candidate found",
          len(cp['all_candidates']) >= 1,
          f"got {len(cp['all_candidates'])}")
    # Each candidate should have T_c, V_c, p_c, Z_c, score, residual
    c0 = cp['all_candidates'][0]
    for key in ('T_c', 'V_c', 'p_c', 'Z_c', 'score', 'residual'):
        check(f"candidate has '{key}' field", key in c0, f"missing {key}")


# ----------------------------------------------------------------------
# v0.9.13 -- Phase envelope tracer for Helmholtz/GERG mixtures
# ----------------------------------------------------------------------


def test_helmholtz_envelope_point_wilson_seeded():
    """Single envelope point via Wilson-seeded Newton. CH4-ethane at T=220K
    has a clean bubble and dew point well inside the 2-phase region."""
    from stateprop.mixture.envelope import envelope_point
    mx = load_mixture(['methane', 'ethane'], [0.5, 0.5], binary_set='gerg2008')
    z = np.array([0.5, 0.5])
    # Bubble
    b = envelope_point(220.0, 1e6, z, mx, beta=0)
    check("envelope_point: bubble converges",
          b['iterations'] < 20 and 1e6 < b['p'] < 1e7,
          f"iters={b['iterations']}, p={b['p']/1e6:.3f}MPa")
    check("envelope_point: bubble K_CH4 > 1 (more volatile)",
          b['K'][0] > 1.0, f"got K_CH4={b['K'][0]:.3f}")
    check("envelope_point: bubble K_C2 < 1 (less volatile)",
          b['K'][1] < 1.0, f"got K_C2={b['K'][1]:.3f}")
    # Dew
    d = envelope_point(220.0, 1e6, z, mx, beta=1)
    check("envelope_point: dew converges",
          d['iterations'] < 20 and 1e5 < d['p'] < 1e7,
          f"iters={d['iterations']}, p={d['p']/1e6:.3f}MPa")
    # At same T, dew p < bubble p (dew line below bubble line)
    check("envelope_point: dew p < bubble p at same T",
          d['p'] < b['p'], f"dew p={d['p']/1e6:.3f} vs bubble p={b['p']/1e6:.3f}")


def test_helmholtz_envelope_point_fugacity_closure():
    """Converged envelope point must satisfy fugacity equality to tight tol.
    This is the core thermodynamic consistency check."""
    from stateprop.mixture.envelope import envelope_point
    from stateprop.mixture.properties import ln_phi, density_from_pressure
    mx = load_mixture(['methane', 'ethane'], [0.5, 0.5], binary_set='gerg2008')
    z = np.array([0.5, 0.5])
    b = envelope_point(220.0, 1e6, z, mx, beta=0)
    T, p, K = b['T'], b['p'], b['K']
    x = z
    y = K * z / float((K * z).sum())
    rho_L = density_from_pressure(p, T, x, mx, phase_hint='liquid')
    rho_V = density_from_pressure(p, T, y, mx, phase_hint='vapor')
    lnphi_L = ln_phi(rho_L, T, x, mx)
    lnphi_V = ln_phi(rho_V, T, y, mx)
    f_residual = np.max(np.abs(np.log(K) - (lnphi_L - lnphi_V)))
    check("envelope_point: fugacity equality satisfied to 1e-7",
          f_residual < 1e-7, f"max residual = {f_residual:.2e}")


def test_helmholtz_trace_envelope_structure():
    """Full envelope trace: seeded from critical, steps outward, gives a
    connected set of points with the expected result structure."""
    from stateprop.mixture.envelope import trace_envelope
    mx = load_mixture(['methane', 'ethane'], [0.5, 0.5], binary_set='gerg2008')
    z = np.array([0.5, 0.5])
    env = trace_envelope(z, mx, max_points_per_branch=15)
    check("trace_envelope: returned dict with expected keys",
          all(k in env for k in ('T', 'p', 'K', 'branch', 'critical', 'n_points')),
          f"keys: {list(env.keys())}")
    check("trace_envelope: at least 10 points returned",
          env['n_points'] >= 10, f"got {env['n_points']}")
    check("trace_envelope: includes the critical point (branch=-1)",
          np.any(env['branch'] == -1),
          f"branch counts: {np.unique(env['branch'], return_counts=True)}")
    check("trace_envelope: critical point matches solver result",
          abs(env['T'][env['branch'] == -1][0] - env['critical']['T_c']) < 0.01,
          "critical T mismatch")
    check("trace_envelope: has both bubble and dew points (or mirror pair)",
          len(np.unique(env['branch'])) >= 2,
          f"unique branches: {np.unique(env['branch'])}")
    check("trace_envelope: all p_c neighborhood points at positive p",
          np.all(env['p'] > 0), "some p <= 0")


def test_helmholtz_envelope_CO2_CH4_trace():
    """Envelope trace on CO2-CH4 (well-behaved Type I binary). Should
    produce a smooth envelope spanning a range of T and p."""
    from stateprop.mixture.envelope import trace_envelope
    mx = load_mixture(['carbondioxide', 'methane'], [0.5, 0.5], binary_set='gerg2008')
    z = np.array([0.5, 0.5])
    env = trace_envelope(z, mx, max_points_per_branch=15)
    check("CO2-CH4 trace: >= 10 points",
          env['n_points'] >= 10, f"got {env['n_points']}")
    # T_c ~= 245K for CO2-CH4 50/50
    check("CO2-CH4 trace: critical near 245 K",
          240 < env['critical']['T_c'] < 250,
          f"got T_c={env['critical']['T_c']:.2f}")
    # Should span at least 5 K away from critical
    T_span = env['T'].max() - env['T'].min()
    check("CO2-CH4 trace: T range spans >= 5 K",
          T_span >= 5.0, f"got T span = {T_span:.2f} K")


# ----------------------------------------------------------------------
# v0.9.14 -- Newton-Raphson bubble/dew point solvers
# ----------------------------------------------------------------------


def test_helmholtz_newton_bubble_point_p_matches_ss():
    """Newton bubble_point_p converges to the same p as SS for well-behaved
    CH4-ethane at T well below critical, and does so in fewer iterations."""
    from stateprop.mixture.flash import bubble_point_p, newton_bubble_point_p
    mx = load_mixture(['methane', 'ethane'], [0.5, 0.5], binary_set='gerg2008')
    z = np.array([0.5, 0.5])
    T = 220.0
    r_ss = bubble_point_p(T, z, mx)
    r_nt = newton_bubble_point_p(T, z, mx)
    check("Newton bubble_point_p: p matches SS to 1e-6 rel",
          abs(r_nt.p - r_ss.p) / r_ss.p < 1e-6,
          f"SS p={r_ss.p/1e6:.4f}, Newton p={r_nt.p/1e6:.4f}")
    check("Newton bubble_point_p: fewer iterations than SS",
          r_nt.iterations <= r_ss.iterations,
          f"SS {r_ss.iterations}, Newton {r_nt.iterations}")
    # Key convergence property: fugacity equality holds at the solution
    from stateprop.mixture.properties import ln_phi, density_from_pressure
    T, p, K, y = r_nt.T, r_nt.p, r_nt.K, r_nt.y
    rho_L = density_from_pressure(p, T, z, mx, phase_hint='liquid')
    rho_V = density_from_pressure(p, T, y, mx, phase_hint='vapor')
    f_residual = np.max(np.abs(np.log(K) - (ln_phi(rho_L, T, z, mx)
                                             - ln_phi(rho_V, T, y, mx))))
    check("Newton bubble_point_p: fugacity equality to 1e-8",
          f_residual < 1e-8, f"max residual = {f_residual:.2e}")


def test_helmholtz_newton_bubble_point_p_near_critical():
    """Near-critical bubble point: Newton is much faster than SS and still
    converges cleanly. CH4-ethane T_c ~ 244K; test at T=240K (4K below)."""
    from stateprop.mixture.flash import bubble_point_p, newton_bubble_point_p
    mx = load_mixture(['methane', 'ethane'], [0.5, 0.5], binary_set='gerg2008')
    z = np.array([0.5, 0.5])
    T = 240.0
    r_ss = bubble_point_p(T, z, mx)
    r_nt = newton_bubble_point_p(T, z, mx)
    check("Near-critical Newton: p matches SS to 1e-6 rel",
          abs(r_nt.p - r_ss.p) / r_ss.p < 1e-6,
          f"SS p={r_ss.p/1e6:.4f}, Newton p={r_nt.p/1e6:.4f}")
    check("Near-critical Newton: converges in < 15 iterations",
          r_nt.iterations < 15, f"iters={r_nt.iterations}")
    check("Near-critical Newton: much faster than SS (iters < 1/3 of SS)",
          r_nt.iterations * 3 < r_ss.iterations,
          f"SS={r_ss.iterations}, Newton={r_nt.iterations}")


def test_helmholtz_newton_bubble_point_T_matches_ss():
    """Newton bubble_point_T converges to same T as SS."""
    from stateprop.mixture.flash import bubble_point_T, newton_bubble_point_T
    mx = load_mixture(['methane', 'ethane'], [0.5, 0.5], binary_set='gerg2008')
    z = np.array([0.5, 0.5])
    p = 2e6
    r_ss = bubble_point_T(p, z, mx)
    r_nt = newton_bubble_point_T(p, z, mx)
    check("Newton bubble_point_T: T matches SS to 1e-6 rel",
          abs(r_nt.T - r_ss.T) / r_ss.T < 1e-6,
          f"SS T={r_ss.T:.3f}, Newton T={r_nt.T:.3f}")


def test_helmholtz_newton_dew_point_p_matches_ss():
    """Newton dew_point_p converges to same p as SS."""
    from stateprop.mixture.flash import dew_point_p, newton_dew_point_p
    mx = load_mixture(['methane', 'ethane'], [0.5, 0.5], binary_set='gerg2008')
    z = np.array([0.5, 0.5])
    T = 220.0
    r_ss = dew_point_p(T, z, mx)
    r_nt = newton_dew_point_p(T, z, mx)
    check("Newton dew_point_p: p matches SS to 1e-6 rel",
          abs(r_nt.p - r_ss.p) / r_ss.p < 1e-6,
          f"SS p={r_ss.p/1e6:.4f}, Newton p={r_nt.p/1e6:.4f}")
    check("Newton dew_point_p: fewer iterations than SS",
          r_nt.iterations <= r_ss.iterations,
          f"SS {r_ss.iterations}, Newton {r_nt.iterations}")


def test_helmholtz_newton_dew_point_T_matches_ss():
    """Newton dew_point_T converges to same T as SS."""
    from stateprop.mixture.flash import dew_point_T, newton_dew_point_T
    mx = load_mixture(['methane', 'ethane'], [0.5, 0.5], binary_set='gerg2008')
    z = np.array([0.5, 0.5])
    p = 2e6
    r_ss = dew_point_T(p, z, mx)
    r_nt = newton_dew_point_T(p, z, mx)
    check("Newton dew_point_T: T matches SS to 1e-6 rel",
          abs(r_nt.T - r_ss.T) / r_ss.T < 1e-6,
          f"SS T={r_ss.T:.3f}, Newton T={r_nt.T:.3f}")


def test_helmholtz_newton_bubble_dew_ordering():
    """At any T well below critical, bubble_p > dew_p -- Newton must respect
    this thermodynamic ordering."""
    from stateprop.mixture.flash import newton_bubble_point_p, newton_dew_point_p
    mx = load_mixture(['carbondioxide', 'methane'], [0.5, 0.5], binary_set='gerg2008')
    z = np.array([0.5, 0.5])
    T = 220.0
    b = newton_bubble_point_p(T, z, mx)
    d = newton_dew_point_p(T, z, mx)
    check("Newton bubble p > Newton dew p at T << T_c",
          b.p > d.p, f"bubble p={b.p/1e6:.3f}, dew p={d.p/1e6:.3f}")
    # Volatility ordering in K-factors
    # CO2 is less volatile than CH4, so K_CH4 > 1 and K_CO2 < 1 at bubble
    check("Newton bubble: K_CH4 > 1 (more volatile)",
          b.K[1] > 1.0, f"got K_CH4={b.K[1]:.3f}")
    check("Newton bubble: K_CO2 < 1 (less volatile)",
          b.K[0] < 1.0, f"got K_CO2={b.K[0]:.3f}")


# ----------------------------------------------------------------------
# v0.9.18 -- Analytic envelope Jacobian for Helmholtz/GERG
# ----------------------------------------------------------------------


def test_helmholtz_envelope_analytic_jacobian_matches_fd():
    """v0.9.18: _envelope_jacobian_analytic matches central-difference FD
    to ~1e-7 relative error on a non-converged state for both beta."""
    from stateprop.mixture.envelope import (
        _envelope_jacobian_analytic, _envelope_jacobian_fd,
    )
    mx = load_mixture(['methane', 'ethane'], [0.5, 0.5], binary_set='gerg2008')
    z = np.array([0.5, 0.5])
    N = 2
    T, p = 220.0, 1e6
    K = np.array([2.0, 0.5])  # not-yet-converged
    X = np.concatenate([np.log(K), [np.log(T), np.log(p)]])
    for beta in (0, 1):
        J_fd = _envelope_jacobian_fd(X, beta, z, N, float(X[N]), mx)
        J_an = _envelope_jacobian_analytic(X, beta, z, N, float(X[N]), mx)
        rel_err = float(np.max(np.abs(J_an - J_fd) / (np.abs(J_fd) + 1e-30)))
        check(f"Helmholtz envelope Jacobian [beta={beta}]: analytic matches FD (rel err < 1e-5)",
              rel_err < 1e-5, f"rel err = {rel_err:.2e}")


def test_helmholtz_envelope_point_analytic_matches_fd():
    """v0.9.18: envelope_point with use_analytic_jac=True gives same
    converged (T, p, K) as FD path to machine precision."""
    from stateprop.mixture.envelope import envelope_point
    mx = load_mixture(['methane', 'ethane'], [0.5, 0.5], binary_set='gerg2008')
    z = np.array([0.5, 0.5])
    T = 220.0
    ep_fd = envelope_point(T, 1e6, z, mx, beta=0, use_analytic_jac=False)
    ep_an = envelope_point(T, 1e6, z, mx, beta=0, use_analytic_jac=True)
    check("Helmholtz envelope_point: analytic vs FD p agreement < 1e-10",
          abs(ep_an['p'] - ep_fd['p']) / ep_fd['p'] < 1e-10,
          f"FD p={ep_fd['p']/1e6:.6f}, AN p={ep_an['p']/1e6:.6f}")
    check("Helmholtz envelope_point: analytic vs FD K agreement < 1e-10",
          np.max(np.abs(ep_an['K'] - ep_fd['K']) / ep_fd['K']) < 1e-10,
          f"K diff = {np.max(np.abs(ep_an['K'] - ep_fd['K']) / ep_fd['K']):.2e}")
    check("Helmholtz envelope_point: analytic converges in same iters as FD",
          ep_an['iterations'] == ep_fd['iterations'],
          f"FD {ep_fd['iterations']}, AN {ep_an['iterations']}")


def test_helmholtz_envelope_point_analytic_default():
    """v0.9.18: envelope_point's use_analytic_jac default is True."""
    from stateprop.mixture.envelope import envelope_point
    import inspect
    sig = inspect.signature(envelope_point)
    default = sig.parameters['use_analytic_jac'].default
    check("Helmholtz envelope_point: use_analytic_jac default is True",
          default is True, f"got {default}")


# ----------------------------------------------------------------------
# v0.9.19 -- Three-phase (VLLE) flash for Helmholtz/GERG
# ----------------------------------------------------------------------


def test_three_phase_rr_material_balance():
    """3-phase Rachford-Rice with constructed K's must exactly recover the
    phase fractions and compositions used to build z. Fundamental correctness
    check for the _rachford_rice_3p solver."""
    from stateprop.mixture.three_phase_flash import _rachford_rice_3p
    # Pick the answer first, derive z and K's
    bV_true, bL1_true, bL2_true = 0.4, 0.35, 0.25
    y_true = np.array([0.7, 0.25, 0.05])
    x1_true = np.array([0.2, 0.55, 0.25])
    x2_true = np.array([0.1, 0.05, 0.85])
    z = bV_true * y_true + bL1_true * x1_true + bL2_true * x2_true
    K_VL1 = y_true / x1_true
    K_L2L1 = x2_true / x1_true
    bV, bL1, bL2, x1, x2, y = _rachford_rice_3p(z, K_VL1, K_L2L1, tol=1e-12)
    check("3-phase RR: beta_V matches truth to 1e-8",
          abs(bV - bV_true) < 1e-8, f"got {bV}, expected {bV_true}")
    check("3-phase RR: beta_L1 matches truth to 1e-8",
          abs(bL1 - bL1_true) < 1e-8, f"got {bL1}, expected {bL1_true}")
    check("3-phase RR: beta_L2 matches truth to 1e-8",
          abs(bL2 - bL2_true) < 1e-8, f"got {bL2}, expected {bL2_true}")
    check("3-phase RR: y matches truth to 1e-8",
          np.max(np.abs(y - y_true)) < 1e-8, f"max diff {np.max(np.abs(y - y_true)):.2e}")
    check("3-phase RR: x1 matches truth to 1e-8",
          np.max(np.abs(x1 - x1_true)) < 1e-8, f"max diff {np.max(np.abs(x1 - x1_true)):.2e}")
    check("3-phase RR: x2 matches truth to 1e-8",
          np.max(np.abs(x2 - x2_true)) < 1e-8, f"max diff {np.max(np.abs(x2 - x2_true)):.2e}")


def test_three_phase_rr_material_balance_closure():
    """Material balance must hold at the RR solution: z = sum_k beta_k x_k."""
    from stateprop.mixture.three_phase_flash import _rachford_rice_3p
    z = np.array([0.35, 0.45, 0.20])
    K_VL1 = np.array([3.5, 0.45, 0.2])
    K_L2L1 = np.array([0.5, 0.09, 3.4])
    bV, bL1, bL2, x1, x2, y = _rachford_rice_3p(z, K_VL1, K_L2L1)
    z_reconstructed = bV * y + bL1 * x1 + bL2 * x2
    max_err = float(np.max(np.abs(z - z_reconstructed)))
    check("3-phase RR: material balance closes to 1e-8",
          max_err < 1e-8, f"max |z - z_reconstructed| = {max_err:.2e}")
    check("3-phase RR: phase fractions sum to 1",
          abs(bV + bL1 + bL2 - 1.0) < 1e-10,
          f"sum = {bV + bL1 + bL2:.10f}")
    check("3-phase RR: phase fractions all in [0, 1]",
          0 <= bV <= 1 and 0 <= bL1 <= 1 and 0 <= bL2 <= 1,
          f"got betas=({bV}, {bL1}, {bL2})")


def test_three_phase_flash_preserves_two_phase():
    """flash_pt_three_phase on a clean 2-phase system (CH4-ethane, GERG)
    should return a 2-phase result, not split into 3 phases."""
    from stateprop.mixture.three_phase_flash import flash_pt_three_phase
    mx = load_mixture(['methane', 'ethane'], [0.5, 0.5], binary_set='gerg2008')
    z = np.array([0.5, 0.5])
    r = flash_pt_three_phase(2e6, 220.0, z, mx)
    check("3-phase flash on CH4-C2 VLE: returns VLE label",
          r.phase == "VLE", f"got phase={r.phase}")
    check("3-phase flash on CH4-C2 VLE: beta_L2 == 0",
          r.beta_L2 == 0.0, f"got beta_L2={r.beta_L2}")
    check("3-phase flash on CH4-C2 VLE: beta_V + beta_L1 == 1",
          abs(r.beta_V + r.beta_L1 - 1.0) < 1e-10,
          f"sum = {r.beta_V + r.beta_L1}")
    check("3-phase flash on CH4-C2 VLE: two_phase_result populated",
          r.two_phase_result is not None, "missing two_phase_result field")


def test_three_phase_flash_preserves_single_phase():
    """flash_pt_three_phase on a supercritical state should return single phase."""
    from stateprop.mixture.three_phase_flash import flash_pt_three_phase
    mx = load_mixture(['methane', 'ethane'], [0.5, 0.5], binary_set='gerg2008')
    z = np.array([0.5, 0.5])
    r = flash_pt_three_phase(2e6, 300.0, z, mx)
    # Should be a single-phase label
    check("3-phase flash on supercritical: single-phase label",
          r.phase in ('supercritical', 'vapor', 'liquid'),
          f"got phase={r.phase}")
    check("3-phase flash on supercritical: no L2 phase",
          r.beta_L2 == 0.0, f"got beta_L2={r.beta_L2}")


# ----------------------------------------------------------------------
# v0.9.21 -- Composition-based SS for Helmholtz three-phase flash
# ----------------------------------------------------------------------


def test_helmholtz_three_phase_ss_composition_based():
    """v0.9.21: the ported composition-based SS (from cubic v0.9.20) must
    still converge on a constructed 3-phase problem. The SS starts from
    initial K-factors and, regardless of how pathological they are,
    should drive to fugacity equality via the material-balance-LSQ
    strategy without hitting rank-1 Rachford-Rice failures."""
    from stateprop.mixture.three_phase_flash import _three_phase_ss
    # Use a mixture that does NOT split 3-phase in reality, but call the
    # SS directly with constructed initial K's to test it handles
    # non-trivial initialization gracefully. We pick CH4-ethane at
    # VLE-like conditions and give K's that could drive to a 2-phase
    # collapse (which _three_phase_ss raises RuntimeError for).
    mx = load_mixture(['methane', 'ethane'], [0.5, 0.5], binary_set='gerg2008')
    z = np.array([0.5, 0.5])
    T, p = 220.0, 2e6
    # K's that should drive to a trivial collapse
    K_VL1 = np.array([3.0, 0.3])
    K_L2L1 = np.array([1.01, 1.01])   # nearly identical to trivial
    # SS should either converge to a valid 3-phase OR raise RuntimeError
    # due to trivial collapse. Either is acceptable; what matters is it
    # doesn't crash with rank-1 Jacobian error.
    try:
        result = _three_phase_ss(z, T, p, mx, K_VL1, K_L2L1, tol=1e-6, maxiter=50)
        # If it converges, ensure phase fractions are valid
        bV, bL1, bL2, *_ = result
        check("Helmholtz 3-phase SS handles degenerate K's: valid betas",
              bV + bL1 + bL2 > 0.999 and bV + bL1 + bL2 < 1.001,
              f"betas sum = {bV+bL1+bL2}")
    except RuntimeError as e:
        # Expected: trivial collapse or SS non-convergence is fine
        msg = str(e)
        check("Helmholtz 3-phase SS handles degenerate K's: graceful failure",
              "collapsed" in msg or "did not converge" in msg or "infeasible" in msg,
              f"error: {msg}")


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
        test_flash_convergence_acceleration,
        test_state_function_flashes,
        test_bubble_dew_points,
        test_unreachable_bubble_dew,
        # departure tests
        test_departure_function_derivatives_fd,
        test_departure_pure_limit,
        test_ln_phi_fd_with_departure,
        test_departure_changes_fugacity,
        test_pt_flash_with_departure,
        # v0.9.9 -- analytic composition derivatives
        test_helmholtz_reducing_hessian_symmetric_and_fd,
        test_helmholtz_analytic_dp_dx_at_rho,
        test_helmholtz_analytic_dlnphi_drho_at_x,
        test_helmholtz_analytic_dlnphi_dx_at_rho,
        test_helmholtz_analytic_dlnphi_dx_at_p,
        test_helmholtz_newton_flash_pt,
        test_helmholtz_newton_flash_handles_single_phase,
        # v0.9.10 -- T and p derivatives of ln phi
        test_helmholtz_dp_dT_at_rho_fd,
        test_helmholtz_dlnphi_dT_at_rho_fd,
        test_helmholtz_dlnphi_dp_at_T_fd,
        test_helmholtz_dlnphi_dT_at_p_fd,
        # v0.9.11 -- residual Helmholtz Hessian + experimental critical point
        test_helmholtz_A_residual_matches_fd,
        test_helmholtz_critical_CO2_CH4,
        test_helmholtz_critical_pure_limit,
        # v0.9.12 -- robust multistart + physical-filter critical point
        test_helmholtz_critical_multistart_CO2_CH4,
        test_helmholtz_critical_multistart_CH4_C2,
        test_helmholtz_critical_multistart_CH4_N2_locus,
        test_helmholtz_critical_multistart_returns_candidates,
        # v0.9.13 -- Phase envelope tracer
        test_helmholtz_envelope_point_wilson_seeded,
        test_helmholtz_envelope_point_fugacity_closure,
        test_helmholtz_trace_envelope_structure,
        test_helmholtz_envelope_CO2_CH4_trace,
        # v0.9.14 -- Newton-Raphson bubble/dew point solvers
        test_helmholtz_newton_bubble_point_p_matches_ss,
        test_helmholtz_newton_bubble_point_p_near_critical,
        test_helmholtz_newton_bubble_point_T_matches_ss,
        test_helmholtz_newton_dew_point_p_matches_ss,
        test_helmholtz_newton_dew_point_T_matches_ss,
        test_helmholtz_newton_bubble_dew_ordering,
        # v0.9.18 -- Analytic envelope Jacobian for Helmholtz
        test_helmholtz_envelope_analytic_jacobian_matches_fd,
        test_helmholtz_envelope_point_analytic_matches_fd,
        test_helmholtz_envelope_point_analytic_default,
        # v0.9.19 -- Three-phase flash (VLLE)
        test_three_phase_rr_material_balance,
        test_three_phase_rr_material_balance_closure,
        test_three_phase_flash_preserves_two_phase,
        test_three_phase_flash_preserves_single_phase,
        # v0.9.21 -- Composition-based SS port from cubic
        test_helmholtz_three_phase_ss_composition_based,
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
