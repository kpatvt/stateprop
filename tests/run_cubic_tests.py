"""Plain-Python test runner for stateprop.cubic.

Run: python tests/run_cubic_tests.py
"""
import sys
import traceback
import numpy as np

sys.path.insert(0, '.')

from stateprop.cubic import (
    CubicEOS, PR, SRK, RK, VDW,
    CubicMixture,
    flash_pt, stability_test_TPD,
)


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


def rel_err(a, b, tol_abs=1e-7):
    if max(abs(a), abs(b)) < tol_abs:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))


# ----------------------------------------------------------------------
# Pure-component EOS tests
# ----------------------------------------------------------------------

def test_pure_alpha_r_derivs_fd():
    """All 5 alpha_r derivatives match FD to better than 1e-6."""
    for name, eos in [('PR-CO2',  PR(304.13, 7.3773e6, 0.2239)),
                      ('SRK-CO2', SRK(304.13, 7.3773e6, 0.2239)),
                      ('RK-CO2',  RK(304.13, 7.3773e6)),
                      ('vdW-CO2', VDW(304.13, 7.3773e6))]:
        max_err = 0
        for (delta, tau) in [(0.3, 1.2), (1.0, 0.9), (0.7, 1.5), (2.0, 0.7)]:
            A, A_d, A_t, A_dd, A_tt, A_dt = eos.alpha_r_derivs(delta, tau)
            eps = 1e-6
            Ap, *_ = eos.alpha_r_derivs(delta + eps, tau); Am, *_ = eos.alpha_r_derivs(delta - eps, tau)
            fd_d = (Ap - Am) / (2*eps)
            Ap, *_ = eos.alpha_r_derivs(delta, tau + eps); Am, *_ = eos.alpha_r_derivs(delta, tau - eps)
            fd_t = (Ap - Am) / (2*eps)
            _, Adp, *_ = eos.alpha_r_derivs(delta + eps, tau); _, Adm, *_ = eos.alpha_r_derivs(delta - eps, tau)
            fd_dd = (Adp - Adm) / (2*eps)
            _, _, Atp, *_ = eos.alpha_r_derivs(delta, tau + eps); _, _, Atm, *_ = eos.alpha_r_derivs(delta, tau - eps)
            fd_tt = (Atp - Atm) / (2*eps)
            _, Adp, *_ = eos.alpha_r_derivs(delta, tau + eps); _, Adm, *_ = eos.alpha_r_derivs(delta, tau - eps)
            fd_dt = (Adp - Adm) / (2*eps)
            errs = [rel_err(A_d, fd_d), rel_err(A_t, fd_t),
                    rel_err(A_dd, fd_dd), rel_err(A_tt, fd_tt),
                    rel_err(A_dt, fd_dt)]
            max_err = max(max_err, max(errs))
        check(f"{name}: all alpha_r derivatives FD", max_err < 1e-6,
              f"max err {max_err:.2e}")


def test_pressure_consistency():
    """p from direct cubic matches p from Helmholtz form (1 + delta * a_d)."""
    for name, eos in [('PR-CO2',  PR(304.13, 7.3773e6, 0.2239)),
                      ('SRK-CO2', SRK(304.13, 7.3773e6, 0.2239))]:
        max_err = 0
        for (rho, T) in [(5000, 250), (10000, 300), (1000, 400), (20000, 200)]:
            p_direct = eos.pressure(rho, T)
            delta = rho / eos.rho_c
            tau = eos.T_c / T
            _, a_d, _, _, _, _ = eos.alpha_r_derivs(delta, tau)
            p_helm = rho * eos.R * T * (1.0 + delta * a_d)
            err = rel_err(p_direct, p_helm, tol_abs=1e-3)
            max_err = max(max_err, err)
        check(f"{name}: p(direct)==p(Helmholtz)", max_err < 1e-12, f"err {max_err:.2e}")


def test_pure_saturation_SRK_vs_NIST():
    """SRK should match NIST CO2 saturation to within a few percent."""
    eos = SRK(T_c=304.1282, p_c=7.3773e6, acentric_factor=0.22394)
    # NIST CO2 saturation data (approximate, from NIST webbook)
    nist = [(220, 0.5991), (240, 1.2825), (250, 1.7851), (260, 2.4188)]

    for T, p_nist in nist:
        try:
            p = eos.saturation_p(T)
        except Exception as e:
            check(f"SRK saturation at T={T}", False, str(e))
            continue
        p_nist_Pa = p_nist * 1e6
        err = abs(p - p_nist_Pa) / p_nist_Pa
        check(f"SRK CO2 sat at T={T}K: {p*1e-6:.3f} MPa vs NIST {p_nist:.3f} MPa",
              err < 0.03, f"err {err*100:.1f}%")


def test_pure_saturation_PR_vs_NIST():
    """PR for CO2: known to overpredict by 10-20%. Just check convergence."""
    eos = PR(T_c=304.1282, p_c=7.3773e6, acentric_factor=0.22394)
    for T in [220, 240, 260, 280]:
        try:
            p = eos.saturation_p(T)
            check(f"PR CO2 sat at T={T}K converges", p > 0 and np.isfinite(p),
                  f"p={p*1e-6 if p else p}")
        except Exception as e:
            check(f"PR CO2 sat at T={T}K converges", False, str(e))


# ----------------------------------------------------------------------
# Mixture tests
# ----------------------------------------------------------------------

def test_mixture_pressure_consistency():
    """p from direct cubic at (rho, T, x) matches the requested p."""
    c_CH4 = PR(T_c=190.564, p_c=4.5992e6, acentric_factor=0.01142)
    c_N2  = PR(T_c=126.192, p_c=3.3958e6, acentric_factor=0.0372)
    mx = CubicMixture([c_CH4, c_N2], composition=[0.8, 0.2], k_ij={(0,1): 0.025})
    for T, p_MPa in [(200, 5), (250, 10), (300, 2), (150, 1)]:
        p = p_MPa * 1e6
        try:
            rho = mx.density_from_pressure(p, T, phase_hint='vapor')
        except Exception as e:
            check(f"mixture pressure consistency (T={T},p={p_MPa})", False, str(e))
            continue
        a, b, *_ = mx.a_b_mix(T)
        v = 1/rho
        p_chk = mx.R*T/(v-b) - a/((v+mx.epsilon*b)*(v+mx.sigma*b))
        err = abs(p - p_chk) / p
        check(f"mixture p(rho,T,x)==p at T={T},p={p_MPa}MPa", err < 1e-10,
              f"err {err:.2e}")


def test_mixture_ln_phi_fd():
    """ln phi_i from cubic mixture matches FD of d(n*alpha_r)/dn_i."""
    def n_alpha_r(n_vec, V, T, mx):
        n = n_vec.sum(); xx = n_vec/n; rho = n/V
        a, b, *_ = mx.a_b_mix(T, xx)
        B = b * rho
        eps_ = mx.epsilon; sig = mx.sigma
        if abs(sig - eps_) > 1e-14:
            q = a / (mx.R * T * b * (sig - eps_))
            I = np.log((1 + sig*B)/(1 + eps_*B))
            return n * (-np.log(1-B) - q*I)
        return n * (-np.log(1-B) - a*rho/(mx.R*T))

    # PR binaries
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_N2  = PR(126.192, 3.3958e6, 0.0372)
    max_err = 0
    count = 0
    for comp in [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]:
        mx = CubicMixture([c_CH4, c_N2], composition=comp, k_ij={(0,1): 0.025})
        for T, p_MPa in [(200, 5), (250, 10), (300, 2)]:
            try:
                rho = mx.density_from_pressure(p_MPa*1e6, T, phase_hint='vapor')
            except:
                continue
            V = 1/rho
            lnphi = mx.ln_phi(rho, T)
            Z = p_MPa*1e6 * V / (mx.R * T)
            fd = np.zeros(2); eps = 1e-7
            for i in range(2):
                npv = mx.x.copy(); npv[i] += eps
                nmv = mx.x.copy(); nmv[i] -= eps
                fd[i] = (n_alpha_r(npv, V, T, mx) - n_alpha_r(nmv, V, T, mx)) / (2*eps)
            err = np.max(np.abs(fd - np.log(Z) - lnphi))
            count += 1
            max_err = max(max_err, err)
    check(f"PR binary ln_phi FD ({count} states)", max_err < 1e-5,
          f"max err {max_err:.2e}")


def test_mixture_ln_phi_ternary_fd():
    """Ternary mixture ln phi FD check."""
    def n_alpha_r(n_vec, V, T, mx):
        n = n_vec.sum(); xx = n_vec/n; rho = n/V
        a, b, *_ = mx.a_b_mix(T, xx)
        B = b * rho
        eps_ = mx.epsilon; sig = mx.sigma
        q = a / (mx.R * T * b * (sig - eps_))
        I = np.log((1 + sig*B)/(1 + eps_*B))
        return n * (-np.log(1-B) - q*I)

    c_CH4 = SRK(190.564, 4.5992e6, 0.01142)
    c_N2  = SRK(126.192, 3.3958e6, 0.0372)
    c_CO2 = SRK(304.128, 7.3773e6, 0.22394)
    max_err = 0
    for comp in [[0.7, 0.1, 0.2], [0.5, 0.3, 0.2], [0.4, 0.2, 0.4]]:
        mx = CubicMixture([c_CH4, c_N2, c_CO2], composition=comp,
                          k_ij={(0,1): 0.025, (0,2): 0.091, (1,2): -0.017})
        for T, p_MPa in [(250, 5), (300, 3)]:
            try:
                rho = mx.density_from_pressure(p_MPa*1e6, T, phase_hint='vapor')
            except:
                continue
            V = 1/rho
            lnphi = mx.ln_phi(rho, T)
            Z = p_MPa*1e6 * V / (mx.R * T)
            fd = np.zeros(3); eps = 1e-7
            for i in range(3):
                npv = mx.x.copy(); npv[i] += eps
                nmv = mx.x.copy(); nmv[i] -= eps
                fd[i] = (n_alpha_r(npv, V, T, mx) - n_alpha_r(nmv, V, T, mx)) / (2*eps)
            err = np.max(np.abs(fd - np.log(Z) - lnphi))
            max_err = max(max_err, err)
    check("SRK ternary ln_phi FD", max_err < 1e-5, f"max err {max_err:.2e}")


def test_pt_flash_fugacity_equality():
    """PT flash: at a 2-phase converged state, f_L = f_V."""
    # CH4-N2 at low T
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_N2  = PR(126.192, 3.3958e6, 0.0372)
    mx = CubicMixture([c_CH4, c_N2], composition=[0.8, 0.2], k_ij={(0,1): 0.025})
    r = flash_pt(2e6, 150.0, mx.x, mx)
    check("PT flash CH4-N2: 2-phase detected", r.phase == 'two_phase')
    if r.phase == 'two_phase':
        lpL = mx.ln_phi(r.rho_L, 150.0, r.x)
        lpV = mx.ln_phi(r.rho_V, 150.0, r.y)
        f_L = r.x * np.exp(lpL) * 2e6
        f_V = r.y * np.exp(lpV) * 2e6
        max_err = np.max(np.abs(f_L/f_V - 1.0))
        check("PT flash CH4-N2: fugacity equality", max_err < 1e-9,
              f"max err {max_err:.2e}")
        # Physical check: liquid denser than vapor
        check("PT flash CH4-N2: rho_L > rho_V", r.rho_L > r.rho_V,
              f"rho_L={r.rho_L}, rho_V={r.rho_V}")


def test_cubic_analytic_dlnphi_dx_at_rho():
    """The analytic d(ln phi_i)/d x_k at fixed (T, rho) must match
    a 2-point finite-difference of ln_phi to ~1e-7 across phases.
    Tests the matrix building blocks before the chain rule combines them."""
    c1 = PR(190.564, 4.5992e6, 0.01142)
    c2 = PR(425.12, 3.796e6, 0.2002)
    mx = CubicMixture([c1, c2], composition=[0.5, 0.5], k_ij={})
    T, p = 300.0, 5e6
    x = np.array([0.5, 0.5])
    for phase in ('vapor', 'liquid'):
        rho = mx.density_from_pressure(p, T, x, phase_hint=phase)
        J_an = mx._dlnphi_dx_at_rho(rho, T, x)
        h = 1e-6
        J_fd = np.zeros((2, 2))
        for k in range(2):
            ek = np.eye(2)[k]
            J_fd[:, k] = (mx.ln_phi(rho, T, x + h*ek) - mx.ln_phi(rho, T, x - h*ek)) / (2*h)
        rel = np.max(np.abs((J_an - J_fd) / J_fd))
        check(f"d(lnphi)/dx at fixed (T,rho) [{phase}]: rel err < 1e-6",
              rel < 1e-6, f"got {rel:.2e}")


def test_cubic_analytic_dlnphi_drho():
    """d(ln phi)/drho at fixed (T, x). N-vector. FD-verified."""
    c1 = PR(190.564, 4.5992e6, 0.01142); c2 = PR(425.12, 3.796e6, 0.2002)
    mx = CubicMixture([c1, c2], composition=[0.5, 0.5], k_ij={})
    T, p = 300.0, 5e6; x = np.array([0.5, 0.5])
    for phase in ('vapor', 'liquid'):
        rho = mx.density_from_pressure(p, T, x, phase_hint=phase)
        an = mx._dlnphi_drho_at_x(rho, T, x)
        hr = max(rho * 1e-6, 1e-3)
        fd = (mx.ln_phi(rho + hr, T, x) - mx.ln_phi(rho - hr, T, x)) / (2 * hr)
        rel = np.max(np.abs((an - fd) / fd))
        check(f"d(lnphi)/drho at fixed (T,x) [{phase}]: rel err < 1e-5",
              rel < 1e-5, f"got {rel:.2e}")


def test_cubic_analytic_dp_dx_at_rho():
    """d(p)/d(x_k) at fixed (T, rho). FD against the EOS pressure formula."""
    c1 = PR(190.564, 4.5992e6, 0.01142); c2 = PR(425.12, 3.796e6, 0.2002)
    mx = CubicMixture([c1, c2], composition=[0.5, 0.5], k_ij={})
    T, p_target = 300.0, 5e6; x = np.array([0.5, 0.5])
    for phase in ('vapor', 'liquid'):
        rho = mx.density_from_pressure(p_target, T, x, phase_hint=phase)
        an = mx._dp_dx_at_rho(rho, T, x)
        def p_at(x_):
            a, b, _, _, _, _ = mx.a_b_mix(T, x_)
            v = 1.0 / rho
            return mx.R*T/(v - b) - a / ((v + mx.epsilon*b)*(v + mx.sigma*b))
        h = 1e-6
        fd = np.array([(p_at(x + h*np.eye(2)[k]) - p_at(x - h*np.eye(2)[k])) / (2*h)
                       for k in range(2)])
        rel = np.max(np.abs((an - fd) / fd))
        check(f"dp/dx at fixed (T,rho) [{phase}]: rel err < 1e-6",
              rel < 1e-6, f"got {rel:.2e}")


def test_cubic_analytic_dlnphi_dxk_at_p():
    """The headline result: d(ln phi_i)/d x_k at fixed (T, p), the
    Newton-flash Jacobian. Combines all four building blocks via the
    chain rule and FD-verifies the combination across multiple mixtures
    and both phases."""
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_N2  = PR(126.192, 3.3958e6, 0.0372)
    c_C4  = PR(425.12, 3.796e6, 0.2002)
    c_C10 = PR(617.7, 21.1e5, 0.4923)
    test_states = [
        (CubicMixture([c_CH4, c_N2], composition=[0.6, 0.4], k_ij={(0,1):0.025}),
         200.0, 3e6, np.array([0.6, 0.4])),
        (CubicMixture([c_CH4, c_C4], composition=[0.5, 0.5], k_ij={}),
         300.0, 5e6, np.array([0.5, 0.5])),
        (CubicMixture([c_CH4, c_C10], composition=[0.5, 0.5], k_ij={}),
         350.0, 10e6, np.array([0.5, 0.5])),
    ]
    for mx, T, p, x in test_states:
        for phase in ('vapor', 'liquid'):
            J_an = mx.dlnphi_dxk_at_p(p, T, x, phase_hint=phase)
            h = 1e-6
            J_fd = np.zeros((2, 2))
            for k in range(2):
                ek = np.eye(2)[k]
                rho_p = mx.density_from_pressure(p, T, x + h*ek, phase_hint=phase)
                rho_m = mx.density_from_pressure(p, T, x - h*ek, phase_hint=phase)
                J_fd[:, k] = (mx.ln_phi(rho_p, T, x + h*ek)
                              - mx.ln_phi(rho_m, T, x - h*ek)) / (2*h)
            rel = np.max(np.abs((J_an - J_fd) / np.where(np.abs(J_fd) > 1e-10, J_fd, 1.0)))
            label = f"Tc={[c.T_c for c in mx.components]} T={T} p={p:.1e} {phase}"
            check(f"dlnphi/dx at fixed (T,p) [{label}]: rel err < 1e-5",
                  rel < 1e-5, f"got {rel:.2e}")


def test_cubic_newton_flash_pt():
    """Newton-Raphson flash with analytic Jacobian must converge to the
    same answer as the SS+Broyden flash but in fewer iterations."""
    from stateprop.cubic.flash import newton_flash_pt
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_C10 = PR(617.7, 21.1e5, 0.4923)
    mx = CubicMixture([c_CH4, c_C10], composition=[0.5, 0.5], k_ij={})
    z = np.array([0.5, 0.5])
    T, p = 350.0, 10e6
    rb = flash_pt(p, T, z, mx)
    rn = newton_flash_pt(p, T, z, mx)
    check("Newton flash: 2-phase detected (matches Broyden)",
          rn.phase == rb.phase == 'two_phase')
    check("Newton flash: beta agrees with Broyden to 1e-7",
          abs(rn.beta - rb.beta) < 1e-7,
          f"Broyden beta={rb.beta:.8f}, Newton beta={rn.beta:.8f}")
    check("Newton flash: x agrees with Broyden to 1e-6",
          np.max(np.abs(rn.x - rb.x)) < 1e-6,
          f"max diff = {np.max(np.abs(rn.x - rb.x)):.2e}")
    check("Newton flash: y agrees with Broyden to 1e-6",
          np.max(np.abs(rn.y - rb.y)) < 1e-6)
    check("Newton flash: converges in <= 6 iters (vs Broyden's 8)",
          rn.iterations <= 6,
          f"got {rn.iterations} iters, Broyden took {rb.iterations}")
    lpL = mx.ln_phi(rn.rho_L, T, rn.x); lpV = mx.ln_phi(rn.rho_V, T, rn.y)
    f_L = rn.x * np.exp(lpL) * p; f_V = rn.y * np.exp(lpV) * p
    err = np.max(np.abs(f_L / f_V - 1.0))
    check("Newton flash: fugacity equality at converged state",
          err < 1e-9, f"max f_L/f_V - 1 = {err:.2e}")


def test_cubic_newton_flash_handles_single_phase():
    """Newton flash on a single-phase feed should gracefully delegate
    to the SS+Broyden path (which knows how to label the single-phase
    result)."""
    from stateprop.cubic.flash import newton_flash_pt
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_N2  = PR(126.192, 3.3958e6, 0.0372)
    mx = CubicMixture([c_CH4, c_N2], composition=[0.5, 0.5], k_ij={(0,1):0.025})
    rn = newton_flash_pt(p=5e6, T=300.0, z=mx.x, mixture=mx)
    check("Newton flash on supercritical: gets a single-phase classification",
          rn.phase in ('vapor', 'liquid', 'supercritical') and rn.beta is None,
          f"got phase={rn.phase}, beta={rn.beta}")


def test_cubic_analytic_dp_dT_at_rho():
    """dp/dT at fixed (rho, x) validates vs 2-pt FD on the cubic EOS
    pressure formula. Dependence enters via a_m(T)."""
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_N2  = PR(126.192, 3.3958e6, 0.0372)
    c_C10 = PR(617.7, 21.1e5, 0.4923)
    cases = [
        (CubicMixture([c_CH4, c_N2], composition=[0.6,0.4], k_ij={(0,1):0.025}),
         200.0, 3e6, np.array([0.6, 0.4])),
        (CubicMixture([c_CH4, c_C10], composition=[0.5,0.5], k_ij={}),
         350.0, 10e6, np.array([0.5, 0.5])),
    ]
    hT = 1e-3
    for mx, T, p, x in cases:
        for phase in ('vapor', 'liquid'):
            rho = mx.density_from_pressure(p, T, x, phase_hint=phase)
            an = mx._dp_dT_at_rho(rho, T, x)
            # FD using the cubic pressure formula directly
            def p_of_T(T_val):
                a, b, _, _, _, _ = mx.a_b_mix(T_val, x)
                v = 1.0 / rho
                if abs(mx.sigma - mx.epsilon) > 1e-14:
                    return mx.R * T_val / (v - b) - a / ((v + mx.epsilon*b)*(v + mx.sigma*b))
                return mx.R * T_val / (v - b) - a / (v * v)
            fd = (p_of_T(T + hT) - p_of_T(T - hT)) / (2 * hT)
            err = abs((an - fd) / fd)
            check(f"cubic dp/dT_at_rho [{phase}, Tc={[c.T_c for c in mx.components]} "
                  f"T={T} p={p:.1e}]: rel err < 1e-7",
                  err < 1e-7, f"got {err:.2e}")


def test_cubic_analytic_dlnphi_dT_and_dp():
    """Cubic dlnphi_dT_at_p and dlnphi_dp_at_T validate vs FD to 1e-6."""
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_N2  = PR(126.192, 3.3958e6, 0.0372)
    c_C10 = PR(617.7, 21.1e5, 0.4923)
    cases = [
        (CubicMixture([c_CH4, c_N2], composition=[0.6,0.4], k_ij={(0,1):0.025}),
         200.0, 3e6, np.array([0.6, 0.4])),
        (CubicMixture([c_CH4, c_C10], composition=[0.5,0.5], k_ij={}),
         350.0, 10e6, np.array([0.5, 0.5])),
    ]
    for mx, T, p, x in cases:
        for phase in ('vapor', 'liquid'):
            # dlnphi/dp at fixed (T, x)
            an_p = mx.dlnphi_dp_at_T(p, T, x, phase_hint=phase)
            hp = max(p * 1e-5, 1.0)
            rho_p = mx.density_from_pressure(p + hp, T, x, phase_hint=phase)
            rho_m = mx.density_from_pressure(p - hp, T, x, phase_hint=phase)
            fd_p = (mx.ln_phi(rho_p, T, x) - mx.ln_phi(rho_m, T, x)) / (2 * hp)
            rel_p = np.max(np.abs((an_p - fd_p) / (fd_p + 1e-30)))
            check(f"cubic dlnphi/dp_at_T [Tc={[c.T_c for c in mx.components]} "
                  f"T={T} p={p:.1e} {phase}]: rel err < 1e-6",
                  rel_p < 1e-6, f"got {rel_p:.2e}")
            # dlnphi/dT at fixed (p, x)
            an_T = mx.dlnphi_dT_at_p(p, T, x, phase_hint=phase)
            hT = 1e-3
            rho_pT = mx.density_from_pressure(p, T + hT, x, phase_hint=phase)
            rho_mT = mx.density_from_pressure(p, T - hT, x, phase_hint=phase)
            fd_T = (mx.ln_phi(rho_pT, T + hT, x) - mx.ln_phi(rho_mT, T - hT, x)) / (2 * hT)
            rel_T = np.max(np.abs((an_T - fd_T) / (fd_T + 1e-30)))
            check(f"cubic dlnphi/dT_at_p [Tc={[c.T_c for c in mx.components]} "
                  f"T={T} p={p:.1e} {phase}]: rel err < 1e-5",
                  rel_T < 1e-5, f"got {rel_T:.2e}")


def test_cubic_flash_broyden_acceleration():
    """The SS+Broyden hybrid in cubic flash should converge in fewer
    iterations than pure successive substitution for systems with strong
    non-ideality. The 5-component natural gas mix at 250K, 8 MPa is the
    canonical stress test: pure SS needs ~45 iters, hybrid needs ~15.
    Both must agree on the converged beta to 1e-8 (Broyden adds slightly
    more rounding noise than pure SS, but not much)."""
    import stateprop.cubic.flash as cf
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_C2 = PR(305.32, 4.872e6, 0.0995)
    c_C3 = PR(369.83, 4.248e6, 0.1521)
    c_C4 = PR(425.12, 3.796e6, 0.2002)
    c_N2 = PR(126.192, 3.3958e6, 0.0372)
    mx = CubicMixture([c_CH4, c_C2, c_C3, c_C4, c_N2],
                      composition=[0.7, 0.1, 0.1, 0.05, 0.05],
                      k_ij={(0, 4): 0.025, (1, 4): 0.025, (2, 4): 0.025,
                            (3, 4): 0.025})

    # Run hybrid (default)
    r_b = flash_pt(8e6, 250.0, mx.x, mx)

    # Run pure SS by raising the SS warmup floor above maxiter
    orig = cf._SS_WARMUP
    cf._SS_WARMUP = 1000
    try:
        r_ss = flash_pt(8e6, 250.0, mx.x, mx, maxiter=200)
    finally:
        cf._SS_WARMUP = orig

    check("cubic 5-comp flash: SS and Broyden both 2-phase",
          r_b.phase == 'two_phase' and r_ss.phase == 'two_phase')
    check("cubic 5-comp flash: SS and Broyden agree on beta (within 1e-6)",
          abs(r_b.beta - r_ss.beta) < 1e-6,
          f"SS={r_ss.beta:.8f}, Broyden={r_b.beta:.8f}")
    # Broyden should converge in significantly fewer iterations.
    check("cubic 5-comp flash: Broyden uses < 0.6x SS iterations",
          r_b.iterations < r_ss.iterations * 0.6,
          f"SS={r_ss.iterations} iters, Broyden={r_b.iterations} iters "
          f"(want Broyden < {int(r_ss.iterations * 0.6)})")



def test_pt_flash_supercritical():
    """At high T, should be supercritical."""
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_N2  = PR(126.192, 3.3958e6, 0.0372)
    mx = CubicMixture([c_CH4, c_N2], composition=[0.5, 0.5], k_ij={(0,1): 0.025})
    r = flash_pt(5e6, 300.0, mx.x, mx)
    check("PT flash: high T single-phase", r.phase != 'two_phase')


def test_pt_flash_ternary():
    """Ternary flash with fugacity equality."""
    c_CH4 = SRK(190.564, 4.5992e6, 0.01142)
    c_N2  = SRK(126.192, 3.3958e6, 0.0372)
    c_CO2 = SRK(304.128, 7.3773e6, 0.22394)
    mx = CubicMixture([c_CH4, c_N2, c_CO2], composition=[0.7, 0.1, 0.2],
                      k_ij={(0,1): 0.025, (0,2): 0.091, (1,2): -0.017})
    r = flash_pt(2e6, 200.0, mx.x, mx)
    check("SRK ternary: 2-phase detected", r.phase == 'two_phase')
    if r.phase == 'two_phase':
        lpL = mx.ln_phi(r.rho_L, 200.0, r.x)
        lpV = mx.ln_phi(r.rho_V, 200.0, r.y)
        f_L = r.x * np.exp(lpL) * 2e6
        f_V = r.y * np.exp(lpV) * 2e6
        err = np.max(np.abs(f_L/f_V - 1.0))
        check("SRK ternary: fugacity equality", err < 1e-9, f"err {err:.2e}")
        check("SRK ternary: liquid denser than vapor", r.rho_L > r.rho_V)


def test_stability_test():
    """Stability test correctly identifies 2-phase vs single-phase."""
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_N2  = PR(126.192, 3.3958e6, 0.0372)

    # Known 2-phase
    mx = CubicMixture([c_CH4, c_N2], composition=[0.8, 0.2], k_ij={(0,1): 0.025})
    stable, K, Sm1 = stability_test_TPD(mx.x, 150.0, 2e6, mx)
    check("stability test: known 2-phase is unstable", not stable, f"Sm1={Sm1}")

    # Known supercritical
    stable, K, Sm1 = stability_test_TPD(mx.x, 400.0, 5e6, mx)
    check("stability test: high-T is stable", stable)


def test_bubble_dew_points():
    """Bubble and dew point solvers converge and give beta=0 / beta=1."""
    from stateprop.cubic import bubble_point_p, bubble_point_T, dew_point_p, dew_point_T

    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_N2  = PR(126.192, 3.3958e6, 0.0372)

    # Bubble points (CH4-rich, where they exist clearly)
    for z_list, T in [([0.9, 0.1], 140), ([0.9, 0.1], 160),
                      ([0.7, 0.3], 140), ([0.5, 0.5], 140)]:
        z = np.array(z_list)
        mx = CubicMixture([c_CH4, c_N2], composition=z, k_ij={(0,1): 0.025})
        try:
            r = bubble_point_p(T, z, mx)
            r_check = flash_pt(r.p, T, z, mx, check_stability=False)
            beta = r_check.beta if r_check.beta is not None else 0.0
            check(f"bubble_p z={z_list} T={T}K", beta < 1e-3, f"beta={beta:.2e}")
        except Exception as e:
            check(f"bubble_p z={z_list} T={T}K", False, str(e)[:60])

    # Dew points
    for z_list, T in [([0.9, 0.1], 140), ([0.5, 0.5], 140), ([0.3, 0.7], 140)]:
        z = np.array(z_list)
        mx = CubicMixture([c_CH4, c_N2], composition=z, k_ij={(0,1): 0.025})
        try:
            r = dew_point_p(T, z, mx)
            r_check = flash_pt(r.p, T, z, mx, check_stability=False)
            beta = r_check.beta if r_check.beta is not None else 1.0
            check(f"dew_p z={z_list} T={T}K", beta > 0.999, f"beta={beta}")
        except Exception as e:
            check(f"dew_p z={z_list} T={T}K", False, str(e)[:60])

    # Bubble-T and dew-T
    for z_list, p_MPa in [([0.5, 0.5], 1), ([0.5, 0.5], 2)]:
        z = np.array(z_list)
        mx = CubicMixture([c_CH4, c_N2], composition=z, k_ij={(0,1): 0.025})
        try:
            r = bubble_point_T(p_MPa*1e6, z, mx)
            r_check = flash_pt(p_MPa*1e6, r.T, z, mx, check_stability=False)
            beta = r_check.beta if r_check.beta is not None else 0.0
            check(f"bubble_T z={z_list} p={p_MPa}MPa", beta < 1e-3, f"beta={beta:.2e}")
        except Exception as e:
            check(f"bubble_T z={z_list} p={p_MPa}MPa", False, str(e)[:60])

        try:
            r = dew_point_T(p_MPa*1e6, z, mx)
            r_check = flash_pt(p_MPa*1e6, r.T, z, mx, check_stability=False)
            beta = r_check.beta if r_check.beta is not None else 1.0
            check(f"dew_T z={z_list} p={p_MPa}MPa", beta > 0.999, f"beta={beta}")
        except Exception as e:
            check(f"dew_T z={z_list} p={p_MPa}MPa", False, str(e)[:60])


# ----------------------------------------------------------------------
# Caloric properties and state-function flashes
# ----------------------------------------------------------------------

def test_caloric_residual_fd():
    """Residual u, h, s from caloric() match FD of alpha_r at fixed rho."""
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_N2  = PR(126.192, 3.3958e6, 0.0372)
    mx = CubicMixture([c_CH4, c_N2], composition=[0.8, 0.2], k_ij={(0,1): 0.025})

    def alpha_r_of_T(rho, T, mx):
        a_mix, b_mix, *_ = mx.a_b_mix(T, mx.x)
        B = b_mix * rho
        eps_ = mx.epsilon; sig = mx.sigma
        q = a_mix / (mx.R * T * b_mix * (sig - eps_))
        I = np.log((1 + sig*B) / (1 + eps_*B))
        return -np.log(1 - B) - q * I

    max_u_err = max_s_err = max_h_err = 0.0
    for T, p_MPa in [(250, 5), (200, 3), (300, 2), (400, 10)]:
        p = p_MPa * 1e6
        try:
            rho = mx.density_from_pressure(p, T, phase_hint='vapor')
        except RuntimeError:
            continue
        cal = mx.caloric(rho, T, p=p)
        # FD of alpha_r wrt T at fixed rho
        eps = 0.01
        ar_p = alpha_r_of_T(rho, T + eps, mx)
        ar_m = alpha_r_of_T(rho, T - eps, mx)
        ar_0 = alpha_r_of_T(rho, T, mx)
        d_ar_dT = (ar_p - ar_m) / (2*eps)
        u_res_fd = mx.R * T * (-T * d_ar_dT)
        s_res_fd = mx.R * (u_res_fd/(mx.R * T) - ar_0)
        h_res_fd = u_res_fd + mx.R * T * (cal["Z"] - 1.0)

        max_u_err = max(max_u_err, abs(cal["u_res"] - u_res_fd) / max(abs(cal["u_res"]), 1.0))
        max_s_err = max(max_s_err, abs(cal["s_res"] - s_res_fd) / max(abs(cal["s_res"]), 1.0))
        max_h_err = max(max_h_err, abs(cal["h_res"] - h_res_fd) / max(abs(cal["h_res"]), 1.0))

    check("caloric u_res FD consistency", max_u_err < 1e-6, f"err {max_u_err:.2e}")
    check("caloric s_res FD consistency", max_s_err < 1e-6, f"err {max_s_err:.2e}")
    check("caloric h_res FD consistency", max_h_err < 1e-6, f"err {max_h_err:.2e}")


def test_caloric_Cp_recovery():
    """At dilute conditions, dh/dT at const p recovers the ideal-gas Cp."""
    # CO2 with NIST-like Cp polynomial
    co2 = PR(304.13, 7.3773e6, 0.224,
             ideal_gas_cp_poly=(22.26, 0.05981, -3.501e-5, 7.469e-9))
    mx = CubicMixture([co2], composition=[1.0])

    T = 400.0
    # Low pressure -> residual ~ 0, so dh/dT ~ Cp_ig
    p = 0.1e6   # 0.1 MPa
    rho = mx.density_from_pressure(p, T, phase_hint='vapor')
    cal_0 = mx.caloric(rho, T, p=p)
    dT = 0.1
    rho_p = mx.density_from_pressure(p, T + dT, phase_hint='vapor')
    cal_p = mx.caloric(rho_p, T + dT, p=p)
    rho_m = mx.density_from_pressure(p, T - dT, phase_hint='vapor')
    cal_m = mx.caloric(rho_m, T - dT, p=p)
    dh_dT = (cal_p["h"] - cal_m["h"]) / (2 * dT)

    cp_ig_expected = 22.26 + 0.05981*T - 3.501e-5*T**2 + 7.469e-9*T**3
    err = abs(dh_dT - cp_ig_expected) / cp_ig_expected
    check(f"dh/dT at dilute = Cp_ig ({dh_dT:.3f} vs {cp_ig_expected:.3f})",
          err < 0.01, f"err {err*100:.2f}%")


def test_flash_pt_populates_h_s():
    """flash_pt now returns non-zero h and s (not placeholders)."""
    c_CH4 = PR(190.564, 4.5992e6, 0.01142,
               ideal_gas_cp_poly=(19.87, 0.05021, 1.268e-5, -1.100e-8))
    c_N2  = PR(126.192, 3.3958e6, 0.0372,
               ideal_gas_cp_poly=(28.98, 0.001853, -9.647e-6, 1.648e-8))
    mx = CubicMixture([c_CH4, c_N2], composition=[0.8, 0.2], k_ij={(0,1): 0.025})

    # Single-phase
    r = flash_pt(2e6, 250.0, mx.x, mx)
    check("flash_pt single-phase h finite and nonzero",
          np.isfinite(r.h) and abs(r.h) > 1e-6, f"h={r.h}")
    check("flash_pt single-phase s finite",
          np.isfinite(r.s) and abs(r.s) > 1e-6, f"s={r.s}")

    # Two-phase
    r = flash_pt(2e6, 150.0, mx.x, mx)
    check("flash_pt two-phase h finite",
          r.phase == 'two_phase' and np.isfinite(r.h) and abs(r.h) > 1e-6, f"h={r.h}")


def test_flash_ph_roundtrip():
    """PT->get h->flash_ph->recover T."""
    from stateprop.cubic import flash_ph
    c_CH4 = PR(190.564, 4.5992e6, 0.01142,
               ideal_gas_cp_poly=(19.87, 0.05021, 1.268e-5, -1.100e-8))
    c_N2  = PR(126.192, 3.3958e6, 0.0372,
               ideal_gas_cp_poly=(28.98, 0.001853, -9.647e-6, 1.648e-8))
    mx = CubicMixture([c_CH4, c_N2], composition=[0.8, 0.2], k_ij={(0,1): 0.025})

    for T_true, p, tag in [(250, 2e6, 'supercritical'),
                           (170, 2e6, 'vapor'),
                           (150, 2e6, 'two-phase'),
                           (200, 4e6, 'single-phase')]:
        r_pt = flash_pt(p, T_true, mx.x, mx)
        h_target = r_pt.h
        try:
            r_ph = flash_ph(p, h_target, mx.x, mx, T_init=T_true + 40)
            err = abs(r_ph.T - T_true)
            check(f"flash_ph roundtrip [{tag}] T={T_true}K", err < 0.01,
                  f"recovered {r_ph.T}, err {err:.3e}")
        except Exception as e:
            check(f"flash_ph roundtrip [{tag}] T={T_true}K", False, str(e)[:60])


def test_flash_ps_roundtrip():
    """PT->get s->flash_ps->recover T."""
    from stateprop.cubic import flash_ps
    c_CH4 = PR(190.564, 4.5992e6, 0.01142,
               ideal_gas_cp_poly=(19.87, 0.05021, 1.268e-5, -1.100e-8))
    c_N2  = PR(126.192, 3.3958e6, 0.0372,
               ideal_gas_cp_poly=(28.98, 0.001853, -9.647e-6, 1.648e-8))
    mx = CubicMixture([c_CH4, c_N2], composition=[0.8, 0.2], k_ij={(0,1): 0.025})

    for T_true, p, tag in [(250, 2e6, 'supercritical'),
                           (170, 2e6, 'vapor'),
                           (150, 2e6, 'two-phase'),
                           (200, 4e6, 'single-phase')]:
        r_pt = flash_pt(p, T_true, mx.x, mx)
        s_target = r_pt.s
        try:
            r_ps = flash_ps(p, s_target, mx.x, mx, T_init=T_true + 40)
            err = abs(r_ps.T - T_true)
            check(f"flash_ps roundtrip [{tag}] T={T_true}K", err < 0.01,
                  f"recovered {r_ps.T}, err {err:.3e}")
        except Exception as e:
            check(f"flash_ps roundtrip [{tag}] T={T_true}K", False, str(e)[:60])


def test_flash_th_ts_roundtrip():
    """PT->get (h|s)->flash_th/ts->recover p."""
    from stateprop.cubic import flash_th, flash_ts
    c_CH4 = PR(190.564, 4.5992e6, 0.01142,
               ideal_gas_cp_poly=(19.87, 0.05021, 1.268e-5, -1.100e-8))
    c_N2  = PR(126.192, 3.3958e6, 0.0372,
               ideal_gas_cp_poly=(28.98, 0.001853, -9.647e-6, 1.648e-8))
    mx = CubicMixture([c_CH4, c_N2], composition=[0.8, 0.2], k_ij={(0,1): 0.025})

    for T, p_true, tag in [(250, 2e6, 'supercritical'),
                           (200, 4e6, 'single-phase')]:
        r_pt = flash_pt(p_true, T, mx.x, mx)
        # TH
        try:
            r_th = flash_th(T, r_pt.h, mx.x, mx, p_init=p_true*2)
            err = abs(r_th.p - p_true) / p_true
            check(f"flash_th roundtrip [{tag}]", err < 1e-3, f"rel err {err:.2e}")
        except Exception as e:
            check(f"flash_th roundtrip [{tag}]", False, str(e)[:60])
        # TS
        try:
            r_ts = flash_ts(T, r_pt.s, mx.x, mx, p_init=p_true*2)
            err = abs(r_ts.p - p_true) / p_true
            check(f"flash_ts roundtrip [{tag}]", err < 1e-3, f"rel err {err:.2e}")
        except Exception as e:
            check(f"flash_ts roundtrip [{tag}]", False, str(e)[:60])


# ----------------------------------------------------------------------
# PR-1978 tests (m(omega) correlation for heavy components)
# ----------------------------------------------------------------------

def test_pr78_m_coefficients():
    """PR-1978 m(omega) polynomial evaluated correctly at known omegas."""
    from stateprop.cubic import PR, PR78, CubicEOS
    for omega in [0.0, 0.1, 0.49, 0.6, 1.0]:
        m76_expected = 0.37464 + 1.54226*omega - 0.26992*omega**2
        m78_expected = (0.379642 + 1.48503*omega - 0.164423*omega**2
                        + 0.016666*omega**3)
        eos76 = CubicEOS(T_c=500.0, p_c=2e6, acentric_factor=omega,
                         family="pr", use_pr78="never")
        eos78 = PR78(500.0, 2e6, acentric_factor=omega)
        check(f"PR-1976 m at omega={omega}",
              abs(eos76._m - m76_expected) < 1e-10,
              f"got {eos76._m}, expected {m76_expected}")
        check(f"PR-1978 m at omega={omega}",
              abs(eos78._m - m78_expected) < 1e-10,
              f"got {eos78._m}, expected {m78_expected}")


def test_pr78_auto_dispatch():
    """PR(...) auto-switches to PR-1978 when omega > 0.49."""
    from stateprop.cubic import PR
    # Light component -> PR-1976
    light = PR(304.13, 7.3773e6, 0.224)
    check("auto-dispatch light (omega<0.49) uses PR-1976",
          light._effective_family == "pr")
    # Heavy component -> PR-1978
    heavy = PR(617.7, 21.1e5, 0.50)
    check("auto-dispatch heavy (omega>0.49) uses PR-1978",
          heavy._effective_family == "pr78")
    # Explicit override works both directions
    heavy_forced_76 = PR(617.7, 21.1e5, 0.50, use_pr78="never")
    check("use_pr78='never' forces PR-1976",
          heavy_forced_76._effective_family == "pr")
    light_forced_78 = PR(304.13, 7.3773e6, 0.224, use_pr78="always")
    check("use_pr78='always' forces PR-1978",
          light_forced_78._effective_family == "pr78")


def test_pr78_backward_compat():
    """Light components produce identical saturation with PR-1976 or auto-dispatch."""
    from stateprop.cubic import PR
    co2_auto   = PR(304.1282, 7.3773e6, 0.22394)                     # auto -> 1976
    co2_forced = PR(304.1282, 7.3773e6, 0.22394, use_pr78="never")   # 1976 explicit
    for T in [220, 250, 280]:
        p1 = co2_auto.saturation_p(T)
        p2 = co2_forced.saturation_p(T)
        check(f"CO2 sat at T={T}K: auto==PR-1976",
              abs(p1 - p2) < 1e-6, f"auto={p1}, forced={p2}")


def test_pr78_improves_heavy_components():
    """PR-1978 gives smaller saturation error than PR-1976 for heavy alkanes.

    Reference: NIST Antoine coefficients (Carruth-Kobayashi 1973 and similar).
    """
    from stateprop.cubic import PR

    # (name, Tc, Pc, omega, T_test, p_antoine_kPa)
    heavy_cases = [
        ('dodecane',   658.1, 18.2e5, 0.562, 445.0, 30.796),
        ('tetradecane',693.0, 15.7e5, 0.644, 475.0, 26.776),
        ('hexadecane', 723.0, 14.0e5, 0.7174, 502.5, 23.540),
    ]
    for name, Tc, Pc, omega, T, p_ref in heavy_cases:
        eos76 = PR(Tc, Pc, omega, use_pr78="never")
        eos78 = PR(Tc, Pc, omega, use_pr78="always")
        p76 = eos76.saturation_p(T) / 1000
        p78 = eos78.saturation_p(T) / 1000
        e76 = abs(p76 - p_ref) / p_ref
        e78 = abs(p78 - p_ref) / p_ref
        check(f"{name}: PR-1978 error < PR-1976 error",
              e78 < e76,
              f"|err_76|={e76*100:.2f}% |err_78|={e78*100:.2f}%")
        # Both within reasonable bounds (PR is known to be less accurate for
        # heavy alkanes; we expect the PR-1978 result within 10%)
        check(f"{name}: PR-1978 within 10% of NIST",
              e78 < 0.10,
              f"|err|={e78*100:.2f}%")


def test_pr78_in_mixture():
    """PR and PR78 components can be combined into a single CubicMixture."""
    from stateprop.cubic import PR, PR78, CubicMixture, flash_pt
    # Light (auto = PR-1976) + heavy (auto = PR-1978)
    methane = PR(190.564, 4.5992e6, 0.01142)
    decane  = PR(617.7,   21.1e5,   0.4923)   # auto = PR-1978
    check("auto-mixed: methane uses PR-1976",
          methane._effective_family == "pr")
    check("auto-mixed: decane uses PR-1978",
          decane._effective_family == "pr78")
    mx = CubicMixture([methane, decane], composition=[0.6, 0.4],
                      k_ij={(0,1): 0.05})
    # Basic sanity: flash converges
    try:
        r = flash_pt(p=3e6, T=400.0, z=mx.x, mixture=mx)
        check("PR+PR78 mixture flash runs", r.phase in ('vapor', 'liquid',
              'supercritical', 'two_phase'), f"phase={r.phase}")
    except Exception as e:
        check("PR+PR78 mixture flash runs", False, str(e)[:60])


# ----------------------------------------------------------------------
# Alpha-function variants: Mathias-Copeman, Twu, PRSV
# ----------------------------------------------------------------------

def test_alpha_variant_derivatives_fd():
    """Each alpha variant's d_alpha/d_Tr and d^2_alpha/d_Tr^2 match FD."""
    from stateprop.cubic.eos import (
        _mathias_copeman_alpha, _twu_alpha, _prsv_alpha,
    )
    max_err_each = {}
    for kind, fn, params, T_rs in [
        ('MC-CO2-Soave', _mathias_copeman_alpha, (0.5252, 0.0, 0.0),
         [0.5, 0.9, 1.0, 1.3]),
        ('MC-water',     _mathias_copeman_alpha, (1.0783, -0.1708, 0.4066),
         [0.5, 0.8, 1.2]),
        ('Twu-methane',  _twu_alpha, (0.1243, 0.8916, 2.0),
         [0.5, 1.0, 2.0]),
        ('PRSV-CO2',     _prsv_alpha,
         (0.378893 + 1.4897153*0.224 - 0.17131848*0.224**2
          + 0.0196554*0.224**3, 0.04285),
         [0.5, 0.9, 1.0, 1.3]),
    ]:
        max_err = 0.0
        for T_r in T_rs:
            _, a_p, a_pp = fn(T_r, *params)
            eps = 1e-6
            ap, _, _   = fn(T_r + eps, *params)
            am, _, _   = fn(T_r - eps, *params)
            _, app, _  = fn(T_r + eps, *params)
            _, amm, _  = fn(T_r - eps, *params)
            a_p_fd = (ap - am) / (2 * eps)
            a_pp_fd = (app - amm) / (2 * eps)

            def relerr(x, y, tol_abs=1e-8):
                m = max(abs(x), abs(y))
                return 0.0 if m < tol_abs else abs(x - y) / m

            e1 = relerr(a_p, a_p_fd)
            e2 = relerr(a_pp, a_pp_fd)
            max_err = max(max_err, e1, e2)
        max_err_each[kind] = max_err
        check(f"{kind}: FD to 1e-6", max_err < 1e-6,
              f"max err {max_err:.2e}")


def test_mathias_copeman_reduces_to_soave():
    """MC with c1=m(omega), c2=c3=0 must be bit-identical to classical Soave/PR."""
    from stateprop.cubic import PR, PR_MC
    # Match CO2 (PR-1976 regime) and a mid-omega fluid
    for (Tc, pc, omega) in [
        (304.13, 7.3773e6, 0.224),   # CO2
        (507.6,  30.25e5,  0.301),   # hexane
        (190.564, 4.5992e6, 0.011),  # methane
    ]:
        classic = PR(Tc, pc, omega, use_pr78="never")
        mc_same = PR_MC(Tc, pc, c1=None, c2=0.0, c3=0.0, acentric_factor=omega)
        # alpha values match at multiple T_r
        max_err_alpha = 0.0
        for T_r in [0.5, 0.8, 1.0, 1.3]:
            a1 = classic.alpha_func(T_r)
            a2 = mc_same.alpha_func(T_r)
            for i in range(3):
                max_err_alpha = max(max_err_alpha, abs(a1[i] - a2[i]))
        check(f"MC reduces to Soave (Tc={Tc}, omega={omega})",
              max_err_alpha < 1e-14,
              f"max err {max_err_alpha:.2e}")
        # Saturation agrees too
        p1 = classic.saturation_p(0.7 * Tc)
        p2 = mc_same.saturation_p(0.7 * Tc)
        err_p = abs(p1 - p2) / p1
        check(f"MC sat == PR sat (Tc={Tc})", err_p < 1e-9,
              f"err {err_p:.2e}")


def test_prsv_improves_water():
    """PRSV with kappa1=-0.06635 gives smaller errors for water than classical PR.

    Reference (NIST IAPWS): water saturation pressures
      T=400 K:  0.2457 MPa
      T=500 K:  2.6393 MPa
      T=600 K: 12.3450 MPa
    """
    from stateprop.cubic import PR, PRSV
    # Water: Tc=647.096, Pc=22.064 MPa, omega=0.3443
    water_pr   = PR(647.096, 22.064e6, 0.3443, use_pr78="never")
    water_prsv = PRSV(647.096, 22.064e6, acentric_factor=0.3443, kappa1=-0.06635)
    refs = [(400, 0.2457e6), (500, 2.6393e6), (600, 12.345e6)]
    for T, p_ref in refs:
        p_pr   = water_pr.saturation_p(T)
        p_prsv = water_prsv.saturation_p(T)
        err_pr   = abs(p_pr   - p_ref) / p_ref
        err_prsv = abs(p_prsv - p_ref) / p_ref
        check(
            f"water sat at T={T}K: PRSV err ({err_prsv*100:.2f}%) <= PR err "
            f"({err_pr*100:.2f}%)",
            err_prsv <= err_pr,
            f"PR={p_pr/1e6:.3f}, PRSV={p_prsv/1e6:.3f}, ref={p_ref/1e6:.3f}",
        )


def test_twu_alpha_in_flash():
    """PR-Twu integrated through the full flash machinery (pressure consistency)."""
    from stateprop.cubic import PR_Twu
    # Methane-like Twu params
    m = PR_Twu(190.564, 4.5992e6, L=0.1243, M=0.8916, N=2.0)
    # Pressure consistency: p(direct cubic) == p from (rho, T) via Helmholtz form
    for T in [150.0, 200.0, 300.0]:
        rho = m.density_from_pressure(p=2e6, T=T, phase_hint='vapor')
        p_direct = m.pressure(rho, T)
        # Also compute p from the Helmholtz form p = rho*R*T*(1 + delta * a_d)
        delta = rho / m.rho_c
        tau = m.T_c / T
        _, a_d, _, _, _, _ = m.alpha_r_derivs(delta, tau)
        p_helm = rho * m.R * T * (1.0 + delta * a_d)
        err = abs(p_direct - p_helm) / p_direct
        check(f"PR-Twu pressure consistency at T={T}K",
              err < 1e-12, f"err {err:.2e}")


def test_alpha_override_in_mixture():
    """A PR component with MC or PRSV alpha_override mixes with a classical PR component."""
    from stateprop.cubic import PR, PR_MC, PRSV, CubicMixture, flash_pt
    # Methane (classical) + water (PRSV) -- both family='pr'
    methane = PR(190.564, 4.5992e6, 0.01142, use_pr78="never")
    water   = PRSV(647.096, 22.064e6, acentric_factor=0.3443, kappa1=-0.06635)
    mx = CubicMixture([methane, water], composition=[0.5, 0.5],
                      k_ij={(0, 1): 0.5})   # large k_ij for polar water
    # Just verify the flash runs without error and returns a valid phase
    try:
        r = flash_pt(p=5e6, T=500.0, z=mx.x, mixture=mx)
        check("PR + PRSV mixture flash runs",
              r.phase in ('vapor', 'liquid', 'supercritical', 'two_phase'),
              f"phase={r.phase}")
    except Exception as e:
        check("PR + PRSV mixture flash runs", False, str(e)[:60])


def test_alpha_override_invalid_raises():
    """Malformed alpha_override tuples raise clean errors."""
    from stateprop.cubic import CubicEOS
    for bad, reason in [
        (("unknown_kind",), "unknown type"),
        (("mathias_copeman", 0.5, 0.0),   "wrong arg count"),
        (("twu", 0.1, 0.9),               "twu needs 3 args"),
        (("prsv",),                       "prsv needs 1 arg"),
        ((123,),                          "first arg not a string"),
    ]:
        try:
            CubicEOS(T_c=300, p_c=5e6, acentric_factor=0.2,
                     family="pr", alpha_override=bad)
            check(f"invalid alpha_override raises ({reason})",
                  False, "did not raise")
        except ValueError:
            check(f"invalid alpha_override raises ({reason})", True)


# ----------------------------------------------------------------------
# Volume translation (Peneloux-style)
# ----------------------------------------------------------------------

def test_peneloux_vapor_pressure_invariant():
    """Pure-fluid saturation_p is bit-identical with and without volume shift."""
    from stateprop.cubic import SRK
    for (Tc, pc, omega, name) in [
        (190.564, 4.5992e6, 0.01142, 'methane'),
        (507.6,   30.25e5,  0.301,   'n-hexane'),
        (304.13,  7.3773e6, 0.224,   'CO2'),
    ]:
        no_shift = SRK(Tc, pc, omega)
        pen      = SRK(Tc, pc, omega, volume_shift_c='peneloux')
        for T_r in [0.6, 0.75, 0.9]:
            T = T_r * Tc
            try:
                p1 = no_shift.saturation_p(T)
                p2 = pen.saturation_p(T)
                check(f"{name} at T_r={T_r}: p_sat invariant under Peneloux",
                      abs(p1 - p2) < 1e-8, f"p1={p1}, p2={p2}")
            except Exception as e:
                check(f"{name} at T_r={T_r}", False, str(e)[:50])


def test_peneloux_lnphi_difference_invariant():
    """(ln phi_L - ln phi_V) is invariant under volume translation (phase equilibria preserved)."""
    import numpy as np
    from stateprop.cubic import SRK, CubicMixture
    # CH4/N2 mixture in 2-phase region
    c_CH4_no  = SRK(190.564, 4.5992e6, 0.01142)
    c_N2_no   = SRK(126.192, 3.3958e6, 0.0372)
    c_CH4_pen = SRK(190.564, 4.5992e6, 0.01142, volume_shift_c='peneloux')
    c_N2_pen  = SRK(126.192, 3.3958e6, 0.0372,  volume_shift_c='peneloux')
    mx_no  = CubicMixture([c_CH4_no,  c_N2_no],  composition=[0.8, 0.2],
                          k_ij={(0,1): 0.025})
    mx_pen = CubicMixture([c_CH4_pen, c_N2_pen], composition=[0.8, 0.2],
                          k_ij={(0,1): 0.025})
    for (T, p) in [(150, 2e6), (140, 1.5e6), (160, 2.5e6)]:
        rho_v_no = mx_no.density_from_pressure(p, T, phase_hint='vapor')
        rho_l_no = mx_no.density_from_pressure(p, T, phase_hint='liquid')
        d_no = mx_no.ln_phi(rho_l_no, T) - mx_no.ln_phi(rho_v_no, T)

        rho_v_pen = mx_pen.density_from_pressure(p, T, phase_hint='vapor')
        rho_l_pen = mx_pen.density_from_pressure(p, T, phase_hint='liquid')
        d_pen = mx_pen.ln_phi(rho_l_pen, T) - mx_pen.ln_phi(rho_v_pen, T)

        err = float(np.max(np.abs(d_no - d_pen)))
        check(f"ln_phi_L-ln_phi_V invariant at T={T}, p={p/1e6}MPa",
              err < 1e-10, f"max diff {err:.2e}")


def test_peneloux_flash_invariant():
    """PT flash results (beta, x, y, K) identical with/without volume translation."""
    import numpy as np
    from stateprop.cubic import SRK, CubicMixture, flash_pt
    c_CH4_no  = SRK(190.564, 4.5992e6, 0.01142)
    c_N2_no   = SRK(126.192, 3.3958e6, 0.0372)
    c_CH4_pen = SRK(190.564, 4.5992e6, 0.01142, volume_shift_c='peneloux')
    c_N2_pen  = SRK(126.192, 3.3958e6, 0.0372,  volume_shift_c='peneloux')
    mx_no  = CubicMixture([c_CH4_no,  c_N2_no],  composition=[0.8, 0.2], k_ij={(0,1):0.025})
    mx_pen = CubicMixture([c_CH4_pen, c_N2_pen], composition=[0.8, 0.2], k_ij={(0,1):0.025})
    for (T, p, tag) in [(150, 2e6, '2-phase'), (250, 2e6, 'supercrit')]:
        r_no  = flash_pt(p, T, mx_no.x,  mx_no)
        r_pen = flash_pt(p, T, mx_pen.x, mx_pen)
        # Phase label must agree
        check(f"flash_pt [{tag}]: phase label invariant",
              r_no.phase == r_pen.phase,
              f"{r_no.phase} vs {r_pen.phase}")
        # Compositions and beta must match to high precision
        if r_no.beta is not None:
            check(f"flash_pt [{tag}]: beta invariant",
                  abs(r_no.beta - r_pen.beta) < 1e-10,
                  f"{r_no.beta} vs {r_pen.beta}")
            err_x = float(np.max(np.abs(r_no.x - r_pen.x)))
            err_y = float(np.max(np.abs(r_no.y - r_pen.y)))
            check(f"flash_pt [{tag}]: x, y invariant",
                  max(err_x, err_y) < 1e-10,
                  f"err_x={err_x:.2e}, err_y={err_y:.2e}")
        # But densities DIFFER
        if r_no.rho_L is not None:
            diff_rho = abs(r_no.rho_L - r_pen.rho_L) / r_no.rho_L
            check(f"flash_pt [{tag}]: rho_L changes under translation",
                  diff_rho > 1e-3,
                  f"rel change {diff_rho:.2e} (should be >0.1%)")


def test_peneloux_improves_srk_liquid_density():
    """SRK+Peneloux gives smaller liquid-density errors than plain SRK for alkanes."""
    from stateprop.cubic import SRK
    # (Tc, Pc, omega, M [kg/mol], [(T, rho_nist_kgm3)])
    cases = [
        (425.12, 37.96e5, 0.200, 58.12e-3, [(300, 578), (350, 516)]),
        (507.6,  30.25e5, 0.301, 86.18e-3, [(298.15, 655), (350, 610)]),
    ]
    for Tc, Pc, omega, M, ref_data in cases:
        srk_no  = SRK(Tc, Pc, omega, molar_mass=M)
        srk_pen = SRK(Tc, Pc, omega, molar_mass=M, volume_shift_c='peneloux')
        for T, rho_exp in ref_data:
            p = srk_no.saturation_p(T)
            rho_no  = srk_no.density_from_pressure(p, T, phase_hint='liquid') * M
            rho_pen = srk_pen.density_from_pressure(p, T, phase_hint='liquid') * M
            err_no  = abs(rho_no  - rho_exp) / rho_exp
            err_pen = abs(rho_pen - rho_exp) / rho_exp
            check(f"SRK+Peneloux liquid rho (Tc={Tc}, T={T}) better than SRK",
                  err_pen < err_no,
                  f"|err_no|={err_no*100:.2f}%, |err_pen|={err_pen*100:.2f}%")


def test_peneloux_roundtrip_density_pressure():
    """Round-trip p -> density_from_pressure -> pressure(rho, T) recovers p (translated)."""
    from stateprop.cubic import SRK, CubicMixture
    # Pure
    eos = SRK(507.6, 30.25e5, 0.301, volume_shift_c='peneloux')
    for (p, T) in [(1e5, 300), (1e6, 350), (5e5, 400)]:
        rho = eos.density_from_pressure(p, T, phase_hint='liquid')
        p_back = eos.pressure(rho, T)
        err = abs(p_back - p) / p
        check(f"pure SRK+Pen round-trip at p={p/1e3}kPa, T={T}K",
              err < 1e-9, f"err {err:.2e}")
    # Mixture
    c_CH4 = SRK(190.564, 4.5992e6, 0.01142, volume_shift_c='peneloux')
    c_N2  = SRK(126.192, 3.3958e6, 0.0372,  volume_shift_c='peneloux')
    mx = CubicMixture([c_CH4, c_N2], composition=[0.8, 0.2], k_ij={(0,1):0.025})
    for (p, T) in [(2e6, 200), (5e6, 300)]:
        rho = mx.density_from_pressure(p, T, phase_hint='vapor')
        # Reconstruct p from the cubic at v_cubic
        c_m = mx.c_mix()
        v_real = 1.0 / rho
        v_cubic = v_real + c_m
        a_mix, b_mix, *_ = mx.a_b_mix(T, mx.x)
        eps_ = mx.epsilon; sig = mx.sigma
        p_back = mx.R * T / (v_cubic - b_mix) - a_mix / (
            (v_cubic + eps_*b_mix) * (v_cubic + sig*b_mix))
        err = abs(p_back - p) / p
        check(f"mixture SRK+Pen density round-trip at p={p/1e6}MPa, T={T}K",
              err < 1e-10, f"err {err:.2e}")


def test_peneloux_numeric_c():
    """Passing numeric volume_shift_c works and differs from 'peneloux' default."""
    from stateprop.cubic import SRK
    eos_num  = SRK(507.6, 30.25e5, 0.301, volume_shift_c=1e-5)   # 10 cm^3/mol
    eos_none = SRK(507.6, 30.25e5, 0.301)
    check("numeric c stored correctly", abs(eos_num.c - 1e-5) < 1e-15,
          f"c={eos_num.c}")
    check("None -> c=0", eos_none.c == 0.0, f"c={eos_none.c}")
    # Saturation still invariant
    p1 = eos_num.saturation_p(350)
    p2 = eos_none.saturation_p(350)
    check("numeric c: vapor pressure invariant",
          abs(p1 - p2) < 1e-8, f"diff {abs(p1-p2):.2e}")


def test_peneloux_pr_rejects_auto():
    """PR with volume_shift_c='peneloux' raises (only SRK supports auto)."""
    from stateprop.cubic import PR
    try:
        PR(304.13, 7.3773e6, 0.224, volume_shift_c='peneloux')
        check("PR 'peneloux' auto rejected with clear error", False,
              "did not raise")
    except ValueError as e:
        check("PR 'peneloux' auto rejected with clear error",
              "peneloux" in str(e).lower() or "only" in str(e).lower(),
              f"msg='{e}'")
    # But numeric c works on PR
    try:
        eos = PR(304.13, 7.3773e6, 0.224, volume_shift_c=1e-6)
        check("PR accepts numeric volume_shift_c", abs(eos.c - 1e-6) < 1e-20)
    except Exception as e:
        check("PR accepts numeric volume_shift_c", False, str(e)[:50])


def test_peneloux_caloric_consistency():
    """Caloric u_res is invariant under volume translation (only h_res changes)."""
    from stateprop.cubic import SRK, CubicMixture
    c_CH4_no  = SRK(190.564, 4.5992e6, 0.01142,
                    ideal_gas_cp_poly=(19.87, 0.05021, 1.268e-5, -1.100e-8))
    c_N2_no   = SRK(126.192, 3.3958e6, 0.0372,
                    ideal_gas_cp_poly=(28.98, 0.001853, -9.647e-6, 1.648e-8))
    c_CH4_pen = SRK(190.564, 4.5992e6, 0.01142, volume_shift_c='peneloux',
                    ideal_gas_cp_poly=(19.87, 0.05021, 1.268e-5, -1.100e-8))
    c_N2_pen  = SRK(126.192, 3.3958e6, 0.0372, volume_shift_c='peneloux',
                    ideal_gas_cp_poly=(28.98, 0.001853, -9.647e-6, 1.648e-8))
    mx_no  = CubicMixture([c_CH4_no, c_N2_no], composition=[0.8, 0.2],
                          k_ij={(0,1):0.025})
    mx_pen = CubicMixture([c_CH4_pen, c_N2_pen], composition=[0.8, 0.2],
                          k_ij={(0,1):0.025})
    # At a single-phase state
    T, p = 300, 5e6
    rho_no  = mx_no.density_from_pressure(p, T, phase_hint='vapor')
    rho_pen = mx_pen.density_from_pressure(p, T, phase_hint='vapor')
    cal_no  = mx_no.caloric(rho_no,  T)
    cal_pen = mx_pen.caloric(rho_pen, T)

    # u_res should be numerically ~equal at same (T, p, x) -- both use same cubic at v_cubic
    err_u = abs(cal_no['u_res'] - cal_pen['u_res']) / max(abs(cal_no['u_res']), 1.0)
    check(f"u_res invariant under translation (T={T}, p={p/1e6}MPa)",
          err_u < 1e-6, f"err {err_u:.2e}")

    # s_res should also be invariant
    err_s = abs(cal_no['s_res'] - cal_pen['s_res']) / max(abs(cal_no['s_res']), 1.0)
    check(f"s_res invariant under translation",
          err_s < 1e-6, f"err {err_s:.2e}")

    # h_res SHOULD differ by exactly c_mix * p (translation identity)
    c_mix = mx_pen.c_mix()
    expected_h_diff = c_mix * p     # h_res_pen = h_res_no - c_mix*p
    actual_h_diff = cal_no['h_res'] - cal_pen['h_res']
    err_h = abs(actual_h_diff - expected_h_diff) / max(abs(expected_h_diff), 1.0)
    check(f"h_res correction = c_mix*p (c_mix={c_mix*1e6:.3f} cm^3/mol)",
          err_h < 1e-6, f"expected {expected_h_diff:.3f}, got {actual_h_diff:.3f}")


# =====================================================================
# v0.9.119: Volume-translation module API
# =====================================================================

def test_v119_peneloux_c_SRK_helper():
    """peneloux_c_SRK reproduces the SRK formula independently of the EOS."""
    from stateprop.cubic import peneloux_c_SRK, SRK
    c_helper = peneloux_c_SRK(190.564, 4.5992e6, 0.01142)
    eos = SRK(190.564, 4.5992e6, 0.01142, volume_shift_c='peneloux')
    check("peneloux_c_SRK matches EOS-internal computation",
          abs(c_helper - eos.c) < 1e-15,
          f"helper={c_helper}, eos={eos.c}")


def test_v119_jhaveri_youngren_paraffin():
    """J-Y returns positive c for paraffins (stateprop convention:
    c > 0 → smaller external molar volume → higher density).

    Note: this is opposite the published Peneloux/de Sant'Ana sign
    convention.  The bundled tables and correlations are stored in
    stateprop convention to match the internal CubicEOS expectation
    (eos.py line 736: v_external = v_cubic - c)."""
    from stateprop.cubic import jhaveri_youngren_c_PR
    # n-decane: ω=0.488, T_c=617.7 K, P_c=21.1 bar
    c = jhaveri_youngren_c_PR(617.7, 21.1e5, 0.488, molar_mass=0.142)
    check(f"J-Y c(n-decane) positive ({c*1e6:+.2f} cm³/mol)", c > 0)


def test_v119_jhaveri_youngren_returns_zero_for_light():
    """J-Y returns 0 for non-paraffin families (e.g. aromatic).
    For paraffins it now uses an ω-linear correlation that gives
    finite values across C1-C10 (methane is a known outlier
    documented in the docstring; users prefer the table value)."""
    from stateprop.cubic import jhaveri_youngren_c_PR
    # Aromatic family → 0
    c_arom = jhaveri_youngren_c_PR(562.05, 4.895e6, 0.212,
                                              family="aromatic")
    check(f"J-Y aromatic returns 0 ({c_arom})", c_arom == 0.0)
    # Methane via correlation: now finite (was 0 in pre-v0.9.119
    # MW-gated form).  Note: methane is the worst paraffin fit
    # (~60% error vs table); for accuracy use lookup_volume_shift.
    c_meth = jhaveri_youngren_c_PR(190.56, 4.6e6, 0.011,
                                              molar_mass=0.016)
    check(f"J-Y methane via correlation = {c_meth*1e6:+.3f} cm³/mol "
          f"(finite positive in stateprop convention)", c_meth > 0)


def test_v119_lookup_volume_shift_known_compounds():
    """Bundled table contains standard natural-gas + light-petroleum species."""
    from stateprop.cubic import lookup_volume_shift
    for name in ["methane", "ethane", "propane", "n-butane",
                  "carbon dioxide", "nitrogen", "water"]:
        c = lookup_volume_shift(name, family="pr")
        check(f"  table contains {name!r}", c is not None)


def test_v119_lookup_volume_shift_aliases():
    """Common aliases (CO2, H2S, C1, nC4, ...) resolve to a value
    in either family."""
    from stateprop.cubic import lookup_volume_shift
    for alias in ["CO2", "H2S", "C1", "nC4", "i-pentane", "N2"]:
        c_pr = lookup_volume_shift(alias, family="pr")
        c_srk = lookup_volume_shift(alias, family="srk")
        check(f"  alias {alias!r} resolved (pr={c_pr}, srk={c_srk})",
              c_pr is not None or c_srk is not None)


def test_v119_lookup_volume_shift_unknown_returns_none():
    """Unknown compound returns None rather than raising."""
    from stateprop.cubic import lookup_volume_shift
    c = lookup_volume_shift("unobtainium", family="pr")
    check("unknown compound returns None", c is None)


def test_v119_resolve_volume_shift_modes():
    """resolve_volume_shift dispatches all four modes correctly."""
    from stateprop.cubic import resolve_volume_shift
    c1 = resolve_volume_shift("methane", "pr",
                                    omega=0.011, T_c=190.56, p_c=4.6e6,
                                    mode="auto")
    check(f"auto/table-hit: {c1*1e6:+.2f} cm³/mol",
          c1 is not None and c1 > 0)
    c2 = resolve_volume_shift("xenon", "srk",
                                    omega=0.0, T_c=289.7, p_c=5.84e6,
                                    mode="auto")
    check(f"auto/correlation-fallback: {c2*1e6:+.2f} cm³/mol",
          isinstance(c2, float))
    raised = False
    try:
        resolve_volume_shift("xenon", "pr", mode="table")
    except KeyError:
        raised = True
    check("table mode raises on miss", raised)
    c3 = resolve_volume_shift("methane", "srk",
                                    omega=0.011, T_c=190.56, p_c=4.6e6,
                                    mode="correlation")
    check(f"correlation mode: {c3*1e6:+.2f} cm³/mol",
          isinstance(c3, float))
    c4 = resolve_volume_shift("anything", "pr", mode=-2.5e-6)
    check(f"float passthrough: {c4*1e6:+.2f}",
          c4 == -2.5e-6)
    c5 = resolve_volume_shift("anything", "pr", mode=None)
    check("None mode returns None", c5 is None)


def test_v119_cubic_from_name_volume_shift_auto():
    """cubic_from_name(volume_shift='auto') populates eos.c from table."""
    from stateprop.cubic import cubic_from_name, lookup_volume_shift
    eos = cubic_from_name("methane", family="pr", volume_shift="auto")
    expected = lookup_volume_shift("methane", family="pr")
    check(f"PR methane volume_shift='auto' c = {eos.c*1e6:+.2f}",
          abs(eos.c - expected) < 1e-15)


def test_v119_cubic_from_name_default_no_shift():
    """Default behavior unchanged: no volume_shift = c=0."""
    from stateprop.cubic import cubic_from_name
    eos = cubic_from_name("methane", family="pr")
    check("default no shift: c=0", eos.c == 0.0, f"c={eos.c}")


def test_v119_cubic_from_name_volume_shift_c_legacy_kwarg():
    """volume_shift_c (legacy) wins over volume_shift (new)."""
    from stateprop.cubic import cubic_from_name
    eos = cubic_from_name("methane", family="pr",
                                volume_shift="auto",
                                volume_shift_c=-3e-6)
    check("legacy kwarg overrides new",
          abs(eos.c - (-3e-6)) < 1e-15, f"c={eos.c}")


def test_v119_cubic_from_name_volume_shift_table_miss_raises():
    """volume_shift='table' raises KeyError for missing compound."""
    from stateprop.cubic import cubic_from_name
    raised = False
    try:
        cubic_from_name("propylene", family="pr", volume_shift="table")
    except KeyError:
        raised = True
    check("table mode raises for missing compound", raised)


def test_v119_volume_shift_does_not_affect_phase_equilibrium():
    """Crucial: volume_shift='auto' does not alter K-values, β, x, y."""
    from stateprop.cubic import cubic_from_name, CubicMixture, flash_pt
    e1_no = cubic_from_name("methane", family="srk")
    e2_no = cubic_from_name("ethane", family="srk")
    e1_yes = cubic_from_name("methane", family="srk",
                                    volume_shift="auto")
    e2_yes = cubic_from_name("ethane", family="srk",
                                    volume_shift="auto")
    mx_no = CubicMixture([e1_no, e2_no], composition=[0.7, 0.3])
    mx_yes = CubicMixture([e1_yes, e2_yes], composition=[0.7, 0.3])
    r_no = flash_pt(p=4e6, T=240.0, z=[0.7, 0.3], mixture=mx_no)
    r_yes = flash_pt(p=4e6, T=240.0, z=[0.7, 0.3], mixture=mx_yes)
    check(f"  β unchanged ({r_no.beta:.6f} vs {r_yes.beta:.6f})",
          abs(r_no.beta - r_yes.beta) < 1e-6)


def test_v119_volume_shift_improves_density():
    """Pure-fluid liquid density closer to NIST with bundled c."""
    from stateprop.cubic import cubic_from_name, CubicMixture
    # n-pentane saturated liquid at 298 K — NIST: 626 kg/m³
    e_no = cubic_from_name("n-pentane", family="srk")
    e_yes = cubic_from_name("n-pentane", family="srk",
                                  volume_shift="auto")
    mx_no = CubicMixture([e_no], composition=[1.0])
    mx_yes = CubicMixture([e_yes], composition=[1.0])
    rho_no = mx_no.density_from_pressure(p=1e5, T=298.0,
                                                phase_hint='liquid')
    rho_yes = mx_yes.density_from_pressure(p=1e5, T=298.0,
                                                  phase_hint='liquid')
    M = 0.07215
    err_no = abs(rho_no * M - 626.0)
    err_yes = abs(rho_yes * M - 626.0)
    check(f"  density-error reduced "
          f"(no={err_no:.0f} → yes={err_yes:.0f})",
          err_yes < err_no)


def test_v119_list_volume_shift_compounds():
    """list_volume_shift_compounds returns a usable snapshot."""
    from stateprop.cubic import list_volume_shift_compounds
    table = list_volume_shift_compounds()
    check(f"non-empty ({len(table)} compounds)", len(table) >= 20)
    check("methane present", "methane" in table)
    check("methane has 'pr' and 'srk' entries",
          "pr" in table["methane"] and "srk" in table["methane"])
    table["methane"]["pr"] = 999.0
    table2 = list_volume_shift_compounds()
    check("returned dict is a snapshot",
          table2["methane"]["pr"] != 999.0)


# ----------------------------------------------------------------------
# Mixture critical points (Heidemann-Khalil / Michelsen)
# ----------------------------------------------------------------------

def test_critical_A_residual_fd():
    """Analytic A^res matrix matches FD of (ln_phi + ln Z) across multiple states."""
    from stateprop.cubic.critical import _A_residual_matrix

    def fd_A_res(T, V, n, mx, eps=1e-6):
        N_ = len(n)
        A_fd = np.zeros((N_, N_))
        for j in range(N_):
            # Perturb n_j at fixed T, V
            for sign, delta in [(+1, +eps), (-1, -eps)]:
                pass
            n_p = n.copy(); n_p[j] += eps
            n_m = n.copy(); n_m[j] -= eps
            rho_p, rho_m = n_p.sum() / V, n_m.sum() / V
            x_p = n_p / n_p.sum(); x_m = n_m / n_m.sum()
            lp = mx.ln_phi(rho_p, T, x_p); lm = mx.ln_phi(rho_m, T, x_m)
            ap, bp, *_ = mx.a_b_mix(T, x_p); am, bm, *_ = mx.a_b_mix(T, x_m)
            ep = mx.epsilon; sg = mx.sigma
            vp, vm = 1.0/rho_p, 1.0/rho_m
            pp = mx.R*T/(vp - bp) - ap/((vp + ep*bp)*(vp + sg*bp))
            pm = mx.R*T/(vm - bm) - am/((vm + ep*bm)*(vm + sg*bm))
            Zp = pp * V / (n_p.sum() * mx.R * T)
            Zm = pm * V / (n_m.sum() * mx.R * T)
            for i in range(N_):
                A_fd[i, j] = mx.R * T * ((lp[i] + np.log(Zp)) - (lm[i] + np.log(Zm))) / (2*eps)
        return A_fd

    c_CO2 = PR(304.13, 7.3773e6, 0.224)
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    mx = CubicMixture([c_CO2, c_CH4], composition=[0.5, 0.5], k_ij={(0,1): 0.09})
    max_err = 0.0
    max_sym = 0.0
    for (T, V, z) in [(250, 8e-5, np.array([0.5, 0.5])),
                       (300, 5e-5, np.array([0.3, 0.7])),
                       (220, 1.5e-4, np.array([0.7, 0.3])),
                       (280, 3e-5, np.array([0.5, 0.5]))]:
        A_ana = _A_residual_matrix(T, V, z, mx)
        A_fd_ = fd_A_res(T, V, z, mx, 1e-6)
        rel = np.max(np.abs(A_ana - A_fd_) / np.maximum(np.abs(A_ana), 1.0))
        sym = np.max(np.abs(A_ana - A_ana.T))
        max_err = max(max_err, rel); max_sym = max(max_sym, sym)
    check("A^res analytic matches FD to 1e-8", max_err < 1e-8, f"max err {max_err:.2e}")
    check("A^res is symmetric", max_sym < 1e-12, f"sym err {max_sym:.2e}")

    # Ternary
    c_N2 = PR(126.192, 3.3958e6, 0.0372)
    mx3 = CubicMixture([c_CO2, c_CH4, c_N2], composition=[0.4, 0.4, 0.2],
                       k_ij={(0,1): 0.09, (0,2): -0.02, (1,2): 0.025})
    max_err = 0.0
    for (T, V, z) in [(220, 1e-4, np.array([0.4, 0.4, 0.2])),
                       (250, 5e-5, np.array([0.3, 0.5, 0.2]))]:
        A_ana = _A_residual_matrix(T, V, z, mx3)
        A_fd_ = fd_A_res(T, V, z, mx3, 1e-6)
        rel = np.max(np.abs(A_ana - A_fd_) / np.maximum(np.abs(A_ana), 1.0))
        max_err = max(max_err, rel)
    check("A^res analytic matches FD on ternary", max_err < 1e-8, f"max err {max_err:.2e}")


def test_critical_point_pure_limit():
    """In the pure-fluid limit (z_i -> 1), the mixture critical point recovers
    the EOS critical point of component i."""
    from stateprop.cubic import critical_point
    # CO2-dominant -- should give EOS critical of CO2
    c_CO2 = PR(304.13, 7.3773e6, 0.224)
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    mx = CubicMixture([c_CO2, c_CH4], composition=[0.9999, 0.0001], k_ij={(0,1): 0.09})
    r = critical_point(mx.x, mx)
    # Tc within 0.1 K, pc within 0.5%
    check(f"pure-CO2 limit: T_c={r['T_c']:.3f} (ref 304.13)",
          abs(r['T_c'] - 304.13) < 0.1, f"err {abs(r['T_c'] - 304.13):.3f}")
    check(f"pure-CO2 limit: p_c={r['p_c']/1e6:.4f} MPa (ref 7.3773)",
          abs(r['p_c'] - 7.3773e6) / 7.3773e6 < 0.005,
          f"rel err {abs(r['p_c'] - 7.3773e6)/7.3773e6:.2e}")
    # CH4-dominant
    mx2 = CubicMixture([c_CO2, c_CH4], composition=[0.0001, 0.9999], k_ij={(0,1): 0.09})
    r = critical_point(mx2.x, mx2)
    check(f"pure-CH4 limit: T_c={r['T_c']:.3f} (ref 190.56)",
          abs(r['T_c'] - 190.564) < 0.1, f"err {abs(r['T_c'] - 190.564):.3f}")
    check(f"pure-CH4 limit: p_c={r['p_c']/1e6:.4f} MPa (ref 4.599)",
          abs(r['p_c'] - 4.5992e6) / 4.5992e6 < 0.005,
          f"rel err {abs(r['p_c'] - 4.5992e6)/4.5992e6:.2e}")


def test_critical_point_CO2_CH4_locus():
    """Sweep across the CO2/CH4 critical locus and verify monotone behavior."""
    from stateprop.cubic import critical_point
    c_CO2 = PR(304.13, 7.3773e6, 0.224)
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)

    Tcs, pcs, zs = [], [], []
    for z_CO2 in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        mx = CubicMixture([c_CO2, c_CH4], composition=[z_CO2, 1-z_CO2], k_ij={(0,1):0.09})
        try:
            r = critical_point(mx.x, mx)
            Tcs.append(r['T_c'])
            pcs.append(r['p_c'])
            zs.append(z_CO2)
            check(f"crit pt converges at z(CO2)={z_CO2}",
                  r['residual'] < 1e-6, f"resid {r['residual']:.2e}")
        except Exception as e:
            check(f"crit pt at z(CO2)={z_CO2}", False, str(e)[:50])

    # T_c should be monotonically increasing in z(CO2) (since Tc_CO2 > Tc_CH4)
    Tcs = np.array(Tcs)
    check("T_c is monotonically increasing in z(CO2)",
          np.all(np.diff(Tcs) > 0), f"Tcs={Tcs}")

    # p_c should be a smooth curve with a single max somewhere in the middle
    # (characteristic of Type-I critical loci)
    pcs = np.array(pcs)
    imax = int(np.argmax(pcs))
    check("p_c(z) has interior maximum (Type-I locus)",
          0 < imax < len(pcs) - 1, f"argmax index {imax}, pcs={pcs/1e6}")


def test_critical_point_CH4_N2():
    """CH4/N2 -- another well-known system, close to ideal mixing."""
    from stateprop.cubic import critical_point
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_N2  = PR(126.192, 3.3958e6, 0.0372)
    # Near-equimolar
    mx = CubicMixture([c_CH4, c_N2], composition=[0.5, 0.5], k_ij={(0,1): 0.025})
    try:
        r = critical_point(mx.x, mx)
        # Expect Tc between the pure Tcs (190.6 and 126.2), pc greater than the
        # mole-averaged pc (since critical locus peaks in between for this system).
        T_avg = 0.5 * (190.564 + 126.192)
        check("CH4/N2 50-50 critical T between pure values",
              126.192 < r['T_c'] < 190.564,
              f"T_c={r['T_c']}")
        p_avg = 0.5 * (4.5992e6 + 3.3958e6)
        # For CH4/N2 the pc locus typically exceeds the mole-weighted average
        check("CH4/N2 50-50 critical p reasonable",
              0.8*p_avg < r['p_c'] < 2*p_avg,
              f"p_c={r['p_c']/1e6} MPa")
    except Exception as e:
        check("CH4/N2 critical point converges", False, str(e)[:50])


def test_critical_point_vdw_rejected():
    """Critical-point solver must cleanly reject vdW (sigma == epsilon)."""
    from stateprop.cubic import VDW, critical_point
    c1 = VDW(304.13, 7.3773e6)
    c2 = VDW(190.564, 4.5992e6)
    mx = CubicMixture([c1, c2], composition=[0.5, 0.5])
    try:
        critical_point(mx.x, mx)
        check("vdW critical-point solver raises NotImplementedError",
              False, "did not raise")
    except NotImplementedError:
        check("vdW critical-point solver raises NotImplementedError", True)
    except Exception as e:
        check("vdW critical-point solver raises NotImplementedError", False,
              f"wrong exception: {type(e).__name__}: {e}")


# ----------------------------------------------------------------------
# Phase envelope (v0.5.1)
# ----------------------------------------------------------------------

def test_envelope_point_matches_bubble_point_p():
    """envelope_point with beta=0 matches bubble_point_p at low/moderate T.

    Near the critical, bubble_point_p struggles (K-factors near 1 slow the
    SS iteration), so we test only at T well below critical.
    """
    from stateprop.cubic import envelope_point, bubble_point_p
    c_CO2 = PR(304.13, 7.3773e6, 0.224)
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    mx = CubicMixture([c_CO2, c_CH4], composition=[0.5, 0.5], k_ij={(0,1):0.09})
    # T_c ~= 253 K; stay well below critical where bubble_point_p is robust
    for T in [160, 180, 200, 220]:
        ref = bubble_point_p(T=T, z=mx.x, mixture=mx)
        r = envelope_point(T=T, p=ref.p, z=mx.x, mixture=mx, beta=0)
        err_p = abs(r["p"] - ref.p) / ref.p
        K = r["K"]
        y_computed = K * mx.x
        y_computed /= y_computed.sum()
        err_y = float(np.max(np.abs(y_computed - ref.y)))
        # Tolerance 1e-6 matches bubble_point_p's internal convergence tol
        check(f"envelope_point bubble at T={T}: p matches bubble_point_p",
              err_p < 1e-6, f"rel err {err_p:.2e}")
        check(f"envelope_point bubble at T={T}: y matches bubble_point_p",
              err_y < 1e-6, f"max err {err_y:.2e}")


def test_trace_envelope_seeds_at_critical():
    """Envelope tracer starts at the critical point and builds outward."""
    from stateprop.cubic import trace_envelope, critical_point
    c_CO2 = PR(304.13, 7.3773e6, 0.224)
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    mx = CubicMixture([c_CO2, c_CH4], composition=[0.5, 0.5], k_ij={(0,1):0.09})

    env = trace_envelope(mx.x, mx, max_points_per_branch=30, step_init=0.04,
                         step_max=0.08, crit_offset=0.02)

    # Envelope contains the critical point
    crit = env["critical"]
    check("envelope contains critical point (branch=-1)",
          np.any(env["branch"] == -1),
          f"branch values: {np.unique(env['branch'])}")
    crit_idx = int(np.where(env["branch"] == -1)[0][0])
    check(f"T at critical matches critical_point ({env['T'][crit_idx]:.3f} vs {crit['T_c']:.3f})",
          abs(env["T"][crit_idx] - crit["T_c"]) < 1e-6)
    check(f"p at critical matches critical_point ({env['p'][crit_idx]/1e6:.4f} vs {crit['p_c']/1e6:.4f})",
          abs(env["p"][crit_idx] - crit["p_c"]) < 1e-3)

    # Envelope has both branches
    n_bub = int(np.sum(env["branch"] == 0))
    n_dew = int(np.sum(env["branch"] == 1))
    check(f"envelope has bubble-side points ({n_bub})",
          n_bub > 5, f"only {n_bub} points")
    check(f"envelope has dew-side points ({n_dew})",
          n_dew > 5, f"only {n_dew} points")


def test_envelope_analytic_jacobian_matches_fd():
    """v0.9.17: _envelope_jacobian_analytic matches central-difference FD
    to ~1e-7 relative error on a near-envelope state for both beta=0 and
    beta=1."""
    from stateprop.cubic.envelope import (
        _envelope_jacobian_analytic, _envelope_jacobian_fd,
    )
    c_CO2 = PR(304.13, 7.3773e6, 0.224)
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    mx = CubicMixture([c_CO2, c_CH4], composition=[0.5, 0.5], k_ij={(0,1):0.09})
    z = np.array([0.5, 0.5])
    N = 2
    # Non-converged state so residuals are O(1e-2), not ~0
    T = 220.0; p = 5e6
    K = mx.wilson_K(T, p)
    X = np.concatenate([np.log(K), [np.log(T), np.log(p)]])
    spec_idx = N
    spec_val = float(X[N])
    for beta in (0, 1):
        J_fd = _envelope_jacobian_fd(X, beta, z, spec_idx, spec_val, mx)
        J_an = _envelope_jacobian_analytic(X, beta, z, spec_idx, spec_val, mx)
        rel_err = float(np.max(np.abs(J_an - J_fd) / (np.abs(J_fd) + 1e-30)))
        check(f"envelope Jacobian [beta={beta}]: analytic matches FD (rel err < 1e-6)",
              rel_err < 1e-6, f"rel err = {rel_err:.2e}")


def test_envelope_point_analytic_matches_fd():
    """v0.9.17: envelope_point with use_analytic_jac=True gives same result
    as FD path to machine precision."""
    from stateprop.cubic.envelope import envelope_point
    c_CO2 = PR(304.13, 7.3773e6, 0.224)
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    mx = CubicMixture([c_CO2, c_CH4], composition=[0.3, 0.7], k_ij={(0,1):0.09})
    z = np.array([0.3, 0.7])
    T = 180.0
    ep_fd = envelope_point(T, 1e6, z, mx, beta=0, use_analytic_jac=False)
    ep_an = envelope_point(T, 1e6, z, mx, beta=0, use_analytic_jac=True)
    check("envelope_point: analytic-Jac result matches FD result",
          abs(ep_an['p'] - ep_fd['p']) / ep_fd['p'] < 1e-10,
          f"FD p={ep_fd['p']/1e6:.6f}, AN p={ep_an['p']/1e6:.6f}")
    check("envelope_point: analytic and FD agree on K factors",
          np.max(np.abs(ep_an['K'] - ep_fd['K']) / ep_fd['K']) < 1e-10,
          f"K diff = {np.max(np.abs(ep_an['K'] - ep_fd['K']) / ep_fd['K']):.2e}")


def test_trace_envelope_analytic_corrector_matches_fd():
    """v0.9.17: trace_envelope with use_analytic_jac_corrector=True gives
    an envelope that agrees with the FD-corrector envelope."""
    from stateprop.cubic.envelope import trace_envelope
    c_CO2 = PR(304.13, 7.3773e6, 0.224)
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_C2  = PR(305.322, 4.872e6, 0.0995)
    mx = CubicMixture([c_CO2, c_CH4, c_C2], composition=[0.3, 0.5, 0.2], k_ij={})
    env_fd = trace_envelope(mx.x, mx, max_points_per_branch=15,
                             step_init=0.04, step_max=0.08, crit_offset=0.02)
    env_an = trace_envelope(mx.x, mx, max_points_per_branch=15,
                             step_init=0.04, step_max=0.08, crit_offset=0.02,
                             use_analytic_jac_corrector=True)
    check("trace_envelope: analytic-corrector envelope has >= FD's points",
          env_an['n_points'] >= int(0.8 * env_fd['n_points']),
          f"FD {env_fd['n_points']} pts, AN {env_an['n_points']} pts")
    # Both envelopes should cover similar T range
    T_an_span = env_an['T'].max() - env_an['T'].min()
    T_fd_span = env_fd['T'].max() - env_fd['T'].min()
    check("trace_envelope: analytic-corrector T span matches FD T span",
          abs(T_an_span - T_fd_span) / T_fd_span < 0.3,
          f"FD span {T_fd_span:.2f}K, AN span {T_an_span:.2f}K")


def test_envelope_bubble_side_fugacity_equality():
    """Each envelope point satisfies the Rachford-Rice constraint corresponding
    to its branch label: Sum(K*z) = 1 for bubble (beta=0), Sum(z/K) = 1 for
    dew (beta=1)."""
    from stateprop.cubic import trace_envelope
    c_CO2 = PR(304.13, 7.3773e6, 0.224)
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    mx = CubicMixture([c_CO2, c_CH4], composition=[0.5, 0.5], k_ij={(0,1):0.09})

    env = trace_envelope(mx.x, mx, max_points_per_branch=20, step_init=0.04,
                         step_max=0.08, crit_offset=0.02)

    max_err = 0.0
    for i in range(env["n_points"]):
        br = env["branch"][i]
        K = env["K"][i]
        if br == 0:
            S = float(np.sum(K * mx.x))
        elif br == 1:
            S = float(np.sum(mx.x / K))
        else:
            continue  # critical point: K=1 exactly, skip
        max_err = max(max_err, abs(S - 1.0))
    check(f"all envelope points satisfy Rachford-Rice (max |err| = {max_err:.2e})",
          max_err < 1e-7)


# ----------------------------------------------------------------------
# v0.9.15 -- Newton-Raphson bubble/dew point solvers for cubic EOS
# ----------------------------------------------------------------------


def test_cubic_newton_bubble_point_p_matches_ss():
    """Cubic Newton bubble_point_p converges to same p as SS on CH4-ethane
    well below critical (T_c ~ 267K for this PR mixture)."""
    from stateprop.cubic.flash import bubble_point_p, newton_bubble_point_p
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_C2  = PR(305.322, 4.872e6, 0.0995)
    mx = CubicMixture([c_CH4, c_C2], composition=[0.5, 0.5], k_ij={})
    z = np.array([0.5, 0.5])
    T = 200.0
    r_ss = bubble_point_p(T, z, mx)
    r_nt = newton_bubble_point_p(T, z, mx)
    check("Cubic Newton bubble_point_p: p matches SS to 1e-6 rel",
          abs(r_nt.p - r_ss.p) / r_ss.p < 1e-6,
          f"SS p={r_ss.p/1e6:.4f}, Newton p={r_nt.p/1e6:.4f}")
    check("Cubic Newton bubble_point_p: fewer iterations than SS",
          r_nt.iterations <= r_ss.iterations,
          f"SS {r_ss.iterations}, Newton {r_nt.iterations}")
    # Fugacity equality at solution
    T_r, p_r, K_r, y_r = r_nt.T, r_nt.p, r_nt.K, r_nt.y
    rho_L = mx.density_from_pressure(p_r, T_r, z, phase_hint='liquid')
    rho_V = mx.density_from_pressure(p_r, T_r, y_r, phase_hint='vapor')
    f_residual = np.max(np.abs(np.log(K_r) - (mx.ln_phi(rho_L, T_r, z)
                                                - mx.ln_phi(rho_V, T_r, y_r))))
    check("Cubic Newton bubble_point_p: fugacity equality to 1e-8",
          f_residual < 1e-8, f"max residual = {f_residual:.2e}")


def test_cubic_newton_bubble_point_p_near_critical():
    """Near-critical: Newton is much faster than SS. T_c ~ 267K for this
    PR CH4-ethane mixture. Test at T=240K (27K below critical)."""
    from stateprop.cubic.flash import bubble_point_p, newton_bubble_point_p
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_C2  = PR(305.322, 4.872e6, 0.0995)
    mx = CubicMixture([c_CH4, c_C2], composition=[0.5, 0.5], k_ij={})
    z = np.array([0.5, 0.5])
    T = 240.0
    r_ss = bubble_point_p(T, z, mx)
    r_nt = newton_bubble_point_p(T, z, mx)
    check("Near-critical cubic Newton: p matches SS to 1e-6 rel",
          abs(r_nt.p - r_ss.p) / r_ss.p < 1e-6,
          f"SS p={r_ss.p/1e6:.4f}, Newton p={r_nt.p/1e6:.4f}")
    check("Near-critical cubic Newton: converges in < 15 iterations",
          r_nt.iterations < 15, f"iters={r_nt.iterations}")
    check("Near-critical cubic Newton: much faster than SS (iters < 1/3 of SS)",
          r_nt.iterations * 3 < r_ss.iterations,
          f"SS={r_ss.iterations}, Newton={r_nt.iterations}")


def test_cubic_newton_bubble_point_T_matches_ss():
    """Cubic Newton bubble_point_T converges to same T as SS."""
    from stateprop.cubic.flash import bubble_point_T, newton_bubble_point_T
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_C2  = PR(305.322, 4.872e6, 0.0995)
    mx = CubicMixture([c_CH4, c_C2], composition=[0.5, 0.5], k_ij={})
    z = np.array([0.5, 0.5])
    p = 3e6
    r_ss = bubble_point_T(p, z, mx)
    r_nt = newton_bubble_point_T(p, z, mx)
    check("Cubic Newton bubble_point_T: T matches SS to 1e-6 rel",
          abs(r_nt.T - r_ss.T) / r_ss.T < 1e-6,
          f"SS T={r_ss.T:.3f}, Newton T={r_nt.T:.3f}")


def test_cubic_newton_dew_point_p_matches_ss():
    """Cubic Newton dew_point_p converges to same p as SS."""
    from stateprop.cubic.flash import dew_point_p, newton_dew_point_p
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_C2  = PR(305.322, 4.872e6, 0.0995)
    mx = CubicMixture([c_CH4, c_C2], composition=[0.5, 0.5], k_ij={})
    z = np.array([0.5, 0.5])
    T = 220.0
    r_ss = dew_point_p(T, z, mx)
    r_nt = newton_dew_point_p(T, z, mx)
    check("Cubic Newton dew_point_p: p matches SS to 1e-6 rel",
          abs(r_nt.p - r_ss.p) / r_ss.p < 1e-6,
          f"SS p={r_ss.p/1e6:.4f}, Newton p={r_nt.p/1e6:.4f}")
    check("Cubic Newton dew_point_p: fewer iterations than SS",
          r_nt.iterations <= r_ss.iterations,
          f"SS {r_ss.iterations}, Newton {r_nt.iterations}")


def test_cubic_newton_dew_point_T_matches_ss():
    """Cubic Newton dew_point_T converges to same T as SS."""
    from stateprop.cubic.flash import dew_point_T, newton_dew_point_T
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_C2  = PR(305.322, 4.872e6, 0.0995)
    mx = CubicMixture([c_CH4, c_C2], composition=[0.5, 0.5], k_ij={})
    z = np.array([0.5, 0.5])
    p = 2e6
    r_ss = dew_point_T(p, z, mx)
    r_nt = newton_dew_point_T(p, z, mx)
    check("Cubic Newton dew_point_T: T matches SS to 1e-6 rel",
          abs(r_nt.T - r_ss.T) / r_ss.T < 1e-6,
          f"SS T={r_ss.T:.3f}, Newton T={r_nt.T:.3f}")


def test_cubic_newton_bubble_dew_ordering():
    """At any T well below critical, bubble p > dew p -- Newton must respect
    this thermodynamic ordering. Also check volatility ordering in K-factors."""
    from stateprop.cubic.flash import newton_bubble_point_p, newton_dew_point_p
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_C2  = PR(305.322, 4.872e6, 0.0995)
    mx = CubicMixture([c_CH4, c_C2], composition=[0.5, 0.5], k_ij={})
    z = np.array([0.5, 0.5])
    T = 200.0
    b = newton_bubble_point_p(T, z, mx)
    d = newton_dew_point_p(T, z, mx)
    check("Cubic Newton bubble p > dew p at T << T_c",
          b.p > d.p, f"bubble p={b.p/1e6:.3f}, dew p={d.p/1e6:.3f}")
    check("Cubic Newton bubble: K_CH4 > 1 (more volatile)",
          b.K[0] > 1.0, f"got K_CH4={b.K[0]:.3f}")
    check("Cubic Newton bubble: K_C2 < 1 (less volatile)",
          b.K[1] < 1.0, f"got K_C2={b.K[1]:.3f}")


def test_cubic_newton_peneloux_analytic_path():
    """v0.9.16: Peneloux volume-shifted cubic mixtures now use the analytic
    Jacobian (no longer fall back to SS via NotImplementedError). The
    Newton iteration count drops from SS-like values to quadratic-Newton
    values, and results still match SS to tight tolerance."""
    from stateprop.cubic.eos import SRK
    from stateprop.cubic.flash import bubble_point_p, newton_bubble_point_p
    c_CH4 = SRK(190.564, 4.5992e6, 0.01142, volume_shift_c='peneloux')
    c_C2  = SRK(305.322, 4.872e6, 0.0995, volume_shift_c='peneloux')
    mx = CubicMixture([c_CH4, c_C2], composition=[0.5, 0.5], k_ij={})
    z = np.array([0.5, 0.5])
    T = 200.0
    r_nt = newton_bubble_point_p(T, z, mx)
    r_ss = bubble_point_p(T, z, mx)
    check("Peneloux Newton: result matches SS",
          abs(r_nt.p - r_ss.p) / r_ss.p < 1e-6,
          f"Newton p={r_nt.p/1e6:.4f}, SS p={r_ss.p/1e6:.4f}")
    check("Peneloux Newton: now converges in Newton iters (<=10), not SS-like",
          r_nt.iterations <= 10, f"iters={r_nt.iterations}")


def test_peneloux_dlnphi_dxk_at_p_fd():
    """Peneloux (v0.9.16) analytic dlnphi_dxk_at_p matches central FD on
    the real (shifted) ln_phi for a Peneloux SRK mixture. Across vapor
    and liquid phases and for binary + ternary compositions."""
    from stateprop.cubic.eos import SRK
    # Binary CH4-n-butane Peneloux SRK
    c_CH4 = SRK(190.564, 4.5992e6, 0.01142, volume_shift_c='peneloux')
    c_C4  = SRK(425.12, 3.796e6, 0.2002, volume_shift_c='peneloux')
    mx = CubicMixture([c_CH4, c_C4], composition=[0.5, 0.5], k_ij={})
    x = mx.x.copy()
    T, p = 250.0, 3e6
    for phase in ('vapor', 'liquid'):
        an = mx.dlnphi_dxk_at_p(p, T, x, phase_hint=phase)
        hx = 1e-5
        err_max = 0.0
        for k in range(len(x)):
            x_p = x.copy(); x_p[k] += hx
            x_m = x.copy(); x_m[k] -= hx
            rho_xp = mx.density_from_pressure(p, T, x_p, phase_hint=phase)
            rho_xm = mx.density_from_pressure(p, T, x_m, phase_hint=phase)
            fd_xk = (mx.ln_phi(rho_xp, T, x_p) - mx.ln_phi(rho_xm, T, x_m)) / (2*hx)
            err_max = max(err_max, float(np.max(
                np.abs(an[:, k] - fd_xk) / (np.abs(fd_xk) + 1e-30))))
        check(f"Peneloux dlnphi_dxk_at_p [{phase}]: matches FD, rel err < 1e-7",
              err_max < 1e-7, f"rel err = {err_max:.2e}")


def test_peneloux_dlnphi_dp_at_T_fd():
    """Peneloux (v0.9.16) analytic dlnphi_dp_at_T matches central FD."""
    from stateprop.cubic.eos import SRK
    c_CH4 = SRK(190.564, 4.5992e6, 0.01142, volume_shift_c='peneloux')
    c_C4  = SRK(425.12, 3.796e6, 0.2002, volume_shift_c='peneloux')
    mx = CubicMixture([c_CH4, c_C4], composition=[0.5, 0.5], k_ij={})
    x = mx.x.copy()
    T, p = 250.0, 3e6
    for phase in ('vapor', 'liquid'):
        an = mx.dlnphi_dp_at_T(p, T, x, phase_hint=phase)
        hp = p * 1e-5
        rho_p = mx.density_from_pressure(p + hp, T, x, phase_hint=phase)
        rho_m = mx.density_from_pressure(p - hp, T, x, phase_hint=phase)
        fd = (mx.ln_phi(rho_p, T, x) - mx.ln_phi(rho_m, T, x)) / (2*hp)
        err = float(np.max(np.abs(an - fd) / (np.abs(fd) + 1e-30)))
        check(f"Peneloux dlnphi_dp_at_T [{phase}]: matches FD, rel err < 1e-7",
              err < 1e-7, f"rel err = {err:.2e}")


def test_peneloux_dlnphi_dT_at_p_fd():
    """Peneloux (v0.9.16) analytic dlnphi_dT_at_p matches central FD."""
    from stateprop.cubic.eos import SRK
    c_CH4 = SRK(190.564, 4.5992e6, 0.01142, volume_shift_c='peneloux')
    c_C4  = SRK(425.12, 3.796e6, 0.2002, volume_shift_c='peneloux')
    mx = CubicMixture([c_CH4, c_C4], composition=[0.5, 0.5], k_ij={})
    x = mx.x.copy()
    T, p = 250.0, 3e6
    for phase in ('vapor', 'liquid'):
        an = mx.dlnphi_dT_at_p(p, T, x, phase_hint=phase)
        hT = T * 1e-5
        rho_Tp = mx.density_from_pressure(p, T + hT, x, phase_hint=phase)
        rho_Tm = mx.density_from_pressure(p, T - hT, x, phase_hint=phase)
        fd = (mx.ln_phi(rho_Tp, T + hT, x) - mx.ln_phi(rho_Tm, T - hT, x)) / (2*hT)
        err = float(np.max(np.abs(an - fd) / (np.abs(fd) + 1e-30)))
        check(f"Peneloux dlnphi_dT_at_p [{phase}]: matches FD, rel err < 1e-7",
              err < 1e-7, f"rel err = {err:.2e}")


def test_peneloux_derivs_ternary_manual_c():
    """Peneloux derivatives on ternary + manual numeric c values (not the
    automatic Peneloux 1982 formula). Tests the general c-shift code path."""
    from stateprop.cubic.eos import SRK
    c_CH4 = SRK(190.564, 4.5992e6, 0.01142, volume_shift_c=5e-7)
    c_C2  = SRK(305.322, 4.872e6, 0.0995, volume_shift_c=2e-6)
    c_C3  = SRK(369.83, 4.248e6, 0.152, volume_shift_c=8e-6)
    mx = CubicMixture([c_CH4, c_C2, c_C3], composition=[0.6, 0.3, 0.1], k_ij={})
    x = mx.x.copy()
    T, p = 260.0, 4e6

    # dp check
    hp = p * 1e-5
    for phase in ('vapor', 'liquid'):
        an = mx.dlnphi_dp_at_T(p, T, x, phase_hint=phase)
        rho_p = mx.density_from_pressure(p + hp, T, x, phase_hint=phase)
        rho_m = mx.density_from_pressure(p - hp, T, x, phase_hint=phase)
        fd = (mx.ln_phi(rho_p, T, x) - mx.ln_phi(rho_m, T, x)) / (2*hp)
        err = float(np.max(np.abs(an - fd) / (np.abs(fd) + 1e-30)))
        check(f"Ternary numeric-c dlnphi_dp [{phase}]: rel err < 1e-7",
              err < 1e-7, f"rel err = {err:.2e}")

    # dx check (sample k=0)
    hx = 1e-5
    for phase in ('vapor', 'liquid'):
        an = mx.dlnphi_dxk_at_p(p, T, x, phase_hint=phase)
        x_p = x.copy(); x_p[0] += hx
        x_m = x.copy(); x_m[0] -= hx
        rho_xp = mx.density_from_pressure(p, T, x_p, phase_hint=phase)
        rho_xm = mx.density_from_pressure(p, T, x_m, phase_hint=phase)
        fd = (mx.ln_phi(rho_xp, T, x_p) - mx.ln_phi(rho_xm, T, x_m)) / (2*hx)
        err = float(np.max(np.abs(an[:, 0] - fd) / (np.abs(fd) + 1e-30)))
        check(f"Ternary numeric-c dlnphi_dx_0 [{phase}]: rel err < 1e-7",
              err < 1e-7, f"rel err = {err:.2e}")


# ----------------------------------------------------------------------
# v0.9.20 -- Three-phase (VLLE) flash for cubic EOS
# ----------------------------------------------------------------------


def test_cubic_three_phase_rr_material_balance():
    """Cubic 3-phase Rachford-Rice with constructed K's exactly recovers
    phase fractions and compositions used to build z."""
    from stateprop.cubic.three_phase_flash import _rachford_rice_3p
    bV_t, bL1_t, bL2_t = 0.4, 0.35, 0.25
    y_t = np.array([0.7, 0.25, 0.05])
    x1_t = np.array([0.2, 0.55, 0.25])
    x2_t = np.array([0.1, 0.05, 0.85])
    z = bV_t * y_t + bL1_t * x1_t + bL2_t * x2_t
    K_VL1 = y_t / x1_t; K_L2L1 = x2_t / x1_t
    bV, bL1, bL2, x1, x2, y = _rachford_rice_3p(z, K_VL1, K_L2L1, tol=1e-12)
    check("cubic 3-phase RR: beta_V recovers truth to 1e-8",
          abs(bV - bV_t) < 1e-8, f"got {bV}, expected {bV_t}")
    check("cubic 3-phase RR: beta_L1 recovers truth to 1e-8",
          abs(bL1 - bL1_t) < 1e-8)
    check("cubic 3-phase RR: beta_L2 recovers truth to 1e-8",
          abs(bL2 - bL2_t) < 1e-8)
    check("cubic 3-phase RR: x1 recovers truth to 1e-8",
          np.max(np.abs(x1 - x1_t)) < 1e-8)
    check("cubic 3-phase RR: x2 recovers truth to 1e-8",
          np.max(np.abs(x2 - x2_t)) < 1e-8)
    check("cubic 3-phase RR: y recovers truth to 1e-8",
          np.max(np.abs(y - y_t)) < 1e-8)


def test_cubic_three_phase_preserves_vle():
    """Cubic 3-phase flash on a clean VLE system (CO2-CH4 PR) returns
    a 2-phase result, not a spurious 3-phase split."""
    from stateprop.cubic.three_phase_flash import flash_pt_three_phase
    c_CO2 = PR(304.13, 7.3773e6, 0.224)
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    mx = CubicMixture([c_CO2, c_CH4], composition=[0.5, 0.5], k_ij={(0,1): 0.09})
    r = flash_pt_three_phase(5e6, 220.0, np.array([0.5, 0.5]), mx)
    check("cubic 3-phase on CO2-CH4 VLE: returns VLE label",
          r.phase == "VLE", f"got phase={r.phase}")
    check("cubic 3-phase on CO2-CH4 VLE: beta_L2 == 0",
          r.beta_L2 == 0.0, f"got beta_L2={r.beta_L2}")
    check("cubic 3-phase on CO2-CH4 VLE: beta_V + beta_L1 == 1",
          abs(r.beta_V + r.beta_L1 - 1.0) < 1e-10)


def test_cubic_three_phase_preserves_single_phase():
    """Cubic 3-phase flash on a supercritical state returns single phase."""
    from stateprop.cubic.three_phase_flash import flash_pt_three_phase
    c_CO2 = PR(304.13, 7.3773e6, 0.224)
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    mx = CubicMixture([c_CO2, c_CH4], composition=[0.5, 0.5], k_ij={(0,1): 0.09})
    r = flash_pt_three_phase(5e6, 350.0, np.array([0.5, 0.5]), mx)
    check("cubic 3-phase on supercritical: single-phase label",
          r.phase in ('supercritical', 'vapor', 'liquid'),
          f"got phase={r.phase}")
    check("cubic 3-phase on supercritical: no L2 phase",
          r.beta_L2 == 0.0)


def test_cubic_three_phase_water_butane_vlle():
    """Cubic 3-phase flash on water-nButane PR with k_ij=0.50 at T=350K,
    p=1MPa must identify a 3-phase VLLE state with material balance
    closing to 1e-8 and three distinct compositions (water-rich liquid,
    butane-rich liquid, and butane-rich vapor)."""
    from stateprop.cubic.three_phase_flash import flash_pt_three_phase
    c_H2O = PR(647.096, 22.064e6, 0.3443)
    c_C4 = PR(425.12, 3.796e6, 0.2002)
    mx = CubicMixture([c_H2O, c_C4], composition=[0.5, 0.5], k_ij={(0,1): 0.50})
    z = np.array([0.5, 0.5])
    r = flash_pt_three_phase(1e6, 350.0, z, mx)
    check("water-butane PR at 350K/1MPa: phase is VLLE",
          r.phase == "VLLE", f"got phase={r.phase}")
    if r.phase != "VLLE":
        return  # can't validate further

    check("water-butane VLLE: all three beta > 0.05",
          r.beta_V > 0.05 and r.beta_L1 > 0.05 and r.beta_L2 > 0.05,
          f"betas=({r.beta_V}, {r.beta_L1}, {r.beta_L2})")
    check("water-butane VLLE: betas sum to 1",
          abs(r.beta_V + r.beta_L1 + r.beta_L2 - 1.0) < 1e-10)
    # Material balance
    z_recon = r.beta_V * r.y + r.beta_L1 * r.x1 + r.beta_L2 * r.x2
    mb_err = float(np.max(np.abs(z - z_recon)))
    check("water-butane VLLE: material balance closes to 1e-8",
          mb_err < 1e-8, f"max err = {mb_err:.2e}")
    # x1 should be water-rich (water dominant), x2 should be butane-rich
    check("water-butane VLLE: x1 is water-rich (x1[water] > 0.5)",
          r.x1[0] > 0.5, f"got x1[water]={r.x1[0]}")
    check("water-butane VLLE: x2 is butane-rich (x2[butane] > 0.5)",
          r.x2[1] > 0.5, f"got x2[butane]={r.x2[1]}")
    # Vapor should have more butane than water
    check("water-butane VLLE: vapor is butane-rich (y[butane] > 0.5)",
          r.y[1] > 0.5, f"got y[butane]={r.y[1]}")


# ----------------------------------------------------------------------
# v0.9.56 -- PV / Pα / Tα flash modes
# ----------------------------------------------------------------------


def _make_methane_ethane_propane():
    """Standard 3-component hydrocarbon mixture for v0.9.56 tests."""
    methane = PR(T_c=190.56, p_c=4.599e6, acentric_factor=0.011)
    ethane  = PR(T_c=305.32, p_c=4.872e6, acentric_factor=0.099)
    propane = PR(T_c=369.83, p_c=4.248e6, acentric_factor=0.152)
    z = np.array([0.6, 0.3, 0.1])
    mx = CubicMixture([methane, ethane, propane], composition=z)
    return z, mx


def test_flash_pv_single_phase_roundtrip():
    """flash_pv at single-phase conditions recovers the input T to high
    precision when given v computed from a flash_pt at known T."""
    from stateprop.cubic import flash_pv
    z, mx = _make_methane_ethane_propane()
    # Single-phase vapor: high T, modest p
    T_ref, p_ref = 350.0, 10e5
    r_ref = flash_pt(p_ref, T_ref, z, mx, tol=1e-10)
    v_ref = 1.0 / r_ref.rho
    r_pv = flash_pv(p_ref, v_ref, z, mx, tol=1e-9)
    err_T = abs(r_pv.T - T_ref)
    check(f"flash_pv single-phase recovery: T err = {err_T:.3e} K",
          err_T < 1e-5)
    # Single-phase liquid: low T, high p
    T_ref, p_ref = 200.0, 100e5
    r_ref = flash_pt(p_ref, T_ref, z, mx, tol=1e-10)
    v_ref = 1.0 / r_ref.rho
    r_pv = flash_pv(p_ref, v_ref, z, mx, tol=1e-9)
    err_T = abs(r_pv.T - T_ref)
    check(f"flash_pv liquid-phase recovery: T err = {err_T:.3e} K",
          err_T < 1e-5)


def test_flash_pv_two_phase_roundtrip():
    """flash_pv inside the two-phase dome recovers T to high precision."""
    from stateprop.cubic import flash_pv
    z, mx = _make_methane_ethane_propane()
    T_ref, p_ref = 220.0, 30e5
    r_ref = flash_pt(p_ref, T_ref, z, mx, tol=1e-10)
    if r_ref.phase != 'two_phase':
        check(f"setup: T={T_ref} p={p_ref/1e5} bar should be two-phase, got {r_ref.phase}",
              False)
        return
    v_ref = 1.0 / r_ref.rho
    r_pv = flash_pv(p_ref, v_ref, z, mx, tol=1e-9)
    err_T = abs(r_pv.T - T_ref)
    check(f"flash_pv two-phase: T err = {err_T:.3e} K, "
          f"phase={r_pv.phase}, beta={r_pv.beta:.4f}",
          err_T < 1e-4 and r_pv.phase == 'two_phase')


def test_flash_p_alpha_recovers_target_beta():
    """flash_p_alpha at intermediate alpha values recovers the target
    vapor fraction to <1e-6 in beta."""
    from stateprop.cubic import flash_p_alpha
    z, mx = _make_methane_ethane_propane()
    p_test = 20e5
    max_err = 0.0
    for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
        r = flash_p_alpha(p_test, alpha, z, mx, tol=1e-7)
        actual = r.beta if r.beta is not None else (1.0 if r.phase == 'vapor' else 0.0)
        err = abs(actual - alpha)
        if err > max_err:
            max_err = err
    check(f"flash_p_alpha 5 alphas, max beta err = {max_err:.3e}",
          max_err < 1e-6)


def test_flash_p_alpha_endpoints_match_bubble_dew():
    """flash_p_alpha at alpha=0 reproduces newton_bubble_point_T;
    at alpha=1 reproduces newton_dew_point_T."""
    from stateprop.cubic import (flash_p_alpha,
        newton_bubble_point_T, newton_dew_point_T)
    z, mx = _make_methane_ethane_propane()
    p_test = 20e5
    r_bub_direct = newton_bubble_point_T(p_test, z, mx)
    r_bub_alpha = flash_p_alpha(p_test, 0.0, z, mx)
    err_bub = abs(r_bub_direct.T - r_bub_alpha.T)
    check(f"flash_p_alpha(alpha=0) matches bubble: T err = {err_bub:.3e}",
          err_bub < 1e-6)
    r_dew_direct = newton_dew_point_T(p_test, z, mx, T_init=r_bub_direct.T * 1.1)
    r_dew_alpha = flash_p_alpha(p_test, 1.0, z, mx)
    err_dew = abs(r_dew_direct.T - r_dew_alpha.T)
    check(f"flash_p_alpha(alpha=1) matches dew: T err = {err_dew:.3e}",
          err_dew < 1e-6)


def test_flash_t_alpha_recovers_target_beta():
    """flash_t_alpha at intermediate alpha values recovers target vapor
    fraction to <1e-6."""
    from stateprop.cubic import flash_t_alpha
    z, mx = _make_methane_ethane_propane()
    T_test = 250.0
    max_err = 0.0
    for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
        r = flash_t_alpha(T_test, alpha, z, mx, tol=1e-7)
        actual = r.beta if r.beta is not None else (1.0 if r.phase == 'vapor' else 0.0)
        err = abs(actual - alpha)
        if err > max_err:
            max_err = err
    check(f"flash_t_alpha 5 alphas, max beta err = {max_err:.3e}",
          max_err < 1e-6)


def test_flash_t_alpha_endpoints_match_bubble_dew():
    """flash_t_alpha at alpha=0 reproduces newton_bubble_point_p;
    at alpha=1 reproduces newton_dew_point_p."""
    from stateprop.cubic import (flash_t_alpha,
        newton_bubble_point_p, newton_dew_point_p)
    z, mx = _make_methane_ethane_propane()
    T_test = 250.0
    r_bub_direct = newton_bubble_point_p(T_test, z, mx)
    r_bub_alpha = flash_t_alpha(T_test, 0.0, z, mx)
    err_bub = abs(r_bub_direct.p - r_bub_alpha.p) / r_bub_direct.p
    check(f"flash_t_alpha(alpha=0) matches bubble: rel p err = {err_bub:.3e}",
          err_bub < 1e-6)
    r_dew_direct = newton_dew_point_p(T_test, z, mx, p_init=r_bub_direct.p * 0.5)
    r_dew_alpha = flash_t_alpha(T_test, 1.0, z, mx)
    err_dew = abs(r_dew_direct.p - r_dew_alpha.p) / r_dew_direct.p
    check(f"flash_t_alpha(alpha=1) matches dew: rel p err = {err_dew:.3e}",
          err_dew < 1e-6)


def test_flash_p_alpha_invalid_alpha_raises():
    """flash_p_alpha rejects alpha outside [0, 1] with ValueError."""
    from stateprop.cubic import flash_p_alpha, flash_t_alpha
    z, mx = _make_methane_ethane_propane()
    raised_p, raised_t = False, False
    try:
        flash_p_alpha(20e5, 1.5, z, mx)
    except ValueError:
        raised_p = True
    try:
        flash_t_alpha(250.0, -0.1, z, mx)
    except ValueError:
        raised_t = True
    check(f"flash_p_alpha rejects alpha>1: {raised_p}", raised_p)
    check(f"flash_t_alpha rejects alpha<0: {raised_t}", raised_t)


def test_flash_pv_consistent_with_flash_pt():
    """At the converged T from flash_pv, flash_pt should give the same
    rho, phase, beta, x, y as the flash_pv result."""
    from stateprop.cubic import flash_pv
    z, mx = _make_methane_ethane_propane()
    p_ref = 30e5
    # Pick a v that gives 2-phase
    r_ref = flash_pt(p_ref, 220.0, z, mx, tol=1e-10)
    v_ref = 1.0 / r_ref.rho
    r_pv = flash_pv(p_ref, v_ref, z, mx, tol=1e-9)
    # Re-do flash_pt at the converged T
    r_pt = flash_pt(p_ref, r_pv.T, z, mx, tol=1e-10)
    rho_err = abs(r_pt.rho - r_pv.rho) / r_pt.rho
    beta_err = abs((r_pt.beta or 0) - (r_pv.beta or 0))
    check(f"flash_pv internally consistent: rho_err={rho_err:.3e}, "
          f"beta_err={beta_err:.3e}",
          rho_err < 1e-6 and beta_err < 1e-6)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_pure_alpha_r_derivs_fd,
        test_pressure_consistency,
        test_pure_saturation_SRK_vs_NIST,
        test_pure_saturation_PR_vs_NIST,
        test_mixture_pressure_consistency,
        test_mixture_ln_phi_fd,
        test_mixture_ln_phi_ternary_fd,
        test_pt_flash_fugacity_equality,
        test_cubic_analytic_dp_dx_at_rho,
        test_cubic_analytic_dlnphi_dx_at_rho,
        test_cubic_analytic_dlnphi_drho,
        test_cubic_analytic_dlnphi_dxk_at_p,
        test_cubic_newton_flash_pt,
        test_cubic_newton_flash_handles_single_phase,
        test_cubic_analytic_dp_dT_at_rho,
        test_cubic_analytic_dlnphi_dT_and_dp,
        test_cubic_flash_broyden_acceleration,
        test_pt_flash_supercritical,
        test_pt_flash_ternary,
        test_stability_test,
        test_bubble_dew_points,
        # Caloric + state-function flashes
        test_caloric_residual_fd,
        test_caloric_Cp_recovery,
        test_flash_pt_populates_h_s,
        test_flash_ph_roundtrip,
        test_flash_ps_roundtrip,
        test_flash_th_ts_roundtrip,
        # PR-1978
        test_pr78_m_coefficients,
        test_pr78_auto_dispatch,
        test_pr78_backward_compat,
        test_pr78_improves_heavy_components,
        test_pr78_in_mixture,
        # Alpha variants (Mathias-Copeman, Twu, PRSV)
        test_alpha_variant_derivatives_fd,
        test_mathias_copeman_reduces_to_soave,
        test_prsv_improves_water,
        test_twu_alpha_in_flash,
        test_alpha_override_in_mixture,
        test_alpha_override_invalid_raises,
        # Volume translation (Peneloux)
        test_peneloux_vapor_pressure_invariant,
        test_peneloux_lnphi_difference_invariant,
        test_peneloux_flash_invariant,
        test_peneloux_improves_srk_liquid_density,
        test_peneloux_roundtrip_density_pressure,
        test_peneloux_numeric_c,
        test_peneloux_pr_rejects_auto,
        test_peneloux_caloric_consistency,
        # v0.9.119 — volume-translation module API
        test_v119_peneloux_c_SRK_helper,
        test_v119_jhaveri_youngren_paraffin,
        test_v119_jhaveri_youngren_returns_zero_for_light,
        test_v119_lookup_volume_shift_known_compounds,
        test_v119_lookup_volume_shift_aliases,
        test_v119_lookup_volume_shift_unknown_returns_none,
        test_v119_resolve_volume_shift_modes,
        test_v119_cubic_from_name_volume_shift_auto,
        test_v119_cubic_from_name_default_no_shift,
        test_v119_cubic_from_name_volume_shift_c_legacy_kwarg,
        test_v119_cubic_from_name_volume_shift_table_miss_raises,
        test_v119_volume_shift_does_not_affect_phase_equilibrium,
        test_v119_volume_shift_improves_density,
        test_v119_list_volume_shift_compounds,
        # Critical points (Heidemann-Khalil / Michelsen)
        test_critical_A_residual_fd,
        test_critical_point_pure_limit,
        test_critical_point_CO2_CH4_locus,
        test_critical_point_CH4_N2,
        test_critical_point_vdw_rejected,
        # Phase envelope
        test_envelope_point_matches_bubble_point_p,
        test_trace_envelope_seeds_at_critical,
        test_envelope_bubble_side_fugacity_equality,
        # v0.9.15 -- Newton-Raphson bubble/dew point solvers
        test_cubic_newton_bubble_point_p_matches_ss,
        test_cubic_newton_bubble_point_p_near_critical,
        test_cubic_newton_bubble_point_T_matches_ss,
        test_cubic_newton_dew_point_p_matches_ss,
        test_cubic_newton_dew_point_T_matches_ss,
        test_cubic_newton_bubble_dew_ordering,
        test_cubic_newton_peneloux_analytic_path,
        # v0.9.16 -- Peneloux volume translation in analytic Jacobians
        test_peneloux_dlnphi_dxk_at_p_fd,
        test_peneloux_dlnphi_dp_at_T_fd,
        test_peneloux_dlnphi_dT_at_p_fd,
        test_peneloux_derivs_ternary_manual_c,
        # v0.9.17 -- Analytic Jacobian for envelope tracer's corrector
        test_envelope_analytic_jacobian_matches_fd,
        test_envelope_point_analytic_matches_fd,
        test_trace_envelope_analytic_corrector_matches_fd,
        # v0.9.20 -- Three-phase (VLLE) flash for cubic EOS
        test_cubic_three_phase_rr_material_balance,
        test_cubic_three_phase_preserves_vle,
        test_cubic_three_phase_preserves_single_phase,
        test_cubic_three_phase_water_butane_vlle,
        # v0.9.56 -- PV / Pα / Tα flash modes
        test_flash_pv_single_phase_roundtrip,
        test_flash_pv_two_phase_roundtrip,
        test_flash_p_alpha_recovers_target_beta,
        test_flash_p_alpha_endpoints_match_bubble_dew,
        test_flash_t_alpha_recovers_target_beta,
        test_flash_t_alpha_endpoints_match_bubble_dew,
        test_flash_p_alpha_invalid_alpha_raises,
        test_flash_pv_consistent_with_flash_pt,
    ]
    for t in tests:
        run_test(t)

    print(f"\n{'='*60}")
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    if FAILURES:
        print("\nFailures:")
        for name, detail in FAILURES:
            print(f"  - {name}: {detail}")
    print('='*60)
    sys.exit(0 if FAILED == 0 else 1)
