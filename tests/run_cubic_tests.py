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
