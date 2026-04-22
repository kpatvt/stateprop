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
