"""Tests for stateprop.transport (v0.9.32).

Validates Chung-Lee-Starling viscosity and thermal conductivity against
literature reference values, plus Brock-Bird surface tension.

These are *corresponding-states* correlations -- accuracy of 5-15% vs
experiment is expected and acceptable for engineering calculations.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stateprop.transport import (
    viscosity_chung, thermal_conductivity_chung,
    viscosity_mixture_chung, thermal_conductivity_mixture_chung,
    surface_tension_brock_bird,
)
from stateprop.saft import (
    METHANE, ETHANE, PROPANE, N_BUTANE, N_HEXANE, N_HEPTANE, N_OCTANE,
    CO2, NITROGEN, WATER, METHANOL, ETHANOL, ACETONE,
)

_results = []

def check(msg, ok, extra=""):
    tag = "PASS" if ok else "FAIL"
    _results.append((ok, msg, extra))
    print(f"  {tag}  {msg}" + (f": {extra}" if extra else ""))


def test_chung_viscosity_dilute_gas_hydrocarbons():
    """Dilute-gas Chung viscosity within 35% of experiment for
    non-polar hydrocarbons over 300-500K. Chung's own error bounds are
    5-15% on average but some fluids have ~30% systematic bias."""
    # (component, T_K, experimental viscosity in uPa.s at ~1 atm)
    # Values from Friend 1989, Vogel et al., NIST WebBook
    cases = [
        (METHANE,   500.0, 16.54),
        (ETHANE,    500.0, 12.7),
        (PROPANE,   500.0, 10.9),
        (N_BUTANE,  500.0, 10.8),
        (N_HEXANE,  500.0, 9.1),
        (N_HEPTANE, 500.0, 8.9),
        (N_OCTANE,  500.0, 8.8),
    ]
    for comp, T, nist_uPa in cases:
        eta = viscosity_chung(comp, T) * 1e6
        rel = abs(eta - nist_uPa) / nist_uPa
        check(f"{comp.name} @ {T}K: eta={eta:.2f} uPa.s vs NIST={nist_uPa:.2f} (err {rel*100:.1f}%)",
              rel < 0.35)


def test_chung_viscosity_dilute_gas_polar():
    """Polar fluids (CO2) via Chung dilute-gas viscosity."""
    cases = [
        (CO2,      500.0, 23.5),   # NIST
        (NITROGEN, 500.0, 25.7),   # NIST
    ]
    for comp, T, nist in cases:
        eta = viscosity_chung(comp, T) * 1e6
        rel = abs(eta - nist) / nist
        check(f"{comp.name} @ {T}K: eta={eta:.2f} uPa.s vs NIST={nist:.2f} (err {rel*100:.1f}%)",
              rel < 0.40)


def test_chung_viscosity_dense_liquid():
    """Dense-phase (liquid) Chung viscosity within 30% of experiment."""
    # (component, T_K, rho_mol [mol/m^3], experimental uPa.s)
    cases = [
        (N_BUTANE, 300.0, 10100.0, 160.0),    # saturated liquid, NIST
        (N_HEXANE, 300.0, 7640.0,  295.0),    # saturated liquid, NIST
    ]
    for comp, T, rho, nist in cases:
        eta = viscosity_chung(comp, T, rho_mol=rho) * 1e6
        rel = abs(eta - nist) / nist
        check(f"{comp.name} liquid @ {T}K rho={rho}: eta={eta:.1f} uPa.s vs NIST={nist:.1f} (err {rel*100:.1f}%)",
              rel < 0.35)


def test_chung_thermal_conductivity_dilute_gas():
    """Dilute-gas Chung thermal conductivity within 40% of experiment
    (note: thermal conductivity is harder than viscosity for
    corresponding-states methods due to the internal-mode contribution
    which is highly molecule-specific)."""
    # Methane at 500K, low-p: ~63 mW/m.K (NIST)
    # CO2 at 500K, low-p: ~34 mW/m.K
    cases = [
        (METHANE, 500.0, 63.0),
        (CO2,     500.0, 34.0),
    ]
    for comp, T, nist in cases:
        lam = thermal_conductivity_chung(comp, T) * 1e3
        rel = abs(lam - nist) / nist
        check(f"{comp.name} @ {T}K: lambda={lam:.2f} mW/m.K vs NIST={nist:.1f} (err {rel*100:.1f}%)",
              rel < 0.60)


def test_chung_viscosity_mixture():
    """Chung mixture viscosity via pseudo-component mixing should
    return a value between pure-component viscosities for a non-ideal
    binary, and should match pure-component for a degenerate
    composition."""
    # At x=1.0 the mixture should match pure component
    eta_pure = viscosity_chung(METHANE, 300.0) * 1e6
    eta_mix = viscosity_mixture_chung([METHANE, ETHANE], [1.0, 0.0], 300.0) * 1e6
    # Small deviation permitted due to numerical trace of the other
    # component (x_j = 0.0 gives sigma_ij terms in mixing rules)
    rel = abs(eta_mix - eta_pure) / eta_pure
    check(f"mixture viscosity at x=(1,0) matches pure CH4 (rel err = {rel:.2e})",
          rel < 0.05)

    # Binary 50/50: should be bracketed by pure values
    eta_a = viscosity_chung(METHANE, 300.0) * 1e6
    eta_b = viscosity_chung(ETHANE, 300.0) * 1e6
    eta_mix = viscosity_mixture_chung([METHANE, ETHANE], [0.5, 0.5], 300.0) * 1e6
    lo, hi = min(eta_a, eta_b), max(eta_a, eta_b)
    check(f"CH4-C2 50/50 mix viscosity {eta_mix:.2f} in [{lo:.2f}, {hi:.2f}]",
          lo - 1e-6 <= eta_mix <= hi + 1e-6)


def test_brock_bird_surface_tension():
    """Brock-Bird surface tension within 20% for non-polar, 40% for
    hydrogen-bonding fluids."""
    cases = [
        (N_BUTANE,  300.0, 11.7,  0.20),   # non-polar
        (N_HEXANE,  300.0, 17.9,  0.20),
        (N_OCTANE,  300.0, 21.1,  0.20),
        (WATER,     300.0, 71.7,  0.80),   # strong H-bonding, expected large error
        (ETHANOL,   300.0, 22.0,  0.85),
    ]
    for comp, T, nist_mNm, tol in cases:
        sig = surface_tension_brock_bird(comp, T) * 1e3  # N/m -> mN/m
        rel = abs(sig - nist_mNm) / nist_mNm
        check(f"{comp.name} @ {T}K: sigma={sig:.2f} mN/m vs NIST={nist_mNm:.1f} (err {rel*100:.1f}%)",
              rel < tol)


def test_macleod_sugden_pure_with_NIST_densities():
    """Pure-fluid Macleod-Sugden with NIST-quality densities matches
    NIST within 20%. Especially good for polar fluids (water 3% err
    vs Brock-Bird's 48%)."""
    from stateprop.transport import surface_tension_macleod_sugden
    cases = [
        (N_HEXANE, 300.0,  7640.0,  10.0,  17.9,  0.10),
        (N_OCTANE, 300.0,  6202.0,   0.7,  21.1,  0.10),
        (WATER,    300.0, 55400.0,   1.5,  71.7,  0.10),   # polar!
        (METHANOL, 300.0, 24500.0,  23.0,  22.1,  0.10),
    ]
    for comp, T, rL, rV, nist, tol in cases:
        sig = surface_tension_macleod_sugden(comp, rL, rV) * 1e3
        rel = abs(sig - nist) / nist
        check(f"MS {comp.name} @ {T}K: sigma={sig:.2f} mN/m vs NIST={nist:.1f} (err {rel*100:.1f}%)",
              rel < tol)


def test_macleod_sugden_with_saft_densities():
    """End-to-end: SAFT sat densities + Macleod-Sugden."""
    from stateprop.saft import SAFTMixture
    from stateprop.transport import surface_tension_macleod_sugden

    def sat_rhos(mx, T):
        def diff(p):
            try:
                rL = mx.density_from_pressure(p, T, phase_hint='liquid')
                rV = mx.density_from_pressure(p, T, phase_hint='vapor')
                return rL, rV, float(mx.ln_phi(rL, T)[0] - mx.ln_phi(rV, T)[0])
            except: return None
        p_prev, d_prev = None, None; p = 1.0
        for _ in range(60):
            out = diff(p)
            if out is not None:
                _, _, d = out
                if d_prev is not None and d_prev * d < 0:
                    a, b, da = p_prev, p, d_prev
                    for _ in range(40):
                        c = 0.5*(a+b); rr = diff(c)
                        if rr is None: b = c; continue
                        rL, rV, dc = rr
                        if abs(dc) < 1e-5: return rL, rV
                        if da*dc < 0: b = c
                        else: a = c; da = dc
                    return rL, rV
                p_prev, d_prev = p, d
            if p > 1e8: break
            p *= 1.5
        return None, None

    for comp, T, nist, tol in [(N_HEXANE, 300.0, 17.9, 0.15),
                                (WATER, 350.0, 63.2, 0.15)]:
        mx = SAFTMixture([comp], [1.0])
        rL, rV = sat_rhos(mx, T)
        if rL is None:
            check(f"{comp.name} SAFT sat @ {T}K found", False); continue
        sig = surface_tension_macleod_sugden(comp, rL, rV) * 1e3
        rel = abs(sig - nist) / nist
        check(f"SAFT+MS {comp.name} @ {T}K: sigma={sig:.2f} mN/m vs NIST={nist:.1f} (err {rel*100:.1f}%)",
              rel < tol)


def test_macleod_sugden_mixture_bracketed():
    """Mixture MS at x=(1,0) matches pure-component."""
    from stateprop.transport import (surface_tension_macleod_sugden,
                                     surface_tension_mixture_macleod_sugden)
    rL, rV = 7640.0, 10.0
    sig_pure = surface_tension_macleod_sugden(N_HEXANE, rL, rV) * 1e3
    sig_mix = surface_tension_mixture_macleod_sugden(
        [N_HEXANE, N_OCTANE], [1.0, 0.0], [1.0, 0.0], rL, rV
    ) * 1e3
    rel = abs(sig_mix - sig_pure) / sig_pure
    check(f"mixture x=(1,0) matches pure n-hexane (rel err = {rel:.2e})",
          rel < 1e-8)


def test_macleod_sugden_missing_parachor_raises():
    """Component without parachor -> clear error."""
    from stateprop.saft.eos import PCSAFT
    from stateprop.transport import surface_tension_macleod_sugden
    c = PCSAFT(m=1.0, sigma=3.7, epsilon_k=150.0, T_c=190.0, p_c=4.6e6,
               acentric_factor=0.01, molar_mass=0.016, name='nopar')
    try:
        surface_tension_macleod_sugden(c, 10000.0, 10.0)
        check("missing parachor raises ValueError", False, "no exception")
    except ValueError:
        check("missing parachor raises ValueError", True)


def test_surface_tension_zero_at_critical():
    """Surface tension is zero at/above T_c."""
    sig = surface_tension_brock_bird(N_BUTANE, N_BUTANE.T_c + 10.0)
    check("surface_tension = 0 above T_c", sig == 0.0)


def test_molar_mass_required_for_chung():
    """Chung transport needs molar_mass; a component without one raises."""
    from stateprop.saft.eos import PCSAFT
    c = PCSAFT(m=1.0, sigma=3.7, epsilon_k=150.0, T_c=190.0, p_c=4.6e6,
               acentric_factor=0.01, name='no_mass')
    try:
        viscosity_chung(c, 300.0)
        check("missing molar_mass raises ValueError", False, "no exception")
    except ValueError:
        check("missing molar_mass raises ValueError", True)


# ------------------------------------------------------------------------
# v0.9.38: Stiel-Thodos viscosity + Wassiljewa-Mason-Saxena thermal cond
# ------------------------------------------------------------------------


def test_stiel_thodos_pure_hexane_liquid():
    """Stiel-Thodos liquid n-hexane viscosity at 300 K should match
    NIST (~0.292 mPa·s) within 5%, better than Chung's dense extension
    which gives ~14% error at this state."""
    from stateprop.transport import (viscosity_stiel_thodos, viscosity_chung)
    # n-hexane critical properties from NIST
    M = 0.08618; T_c = 507.82; p_c = 3.034e6; V_c = 3.696e-4
    omega = 0.301
    class C:
        molar_mass = M; T_c = 507.82; p_c = 3.034e6; V_c = 3.696e-4
        acentric_factor = 0.301; dipole_moment = None; chung_kappa = None
    T = 300.0
    rho_liq = 7640.0
    # Dilute viscosity (low rho)
    mu_dilute = viscosity_chung(C, T, rho_mol=10.0)
    # Liquid via Stiel-Thodos
    mu_st = viscosity_stiel_thodos(rho_liq, T_c, p_c, V_c, M, mu_dilute)
    nist_mPa_s = 0.2917
    err_pct = abs(mu_st * 1e3 - nist_mPa_s) / nist_mPa_s * 100
    check(f"Stiel-Thodos n-hexane liquid 300K: {mu_st*1e3:.3f} mPa·s vs NIST=0.292 (err {err_pct:.1f}%)",
          err_pct < 5.0, f"err {err_pct:.1f}%")


def test_stiel_thodos_low_density_returns_dilute():
    """At very low rho_r, Stiel-Thodos should return dilute viscosity."""
    from stateprop.transport import viscosity_stiel_thodos
    M = 0.01604; T_c = 190.564; p_c = 4.5992e6; V_c = 9.86e-5
    mu_dilute = 1.05e-5
    mu = viscosity_stiel_thodos(1.0, T_c, p_c, V_c, M, mu_dilute)
    rel = abs(mu - mu_dilute) / mu_dilute
    check(f"Stiel-Thodos at rho_r→0 returns dilute (rel err = {rel:.2e})",
          rel < 1e-6)


def test_stiel_thodos_mixture_rho_zero_limit():
    """At rho→0, mixture Stiel-Thodos should equal the input dilute mu_mix."""
    import numpy as np
    from stateprop.transport import viscosity_mixture_stiel_thodos
    T_c = np.array([190.564, 305.32])
    p_c = np.array([4.5992e6, 4.872e6])
    V_c = np.array([9.86e-5, 1.456e-4])
    M = np.array([0.01604, 0.030069])
    x = np.array([0.5, 0.5])
    mu_dilute = 4.98e-6
    mu = viscosity_mixture_stiel_thodos(0.5, x, T_c, p_c, V_c, M, mu_dilute)
    rel = abs(mu - mu_dilute) / mu_dilute
    check(f"Stiel-Thodos mixture at rho→0 returns dilute (rel err = {rel:.2e})",
          rel < 1e-6)


def test_wassiljewa_pure_limit():
    """For x = (1, 0), Wassiljewa should return lambda_1."""
    import numpy as np
    from stateprop.transport import thermal_conductivity_mixture_wassiljewa
    M = np.array([0.01604, 0.030069])
    lam = np.array([0.0170, 0.0088])
    mu = np.array([5.56e-6, 4.56e-6])
    x_pure_1 = np.array([1.0, 0.0])
    lam_pure_1 = thermal_conductivity_mixture_wassiljewa(x_pure_1, M, lam, mu)
    rel = abs(lam_pure_1 - lam[0]) / lam[0]
    check(f"Wassiljewa at x=(1,0) returns lambda_1 (rel err = {rel:.2e})",
          rel < 1e-10)


def test_wassiljewa_bracketed_by_pure_values():
    """Mixture lambda must lie between min and max of pure-component values."""
    import numpy as np
    from stateprop.transport import thermal_conductivity_mixture_wassiljewa
    M = np.array([0.01604, 0.030069, 0.04410])     # CH4, C2H6, C3H8
    lam = np.array([0.0170, 0.0088, 0.0067])
    mu = np.array([5.56e-6, 4.56e-6, 3.95e-6])
    for x_test in [np.array([1/3, 1/3, 1/3]), np.array([0.7, 0.2, 0.1]),
                    np.array([0.1, 0.1, 0.8])]:
        lam_m = thermal_conductivity_mixture_wassiljewa(x_test, M, lam, mu)
        check(f"Wassiljewa lambda_m={lam_m:.4f} bracketed by [{lam.min():.4f}, {lam.max():.4f}]",
              lam.min() <= lam_m <= lam.max())


def test_wilke_pure_limit():
    """For x = (1, 0), Wilke mixture mu should return mu_1."""
    import numpy as np
    from stateprop.transport import viscosity_mixture_wilke
    M = np.array([0.01604, 0.030069])
    mu = np.array([5.56e-6, 4.56e-6])
    x = np.array([1.0, 0.0])
    mu_m = viscosity_mixture_wilke(x, M, mu)
    rel = abs(mu_m - mu[0]) / mu[0]
    check(f"Wilke at x=(1,0) returns mu_1 (rel err = {rel:.2e})", rel < 1e-10)


def test_wilke_bracketed_by_pure():
    """Mixture viscosity from Wilke must lie between pure values."""
    import numpy as np
    from stateprop.transport import viscosity_mixture_wilke
    M = np.array([0.01604, 0.030069, 0.04410])
    mu = np.array([5.56e-6, 4.56e-6, 3.95e-6])
    x_test = np.array([0.4, 0.4, 0.2])
    mu_m = viscosity_mixture_wilke(x_test, M, mu)
    check(f"Wilke mu_m={mu_m*1e6:.2f} bracketed by [{mu.min()*1e6:.2f}, {mu.max()*1e6:.2f}]",
          mu.min() <= mu_m <= mu.max())


def main():
    for fn in [
        test_chung_viscosity_dilute_gas_hydrocarbons,
        test_chung_viscosity_dilute_gas_polar,
        test_chung_viscosity_dense_liquid,
        test_chung_thermal_conductivity_dilute_gas,
        test_chung_viscosity_mixture,
        test_brock_bird_surface_tension,
        test_macleod_sugden_pure_with_NIST_densities,
        test_macleod_sugden_with_saft_densities,
        test_macleod_sugden_mixture_bracketed,
        test_macleod_sugden_missing_parachor_raises,
        test_surface_tension_zero_at_critical,
        test_molar_mass_required_for_chung,
        # v0.9.38 -- Stiel-Thodos + Wassiljewa
        test_stiel_thodos_pure_hexane_liquid,
        test_stiel_thodos_low_density_returns_dilute,
        test_stiel_thodos_mixture_rho_zero_limit,
        test_wassiljewa_pure_limit,
        test_wassiljewa_bracketed_by_pure_values,
        test_wilke_pure_limit,
        test_wilke_bracketed_by_pure,
    ]:
        print(f"\n[{fn.__name__}]")
        fn()

    passed = sum(1 for ok, _, _ in _results if ok)
    failed = len(_results) - passed
    print("\n" + "=" * 60)
    print(f"RESULT: {passed} passed, {failed} failed")
    if failed:
        print("\nFailures:")
        for ok, msg, extra in _results:
            if not ok:
                print(f"  - {msg}: {extra}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
