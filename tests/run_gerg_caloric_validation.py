"""GERG-2008 caloric property cross-validation.

Validates enthalpy (h), entropy (s), heat capacities (cp, cv), and speed of
sound (w) for the three GERG-2008 components that also have dedicated
reference equations in the package (Span/IAPWS):

    - Nitrogen        (GERG-2008 vs Span et al. 2000)
    - Carbon Dioxide  (GERG-2008 vs Span & Wagner 1996)
    - Water           (GERG-2008 vs IAPWS-95)

Plus absolute-value comparison against NIST Webbook reference points where
available.

METHODOLOGY NOTES

1. Absolute h and s cannot be compared directly because each formulation
   chooses its own reference state (for nitrogen: Span 2000 sets h, s to
   zero at the NBP; IAPWS-95 sets them to zero at the triple point of
   water; GERG-2008 uses a different set of reference-state constants).
   We therefore validate:
     - cp, cv, w (state functions independent of reference offsets)
     - Δh, Δs along an isobar (differences cancel reference-state offsets)

2. cp, cv, w are compared directly between formulations and against NIST.

3. Thermodynamic identities (cp - cv = T*(dp/dT)^2 / (rho^2 * dp/drho);
   w^2 = cp/cv * dp/drho) are checked to ensure internal consistency.

PRE-v0.6.3 BUG FIX: The packaged `stateprop/fluids/nitrogen.json` (Span et
al. 2000) originally had INCORRECT NEGATIVE exponents in the ideal-gas
polynomial, yielding cv^0 = 15.37 J/(mol K) at 300 K (should be ~20.78)
and unphysical negative cv above 400 K. This was fixed in v0.6.3 by
flipping the exponent signs, which now gives Span N2 cp = 29.17 J/(mol K)
at 300 K matching NIST to 0.16%. This caloric validation suite would
have caught this bug on day one had it existed — and now serves as a
regression check to prevent reintroduction.

Run: python tests/run_gerg_caloric_validation.py
"""
import sys
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

from stateprop.fluid import load_fluid
from stateprop.properties import (
    enthalpy, entropy, cp, cv, speed_of_sound, pressure,
)
from stateprop.saturation import density_from_pressure


PASSED = 0
FAILED = 0
FAILURES = []


def check(label, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        FAILURES.append((label, detail))
        print(f"  FAIL  {label}: {detail}")


def run_test(fn):
    print(f"\n[{fn.__name__}]")
    try:
        fn()
    except Exception as e:
        global FAILED
        FAILED += 1
        FAILURES.append((fn.__name__, f"EXCEPTION: {type(e).__name__}: {e}"))
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# NIST absolute comparison (ground truth from NIST Webbook)
# ---------------------------------------------------------------------------

def test_nitrogen_cp_and_w_vs_NIST():
    """GERG-2008 nitrogen cp and speed of sound vs NIST Webbook values
    at four gas-phase states across 200-500 K at 1 atm.

    Expected accuracy: GERG-2008 simplified EOS matches NIST to ~0.1-0.5%
    for cp and w in the gas phase (worse in dense liquid)."""
    fl = load_fluid("gerg2008/nitrogen")
    # NIST Webbook reference points at (T, p) = (..., 1 atm)
    points = [
        # (T [K],  p [Pa],   cp_NIST [J/(mol K)], w_NIST [m/s])
        (200.0,   101325.0,  29.127,              287.9),
        (300.0,   101325.0,  29.124,              353.0),
        (400.0,   101325.0,  29.249,              407.9),
        (500.0,   101325.0,  29.582,              456.1),
    ]
    for T, p, cp_nist, w_nist in points:
        rho = density_from_pressure(p, T, fl, phase="vapor")
        cp_calc = cp(rho, T, fl)
        w_calc = speed_of_sound(rho, T, fl)
        rel_cp = abs(cp_calc - cp_nist) / cp_nist
        rel_w = abs(w_calc - w_nist) / w_nist
        check(f"N2 T={T}K, 1atm: cp = {cp_calc:.4f} (NIST {cp_nist}, diff {rel_cp:.2%})",
              rel_cp < 5e-3, f"rel err {rel_cp:.2e}")
        check(f"N2 T={T}K, 1atm: w  = {w_calc:.3f} (NIST {w_nist}, diff {rel_w:.2%})",
              rel_w < 5e-3, f"rel err {rel_w:.2e}")


def test_water_cp_and_w_vs_iapws_reference():
    """GERG-2008 water cp and speed of sound vs the packaged IAPWS-95
    reference equation (which is itself the NIST reference implementation
    for water). Tests at the same (T, p) so we're comparing computed state
    functions at physically equivalent states.

    Tolerance 2% reflects the documented accuracy of the GERG-2008 simplified
    water EOS (16 terms) relative to IAPWS-95 (56 terms)."""
    fl_ref = load_fluid("water")
    fl_gerg = load_fluid("gerg2008/water")
    points = [
        # (T [K], p [Pa], phase_hint)
        (500.0,  1e5,   "vapor"),
        (700.0,  1e5,   "vapor"),
        (700.0,  1e6,   "vapor"),
        (1000.0, 1e6,   "vapor"),
    ]
    for T, p, phase in points:
        rho_ref  = density_from_pressure(p, T, fl_ref,  phase=phase)
        rho_gerg = density_from_pressure(p, T, fl_gerg, phase=phase)
        cp_ref  = cp(rho_ref, T, fl_ref)
        cp_gerg = cp(rho_gerg, T, fl_gerg)
        w_ref  = speed_of_sound(rho_ref, T, fl_ref)
        w_gerg = speed_of_sound(rho_gerg, T, fl_gerg)
        rel_cp = abs(cp_gerg - cp_ref) / cp_ref
        rel_w  = abs(w_gerg  - w_ref)  / w_ref
        check(f"H2O T={T}K, p={p/1e5:.0f}bar: cp IAPWS={cp_ref:.3f}, GERG={cp_gerg:.3f} "
              f"(diff {rel_cp:.2%})",
              rel_cp < 2e-2, f"rel err {rel_cp:.2e}")
        check(f"H2O T={T}K, p={p/1e5:.0f}bar:  w IAPWS={w_ref:.2f},  GERG={w_gerg:.2f} "
              f"(diff {rel_w:.2%})",
              rel_w < 2e-2, f"rel err {rel_w:.2e}")


def test_co2_cp_and_w_vs_span96_reference():
    """GERG-2008 CO2 cp and speed of sound vs Span 1996 reference at
    same (T, p). Span 1996 is the NIST reference implementation for CO2.

    At dense supercritical states, the simplified GERG-2008 form deviates
    by a few percent from Span 1996 (documented GERG-2008 accuracy limit)."""
    fl_ref = load_fluid("carbondioxide")
    fl_gerg = load_fluid("gerg2008/carbondioxide")
    points = [
        # (T, p, phase_hint, tol_cp, tol_w)
        (300.0, 1e5,   "vapor", 2e-2, 1e-2),
        (350.0, 1e5,   "vapor", 2e-2, 1e-2),
        (500.0, 1e6,   "vapor", 2e-2, 1e-2),
        (750.0, 1e7,   "vapor", 2e-2, 1e-2),
    ]
    for T, p, phase, tol_cp, tol_w in points:
        rho_ref  = density_from_pressure(p, T, fl_ref,  phase=phase)
        rho_gerg = density_from_pressure(p, T, fl_gerg, phase=phase)
        cp_ref  = cp(rho_ref,  T, fl_ref)
        cp_gerg = cp(rho_gerg, T, fl_gerg)
        w_ref  = speed_of_sound(rho_ref,  T, fl_ref)
        w_gerg = speed_of_sound(rho_gerg, T, fl_gerg)
        rel_cp = abs(cp_gerg - cp_ref) / cp_ref
        rel_w  = abs(w_gerg  - w_ref)  / w_ref
        check(f"CO2 T={T}K, p={p/1e5:.0f}bar: cp Span96={cp_ref:.3f}, GERG={cp_gerg:.3f} "
              f"(diff {rel_cp:.2%}, tol {tol_cp:.0%})",
              rel_cp < tol_cp, f"rel err {rel_cp:.2e}")
        check(f"CO2 T={T}K, p={p/1e5:.0f}bar:  w Span96={w_ref:.2f},  GERG={w_gerg:.2f} "
              f"(diff {rel_w:.2%}, tol {tol_w:.0%})",
              rel_w < tol_w, f"rel err {rel_w:.2e}")


# ---------------------------------------------------------------------------
# Cross-EOS comparison (GERG-2008 vs Span/IAPWS reference equations)
# ---------------------------------------------------------------------------

def test_water_caloric_gerg_vs_iapws95():
    """GERG-2008 water cp, cv, w compared against IAPWS-95 reference
    at the same (T, rho) states. These are intensive properties
    independent of reference state, so direct comparison is meaningful."""
    fl_ref = load_fluid("water")
    fl_gerg = load_fluid("gerg2008/water")
    # Supercritical states (avoid 2-phase ambiguity)
    points = [
        (700.0, 100.0),     # low density supercritical
        (700.0, 1000.0),    # moderate density
        (1000.0, 5000.0),   # high T, high rho
    ]
    for T, rho in points:
        cp_ref = cp(rho, T, fl_ref)
        cp_gerg = cp(rho, T, fl_gerg)
        cv_ref = cv(rho, T, fl_ref)
        cv_gerg = cv(rho, T, fl_gerg)
        w_ref = speed_of_sound(rho, T, fl_ref)
        w_gerg = speed_of_sound(rho, T, fl_gerg)
        rel_cp = abs(cp_ref - cp_gerg) / cp_ref
        rel_cv = abs(cv_ref - cv_gerg) / cv_ref
        rel_w  = abs(w_ref  - w_gerg)  / w_ref
        check(f"H2O T={T}K, rho={rho}: cp IAPWS={cp_ref:.3f}, GERG={cp_gerg:.3f} "
              f"(diff {rel_cp:.2%})",
              rel_cp < 2e-2, f"rel err {rel_cp:.2e}")
        check(f"H2O T={T}K, rho={rho}: cv IAPWS={cv_ref:.3f}, GERG={cv_gerg:.3f} "
              f"(diff {rel_cv:.2%})",
              rel_cv < 2e-2, f"rel err {rel_cv:.2e}")
        check(f"H2O T={T}K, rho={rho}:  w IAPWS={w_ref:.2f},  GERG={w_gerg:.2f} "
              f"(diff {rel_w:.2%})",
              rel_w < 2e-2, f"rel err {rel_w:.2e}")


def test_co2_caloric_gerg_vs_span96():
    """GERG-2008 CO2 cp, cv, w compared against Span-Wagner 1996 at
    the same (T, rho) states.

    Span 1996 has 39 terms + non-analytic critical-enhancement; GERG-2008
    has 22 simplified terms. Expect tight agreement at moderate density,
    degrading near critical and in dense liquid."""
    fl_ref = load_fluid("carbondioxide")
    fl_gerg = load_fluid("gerg2008/carbondioxide")
    # Supercritical; avoid near-critical where 39-term Span has edge accuracy
    points = [
        # (T, rho, tolerance)
        (350.0, 100.0,  2e-3),
        (350.0, 1000.0, 5e-3),
        (500.0, 1000.0, 2e-3),
        (500.0, 5000.0, 5e-2),    # high-density, larger tolerance
    ]
    for T, rho, tol in points:
        cp_ref = cp(rho, T, fl_ref)
        cp_gerg = cp(rho, T, fl_gerg)
        w_ref = speed_of_sound(rho, T, fl_ref)
        w_gerg = speed_of_sound(rho, T, fl_gerg)
        rel_cp = abs(cp_ref - cp_gerg) / cp_ref
        rel_w = abs(w_ref - w_gerg) / w_ref
        check(f"CO2 T={T}K, rho={rho:.0f}: cp Span={cp_ref:.3f}, GERG={cp_gerg:.3f} "
              f"(diff {rel_cp:.2%}, tol {tol:.0%})",
              rel_cp < tol, f"rel err {rel_cp:.2e}")
        check(f"CO2 T={T}K, rho={rho:.0f}:  w Span={w_ref:.2f},  GERG={w_gerg:.2f} "
              f"(diff {rel_w:.2%}, tol {tol:.0%})",
              rel_w < tol, f"rel err {rel_w:.2e}")


def test_nitrogen_caloric_gerg_vs_span():
    """GERG-2008 vs Span 2000 nitrogen cp, cv, and speed of sound.

    With the ideal-gas polynomial in fluids/nitrogen.json fixed in v0.6.3,
    both formulations agree to better than 1% across the gas phase and
    moderate-density states."""
    fl_ref = load_fluid("nitrogen")
    fl_gerg = load_fluid("gerg2008/nitrogen")
    points = [
        # (T, rho)
        (200.0, 100.0),
        (300.0, 100.0),
        (300.0, 1000.0),
        (500.0, 1000.0),
        (200.0, 10000.0),
        (150.0, 20000.0),
    ]
    for T, rho in points:
        cp_ref = cp(rho, T, fl_ref)
        cp_gerg = cp(rho, T, fl_gerg)
        cv_ref = cv(rho, T, fl_ref)
        cv_gerg = cv(rho, T, fl_gerg)
        w_ref = speed_of_sound(rho, T, fl_ref)
        w_gerg = speed_of_sound(rho, T, fl_gerg)
        rel_cp = abs(cp_ref - cp_gerg) / cp_ref
        rel_cv = abs(cv_ref - cv_gerg) / cv_ref
        rel_w = abs(w_ref - w_gerg) / w_ref
        check(f"N2 T={T}K, rho={rho:.0f}: cp Span={cp_ref:.3f}, GERG={cp_gerg:.3f} "
              f"(diff {rel_cp:.2%})",
              rel_cp < 1e-2, f"rel err {rel_cp:.2e}")
        check(f"N2 T={T}K, rho={rho:.0f}: cv Span={cv_ref:.3f}, GERG={cv_gerg:.3f} "
              f"(diff {rel_cv:.2%})",
              rel_cv < 1e-2, f"rel err {rel_cv:.2e}")
        check(f"N2 T={T}K, rho={rho:.0f}:  w Span={w_ref:.2f},  GERG={w_gerg:.2f} "
              f"(diff {rel_w:.2%})",
              rel_w < 1e-2, f"rel err {rel_w:.2e}")


# ---------------------------------------------------------------------------
# Enthalpy and entropy differences (reference-state-independent)
# ---------------------------------------------------------------------------

def test_water_delta_h_delta_s_isobar():
    """Δh and Δs between two temperatures at constant pressure must agree
    between GERG and IAPWS up to the GERG simplified-EOS accuracy."""
    fl_ref = load_fluid("water")
    fl_gerg = load_fluid("gerg2008/water")
    p = 1e6   # 1 MPa (supercritical above 500K)
    T1, T2 = 500.0, 700.0

    rho1_r = density_from_pressure(p, T1, fl_ref, phase="vapor")
    rho2_r = density_from_pressure(p, T2, fl_ref, phase="vapor")
    rho1_g = density_from_pressure(p, T1, fl_gerg, phase="vapor")
    rho2_g = density_from_pressure(p, T2, fl_gerg, phase="vapor")

    dh_ref  = enthalpy(rho2_r, T2, fl_ref)  - enthalpy(rho1_r, T1, fl_ref)
    dh_gerg = enthalpy(rho2_g, T2, fl_gerg) - enthalpy(rho1_g, T1, fl_gerg)
    ds_ref  = entropy(rho2_r, T2, fl_ref)   - entropy(rho1_r, T1, fl_ref)
    ds_gerg = entropy(rho2_g, T2, fl_gerg)  - entropy(rho1_g, T1, fl_gerg)

    rel_h = abs(dh_ref - dh_gerg) / abs(dh_ref)
    rel_s = abs(ds_ref - ds_gerg) / abs(ds_ref)
    check(f"H2O Δh {T1}->{T2}K at {p/1e6}MPa: IAPWS={dh_ref:.2f}, GERG={dh_gerg:.2f} "
          f"J/mol (rel {rel_h:.3%})",
          rel_h < 2e-2, f"rel err {rel_h:.2e}")
    check(f"H2O Δs {T1}->{T2}K at {p/1e6}MPa: IAPWS={ds_ref:.4f}, GERG={ds_gerg:.4f} "
          f"J/(mol K) (rel {rel_s:.3%})",
          rel_s < 2e-2, f"rel err {rel_s:.2e}")


def test_co2_delta_h_isobar():
    """Δh along an isobar: GERG vs Span 1996 for CO2."""
    fl_ref = load_fluid("carbondioxide")
    fl_gerg = load_fluid("gerg2008/carbondioxide")
    p = 1e6
    T1, T2 = 350.0, 500.0

    rho1_r = density_from_pressure(p, T1, fl_ref, phase="vapor")
    rho2_r = density_from_pressure(p, T2, fl_ref, phase="vapor")
    rho1_g = density_from_pressure(p, T1, fl_gerg, phase="vapor")
    rho2_g = density_from_pressure(p, T2, fl_gerg, phase="vapor")

    dh_ref = enthalpy(rho2_r, T2, fl_ref) - enthalpy(rho1_r, T1, fl_ref)
    dh_gerg = enthalpy(rho2_g, T2, fl_gerg) - enthalpy(rho1_g, T1, fl_gerg)
    rel_h = abs(dh_ref - dh_gerg) / abs(dh_ref)
    check(f"CO2 Δh {T1}->{T2}K at {p/1e6}MPa: Span={dh_ref:.2f}, GERG={dh_gerg:.2f} "
          f"J/mol (rel {rel_h:.3%})",
          rel_h < 2e-2, f"rel err {rel_h:.2e}")


# ---------------------------------------------------------------------------
# Thermodynamic identity checks (internal consistency)
# ---------------------------------------------------------------------------

def test_thermodynamic_identities_gerg_components():
    """Verify two thermodynamic identities for GERG-2008 fluids:
       1. cp - cv = T * (dp/dT)^2 / (rho^2 * dp/drho)_T                 (Mayer)
       2. w^2   = (cp/cv) * (dp/drho)_T / M                             (sound speed)

    These hold for any thermodynamically consistent EOS. Failure would
    indicate a derivative bug or missing term."""
    for key, T, rho in [
        ("methane",       300.0, 100.0),
        ("ethane",        300.0, 100.0),
        ("nitrogen",      300.0, 100.0),
        ("carbondioxide", 350.0, 100.0),
        ("water",         700.0, 100.0),
        ("hydrogen",      300.0, 100.0),
    ]:
        fl = load_fluid(f"gerg2008/{key}")
        cp_v = cp(rho, T, fl)
        cv_v = cv(rho, T, fl)
        w_v = speed_of_sound(rho, T, fl)

        # Finite-difference (dp/dT)_rho and (dp/drho)_T
        dT = T * 1e-5
        drho = rho * 1e-5
        dpdT = (pressure(rho, T + dT, fl) - pressure(rho, T - dT, fl)) / (2 * dT)
        dpdrho = (pressure(rho + drho, T, fl) - pressure(rho - drho, T, fl)) / (2 * drho)

        # Mayer relation: cp - cv = T * (dp/dT)^2 / (rho^2 * dp/drho)
        mayer_rhs = T * dpdT**2 / (rho**2 * dpdrho)
        mayer_lhs = cp_v - cv_v
        rel_mayer = abs(mayer_rhs - mayer_lhs) / abs(mayer_lhs)

        # Sound speed identity: w^2 * M = (cp/cv) * dp/drho
        # (M in kg/mol, so w is in m/s)
        w2M = w_v**2 * fl.molar_mass
        w2_identity = (cp_v / cv_v) * dpdrho
        rel_w2 = abs(w2M - w2_identity) / abs(w2_identity)

        check(f"{key:15s} Mayer relation (cp-cv): {mayer_lhs:.4f} vs "
              f"{mayer_rhs:.4f} (rel {rel_mayer:.2e})",
              rel_mayer < 1e-5, f"rel err {rel_mayer:.2e}")
        check(f"{key:15s} Sound speed identity: w^2*M = {w2M:.2f} vs "
              f"(cp/cv)*dpdρ = {w2_identity:.2f} (rel {rel_w2:.2e})",
              rel_w2 < 1e-5, f"rel err {rel_w2:.2e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_nitrogen_cp_and_w_vs_NIST,
        test_water_cp_and_w_vs_iapws_reference,
        test_co2_cp_and_w_vs_span96_reference,
        test_water_caloric_gerg_vs_iapws95,
        test_co2_caloric_gerg_vs_span96,
        test_nitrogen_caloric_gerg_vs_span,
        test_water_delta_h_delta_s_isobar,
        test_co2_delta_h_isobar,
        test_thermodynamic_identities_gerg_components,
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
