"""GERG-2008 numerical validation tests.

Distinct from run_gerg_tests.py (which is a structural integration test):
this file validates that the numerical output of the GERG-2008
implementation matches independent references.

Three classes of validation:

1. CROSS-EOS COMPARISON
   For nitrogen, CO2, and water the package contains both:
   - The dedicated reference equation (Span 2000 / Span 1996 / IAPWS-95)
   - The GERG-2008 simplified pure-component equation
   They should agree to within the GERG-2008 simplified form's accuracy
   envelope: very tight at low/moderate density (< 0.1%), looser near
   critical and at very high liquid-phase densities.

2. PHYSICAL REFERENCE POINTS
   Methane density at NTP should match NIST tabulated values to better
   than 0.1%.

3. MIXTURE REDUCTION CONSISTENCY
   A single-component "mixture" with composition [1.0] should produce
   pressure values identical (to machine precision) to the pure-fluid
   evaluation -- this is a regression check on the mixture reduction
   collapsing correctly to the pure-fluid limit.

Run: python tests/run_gerg_validation.py
"""
import sys
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

from stateprop.fluid import load_fluid
from stateprop.properties import pressure as fluid_pressure
from stateprop.saturation import density_from_pressure
from stateprop.mixture.mixture import load_mixture
from stateprop.mixture.properties import pressure as mix_pressure


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
# CROSS-EOS COMPARISON
# ---------------------------------------------------------------------------

def test_nitrogen_gerg_vs_span():
    """GERG-2008 nitrogen vs Span et al. (2000) reference EOS.

    Both formulations share the same Wagner functional form, so agreement
    should be very tight (better than 0.1%) across the entire validity
    range, with slightly larger spread at high density.
    """
    fl_ref = load_fluid("nitrogen")
    fl_gerg = load_fluid("gerg2008/nitrogen")
    T = 300.0
    cases = [
        # (rho mol/m^3, max relative pressure difference)
        (10.0,    1e-4),    # near ideal-gas
        (100.0,   1e-4),
        (1000.0,  1e-3),    # ~2.5 MPa
        (5000.0,  1e-3),    # ~12.7 MPa
        (15000.0, 1e-3),    # ~52 MPa, very dense
    ]
    for rho, tol in cases:
        p_ref = fluid_pressure(rho, T, fl_ref)
        p_gerg = fluid_pressure(rho, T, fl_gerg)
        rel = abs(p_gerg - p_ref) / abs(p_ref)
        check(f"N2 at T={T}K, rho={rho:.0f}: |rel diff| = {rel:.2e} (tol={tol})",
              rel < tol, f"got {rel:.2e}")


def test_water_gerg_vs_iapws95():
    """GERG-2008 water vs IAPWS-95 reference EOS (supercritical region).

    GERG-2008 uses 16 terms, IAPWS-95 uses 56 terms with non-analytic
    critical-enhancement. Agreement is excellent in the supercritical
    region (away from critical and saturation curve).
    """
    fl_ref = load_fluid("water")
    fl_gerg = load_fluid("gerg2008/water")
    T = 700.0   # supercritical, well above T_c=647 K
    cases = [
        (10.0,    1e-4),
        (100.0,   1e-3),
        (1000.0,  1e-3),
        (5000.0,  2e-3),
        (15000.0, 2e-3),
    ]
    for rho, tol in cases:
        p_ref = fluid_pressure(rho, T, fl_ref)
        p_gerg = fluid_pressure(rho, T, fl_gerg)
        rel = abs(p_gerg - p_ref) / abs(p_ref)
        check(f"H2O at T={T}K, rho={rho:.0f}: |rel diff| = {rel:.2e} (tol={tol})",
              rel < tol, f"got {rel:.2e}")


def test_co2_gerg_vs_span():
    """GERG-2008 CO2 vs Span & Wagner (1996) reference EOS.

    Span 1996 has 39 terms including non-analytic critical-enhancement;
    GERG-2008 has 22 simpler terms. Agreement is excellent at low to
    moderate density but degrades to a few percent at very high liquid-
    phase densities (this is the documented limitation of the GERG-2008
    simplified form, NOT a bug).
    """
    fl_ref = load_fluid("carbondioxide")
    fl_gerg = load_fluid("gerg2008/carbondioxide")
    T = 320.0   # supercritical
    cases = [
        # Looser tolerances at high density reflect the documented
        # accuracy difference between the two formulations.
        (10.0,    1e-4),
        (100.0,   1e-4),
        (1000.0,  1e-3),
        (5000.0,  1e-2),    # 1% near critical density
        (15000.0, 1e-1),    # 10% in dense liquid (GERG-2008 accuracy limit)
    ]
    for rho, tol in cases:
        p_ref = fluid_pressure(rho, T, fl_ref)
        p_gerg = fluid_pressure(rho, T, fl_gerg)
        rel = abs(p_gerg - p_ref) / abs(p_ref)
        check(f"CO2 at T={T}K, rho={rho:.0f}: |rel diff| = {rel:.2e} (tol={tol})",
              rel < tol, f"got {rel:.2e}")


# ---------------------------------------------------------------------------
# PHYSICAL REFERENCE POINTS
# ---------------------------------------------------------------------------

def test_methane_density_at_NTP():
    """Methane density at 20 C, 1 atm.

    Reference: NIST Webbook gives 0.6680 kg/m^3 at 20 C, 1 atm.
    Compressibility factor Z ~ 0.998 reflects the small attractive
    contribution.
    """
    fl = load_fluid("gerg2008/methane")
    T, p = 293.15, 101325.0
    rho_n = density_from_pressure(p, T, fl, phase="vapor")
    rho_kg = rho_n * fl.molar_mass
    Z = p / (rho_n * fl.R * T)
    NIST_rho_kg = 0.6680
    rel = abs(rho_kg - NIST_rho_kg) / NIST_rho_kg
    check(f"CH4 density at 20C, 1 atm = {rho_kg:.4f} kg/m^3 (NIST: {NIST_rho_kg})",
          rel < 5e-4, f"rel err = {rel:.2e}")
    check(f"  Z = {Z:.5f} (expect ~0.998)",
          0.997 < Z < 0.999)


def test_nitrogen_density_at_NTP():
    """Nitrogen density at 20 C, 1 atm.
    Reference: NIST gives 1.1648 kg/m^3 at 20 C, 1 atm.
    """
    fl = load_fluid("gerg2008/nitrogen")
    T, p = 293.15, 101325.0
    rho_n = density_from_pressure(p, T, fl, phase="vapor")
    rho_kg = rho_n * fl.molar_mass
    NIST_rho_kg = 1.1648
    rel = abs(rho_kg - NIST_rho_kg) / NIST_rho_kg
    check(f"N2 density at 20C, 1 atm = {rho_kg:.4f} kg/m^3 (NIST: {NIST_rho_kg})",
          rel < 5e-4, f"rel err = {rel:.2e}")


def test_co2_density_at_NTP():
    """CO2 density at 20 C, 1 atm.
    Reference: NIST gives 1.8393 kg/m^3 at 20 C, 1 atm.
    """
    fl = load_fluid("gerg2008/carbondioxide")
    T, p = 293.15, 101325.0
    rho_n = density_from_pressure(p, T, fl, phase="vapor")
    rho_kg = rho_n * fl.molar_mass
    NIST_rho_kg = 1.8393
    rel = abs(rho_kg - NIST_rho_kg) / NIST_rho_kg
    check(f"CO2 density at 20C, 1 atm = {rho_kg:.4f} kg/m^3 (NIST: {NIST_rho_kg})",
          rel < 1e-3, f"rel err = {rel:.2e}")


# ---------------------------------------------------------------------------
# MIXTURE REDUCTION CONSISTENCY
# ---------------------------------------------------------------------------

def test_single_component_mixture_collapses_to_pure_fluid():
    """A 'mixture' with composition [1.0] should give pressure values
    identical (to machine precision) to direct pure-fluid evaluation.

    This catches any divergence in the mixture-reduction code path that
    would silently produce different results for the same physical state.
    """
    for key in ["methane", "ethane", "nitrogen", "carbondioxide"]:
        fl = load_fluid(f"gerg2008/{key}")
        mix = load_mixture([f"gerg2008/{key}"], [1.0], binary_set=None)
        x = np.array([1.0])
        for T, rho in [(300.0, 100.0), (300.0, 5000.0), (200.0, 10000.0)]:
            p_fluid = fluid_pressure(rho, T, fl)
            p_mix = mix_pressure(rho, T, x, mix)
            # Should be exactly equal (zero relative error)
            rel = abs(p_fluid - p_mix) / abs(p_fluid) if p_fluid != 0 else 0.0
            check(f"{key:14s} T={T}K, rho={rho:>6.0f}: pure==mixture[1.0]"
                  f" (p={p_fluid:.4e}, rel diff={rel:.2e})",
                  rel < 1e-12)


# ---------------------------------------------------------------------------
# Z -> 1 LIMIT FOR ALL 21 GERG COMPONENTS
# ---------------------------------------------------------------------------

def test_all_components_ideal_gas_limit():
    """At very low density, every GERG-2008 fluid should give Z within
    1e-4 of unity. This catches gross errors in any individual component's
    polynomial coefficients.

    Heavy n-alkanes (n-heptane through n-decane) have large negative
    second virial coefficients (~-1000 to -2000 cm^3/mol at 400 K), so
    Z-1 ~ B*rho can exceed 1e-3 even at rho=1 mol/m^3. We use rho=0.01
    mol/m^3 so that even the heaviest alkane gives |Z-1| < 1e-4."""
    KEYS = [
        "methane", "nitrogen", "carbondioxide", "ethane", "propane",
        "nbutane", "isobutane", "npentane", "isopentane", "nhexane",
        "nheptane", "noctane", "nnonane", "ndecane", "hydrogen",
        "oxygen", "carbonmonoxide", "water", "hydrogensulfide",
        "helium", "argon",
    ]
    T = 400.0       # high enough to be vapor for all 21
    rho = 0.01      # extremely dilute
    for key in KEYS:
        fl = load_fluid(f"gerg2008/{key}")
        p = fluid_pressure(rho, T, fl)
        Z = p / (rho * fl.R * T)
        check(f"{key:18s}: Z = {Z:.7f} at low density",
              abs(Z - 1.0) < 1e-4, f"|Z-1| = {abs(Z-1):.2e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_nitrogen_gerg_vs_span,
        test_water_gerg_vs_iapws95,
        test_co2_gerg_vs_span,
        test_methane_density_at_NTP,
        test_nitrogen_density_at_NTP,
        test_co2_density_at_NTP,
        test_single_component_mixture_collapses_to_pure_fluid,
        test_all_components_ideal_gas_limit,
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
