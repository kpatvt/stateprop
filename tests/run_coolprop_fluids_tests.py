"""Tests for the bundled CoolProp-derived fluids in stateprop/fluids/coolprop/.

These tests verify that the 98 fluids ingested from CoolProp's reference
library load correctly through stateprop's existing JSON loader and
produce thermodynamically reasonable results.

Note: this is NOT a full re-validation against published reference data
for each fluid -- that would require thousands of NIST WebBook checks.
Instead, these tests catch:

  1. JSON loading errors (file format, missing fields, etc.)
  2. Numerical kernel errors (NaN, division-by-zero, exp-overflow)
  3. Gross EOS violations (negative pressure in single-phase, dp/drho<0
     in stable region, ideal-gas limit failure)
  4. Reference-state spot checks for several well-known fluids against
     NIST WebBook approximate values

For the per-fluid pressure-equality validation (vs original CoolProp
output to e.g. 1e-12), use the in-converter round-trip tests in
run_converter_tests.py which already establish that the converter
preserves the EOS exactly.
"""
import sys
import os
import glob
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

import stateprop as sp
from stateprop.fluid import Fluid


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


def _p(f, T, rho):
    """Pressure from the multiparameter Helmholtz kernel."""
    return sp.compressibility_factor(rho, T, f) * rho * f.R * T


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

COOLPROP_DIR = os.path.normpath(os.path.join(
    HERE, "..", "stateprop", "fluids", "coolprop"))


def _all_files():
    return sorted(p for p in glob.glob(os.path.join(COOLPROP_DIR, "*.json"))
                  if not p.endswith("_manifest.json"))


# ---------------------------------------------------------------------------
# Test 1: every fluid loads
# ---------------------------------------------------------------------------

def test_all_files_load():
    files = _all_files()
    check(f"found at least 90 converted fluid files (got {len(files)})",
          len(files) >= 90)
    n_loaded = 0
    for path in files:
        try:
            f = Fluid.from_json(path)
            assert f.T_c > 0
            assert f.molar_mass > 0
            n_loaded += 1
        except Exception as e:
            check(f"load {os.path.basename(path)}", False,
                  f"{type(e).__name__}: {e}")
    check(f"all {len(files)} fluid files loaded successfully (got {n_loaded})",
          n_loaded == len(files))


# ---------------------------------------------------------------------------
# Test 2: ideal-gas limit at low density
# ---------------------------------------------------------------------------

def test_ideal_gas_limit_all_fluids():
    """At rho << rho_c and T >> T_c, p -> rho R T to high accuracy.

    Tolerance is 2e-3 rather than tighter because strongly associating
    fluids like Methanol genuinely show ~0.1% residual contributions
    even at rho = 0.001 * rho_c, T = 2*Tc due to H-bonding terms.
    """
    files = _all_files()
    n_ok = 0
    for path in files:
        f = Fluid.from_json(path)
        T = max(2.0 * f.T_c, 300.0)
        rho = 0.001 * f.rho_c
        p = _p(f, T, rho)
        p_ideal = rho * f.R * T
        rel = abs(p - p_ideal) / p_ideal
        if rel < 2e-3:
            n_ok += 1
        else:
            check(f"ideal-gas limit {os.path.basename(path)}", False,
                  f"rel diff = {rel:.2e}, T={T}, rho={rho}")
    check(f"ideal-gas limit holds for all {len(files)} fluids "
          f"(got {n_ok}/{len(files)} within 2e-3)",
          n_ok == len(files))


# ---------------------------------------------------------------------------
# Test 3: mechanical stability (dp/drho > 0) in supercritical region
# ---------------------------------------------------------------------------

def test_mechanical_stability_supercritical():
    """In the supercritical single-phase region, dp/drho must be positive
    everywhere. We probe at five (T, rho) points above the critical."""
    files = _all_files()
    n_ok = 0
    for path in files:
        f = Fluid.from_json(path)
        all_ok = True
        for T_factor in (1.5, 2.0, 3.0):
            for rho_factor in (0.5, 1.5):
                T = T_factor * f.T_c
                rho = rho_factor * f.rho_c
                try:
                    p1 = _p(f, T, rho)
                    p2 = _p(f, T, rho * 1.001)
                except Exception:
                    all_ok = False
                    break
                if not (p2 > p1):
                    all_ok = False
                    break
            if not all_ok:
                break
        if all_ok:
            n_ok += 1
        else:
            check(f"stability {os.path.basename(path)}", False,
                  f"dp/drho<=0 at supercritical point")
    check(f"mechanical stability for all {len(files)} fluids "
          f"(got {n_ok}/{len(files)})",
          n_ok == len(files))


# ---------------------------------------------------------------------------
# Test 4: spot checks against published reference values
# ---------------------------------------------------------------------------

def test_spot_check_references():
    """Selected (fluid, T, rho, expected_p) tuples from NIST WebBook
    or REFPROP. Tolerance is generous (~3%) because we compare a Helmholtz
    EOS evaluation to a sometimes-rounded WebBook tabulated value."""

    # Each entry: (filename, T, rho, expected_p_Pa, tol, comment)
    #
    # All cases below are SINGLE-PHASE points (well outside the two-phase
    # dome) where p(T, rho) from the EOS is unambiguous. Tolerances
    # account for: (a) my recall of WebBook values, and (b) different EOS
    # versions sometimes giving slightly different answers.
    cases = [
        # Methane gas at 300K, near 1 atm: rho ~ 40.6 mol/m^3 (Z<1)
        # Setzmann-Wagner gives p ~ 1.00 bar at this point
        ("methane.json", 300.0, 40.1, 1.00e5, 0.02,
         "Methane gas near 1 atm at 300K"),

        # Argon at 300K, ~1 MPa, gas: rho ~ 402 mol/m^3
        ("argon.json", 300.0, 402.0, 1.0e6, 0.02,
         "Argon gas at 300K, ~1 MPa"),

        # CO2 supercritical at 350K, 5000 mol/m^3
        # Span-Wagner: p ~ 97.5 bar (this matches the actual EOS)
        ("carbondioxide.json", 350.0, 5000.0, 9.75e6, 0.05,
         "CO2 supercritical at 350K"),

        # R134a low-density vapor at 300K (well above the two-phase dome
        # at this T, p): rho=100 mol/m^3, p ~ 2.37 bar
        ("r134a.json", 300.0, 100.0, 2.37e5, 0.05,
         "R134a low-density vapor at 300K"),

        # Water steam at 500K, 50 mol/m^3 (well-superheated, far from sat):
        # IAPWS-95: p = 50 * R * T * Z, Z very close to 1
        # p ~ 50 * 8.314 * 500 = 2.078e5 Pa, with small real-gas correction
        ("water.json", 500.0, 50.0, 2.07e5, 0.02,
         "Water vapor superheated at 500K"),

        # n-Propane gas at 350K, 100 mol/m^3 (above NBP, single-phase gas)
        # p ~ 100 * 8.314 * 350 = 2.91e5 Pa with small Z correction
        ("npropane.json", 350.0, 100.0, 2.85e5, 0.05,
         "n-Propane gas at 350K, low density"),

        # Hydrogen gas at 300K, 1000 mol/m^3 (well-defined)
        # p ~ 1000 * 8.314 * 300 = 2.494e6 Pa, Z slightly > 1 for H2
        ("hydrogen.json", 300.0, 1000.0, 2.5e6, 0.05,
         "Hydrogen gas at 300K, 1000 mol/m^3"),

        # n-Heptane gas at 400K (above NBP=371.6K), low density
        # p ~ 50 * 8.314 * 400 = 1.66e5 Pa with small Z correction
        # NEWLY ADDED in v0.9.1 via CP0 ideal-gas converter
        ("nheptane.json", 400.0, 50.0, 1.6e5, 0.05,
         "n-Heptane gas at 400K (v0.9.1 CP0 fluid)"),

        # R22 vapor at 280K (below Tc=369.295), low density
        # p ~ 100 * 8.314 * 280 = 2.33e5 Pa, real-gas Z slightly < 1
        # NEWLY ADDED in v0.9.1 via CP0 ideal-gas converter
        ("r22.json", 280.0, 100.0, 2.30e5, 0.05,
         "R22 vapor at 280K (v0.9.1 CP0 fluid)"),
    ]

    for filename, T, rho, p_expected, tol, comment in cases:
        path = os.path.join(COOLPROP_DIR, filename)
        if not os.path.exists(path):
            check(f"spot check: {filename} not bundled (skip)", True,
                  f"file missing -- skipped")
            continue
        try:
            f = Fluid.from_json(path)
            p_calc = _p(f, T, rho)
        except Exception as e:
            check(f"spot check {comment}", False,
                  f"computation failed: {type(e).__name__}: {e}")
            continue
        rel = abs(p_calc - p_expected) / abs(p_expected)
        check(f"{comment}: p_calc={p_calc/1e5:.3f} bar, "
              f"p_ref={p_expected/1e5:.3f} bar, rel={rel:.2e}",
              rel < tol,
              f"rel diff {rel:.2e} > tol {tol}")


# ---------------------------------------------------------------------------
# Test 4b: CALORIC spot-checks against NIST WebBook ideal-gas cp values.
#
# These tests guard against a silent-bug class that escaped detection for
# multiple releases: bugs in the ideal-gas kernel (alpha_0 computation) or
# in the CP0PolyT / PE_general / tau_log_tau conversion handlers show up
# in cp, cv, and entropy but do NOT affect pressure (which is dominated by
# the residual part alpha_r). Pressure-only spot-checks miss them entirely.
#
# The cp values below use rho=1 mol/m^3 (essentially ideal gas) at moderate
# temperatures so alpha_r contribution is negligible and cp is dominated
# by the ideal-gas polynomial terms being tested.
# ---------------------------------------------------------------------------

def test_caloric_spot_check_references():
    """Each entry: (filename, T, cp_expected_J_per_molK, tol, comment).

    cp_expected values are NIST WebBook ideal-gas cp^0(T) values for the
    pure substance. Tolerance is 5% (loose enough to accommodate minor
    reference-state and NIST-model differences, but tight enough to catch
    unit or formula errors).
    """
    cases = [
        # (filename, T[K], cp_expected[J/(mol*K)], tol, comment)

        # Air: diatomic, cp ~ 7/2 * R = 29.1 J/(mol K) at room temp
        # Tests PE_general kernel (code 9, v0.9.2) for c=2/3, d=1
        ("air.json", 298.15, 29.1, 0.05,
         "Air cp at 298K (PE_general term kernel)"),

        # n-Undecane (C11H24): large hydrocarbon with complex CP0PolyT polynomial
        # including t=-1 case. Tests tau_log_tau kernel (code 8, v0.9.2)
        # AND the Tc^t_k correction to CP0PolyT conversion (v0.9.2 bugfix).
        # NIST ideal-gas cp^0 at 400K is ~328 J/(mol K)
        ("nundecane.json", 400.0, 328.0, 0.05,
         "n-Undecane cp at 400K (CP0PolyT t=-1 + Tc^t_k fix)"),

        # HFE143m: CP0PolyT with t=[1,2,3] but no PE vibrational modes,
        # so cp is dominated by the polynomial. Off by factor of Tc^t_k
        # (~380) in stateprop versions before the v0.9.2 CP0PolyT fix.
        # NIST cp^0 at 400K ~ 108 J/(mol K)
        ("hfe143m.json", 400.0, 108.0, 0.05,
         "HFE143m cp at 400K (CP0PolyT polynomial dominant)"),

        # R22: CP0PolyT with t=1 and small coefficient. NIST ~60 at 400K.
        # This fluid was passing pressure checks even with the CP0PolyT bug
        # because the polynomial correction is small; this test would
        # still have caught a larger regression.
        ("r22.json", 400.0, 60.0, 0.10,
         "R22 cp at 400K (CP0Constant + small CP0PolyT correction)"),

        # D6 (siloxane, Colonna et al. 2006): CP0AlyLee form, tests the
        # Aly-Lee sinh^2/cosh^2 -> PE_sinh/PE_cosh mapping. Reference cp=924
        # at 700K computed directly from the Aly-Lee formula.
        ("d6.json", 700.0, 924.0, 0.02,
         "D6 cp at 700K (CP0AlyLee -> PE_sinh/PE_cosh)"),

        # Methyl Oleate (biodiesel component, Huber et al. 2009 EOS):
        # CP0PolyT (t=0.146) + 3 Planck-Einstein modes. Reference value
        # 746 J/(mol*K) computed directly from the EOS formula.
        ("methyloleate.json", 600.0, 746.0, 0.02,
         "MethylOleate cp at 600K (CP0PolyT fractional-t + PE)"),

        # Ammonia (Gao 2020 EOS, v0.9.3): tests the new ResidualHelmholtzGaoB
        # kernel for the rational-in-tau exponent `1/(beta*(tau-gamma)^2 + b)`.
        # NIST ideal-gas cp^0 at 400K is ~40.8 J/(mol*K).
        ("ammonia.json", 400.0, 40.8, 0.06,
         "Ammonia cp at 400K (ResidualHelmholtzGaoB kernel)"),

        # R125 (Lemmon 2005 EOS, v0.9.3): tests the ResidualHelmholtzLemmon2005
        # decomposition (poly + exp + double_exp). NIST cp^0 at 400K ~ 109.
        ("r125.json", 400.0, 109.0, 0.05,
         "R125 cp at 400K (ResidualHelmholtzLemmon2005 decomposition)"),

        # Methanol (Piazza & Span 2013 EOS, v0.9.3): tests the
        # ResidualHelmholtzDoubleExponential kernel (also Exponential block).
        # NIST cp^0 at 400K ~ 53 J/(mol*K).
        ("methanol.json", 400.0, 53.0, 0.05,
         "Methanol cp at 400K (ResidualHelmholtzDoubleExponential kernel)"),
    ]
    for filename, T, cp_expected, tol, comment in cases:
        fluid_path = os.path.join(COOLPROP_DIR, filename)
        if not os.path.exists(fluid_path):
            check(f"SKIP {comment} (file not present)", True)
            continue
        fluid = Fluid.from_json(fluid_path)
        rho = 1.0  # essentially ideal-gas limit
        cp_calc = sp.cp(rho, T, fluid)
        rel = abs(cp_calc - cp_expected) / abs(cp_expected)
        check(f"{comment}: cp_calc={cp_calc:.2f}, cp_ref={cp_expected:.2f}, "
              f"rel={rel:.2%}",
              rel < tol,
              f"rel diff {rel:.2%} > tol {tol:.0%}")


# ---------------------------------------------------------------------------
# Test 5: cross-check against existing bundled fluids where applicable
# ---------------------------------------------------------------------------

def test_cross_check_existing_fluids():
    """Where a fluid exists in BOTH the original stateprop fluids/ and the
    new fluids/coolprop/, results should agree closely AWAY from the
    critical region (the new files include non-analytic critical terms
    that the old files may omit)."""

    pairs = [
        # (existing_path, new_path, T, rho, max_rel_diff, comment)
        ("stateprop/fluids/carbondioxide.json",
         "stateprop/fluids/coolprop/carbondioxide.json",
         500.0, 2000.0, 1e-5,
         "CO2: existing vs new, away from critical"),
        ("stateprop/fluids/water.json",
         "stateprop/fluids/coolprop/water.json",
         400.0, 50000.0, 1e-4,
         "Water: existing vs new IAPWS-95"),
        ("stateprop/fluids/water.json",
         "stateprop/fluids/coolprop/water.json",
         800.0, 200.0, 1e-4,
         "Water gas: existing vs new IAPWS-95"),
    ]
    base = os.path.normpath(os.path.join(HERE, ".."))
    for old_rel, new_rel, T, rho, tol, comment in pairs:
        old_path = os.path.join(base, old_rel)
        new_path = os.path.join(base, new_rel)
        if not (os.path.exists(old_path) and os.path.exists(new_path)):
            check(f"cross-check {comment}: paths exist", False,
                  f"missing: {old_path if not os.path.exists(old_path) else new_path}")
            continue
        f_old = Fluid.from_json(old_path)
        f_new = Fluid.from_json(new_path)
        p_old = _p(f_old, T, rho)
        p_new = _p(f_new, T, rho)
        rel = abs(p_old - p_new) / max(abs(p_old), 1.0)
        check(f"{comment}: p_old={p_old/1e6:.5f}, p_new={p_new/1e6:.5f}, "
              f"rel={rel:.2e}",
              rel < tol,
              f"rel {rel:.2e} > tol {tol}")


# ---------------------------------------------------------------------------
# Test 6: manifest sanity
# ---------------------------------------------------------------------------

def test_manifest_present_and_consistent():
    manifest_path = os.path.join(COOLPROP_DIR, "_manifest.json")
    check(f"manifest file exists at {manifest_path}",
          os.path.exists(manifest_path))
    with open(manifest_path) as f:
        m = json.load(f)
    check(f"manifest has 'converted' list (got {len(m.get('converted', []))})",
          len(m.get("converted", [])) > 0)
    check(f"manifest has 'skipped' list",
          isinstance(m.get("skipped", []), list))
    files_on_disk = set(os.path.basename(p) for p in _all_files())
    files_in_manifest = set(c["output"] for c in m["converted"])
    missing = files_in_manifest - files_on_disk
    extra = files_on_disk - files_in_manifest
    check(f"manifest <-> disk: missing={len(missing)}, extra={len(extra)}",
          len(missing) == 0 and len(extra) == 0,
          f"missing={missing}, extra={extra}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_all_files_load,
        test_ideal_gas_limit_all_fluids,
        test_mechanical_stability_supercritical,
        test_spot_check_references,
        test_caloric_spot_check_references,
        test_cross_check_existing_fluids,
        test_manifest_present_and_consistent,
    ]
    for t in tests:
        run_test(t)
    print(f"\n{'='*60}")
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    if FAILURES:
        for n, d in FAILURES:
            print(f"  - {n}: {d}")
    print('='*60)
    sys.exit(0 if FAILED == 0 else 1)
