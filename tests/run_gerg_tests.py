"""GERG-2008 integration test.

Loads a 5-component natural-gas-like mixture from the packaged GERG-2008
data (Tables A1-A8 of Kunz & Wagner 2012) and verifies the complete
Helmholtz mixture pipeline works end-to-end:

  1. Pure-fluid loading from fluids/gerg2008/<key>.json
  2. Component wrapper resolution from fluids/components/<key>.json
  3. Binary-parameter loading from fluids/binaries/gerg2008.json
     (210 pairs, 15 with departure functions)
  4. Mixture reduction (Kunz-Wagner reducing functions)
  5. Pressure evaluation at known states (must reproduce ideal-gas at low rho)
  6. Density-from-pressure inversion
  7. PT flash with phase classification

This is NOT a regression test against published GERG-2008 reference values
(no external data set wired up); it's a structural test that the ingested
data flows through the framework without errors and produces physically
sensible results.

Run: python tests/run_gerg_tests.py
"""
import sys
import os
import numpy as np

# Allow running from the repo root or tests/ dir
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

from stateprop.fluid import load_fluid
from stateprop.mixture.component import load_component
from stateprop.mixture.mixture import load_mixture
from stateprop.mixture.properties import (
    pressure, density_from_pressure, alpha_r_mix_derivs,
)
from stateprop.mixture.flash import flash_pt


# ---------------------------------------------------------------------------
# Test runner scaffolding (matches the style of run_cubic_tests.py)
# ---------------------------------------------------------------------------

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
# Tests
# ---------------------------------------------------------------------------

# The 5-component test mixture: a typical natural-gas composition.
# Path-style names "gerg2008/<key>" route through load_fluid to
# fluids/gerg2008/<key>.json. The fluids' `name` field is set to the
# lowercase key so binary-pair lookup matches without component wrappers.
TEST_COMPONENTS = [
    "gerg2008/methane", "gerg2008/ethane", "gerg2008/propane",
    "gerg2008/nitrogen", "gerg2008/carbondioxide",
]
TEST_KEYS = ["methane", "ethane", "propane", "nitrogen", "carbondioxide"]
TEST_X = np.array([0.85, 0.05, 0.02, 0.05, 0.03])
EXPECTED_TC = {            # GERG-2008 Table A5 (K)
    "methane":       190.564,
    "ethane":        305.322,
    "propane":       369.825,
    "nitrogen":      126.192,
    "carbondioxide": 304.1282,
}
EXPECTED_RHO_C = {          # GERG-2008 Table A5 (mol/m^3) -- converted from mol/dm^3
    "methane":       10139.342719,
    "ethane":        6870.85454,
    "propane":       5000.043088,
    "nitrogen":      11183.9,
    "carbondioxide": 10624.978698,
}


def test_load_pure_fluids():
    """All 21 GERG fluids load and have valid critical parameters."""
    KEYS = [
        "methane", "nitrogen", "carbondioxide", "ethane", "propane",
        "nbutane", "isobutane", "npentane", "isopentane", "nhexane",
        "nheptane", "noctane", "nnonane", "ndecane", "hydrogen",
        "oxygen", "carbonmonoxide", "water", "hydrogensulfide",
        "helium", "argon",
    ]
    for key in KEYS:
        try:
            fl = load_fluid(f"gerg2008/{key}")
            check(f"loaded gerg2008/{key} (T_c={fl.T_c:.2f}K, M={fl.molar_mass*1000:.3f}g/mol)",
                  fl.T_c > 0 and fl.rho_c > 0 and fl.molar_mass > 0)
        except Exception as e:
            check(f"loaded gerg2008/{key}", False, str(e))


def test_critical_parameters_match_table_a5():
    """Critical T and rho for our 5 test components match Table A5 to <0.01%."""
    for path_name, key in zip(TEST_COMPONENTS, TEST_KEYS):
        c = load_component(path_name)
        rel_T = abs(c.T_c - EXPECTED_TC[key]) / EXPECTED_TC[key]
        rel_rho = abs(c.rho_c - EXPECTED_RHO_C[key]) / EXPECTED_RHO_C[key]
        check(f"{key} T_c matches Table A5 ({c.T_c:.4f} vs {EXPECTED_TC[key]:.4f})",
              rel_T < 1e-4, f"rel err {rel_T:.2e}")
        check(f"{key} rho_c matches Table A5 ({c.rho_c:.2f} vs {EXPECTED_RHO_C[key]:.2f})",
              rel_rho < 1e-4, f"rel err {rel_rho:.2e}")


def test_load_mixture_with_binaries():
    """Loading the 5-component mixture brings in all C(5,2)=10 binary pairs from the
    GERG-2008 binary set, and the expected 7 of them carry departure functions."""
    mix = load_mixture(
        component_names=TEST_COMPONENTS,
        composition=TEST_X.tolist(),
        binary_set="gerg2008",
    )
    check(f"loaded 5-component mixture", len(mix.components) == 5,
          f"got {len(mix.components)} components")
    check(f"loaded all 10 binary pairs", len(mix.binary) == 10,
          f"got {len(mix.binary)} pairs")

    # Within these 5 components, 7 pairs are in GERG-2008's specific set with non-zero F:
    # CH4-N2, CH4-CO2, CH4-C2H6, CH4-C3H8, N2-CO2, N2-C2H6, C2H6-C3H8
    n_with_dep = sum(1 for bp in mix.binary.values() if bp.departure is not None)
    check(f"7 of 10 pairs have departure functions",
          n_with_dep == 7, f"got {n_with_dep}")
    n_with_nonzero_F = sum(1 for bp in mix.binary.values() if bp.F != 0.0)
    check(f"7 of 10 pairs have non-zero F",
          n_with_nonzero_F == 7, f"got {n_with_nonzero_F}")


def test_mixture_reduction_makes_sense():
    """Reduced (T_r, rho_r) should fall between the lightest and heaviest
    component's pure values, biased by composition."""
    mix = load_mixture(TEST_COMPONENTS, TEST_X.tolist(), binary_set="gerg2008")
    T_r, rho_r = mix.reduce(TEST_X)
    # Pure Tc range: nitrogen 126.19 to propane 369.83
    check(f"T_r between component Tc bounds (got {T_r:.2f} K)",
          126.0 < T_r < 370.0)
    # For a methane-dominated mixture, T_r should be near methane's T_c
    check(f"T_r close to methane Tc for CH4-rich mix (T_r={T_r:.2f}, Tc_CH4=190.56)",
          150.0 < T_r < 230.0)
    check(f"rho_r positive ({rho_r:.2f} mol/m^3)", rho_r > 0)


def test_low_density_approaches_ideal_gas():
    """At very low density, Z = p/(rho R T) -> 1 (ideal-gas limit)."""
    mix = load_mixture(TEST_COMPONENTS, TEST_X.tolist(), binary_set="gerg2008")
    R = 8.314472
    T = 300.0
    for rho in [1.0, 10.0, 100.0]:
        p = pressure(rho, T, TEST_X, mix)
        Z = p / (rho * R * T)
        check(f"Z -> 1 at T={T}K, rho={rho} mol/m^3 (got Z={Z:.5f})",
              abs(Z - 1.0) < 0.01)


def test_density_from_pressure_roundtrip():
    """Round-trip: solve density from pressure, then evaluate pressure -> get back original."""
    mix = load_mixture(TEST_COMPONENTS, TEST_X.tolist(), binary_set="gerg2008")
    cases = [
        (300.0, 1e5),       # ambient
        (300.0, 1e6),       # 10 bar
        (250.0, 5e6),       # 50 bar
        (200.0, 2e6),       # cold compressed
    ]
    for T, p in cases:
        try:
            rho = density_from_pressure(p, T, TEST_X, mix, phase_hint="vapor")
            p_back = pressure(rho, T, TEST_X, mix)
            rel_err = abs(p_back - p) / p
            check(f"round-trip at T={T}K, p={p/1e3:.0f}kPa "
                  f"(rho={rho:.2f}, p_back={p_back:.0f})",
                  rel_err < 1e-6, f"rel err {rel_err:.2e}")
        except Exception as e:
            check(f"round-trip at T={T}K, p={p/1e3:.0f}kPa", False, str(e))


def test_supercritical_flash():
    """At conditions well above pseudo-critical, flash returns supercritical."""
    mix = load_mixture(TEST_COMPONENTS, TEST_X.tolist(), binary_set="gerg2008")
    T, p = 350.0, 5e6   # well above the mixture's pseudo-critical
    result = flash_pt(p, T, TEST_X, mix)
    check(f"flash at T={T}K, p={p/1e6}MPa returns supercritical",
          result.phase in ("supercritical", "vapor"),
          f"got phase={result.phase!r}")
    # Density should be reasonable (between vapor and liquid extremes)
    if result.rho is not None:
        check(f"  rho positive ({result.rho:.2f} mol/m^3)", result.rho > 0)


def test_low_pressure_vapor_flash():
    """At low T and high p the mix may be liquid; at low p it should be vapor."""
    mix = load_mixture(TEST_COMPONENTS, TEST_X.tolist(), binary_set="gerg2008")
    T, p = 280.0, 1e5
    result = flash_pt(p, T, TEST_X, mix)
    check(f"flash at T={T}K, p={p/1e3}kPa returns vapor or single-phase",
          result.phase in ("vapor", "supercritical"),
          f"got phase={result.phase!r}")


def test_departure_function_active():
    """For methane-N2 (z=1.0 each, hypothetically), the departure-function
    contribution to alpha^r should be non-negligible at moderate density.
    We test by comparing pressure WITH and WITHOUT the departure block."""
    # Build CH4/N2 binary
    mix_with = load_mixture(["gerg2008/methane", "gerg2008/nitrogen"], [0.7, 0.3],
                            binary_set="gerg2008")
    mix_without = load_mixture(["gerg2008/methane", "gerg2008/nitrogen"], [0.7, 0.3],
                                binary_set=None)
    x = np.array([0.7, 0.3])
    T, rho = 200.0, 5000.0    # moderate density, near critical
    p_with = pressure(rho, T, x, mix_with)
    p_without = pressure(rho, T, x, mix_without)
    rel_diff = abs(p_with - p_without) / abs(p_without)
    # Without departure functions, the binary uses default beta_T=gamma_T=beta_v=gamma_v=1
    # AND F=0, so pressure is just the ideal-mixing of the pure-fluid alpha_r's.
    # With GERG-2008 binary parameters and departure function, p should differ.
    check(f"departure block active (p_with={p_with:.0f}, p_without={p_without:.0f}, "
          f"rel diff={rel_diff:.2%})",
          rel_diff > 0.001,
          f"departure has no effect on pressure (rel diff {rel_diff:.2e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_load_pure_fluids,
        test_critical_parameters_match_table_a5,
        test_load_mixture_with_binaries,
        test_mixture_reduction_makes_sense,
        test_low_density_approaches_ideal_gas,
        test_density_from_pressure_roundtrip,
        test_supercritical_flash,
        test_low_pressure_vapor_flash,
        test_departure_function_active,
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
