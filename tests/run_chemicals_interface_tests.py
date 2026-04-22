"""Tests for stateprop.cubic.from_chemicals.

Works whether or not the `chemicals` library is installed:
    - If chemicals IS installed: tests cover the real databank (26,000 chemicals)
    - If chemicals is NOT installed: tests cover the built-in fallback table

The fallback-table tests should pass in both cases (since even with chemicals
installed, the stateprop interface returns identical structures).
"""
import sys
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

from stateprop.cubic import (
    PR, SRK,                                      # original factories
    PR_from_name, SRK_from_name, cubic_from_name, # new name-based factories
    cubic_mixture_from_names,
    lookup_pure_component,
    chemicals_available,
    CubicMixture,
)
from stateprop.cubic.flash import flash_pt

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
# Lookup tests
# ---------------------------------------------------------------------------

def test_lookup_basic_fluids():
    """All 21 GERG-2008 components can be looked up."""
    names = [
        "methane", "ethane", "propane", "n-butane", "isobutane",
        "n-pentane", "isopentane", "n-hexane", "n-heptane", "n-octane",
        "n-nonane", "n-decane", "nitrogen", "oxygen", "carbon monoxide",
        "carbon dioxide", "water", "hydrogen sulfide",
        "hydrogen", "helium", "argon",
    ]
    for name in names:
        info = lookup_pure_component(name)
        check(f"lookup '{name}': Tc={info['T_c']:.2f}K, Pc={info['p_c']/1e6:.2f}MPa, "
              f"ω={info['omega']:+.3f}, M={info['M']*1000:.3f}g/mol "
              f"(source={info['source']})",
              info["T_c"] > 0 and info["p_c"] > 0 and info["M"] > 0,
              f"bad info: {info}")


def test_lookup_aliases():
    """Common aliases (formulas, short names) resolve to correct fluids."""
    pairs = [
        ("methane",         "CH4"),
        ("methane",         "c1"),
        ("nitrogen",        "N2"),
        ("carbondioxide",   "CO2"),
        ("carbondioxide",   "Carbon Dioxide"),
        ("water",           "H2O"),
        ("hydrogen",        "H2"),
        ("nbutane",         "butane"),
        ("nbutane",         "n-butane"),
        ("isobutane",       "i-butane"),
        ("nhexane",         "hexane"),
        ("helium",          "He"),
        ("argon",           "Ar"),
    ]
    for canonical_name, alias in pairs:
        a = lookup_pure_component(canonical_name)
        b = lookup_pure_component(alias)
        check(f"alias '{alias}' -> same Tc as '{canonical_name}' "
              f"({a['T_c']:.2f} vs {b['T_c']:.2f})",
              abs(a["T_c"] - b["T_c"]) < 1e-6,
              f"Tc differs: {a['T_c']} vs {b['T_c']}")


def test_lookup_critical_parameters_sensible():
    """Spot-check critical parameters against well-known values."""
    # Methane: Tc = 190.56 K, Pc = 4.60 MPa, omega = 0.011
    m = lookup_pure_component("methane")
    check(f"methane Tc = {m['T_c']:.2f} (expected ~190.56)",
          189 < m["T_c"] < 192)
    check(f"methane Pc = {m['p_c']/1e6:.3f} MPa (expected ~4.6)",
          4.5e6 < m["p_c"] < 4.7e6)
    check(f"methane omega = {m['omega']:+.4f} (expected ~0.011)",
          0.0 < m["omega"] < 0.03)

    # Water: Tc = 647 K, Pc = 22 MPa, omega = 0.344
    w = lookup_pure_component("water")
    check(f"water Tc = {w['T_c']:.2f} (expected ~647)",
          645 < w["T_c"] < 648)
    check(f"water Pc = {w['p_c']/1e6:.3f} MPa (expected ~22.06)",
          21.5e6 < w["p_c"] < 22.5e6)


def test_unknown_fluid_raises():
    """An identifier that's in neither chemicals nor the fallback raises KeyError."""
    try:
        lookup_pure_component("xylzzzene_not_a_real_fluid_32167")
        check("unknown fluid raises", False, "no exception raised")
    except (KeyError, ValueError):
        check("unknown fluid raises correctly", True)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

def test_PR_from_name_matches_manual_construction():
    """PR_from_name(name) should give identical EOS to PR(Tc, pc, omega) with
    matching values."""
    info = lookup_pure_component("methane")
    eos_named = PR_from_name("methane")
    eos_manual = PR(T_c=info["T_c"], p_c=info["p_c"], acentric_factor=info["omega"])
    # Compare a pressure point
    T, rho = 300.0, 5000.0
    p_named = eos_named.pressure(rho, T)
    p_manual = eos_manual.pressure(rho, T)
    check(f"PR_from_name('methane').pressure == PR(manual).pressure "
          f"({p_named:.2e} vs {p_manual:.2e})",
          abs(p_named - p_manual) / p_manual < 1e-12,
          f"p_named={p_named}, p_manual={p_manual}")


def test_SRK_from_name():
    """SRK family via SRK_from_name."""
    eos = SRK_from_name("ethane")
    T, rho = 300.0, 5000.0
    p = eos.pressure(rho, T)
    check(f"SRK ethane pressure at T=300K, rho=5000 = {p:.3e} Pa",
          1e6 < p < 5e7, f"p={p}")


def test_cubic_from_name_family_dispatch():
    """family='pr' / 'srk' / 'rk' / 'vdw' all produce valid EOS."""
    for fam in ["pr", "pr78", "srk", "rk", "vdw"]:
        eos = cubic_from_name("methane", family=fam)
        p = eos.pressure(5000.0, 300.0)
        check(f"cubic_from_name('methane', family='{fam}') gives "
              f"p={p:.2e} Pa at T=300K, rho=5000",
              1e6 < p < 5e7,
              f"p={p}")


def test_cubic_from_name_unknown_family():
    """Unknown family raises ValueError."""
    try:
        cubic_from_name("methane", family="bogus")
        check("unknown family raises", False, "no exception")
    except ValueError:
        check("unknown family raises ValueError correctly", True)


# ---------------------------------------------------------------------------
# Mixture factory tests
# ---------------------------------------------------------------------------

def test_mixture_from_names_5component_natural_gas():
    """Build a 5-component natural-gas mixture and run a PT flash."""
    mix = cubic_mixture_from_names(
        ["methane", "ethane", "propane", "nitrogen", "carbon dioxide"],
        composition=[0.85, 0.05, 0.02, 0.05, 0.03],
        family="pr",
        k_ij={(0, 3): 0.025, (0, 4): 0.09, (3, 4): -0.017},
    )
    check("5-component mixture has 5 components",
          len(mix.components) == 5, f"got {len(mix.components)}")

    # Check each component's Tc matches a spot value
    expected_Tc = [190.56, 305.32, 369.83, 126.19, 304.13]
    for i, expected in enumerate(expected_Tc):
        got = mix.components[i].T_c
        check(f"component {i} Tc = {got:.2f} (expected ~{expected})",
              abs(got - expected) < 0.5)

    # Run a PT flash as integration test
    z = np.array([0.85, 0.05, 0.02, 0.05, 0.03])
    r = flash_pt(5e6, 300.0, z, mix, tol=1e-9)
    check(f"flash_pt on name-built mixture: phase={r.phase}, "
          f"rho={r.rho:.2f} mol/m^3",
          r.phase in ("supercritical", "vapor", "liquid", "two_phase")
          and r.rho > 0)


def test_mixture_from_names_matches_manual():
    """A mixture built from names should give identical properties to one
    built with explicit Tc/Pc/omega values."""
    # Manual build
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    c_N2  = PR(126.192, 3.3958e6, 0.0372)
    c_CO2 = PR(304.128, 7.3773e6, 0.22394)
    mix_manual = CubicMixture([c_CH4, c_N2, c_CO2], composition=[0.7, 0.2, 0.1],
                               k_ij={(0, 1): 0.025, (0, 2): 0.09})

    # Name-based build (Note: fallback-table values may differ at 4th-6th
    # decimal from manual values above; tolerance loosened accordingly)
    mix_named = cubic_mixture_from_names(
        ["methane", "nitrogen", "carbon dioxide"],
        composition=[0.7, 0.2, 0.1],
        family="pr",
        k_ij={(0, 1): 0.025, (0, 2): 0.09},
    )

    z = np.array([0.7, 0.2, 0.1])
    T, p = 300.0, 5e6
    r_manual = flash_pt(p, T, z, mix_manual, tol=1e-9)
    r_named = flash_pt(p, T, z, mix_named, tol=1e-9)
    rel_rho = abs(r_manual.rho - r_named.rho) / r_manual.rho
    rel_h = abs(r_manual.h - r_named.h) / max(1.0, abs(r_manual.h))
    # Values should be near-identical since fallback table is from same
    # references as the hardcoded values
    check(f"name-built vs manual density match (rho_manual={r_manual.rho:.2f}, "
          f"rho_named={r_named.rho:.2f}, rel diff {rel_rho:.2e})",
          rel_rho < 1e-3, f"rel diff {rel_rho:.2e}")


def test_mixture_from_names_default_composition():
    """Omitting composition should default to equal mole fractions."""
    mix = cubic_mixture_from_names(["methane", "ethane"], family="pr")
    z = np.array([0.5, 0.5])
    T, p = 300.0, 1e5
    r = flash_pt(p, T, z, mix, tol=1e-9)
    check(f"default-composition (equimolar) mixture flashes OK: phase={r.phase}",
          r.phase in ("vapor", "supercritical"),
          f"got phase={r.phase}")


# ---------------------------------------------------------------------------
# Edge cases: volume-shift and other pass-through kwargs
# ---------------------------------------------------------------------------

def test_kwargs_passthrough():
    """Extra kwargs should reach the underlying EOS factory.

    Test by setting a volume-shift parameter and checking that the EOS
    stored it. The CubicEOS dataclass exposes `volume_shift_c`.
    """
    eos = PR_from_name("propane", volume_shift_c=-5e-6)
    check(f"volume_shift_c kwarg reached the EOS: "
          f"got {eos.volume_shift_c}",
          eos.volume_shift_c == -5e-6)


# ---------------------------------------------------------------------------
# UV flash on a name-built mixture (integration with v0.7.0 UV flash)
# ---------------------------------------------------------------------------

def test_name_built_mixture_uv_flash():
    """The name-built mixture should work with the UV flash introduced
    in v0.7.0 -- end-to-end integration test."""
    from stateprop.cubic.flash import flash_uv
    mix = cubic_mixture_from_names(
        ["methane", "ethane", "propane", "nitrogen", "carbon dioxide"],
        composition=[0.85, 0.05, 0.02, 0.05, 0.03],
        family="pr",
        k_ij={(0, 3): 0.025, (0, 4): 0.09},
    )
    z = np.array([0.85, 0.05, 0.02, 0.05, 0.03])
    T_true, p_true = 300.0, 5e6
    r = flash_pt(p_true, T_true, z, mix, tol=1e-9)
    u = r.h - r.p / r.rho
    v = 1.0 / r.rho
    r_uv = flash_uv(u, v, z, mix)
    rel_T = abs(r_uv.T - T_true) / T_true
    rel_p = abs(r_uv.p - p_true) / p_true
    check(f"name-built mixture UV round-trip: T_err={rel_T:.1e}, "
          f"p_err={rel_p:.1e}",
          rel_T < 1e-4 and rel_p < 1e-4,
          f"T err {rel_T:.2e}, p err {rel_p:.2e}")


# ---------------------------------------------------------------------------
# Check chemicals-library detection
# ---------------------------------------------------------------------------

def test_chemicals_availability_flag():
    """`chemicals_available()` returns a boolean."""
    flag = chemicals_available()
    check(f"chemicals_available() returns bool (got {flag!r}, type {type(flag).__name__})",
          isinstance(flag, bool))


def test_source_labeling():
    """The lookup dict's `source` field is 'chemicals' when chemicals is
    installed, otherwise 'fallback'. Either way it's a string in
    {'chemicals', 'fallback'}."""
    info = lookup_pure_component("methane")
    expected = "chemicals" if chemicals_available() else "fallback"
    check(f"lookup source label = '{info['source']}' (expected '{expected}')",
          info["source"] == expected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Lookup
        test_lookup_basic_fluids,
        test_lookup_aliases,
        test_lookup_critical_parameters_sensible,
        test_unknown_fluid_raises,
        # Factories
        test_PR_from_name_matches_manual_construction,
        test_SRK_from_name,
        test_cubic_from_name_family_dispatch,
        test_cubic_from_name_unknown_family,
        # Mixtures
        test_mixture_from_names_5component_natural_gas,
        test_mixture_from_names_matches_manual,
        test_mixture_from_names_default_composition,
        # Kwargs / integration
        test_kwargs_passthrough,
        test_name_built_mixture_uv_flash,
        # Meta
        test_chemicals_availability_flag,
        test_source_labeling,
    ]
    for t in tests:
        run_test(t)
    print(f"\n{'='*60}")
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    if not chemicals_available():
        print("NOTE: chemicals library NOT installed; tests used fallback table.")
        print("      Install with `pip install chemicals` for full databank access.")
    if FAILURES:
        print("\nFailures:")
        for name, detail in FAILURES:
            print(f"  - {name}: {detail}")
    print('='*60)
    sys.exit(0 if FAILED == 0 else 1)
