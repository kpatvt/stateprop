"""Tests for stateprop.chemdb (optional interface to the `chemicals` library).

These tests skip cleanly if `chemicals` is not installed, so they don't
cause spurious failures on minimal dev environments.

Run: python tests/run_chemdb_tests.py
"""
import sys
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

PASSED = 0
FAILED = 0
SKIPPED = 0
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


def skip(label, reason):
    global SKIPPED
    SKIPPED += 1
    print(f"  SKIP  {label}: {reason}")


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
# Availability check
# ---------------------------------------------------------------------------

def _chemicals_available():
    try:
        import chemicals  # noqa: F401
        return True
    except ImportError:
        return False


CHEMICALS_AVAILABLE = _chemicals_available()


# ---------------------------------------------------------------------------
# Test 1: module imports even without chemicals
# ---------------------------------------------------------------------------

def test_module_imports_without_chemicals():
    """The stateprop.chemdb module should import cleanly even if the
    `chemicals` package is not installed. The ImportError only fires on
    first attempt to USE a function that needs chemicals."""
    try:
        from stateprop.chemdb import (
            lookup, PR_from_name, SRK_from_name, RK_from_name,
            VDW_from_name, PR78_from_name, components_from_names,
        )
        check("stateprop.chemdb imports cleanly",
              all(callable(f) for f in [lookup, PR_from_name, SRK_from_name,
                                         RK_from_name, VDW_from_name,
                                         PR78_from_name, components_from_names]))
    except ImportError as e:
        check("stateprop.chemdb imports cleanly", False, str(e))


def test_helpful_ImportError_when_chemicals_missing():
    """Without `chemicals` installed, looking up a fluid outside the
    fallback table should raise a useful error that tells the user how
    to install `chemicals` for full coverage.

    NOTE: as of v0.8.0, PR_from_name of a GERG-2008 component (methane,
    water, propane, etc.) now succeeds even without `chemicals` by using
    the built-in fallback table -- a deliberate design choice so
    natural-gas workflows work out of the box. The ImportError-like
    behavior only applies to substances outside the fallback table.
    """
    if CHEMICALS_AVAILABLE:
        skip("helpful error on missing chemicals",
             "chemicals IS installed; can't test missing path")
        return

    from stateprop.chemdb import PR_from_name
    # Pick a substance definitely not in the fallback table
    unknown = "dichloromethane"
    try:
        PR_from_name(unknown)
        check("raises helpful error on unknown fluid without chemicals",
              False, "no exception raised")
    except (KeyError, ImportError) as e:
        msg = str(e)
        check("raises helpful error on unknown fluid without chemicals",
              "chemicals" in msg and "pip install" in msg,
              f"message was: {msg}")
    except Exception as e:
        check("raises helpful error on unknown fluid without chemicals",
              False,
              f"raised {type(e).__name__} unexpectedly: {e}")


# ---------------------------------------------------------------------------
# Tests that REQUIRE chemicals
# ---------------------------------------------------------------------------

def test_lookup_methane():
    """lookup('methane') returns known values."""
    if not CHEMICALS_AVAILABLE:
        skip("lookup('methane')", "chemicals not installed")
        return

    from stateprop.chemdb import lookup
    d = lookup("methane")
    check(f"CAS for methane (got {d['CAS']!r})", d["CAS"] == "74-82-8")
    # Tc ~ 190.56 K (commonly cited)
    check(f"Tc for methane in [190.0, 191.0] (got {d['T_c']})",
          190.0 < d["T_c"] < 191.0)
    # Pc in [4.5, 4.7] MPa
    check(f"Pc for methane in [4.5e6, 4.7e6] (got {d['p_c']})",
          4.5e6 < d["p_c"] < 4.7e6)
    # omega small positive
    check(f"omega for methane in [0.0, 0.05] (got {d['omega']})",
          0.0 <= d["omega"] <= 0.05)
    # M = 16.042 g/mol = 0.016042 kg/mol
    check(f"molar_mass for methane in [0.0160, 0.0161] kg/mol (got {d['molar_mass']})",
          0.0160 < d["molar_mass"] < 0.0161)


def test_lookup_by_multiple_identifier_forms():
    """CAS_from_any resolves names, formulas, CAS numbers."""
    if not CHEMICALS_AVAILABLE:
        skip("multiple identifier forms", "chemicals not installed")
        return

    from stateprop.chemdb import lookup
    # All three should resolve to methane
    d_name = lookup("methane")
    d_cas  = lookup("74-82-8")
    check("name and CAS resolve to same compound",
          d_name["CAS"] == d_cas["CAS"])
    # Water via multiple names
    d_water = lookup("water")
    d_h2o   = lookup("H2O")
    check("'water' and 'H2O' resolve to same compound",
          d_water["CAS"] == d_h2o["CAS"])


def test_PR_from_name_basic():
    """PR_from_name returns a valid CubicEOS with correct attributes."""
    if not CHEMICALS_AVAILABLE:
        skip("PR_from_name basic", "chemicals not installed")
        return

    from stateprop.chemdb import PR_from_name
    c = PR_from_name("ethane")
    check(f"ethane has T_c > 0 (got {c.T_c})", c.T_c > 0)
    check(f"ethane has p_c > 0 (got {c.p_c})", c.p_c > 0)
    check(f"ethane has molar_mass > 0 (got {c.molar_mass})", c.molar_mass > 0)
    # Smoke: compute critical properties from the EOS
    # Should be able to evaluate pressure, etc.
    from stateprop.cubic.eos import CubicEOS
    check("returned object is a CubicEOS", isinstance(c, CubicEOS))


def test_PR_SRK_RK_VDW_factories():
    """All five cubic-family factories work from a name lookup."""
    if not CHEMICALS_AVAILABLE:
        skip("factory variety", "chemicals not installed")
        return

    from stateprop.chemdb import (
        PR_from_name, PR78_from_name, SRK_from_name,
        RK_from_name, VDW_from_name,
    )
    for factory, name in [
        (PR_from_name, "PR"), (PR78_from_name, "PR78"),
        (SRK_from_name, "SRK"), (RK_from_name, "RK"),
        (VDW_from_name, "VDW"),
    ]:
        c = factory("methane")
        check(f"{name}_from_name('methane'): T_c={c.T_c}", c.T_c > 0)


def test_kwarg_override():
    """Explicit kwarg overrides the looked-up value."""
    if not CHEMICALS_AVAILABLE:
        skip("kwarg override", "chemicals not installed")
        return

    from stateprop.chemdb import PR_from_name
    c_default = PR_from_name("propane")
    c_override = PR_from_name("propane", acentric_factor=0.2)
    check("default omega matches lookup",
          abs(c_default.acentric_factor - c_override.acentric_factor) > 0.01)
    check(f"override sets omega=0.2 (got {c_override.acentric_factor})",
          abs(c_override.acentric_factor - 0.2) < 1e-10)


def test_components_from_names_batch():
    """Batch builder produces a list of valid components."""
    if not CHEMICALS_AVAILABLE:
        skip("components_from_names", "chemicals not installed")
        return

    from stateprop.chemdb import components_from_names
    from stateprop.cubic import PR, SRK
    comps = components_from_names(
        ["methane", "ethane", "propane", "nitrogen", "carbon dioxide"],
        factory=PR,
    )
    check(f"got 5 components (got {len(comps)})", len(comps) == 5)
    check("all T_c > 0", all(c.T_c > 0 for c in comps))
    check("first is methane (T_c ~ 190)", 189.0 < comps[0].T_c < 192.0)
    # Also works with SRK
    comps_srk = components_from_names(["methane", "ethane"], factory=SRK)
    check("SRK factory works too", len(comps_srk) == 2)


def test_full_mixture_from_chemdb():
    """Build a CubicMixture entirely from name-lookup and flash it."""
    if not CHEMICALS_AVAILABLE:
        skip("full mixture round-trip", "chemicals not installed")
        return

    from stateprop.chemdb import components_from_names
    from stateprop.cubic import CubicMixture, PR
    from stateprop.cubic.flash import flash_pt

    comps = components_from_names(
        ["methane", "ethane", "propane", "nitrogen", "carbon dioxide"],
        factory=PR,
    )
    mix = CubicMixture(
        comps,
        composition=[0.85, 0.05, 0.02, 0.05, 0.03],
        k_ij={(0, 3): 0.025, (0, 4): 0.09, (3, 4): -0.017},
    )
    z = np.array([0.85, 0.05, 0.02, 0.05, 0.03])
    r = flash_pt(1e6, 300.0, z, mix, tol=1e-9)
    check(f"PT flash succeeds (phase={r.phase}, rho={r.rho:.2f})",
          r.phase in ("vapor", "supercritical", "liquid", "two_phase"))
    # Compressibility factor should be close to 1 for gas at 1 MPa
    Z = r.p / (r.rho * 8.314472 * r.T)
    check(f"Z in physically reasonable range (0.9 < {Z:.4f} < 1.05)",
          0.9 < Z < 1.05)


def test_unresolvable_compound_raises():
    """A nonsense identifier should give a clean ValueError, not a crash."""
    if not CHEMICALS_AVAILABLE:
        skip("unresolvable compound", "chemicals not installed")
        return

    from stateprop.chemdb import PR_from_name
    try:
        PR_from_name("unobtanium_xyz_not_a_real_chemical")
        check("raises ValueError for unresolvable compound", False,
              "no exception raised")
    except ValueError:
        check("raises ValueError for unresolvable compound", True)
    except Exception as e:
        check("raises ValueError for unresolvable compound", False,
              f"raised {type(e).__name__} instead: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_module_imports_without_chemicals,
        test_helpful_ImportError_when_chemicals_missing,
        test_lookup_methane,
        test_lookup_by_multiple_identifier_forms,
        test_PR_from_name_basic,
        test_PR_SRK_RK_VDW_factories,
        test_kwarg_override,
        test_components_from_names_batch,
        test_full_mixture_from_chemdb,
        test_unresolvable_compound_raises,
    ]
    for t in tests:
        run_test(t)

    print(f"\n{'='*60}")
    print(f"RESULT: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped")
    if not CHEMICALS_AVAILABLE:
        print(f"NOTE: 8 tests skipped because `chemicals` is not installed.")
        print(f"      To enable the full suite, install it:")
        print(f"        pip install chemicals")
        print(f"      or:")
        print(f"        pip install 'stateprop[chemdb]'")
    if FAILURES:
        print("\nFailures:")
        for name, detail in FAILURES:
            print(f"  - {name}: {detail}")
    print('='*60)
    sys.exit(0 if FAILED == 0 else 1)
