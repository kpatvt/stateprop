"""UV and TV flash tests (v0.7.0).

The UV flash is the natural-variable flash used in dynamic / transient
simulation. Given internal energy u and molar volume v, solve for
temperature T and phase distribution. The TV flash is its single-temperature
companion: given T and v, find p and phase distribution.

This file exercises both the Helmholtz-mixture (stateprop.mixture.flash)
and cubic-mixture (stateprop.cubic.flash) implementations.

Test strategy:
    1. Round-trip: take a known (T, p) -> flash_pt -> (u, v)
       -> flash_uv -> (T', p'), check T' == T and p' == p.
    2. Run across single-phase (vapor, liquid, supercritical) and
       two-phase regions.
    3. Verify the pure-fluid limit: a composition-[1.0] mixture should
       round-trip identically.
    4. Verify internal consistency at the returned state:
       u(result) == u_target (checks we didn't converge to wrong root).
"""
import sys
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

# Helmholtz path
from stateprop.mixture.mixture import load_mixture
from stateprop.mixture.flash import (
    flash_pt as mix_flash_pt,
    flash_tv as mix_flash_tv,
    flash_uv as mix_flash_uv,
)

# Cubic path
from stateprop.cubic import CubicMixture, PR, SRK
from stateprop.cubic import flash_pt as cubic_flash_pt
from stateprop.cubic.flash import (
    flash_tv as cubic_flash_tv,
    flash_uv as cubic_flash_uv,
)


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
# Test fixtures
# ---------------------------------------------------------------------------

def ng_helmholtz_mixture():
    """5-component GERG-2008 natural-gas mixture."""
    return load_mixture(
        ["gerg2008/methane", "gerg2008/ethane", "gerg2008/propane",
         "gerg2008/nitrogen", "gerg2008/carbondioxide"],
        [0.85, 0.05, 0.02, 0.05, 0.03],
        binary_set="gerg2008",
    )


def ng_cubic_mixture(eos_factory=PR):
    """5-component cubic (PR default) natural-gas mixture."""
    c_CH4  = eos_factory(190.564, 4.5992e6, 0.01142)
    c_C2H6 = eos_factory(305.322, 4.8722e6, 0.0995)
    c_C3H8 = eos_factory(369.825, 4.2472e6, 0.1521)
    c_N2   = eos_factory(126.192, 3.3958e6, 0.0372)
    c_CO2  = eos_factory(304.128, 7.3773e6, 0.22394)
    return CubicMixture(
        [c_CH4, c_C2H6, c_C3H8, c_N2, c_CO2],
        composition=[0.85, 0.05, 0.02, 0.05, 0.03],
        k_ij={(0, 3): 0.025, (0, 4): 0.09, (3, 4): -0.017},
    )


TEST_Z = np.array([0.85, 0.05, 0.02, 0.05, 0.03])


# ---------------------------------------------------------------------------
# Helmholtz-mixture UV tests
# ---------------------------------------------------------------------------

def test_helmholtz_tv_round_trip_single_phase():
    """flash_tv recovers pressure at known (T, rho) for single-phase states."""
    mix = ng_helmholtz_mixture()
    cases = [
        (300.0, 1e5,  "ambient"),
        (300.0, 5e6,  "compressed gas"),
        (350.0, 5e6,  "warm compressed"),
        (250.0, 1e5,  "cold vapor"),
        (400.0, 10e6, "hot compressed"),
    ]
    for T, p, name in cases:
        r = mix_flash_pt(p, T, TEST_Z, mix, tol=1e-10)
        v_target = 1.0 / r.rho
        r_tv = mix_flash_tv(T, v_target, TEST_Z, mix)
        rel_p = abs(r_tv.p - p) / p
        check(f"Helmholtz TV {name} (T={T}K, p={p:.0e}Pa): "
              f"recovered p={r_tv.p:.4e}, rel err {rel_p:.2e}",
              rel_p < 1e-5, f"rel err {rel_p:.2e}")


def test_helmholtz_uv_round_trip_single_phase():
    """flash_uv recovers temperature at known (u, v) for single-phase states."""
    mix = ng_helmholtz_mixture()
    cases = [
        (300.0, 1e5,  "ambient gas"),
        (300.0, 5e6,  "compressed gas"),
        (350.0, 5e6,  "warm compressed"),
        (200.0, 5e6,  "cold compressed liquid"),
        (250.0, 1e5,  "cold vapor"),
        (400.0, 10e6, "hot compressed"),
    ]
    for T_true, p_true, name in cases:
        r = mix_flash_pt(p_true, T_true, TEST_Z, mix, tol=1e-10)
        u_target = r.h - r.p / r.rho
        v_target = 1.0 / r.rho
        r_uv = mix_flash_uv(u_target, v_target, TEST_Z, mix)
        rel_T = abs(r_uv.T - T_true) / T_true
        rel_p = abs(r_uv.p - p_true) / p_true
        check(f"Helmholtz UV {name}: T recovered ({r_uv.T:.3f} vs {T_true}, "
              f"rel {rel_T:.1e})",
              rel_T < 1e-4, f"T rel err {rel_T:.2e}")
        check(f"Helmholtz UV {name}: p recovered ({r_uv.p:.4e} vs {p_true:.0e}, "
              f"rel {rel_p:.1e})",
              rel_p < 1e-4, f"p rel err {rel_p:.2e}")


def test_helmholtz_uv_identity_at_solution():
    """The returned MixtureFlashResult must satisfy u(result) = u_target."""
    mix = ng_helmholtz_mixture()
    for T, p in [(300.0, 1e5), (300.0, 5e6), (350.0, 5e6)]:
        r = mix_flash_pt(p, T, TEST_Z, mix, tol=1e-10)
        u_target = r.h - r.p / r.rho
        v_target = 1.0 / r.rho
        r_uv = mix_flash_uv(u_target, v_target, TEST_Z, mix)
        u_got = r_uv.h - r_uv.p / r_uv.rho
        rel_u = abs(u_got - u_target) / max(1.0, abs(u_target))
        v_got = 1.0 / r_uv.rho
        rel_v = abs(v_got - v_target) / v_target
        check(f"Helmholtz UV identity u at T={T}, p={p:.0e}: "
              f"u_got={u_got:.3f} vs u_target={u_target:.3f}, rel {rel_u:.1e}",
              rel_u < 1e-4, f"rel err {rel_u:.2e}")
        check(f"Helmholtz UV identity v at T={T}, p={p:.0e}: "
              f"rel {rel_v:.1e}",
              rel_v < 1e-4, f"rel err {rel_v:.2e}")


def test_helmholtz_uv_composition_sensitivity():
    """UV flash for three different compositions of same components."""
    for z in [
        [0.99, 0.005, 0.002, 0.002, 0.001],    # nearly pure methane
        [0.5, 0.2, 0.1, 0.1, 0.1],              # rich in heavies
        [0.3, 0.1, 0.0, 0.3, 0.3],              # N2/CO2 heavy
    ]:
        z_arr = np.array(z)
        z_arr = z_arr / z_arr.sum()
        mix = load_mixture(
            ["gerg2008/methane", "gerg2008/ethane", "gerg2008/propane",
             "gerg2008/nitrogen", "gerg2008/carbondioxide"],
            z_arr.tolist(), binary_set="gerg2008",
        )
        T, p = 300.0, 2e6
        r = mix_flash_pt(p, T, z_arr, mix, tol=1e-10)
        u_target = r.h - r.p / r.rho
        v_target = 1.0 / r.rho
        r_uv = mix_flash_uv(u_target, v_target, z_arr, mix)
        rel_T = abs(r_uv.T - T) / T
        check(f"Helmholtz UV composition z=[{z[0]:.2f},{z[1]:.2f},...]: "
              f"T recovered rel {rel_T:.1e}",
              rel_T < 1e-4, f"rel err {rel_T:.2e}")


# ---------------------------------------------------------------------------
# Cubic-mixture UV tests
# ---------------------------------------------------------------------------

def test_cubic_PR_uv_round_trip_single_phase():
    """Peng-Robinson UV round-trip at 5 single-phase states."""
    mix = ng_cubic_mixture(PR)
    cases = [
        (300.0, 1e5,  "ambient gas"),
        (300.0, 5e6,  "compressed gas"),
        (350.0, 5e6,  "warm compressed"),
        (250.0, 1e5,  "cold vapor"),
        (400.0, 10e6, "hot compressed"),
    ]
    for T_true, p_true, name in cases:
        r = cubic_flash_pt(p_true, T_true, TEST_Z, mix, tol=1e-9)
        u_target = r.h - r.p / r.rho
        v_target = 1.0 / r.rho
        r_uv = cubic_flash_uv(u_target, v_target, TEST_Z, mix)
        rel_T = abs(r_uv.T - T_true) / T_true
        rel_p = abs(r_uv.p - p_true) / p_true
        check(f"PR UV {name}: T rel {rel_T:.1e}, p rel {rel_p:.1e}",
              rel_T < 1e-4 and rel_p < 1e-4,
              f"T rel {rel_T:.2e}, p rel {rel_p:.2e}")


def test_cubic_SRK_uv_round_trip():
    """Soave-Redlich-Kwong UV round-trip (different EOS, same interface)."""
    mix = ng_cubic_mixture(SRK)
    cases = [
        (300.0, 1e5,  "ambient gas"),
        (350.0, 5e6,  "warm compressed"),
        (400.0, 10e6, "hot compressed"),
    ]
    for T_true, p_true, name in cases:
        r = cubic_flash_pt(p_true, T_true, TEST_Z, mix, tol=1e-9)
        u_target = r.h - r.p / r.rho
        v_target = 1.0 / r.rho
        r_uv = cubic_flash_uv(u_target, v_target, TEST_Z, mix)
        rel_T = abs(r_uv.T - T_true) / T_true
        rel_p = abs(r_uv.p - p_true) / p_true
        check(f"SRK UV {name}: T rel {rel_T:.1e}, p rel {rel_p:.1e}",
              rel_T < 1e-4 and rel_p < 1e-4,
              f"T rel {rel_T:.2e}, p rel {rel_p:.2e}")


def test_cubic_PR_uv_two_phase():
    """PR UV flash lands correctly in a two-phase region.

    At T=200K, p=5MPa the PR mix is two-phase. The UV flash should
    (a) converge, (b) recover T and p to <1e-3, and (c) return a
    two-phase result with sensible beta, x, y."""
    mix = ng_cubic_mixture(PR)
    T_true, p_true = 200.0, 5e6
    r = cubic_flash_pt(p_true, T_true, TEST_Z, mix, tol=1e-9)
    check(f"ground-truth flash_pt at T={T_true}K, p={p_true:.0e}Pa "
          f"returns two_phase (got {r.phase})",
          r.phase == "two_phase")

    u_target = r.h - r.p / r.rho
    v_target = 1.0 / r.rho
    r_uv = cubic_flash_uv(u_target, v_target, TEST_Z, mix)
    rel_T = abs(r_uv.T - T_true) / T_true
    rel_p = abs(r_uv.p - p_true) / p_true
    check(f"PR UV two-phase T recovered ({r_uv.T:.3f} vs {T_true}, rel {rel_T:.1e})",
          rel_T < 1e-4, f"rel {rel_T:.2e}")
    check(f"PR UV two-phase p recovered ({r_uv.p:.4e} vs {p_true:.0e}, rel {rel_p:.1e})",
          rel_p < 1e-4, f"rel {rel_p:.2e}")
    check(f"PR UV two-phase returns two_phase (got {r_uv.phase})",
          r_uv.phase == "two_phase")
    if r_uv.phase == "two_phase":
        check(f"  beta in [0,1] (got {r_uv.beta:.4f})",
              0.0 < r_uv.beta < 1.0)
        check(f"  rho_V < rho_L (got V={r_uv.rho_V:.2f}, L={r_uv.rho_L:.2f})",
              r_uv.rho_V < r_uv.rho_L)


# ---------------------------------------------------------------------------
# Dynamic-simulation scenario: sudden volume change followed by UV solve
# ---------------------------------------------------------------------------

def test_uv_dynamic_scenario_compression():
    """Simulate an adiabatic compression:
        1. Start at known (T0, p0), compute u0, v0.
        2. Adiabatically reduce v by 50% (v1 = v0/2), conserving u.
        3. UV flash with (u0, v1) should give T > T0 (temperature rises
           under compression) and p > p0.
    This is not a strict regression test for a specific answer but a
    physics sanity check for the dynamic-simulation use case."""
    mix = ng_helmholtz_mixture()
    T0, p0 = 300.0, 1e5
    r0 = mix_flash_pt(p0, T0, TEST_Z, mix, tol=1e-10)
    u0 = r0.h - r0.p / r0.rho
    v0 = 1.0 / r0.rho
    v1 = 0.5 * v0    # halve molar volume

    r1 = mix_flash_uv(u0, v1, TEST_Z, mix, T_init=T0)
    check(f"compression T1 > T0 (T0={T0}, T1={r1.T:.2f})", r1.T > T0)
    check(f"compression p1 > p0 (p0={p0:.0e}, p1={r1.p:.3e})", r1.p > p0)
    u1 = r1.h - r1.p / r1.rho
    rel_u = abs(u1 - u0) / max(1.0, abs(u0))
    check(f"compression preserves u (u0={u0:.2f}, u1={u1:.2f}, rel {rel_u:.1e})",
          rel_u < 1e-4, f"rel {rel_u:.2e}")


# ---------------------------------------------------------------------------
# Pure-component limit (composition=[1.0]) should work for both backends
# ---------------------------------------------------------------------------

def test_uv_pure_component_limit_helmholtz():
    """A 1-component 'mixture' should UV-round-trip just as well as multi."""
    mix = load_mixture(["gerg2008/methane"], [1.0], binary_set=None)
    z = np.array([1.0])
    T_true, p_true = 300.0, 5e6
    r = mix_flash_pt(p_true, T_true, z, mix, tol=1e-10)
    u_target = r.h - r.p / r.rho
    v_target = 1.0 / r.rho
    r_uv = mix_flash_uv(u_target, v_target, z, mix)
    rel_T = abs(r_uv.T - T_true) / T_true
    rel_p = abs(r_uv.p - p_true) / p_true
    check(f"pure CH4 UV: T rel {rel_T:.1e}, p rel {rel_p:.1e}",
          rel_T < 1e-4 and rel_p < 1e-4,
          f"T rel {rel_T:.2e}, p rel {rel_p:.2e}")


def test_uv_pure_component_limit_cubic():
    """PR single-component UV round-trip."""
    c_CH4 = PR(190.564, 4.5992e6, 0.01142)
    mix = CubicMixture([c_CH4], composition=[1.0])
    z = np.array([1.0])
    T_true, p_true = 300.0, 5e6
    r = cubic_flash_pt(p_true, T_true, z, mix, tol=1e-9)
    u_target = r.h - r.p / r.rho
    v_target = 1.0 / r.rho
    r_uv = cubic_flash_uv(u_target, v_target, z, mix)
    rel_T = abs(r_uv.T - T_true) / T_true
    rel_p = abs(r_uv.p - p_true) / p_true
    check(f"pure CH4 PR UV: T rel {rel_T:.1e}, p rel {rel_p:.1e}",
          rel_T < 1e-4 and rel_p < 1e-4,
          f"T rel {rel_T:.2e}, p rel {rel_p:.2e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Helmholtz
        test_helmholtz_tv_round_trip_single_phase,
        test_helmholtz_uv_round_trip_single_phase,
        test_helmholtz_uv_identity_at_solution,
        test_helmholtz_uv_composition_sensitivity,
        # Cubic
        test_cubic_PR_uv_round_trip_single_phase,
        test_cubic_SRK_uv_round_trip,
        test_cubic_PR_uv_two_phase,
        # Dynamic scenario
        test_uv_dynamic_scenario_compression,
        # Pure-component limit
        test_uv_pure_component_limit_helmholtz,
        test_uv_pure_component_limit_cubic,
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
