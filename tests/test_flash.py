"""
Tests for flash algorithms and phase envelope tracing.

The core correctness criterion is *round-trip consistency*: if we pick any
thermodynamic state point, compute some (p, h) or (p, s) there, flash back
with flash_ph or flash_ps, we should recover the original state.

For pure fluids, the saturation curve is 1-D and we also check that the
phase envelope generator returns a monotonically-increasing T and p along
its grid, converges to the critical point at the top, and produces a dome
that closes cleanly in (T, s) coordinates.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

import stateprop as he


def _check(label, got, want, rtol=1e-5, atol=1e-10):
    diff = abs(got - want)
    rel = diff / (abs(want) + atol)
    ok = diff <= atol or rel <= rtol
    status = "PASS" if ok else "FAIL"
    print(f"    [{status}] {label:40s} got={got: .6e}  want={want: .6e}  rel={rel: .2e}")
    return ok


# ---------------------------------------------------------------------------
# Flash PT
# ---------------------------------------------------------------------------

def test_flash_pt():
    """PT flash: pick a range of states, verify density and phase classification."""
    fl = he.load_fluid("water")
    ok = True
    print("Water PT flashes:")

    # Subcooled liquid: 300 K, 1 MPa
    r = he.flash_pt(1e6, 300.0, fl)
    print(f"  300 K, 1 MPa:   phase={r.phase}, rho={r.rho*fl.molar_mass:.2f} kg/m^3, "
          f"h={r.h/fl.molar_mass*1e-3:.3f} kJ/kg")
    ok &= r.phase == "liquid"
    ok &= _check("300K 1MPa rho_kg", r.rho * fl.molar_mass, 996.77, rtol=2e-3)

    # Superheated vapor: 500 K, 0.1 MPa
    r = he.flash_pt(0.1e6, 500.0, fl)
    print(f"  500 K, 0.1 MPa: phase={r.phase}, rho={r.rho*fl.molar_mass:.4f} kg/m^3")
    ok &= r.phase == "vapor"

    # Supercritical: 700 K, 30 MPa
    r = he.flash_pt(30e6, 700.0, fl)
    print(f"  700 K, 30 MPa:  phase={r.phase}, rho={r.rho*fl.molar_mass:.2f} kg/m^3")
    ok &= r.phase == "supercritical"

    return ok


# ---------------------------------------------------------------------------
# Flash PH: round trip from (rho, T) -> (p, h) -> T, rho, etc.
# ---------------------------------------------------------------------------

def test_flash_ph_roundtrip():
    """For each state point, compute p, h, then flash_ph(p, h) and compare."""
    fl = he.load_fluid("water")
    ok = True
    print("PH flash round-trip tests (water):")

    # (T, p_MPa, expected phase)
    test_points = [
        (300.0, 1.0,   "liquid"),
        (300.0, 100.0, "liquid"),
        (500.0, 0.1,   "vapor"),
        (500.0, 5.0,   "liquid"),    # At 5 MPa, water saturates at ~537 K; 500 K < T_sat -> liquid
        (400.0, 0.1,   "vapor"),
        (700.0, 30.0,  "supercritical"),
        (900.0, 100.0, "supercritical"),
    ]

    for T, p_MPa, expected in test_points:
        p = p_MPa * 1e6
        # Forward: PT -> h
        fwd = he.flash_pt(p, T, fl)
        # Back: PH -> T', phase
        back = he.flash_ph(p, fwd.h, fl)
        T_err = abs(back.T - T) / T
        print(f"  T={T:.1f}, p={p_MPa:g} MPa: fwd phase={fwd.phase}, "
              f"h={fwd.h/fl.molar_mass*1e-3:.3f} kJ/kg; "
              f"back T={back.T:.4f}, phase={back.phase}, rel_err={T_err:.2e}")
        ok &= fwd.phase == expected
        ok &= back.phase == expected
        ok &= _check(f"T round-trip @ T={T},p={p_MPa}MPa", back.T, T, rtol=1e-6)

    return ok


def test_flash_ph_two_phase():
    """Inside the dome, flash_ph should return a two-phase result with the
    correct quality."""
    fl = he.load_fluid("water")
    ok = True
    print("PH flash in two-phase region (water):")

    # At T_sat = 450 K, p_sat = 0.9322 MPa (IAPWS-95 Table 8)
    T_sat = 450.0
    rho_L, rho_V, p_sat = he.saturation_pT(T_sat, fl)
    h_L = he.enthalpy(rho_L, T_sat, fl)
    h_V = he.enthalpy(rho_V, T_sat, fl)

    # Choose a target at quality = 0.3
    x_target = 0.3
    h_target = (1.0 - x_target) * h_L + x_target * h_V

    r = he.flash_ph(p_sat, h_target, fl)
    print(f"  p_sat={p_sat*1e-6:.4f} MPa, x_target={x_target}, h_target={h_target/fl.molar_mass*1e-3:.3f} kJ/kg")
    print(f"    -> phase={r.phase}, T={r.T:.4f}, quality={r.quality:.6f}")
    ok &= r.phase == "two_phase"
    ok &= _check("quality", r.quality, x_target, rtol=1e-8)
    ok &= _check("T (should equal T_sat)", r.T, T_sat, rtol=1e-6)

    return ok


# ---------------------------------------------------------------------------
# Flash PS
# ---------------------------------------------------------------------------

def test_flash_ps_roundtrip():
    """PS flash round-trip tests."""
    fl = he.load_fluid("water")
    ok = True
    print("PS flash round-trip tests (water):")

    test_points = [
        (300.0, 1.0,   "liquid"),
        (500.0, 0.1,   "vapor"),
        (700.0, 30.0,  "supercritical"),
    ]

    for T, p_MPa, expected in test_points:
        p = p_MPa * 1e6
        fwd = he.flash_pt(p, T, fl)
        back = he.flash_ps(p, fwd.s, fl)
        print(f"  T={T:.1f}, p={p_MPa:g} MPa: s={fwd.s/fl.molar_mass*1e-3:.5f} kJ/kg-K; "
              f"back T={back.T:.4f}, phase={back.phase}")
        ok &= back.phase == expected
        ok &= _check(f"T round-trip @ T={T},p={p_MPa}MPa", back.T, T, rtol=1e-6)

    return ok


def test_isentropic_process():
    """A practical application: isentropic expansion from a superheated state."""
    fl = he.load_fluid("water")
    print("Isentropic expansion of steam (practical turbine-stage example):")
    # Inlet: 700 K, 10 MPa
    inlet = he.flash_pt(10e6, 700.0, fl)
    print(f"  Inlet:  T={inlet.T:.1f} K, p=10 MPa, h={inlet.h/fl.molar_mass*1e-3:.2f} kJ/kg, "
          f"s={inlet.s/fl.molar_mass*1e-3:.5f} kJ/kg-K")
    # Exit: isentropic expansion to 0.01 MPa
    exit = he.flash_ps(0.01e6, inlet.s, fl)
    h_drop = (inlet.h - exit.h) / fl.molar_mass * 1e-3
    print(f"  Outlet: phase={exit.phase}, T={exit.T:.2f} K, "
          f"h={exit.h/fl.molar_mass*1e-3:.2f} kJ/kg"
          + (f", x={exit.quality:.4f}" if exit.quality is not None else ""))
    print(f"  Specific work = h_inlet - h_outlet_s = {h_drop:.2f} kJ/kg")

    ok = True
    ok &= exit.phase == "two_phase"     # Low-p steam turbine exit is wet
    # Sanity: specific work should be in the right ballpark. For steam expansion
    # from ~700 K, 10 MPa to 0.01 MPa, isentropic work is ~1170 kJ/kg.
    ok &= 900.0 < h_drop < 1400.0
    print(f"    [{'PASS' if ok else 'FAIL'}] exit wet and specific work in expected range")
    return ok


# ---------------------------------------------------------------------------
# Flash TV and UV
# ---------------------------------------------------------------------------

def test_flash_tv():
    """TV flash: trivial single phase, and lever-rule check in two-phase."""
    fl = he.load_fluid("carbondioxide")
    ok = True
    print("TV flash tests (CO2):")

    # Single phase -- T=350 K, v = 1e-4 m^3/mol (dense)
    T = 350.0
    v = 1.0e-4
    r = he.flash_tv(T, v, fl)
    print(f"  T={T}, v={v}: phase={r.phase}, p={r.p*1e-6:.4f} MPa, rho={r.rho:.2f} mol/m^3")
    ok &= r.phase == "supercritical"
    ok &= _check("rho = 1/v", r.rho, 1.0 / v, rtol=1e-12)

    # Two-phase: T=270 K inside the CO2 dome
    T = 270.0
    rho_L, rho_V, p_sat = he.saturation_pT(T, fl)
    v_mid = 0.5 * (1.0 / rho_L + 1.0 / rho_V)   # midway (by volume) between L and V
    r = he.flash_tv(T, v_mid, fl)
    print(f"  T={T}, v midway: phase={r.phase}, quality={r.quality:.4f}")
    ok &= r.phase == "two_phase"
    # Quality should be whatever it is for v_mid between v_L and v_V
    x_expected = (v_mid - 1.0 / rho_L) / (1.0 / rho_V - 1.0 / rho_L)
    ok &= _check("quality from lever rule", r.quality, x_expected, rtol=1e-12)
    return ok


def test_flash_uv():
    """UV flash: round-trip consistency."""
    fl = he.load_fluid("carbondioxide")
    print("UV flash round-trip tests (CO2):")
    ok = True

    # Pick several states, compute (u, v), flash back.
    test_points = [
        (350.0, 3000.0),    # supercritical moderate density
        (310.0, 500.0),     # supercritical gas-like
        (400.0, 10000.0),   # supercritical dense
    ]
    for T, rho in test_points:
        u = he.internal_energy(rho, T, fl)
        v = 1.0 / rho
        r = he.flash_uv(u, v, fl, T_init=T * 1.2)  # start far from T
        print(f"  T={T}, rho={rho}: u={u:.2f}, v={v:.4e} -> T_back={r.T:.4f}")
        ok &= _check(f"UV T round-trip T={T},rho={rho}", r.T, T, rtol=1e-4)

    return ok


# ---------------------------------------------------------------------------
# Phase envelope
# ---------------------------------------------------------------------------

def test_phase_envelope_co2():
    """Trace CO2 phase envelope; check structure and critical-point closure."""
    fl = he.load_fluid("carbondioxide")
    env = he.trace_phase_envelope(fl, n_points=40)
    ok = True

    print(f"CO2 phase envelope ({len(env.T)} points):")
    print(f"  T range: {env.T.min():.2f} -- {env.T.max():.2f} K "
          f"(T_c = {fl.T_c:.2f})")
    print(f"  p range: {env.p.min()*1e-6:.4f} -- {env.p.max()*1e-6:.4f} MPa "
          f"(p_c = {fl.p_c*1e-6:.4f})")

    # Structural checks
    ok &= len(env.T) > 20
    # T should be sorted increasing
    ok &= np.all(np.diff(env.T) >= 0)
    # p should be sorted increasing (monotone on the sat curve)
    ok &= np.all(np.diff(env.p) >= 0)
    # Top of the dome should be the critical point
    ok &= _check("T_top = T_c", env.T[-1], fl.T_c, rtol=1e-4)
    ok &= _check("p_top = p_c", env.p[-1], fl.p_c, rtol=1e-3)
    # rho_L should decrease as T rises, rho_V should increase
    ok &= np.all(np.diff(env.rho_L) <= 1e-6)   # non-increasing
    ok &= np.all(np.diff(env.rho_V) >= -1e-6)  # non-decreasing
    # At the top they meet at rho_c
    ok &= _check("rho_L top = rho_c", env.rho_L[-1], fl.rho_c, rtol=1e-6)
    ok &= _check("rho_V top = rho_c", env.rho_V[-1], fl.rho_c, rtol=1e-6)

    return ok


def test_phase_envelope_water():
    """Same for water, additionally check dome coordinates in mass-based T-s."""
    fl = he.load_fluid("water")
    env = he.trace_phase_envelope(fl, n_points=50)
    ok = True

    print(f"Water phase envelope ({len(env.T)} points):")
    mb = env.as_mass_based()
    print(f"  At low T (~275 K):  p_sat = {mb['p_MPa'][0]:.6f} MPa, "
          f"rho_L = {mb['rho_L'][0]:.3f} kg/m^3, h_vap = {mb['h_vap'][0]:.2f} kJ/kg")
    # Find the closest-to-critical point
    idx = -1 if env.T[-1] == fl.T_c else -1
    print(f"  Near T_c:           p_sat = {mb['p_MPa'][-2]:.6f} MPa, "
          f"rho_L = {mb['rho_L'][-2]:.3f}, rho_V = {mb['rho_V'][-2]:.3f} kg/m^3")

    # T-s dome: check that x, y form a closed curve with matching endpoints
    x, y = env.dome_coordinates("s_kg", "T")
    ok &= len(x) == 2 * len(env.T)
    # Left and right "feet" of the dome should be at the lowest T
    ok &= _check("T at start of dome", y[0],  env.T[0],  rtol=1e-12)
    ok &= _check("T at end   of dome", y[-1], env.T[0],  rtol=1e-12)

    return ok


# ---------------------------------------------------------------------------
# TH and TS flashes
# ---------------------------------------------------------------------------

def test_flash_ts_roundtrip():
    """TS flash: round-trip from (rho, T) -> (T, s) -> p, phase."""
    fl = he.load_fluid("water")
    ok = True
    print("TS flash round-trip tests (water):")

    # (T, p_MPa, expected phase)
    test_points = [
        (300.0,   1.0,  "liquid"),       # subcooled liquid, s < s_L_sat
        (300.0, 100.0,  "liquid"),       # compressed liquid
        (500.0,   0.1,  "vapor"),        # superheated vapor, s > s_V_sat
        (400.0,   0.1,  "vapor"),
        (700.0,  30.0,  "supercritical"),
        (900.0, 100.0,  "supercritical"),
    ]

    for T, p_MPa, expected in test_points:
        p = p_MPa * 1e6
        fwd = he.flash_pt(p, T, fl)
        back = he.flash_ts(T, fwd.s, fl)
        p_err = abs(back.p - p) / p
        print(f"  T={T:.1f}, p={p_MPa:g} MPa: fwd s={fwd.s/fl.molar_mass*1e-3:.5f} kJ/kg-K; "
              f"back p={back.p*1e-6:.5f} MPa, phase={back.phase}, rel_err={p_err:.2e}")
        ok &= back.phase == expected
        ok &= _check(f"p round-trip T={T},p={p_MPa}MPa", back.p, p, rtol=1e-6)

    return ok


def test_flash_ts_two_phase():
    """TS flash in the two-phase region: verify lever-rule quality."""
    fl = he.load_fluid("water")
    ok = True
    print("TS flash in two-phase region (water):")

    # At T=450 K (saturation pressure 0.932 MPa)
    T = 450.0
    rho_L, rho_V, p_sat = he.saturation_pT(T, fl)
    s_L = he.entropy(rho_L, T, fl)
    s_V = he.entropy(rho_V, T, fl)

    for x_target in [0.0, 0.1, 0.5, 0.9, 1.0]:
        s_target = (1.0 - x_target) * s_L + x_target * s_V
        r = he.flash_ts(T, s_target, fl)
        if x_target == 0.0 or x_target == 1.0:
            # Right at the boundary, either single-phase-saturated or two-phase are OK
            print(f"  x_target={x_target}: phase={r.phase}")
        else:
            print(f"  x_target={x_target}: phase={r.phase}, "
                  f"quality={r.quality:.6f}, p={r.p*1e-6:.5f} MPa")
            ok &= r.phase == "two_phase"
            ok &= _check(f"quality at x={x_target}", r.quality, x_target, rtol=1e-8)
            ok &= _check("p = p_sat", r.p, p_sat, rtol=1e-6)

    return ok


def test_flash_th_roundtrip():
    """TH flash: round-trip tests.

    For compressed-liquid states where h > h_L_sat (common), we must pass
    phase_hint='liquid' since the ambiguous region defaults to two-phase.
    """
    fl = he.load_fluid("water")
    ok = True
    print("TH flash round-trip tests (water):")

    # Cases where TH is unambiguous (h > h_V_sat or supercritical)
    unambig = [
        (500.0,   0.1,  "vapor"),        # superheated vapor, h > h_V_sat
        (400.0,   0.1,  "vapor"),
        (700.0,  30.0,  "supercritical"),
        (900.0, 100.0,  "supercritical"),
    ]
    for T, p_MPa, expected in unambig:
        p = p_MPa * 1e6
        fwd = he.flash_pt(p, T, fl)
        back = he.flash_th(T, fwd.h, fl)
        p_err = abs(back.p - p) / p
        print(f"  T={T:.1f}, p={p_MPa:g} MPa: phase={back.phase}, "
              f"p_recov={back.p*1e-6:.5f} MPa, rel_err={p_err:.2e}")
        ok &= back.phase == expected
        ok &= _check(f"p round-trip T={T},p={p_MPa}MPa", back.p, p, rtol=1e-6)

    # Compressed liquid cases: require phase_hint='liquid'
    ambig = [
        (300.0,   1.0,  "liquid"),
        (300.0, 100.0,  "liquid"),
    ]
    for T, p_MPa, expected in ambig:
        p = p_MPa * 1e6
        fwd = he.flash_pt(p, T, fl)
        back = he.flash_th(T, fwd.h, fl, phase_hint="liquid")
        p_err = abs(back.p - p) / p
        print(f"  T={T:.1f}, p={p_MPa:g} MPa (phase_hint='liquid'): "
              f"p_recov={back.p*1e-6:.5f} MPa, rel_err={p_err:.2e}")
        ok &= back.phase == expected
        ok &= _check(f"p round-trip w/hint T={T},p={p_MPa}MPa", back.p, p, rtol=1e-6)

    return ok


def test_flash_th_ambiguous_defaults_twophase():
    """TH without a phase hint in the h_L < h < h_V region: default to two-phase."""
    fl = he.load_fluid("water")
    ok = True
    print("TH flash ambiguous case -> default two-phase:")

    # At T=300K, liquid at 10 MPa has h > h_L_sat but < h_V_sat -- ambiguous.
    state = he.flash_pt(10e6, 300.0, fl)
    r = he.flash_th(300.0, state.h, fl)  # no hint
    print(f"  T=300K, h from compressed liquid: default result phase={r.phase}, "
          f"x={r.quality:.6f}")
    ok &= r.phase == "two_phase"
    # Quality should be small (h just barely above h_L_sat)
    ok &= 0.0 <= r.quality < 0.1

    # And inside the true dome (T=450K, x=0.5)
    T = 450.0
    rL, rV, p_sat = he.saturation_pT(T, fl)
    hL, hV = he.enthalpy(rL, T, fl), he.enthalpy(rV, T, fl)
    h_mid = 0.5 * (hL + hV)
    r = he.flash_th(T, h_mid, fl)
    print(f"  T=450K, mid-dome h: phase={r.phase}, x={r.quality:.6f}, p={r.p*1e-6:.5f} MPa")
    ok &= r.phase == "two_phase"
    ok &= _check("quality at mid-dome", r.quality, 0.5, rtol=1e-8)
    ok &= _check("p = p_sat", r.p, p_sat, rtol=1e-6)

    return ok


def test_flash_ts_co2():
    """TS flash on a second fluid (CO2) to exercise the code on a different EOS."""
    fl = he.load_fluid("carbondioxide")
    ok = True
    print("TS flash round-trip tests (CO2):")

    # p_sat at 260 K ~ 2.42 MPa, at 280 K ~ 4.15 MPa. Choose pressures
    # that definitely put us in the expected phase.
    test_points = [
        (260.0,   5.0, "liquid"),        # subcooled liquid (p > p_sat=2.4 MPa)
        (280.0,   6.0, "liquid"),        # subcooled liquid (p > p_sat=4.15 MPa)
        (310.0,  10.0, "supercritical"), # supercritical dense (T > T_c=304)
        (320.0,   0.1, "supercritical"), # low-density supercritical
    ]
    for T, p_MPa, expected in test_points:
        p = p_MPa * 1e6
        fwd = he.flash_pt(p, T, fl)
        back = he.flash_ts(T, fwd.s, fl)
        p_err = abs(back.p - p) / p
        print(f"  T={T:.1f}, p={p_MPa:g} MPa: fwd phase={fwd.phase}, "
              f"back p={back.p*1e-6:.5f} MPa, phase={back.phase}, rel_err={p_err:.2e}")
        ok &= back.phase == expected
        ok &= _check(f"p round-trip T={T},p={p_MPa}MPa", back.p, p, rtol=1e-5)

    return ok


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

def run_all():
    results = {}
    tests = [
        ("PT flash classification",           test_flash_pt),
        ("PH flash round-trip",               test_flash_ph_roundtrip),
        ("PH flash in two-phase dome",        test_flash_ph_two_phase),
        ("PS flash round-trip",               test_flash_ps_roundtrip),
        ("PS flash isentropic expansion",     test_isentropic_process),
        ("TV flash",                          test_flash_tv),
        ("UV flash round-trip",               test_flash_uv),
        ("TS flash round-trip",               test_flash_ts_roundtrip),
        ("TS flash in two-phase dome",        test_flash_ts_two_phase),
        ("TS flash (CO2)",                    test_flash_ts_co2),
        ("TH flash round-trip",               test_flash_th_roundtrip),
        ("TH ambiguous -> two-phase default", test_flash_th_ambiguous_defaults_twophase),
        ("CO2 phase envelope",                test_phase_envelope_co2),
        ("Water phase envelope",              test_phase_envelope_water),
    ]
    for name, fn in tests:
        print()
        print("=" * 72)
        print(name)
        print("=" * 72)
        results[name] = fn()

    print()
    print("=" * 72)
    print("Summary:")
    all_ok = True
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok &= ok
    print()
    print(f"OVERALL: {'ALL TESTS PASSED' if all_ok else 'FAILURES DETECTED'}")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
