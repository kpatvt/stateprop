"""
Tests beyond CO2:

  1. Nitrogen (Span 2000): basic property evaluation and consistency.
  2. Synthetic ideal-gas EOS exercising PE_cosh and PE_sinh terms
     (used by GERG-2008): check derivatives by finite differences.
  3. Saturation curve for nitrogen against NIST reference values.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

import stateprop as h
from stateprop import load_fluid, saturation_pT, Fluid
from stateprop.core import alpha_0_derivs


def _check(label, got, want, rtol=1e-6, atol=1e-12):
    diff = abs(got - want)
    rel = diff / (abs(want) + atol)
    ok = diff <= atol or rel <= rtol
    print(f"    [{'PASS' if ok else 'FAIL'}] {label:28s} "
          f"got={got: .6e}  want={want: .6e}  rel={rel: .2e}")
    return ok


def test_nitrogen_properties():
    """Sanity-check nitrogen."""
    fl = load_fluid("nitrogen")
    print(f"Loaded {fl}")
    ok = True

    # Standard reference state: T=298.15 K, p=101.325 kPa.  Z close to 1.
    T = 298.15
    from stateprop.saturation import density_from_pressure
    rho = density_from_pressure(101325.0, T, fl, phase="vapor")
    Z = h.compressibility_factor(rho, T, fl)
    print(f"  N2 at T=298.15 K, p=101.325 kPa: rho = {rho:.4f} mol/m^3, Z = {Z:.6f}")
    ok &= _check("N2 rho_ref", rho, 40.87, rtol=0.002)
    ok &= _check("N2 Z_ref",   Z,   0.99976, rtol=1e-3)

    # Thermodynamic identity: h = u + p/rho (exact)
    u = h.internal_energy(rho, T, fl)
    h_val = h.enthalpy(rho, T, fl)
    p = h.pressure(rho, T, fl)
    ok &= _check("N2 h = u + p/rho", h_val, u + p / rho, rtol=1e-10)

    # Supercritical N2 at T=300 K, moderate pressure
    T = 300.0
    rho = density_from_pressure(10e6, T, fl, phase="vapor")
    print(f"  N2 at T=300 K, p=10 MPa: rho = {rho:.1f} mol/m^3  (NIST ~3926)")
    ok &= _check("N2 rho @10MPa,300K", rho, 3926.0, rtol=0.02)

    # cp > cv always
    cp_val = h.cp(rho, T, fl)
    cv_val = h.cv(rho, T, fl)
    print(f"    cp = {cp_val:.3f}, cv = {cv_val:.3f} J/(mol K)")
    ok &= cp_val > cv_val
    return ok


def test_nitrogen_saturation():
    """Nitrogen saturation at a few points."""
    fl = load_fluid("nitrogen")
    ok = True
    print("Nitrogen saturation:")
    # Reference from NIST REFPROP:
    reference = {
        70.0:  (3.8574e4,  29937.0, 68.59),
        80.0:  (1.3698e5,  28341.0, 222.29),
        100.0: (7.7781e5,  24607.0, 1122.20),
        110.0: (1.4671e6,  22330.0, 2181.20),
    }
    for T, (p_ref, rL_ref, rV_ref) in reference.items():
        rL, rV, p = saturation_pT(T, fl)
        print(f"  T={T:6.2f}  p={p*1e-6:7.4f} MPa (ref {p_ref*1e-6:.4f})"
              f"  rho_L={rL:8.2f} (ref {rL_ref:.1f})"
              f"  rho_V={rV:8.3f} (ref {rV_ref:.2f})")
        # 2% tolerance is reasonable given we omit 2 non-analytic near-critical terms
        ok &= _check(f"N2 p_sat @ {T} K", p, p_ref, rtol=0.02)
        ok &= _check(f"N2 rho_L @ {T} K", rL, rL_ref, rtol=0.02)
        ok &= _check(f"N2 rho_V @ {T} K", rV, rV_ref, rtol=0.05)
    return ok


def test_pe_cosh_sinh_terms():
    """Verify PE_cosh and PE_sinh ideal-term codes against finite differences.

    We build a synthetic fluid whose ideal-gas contribution includes both term
    types, then check that the analytic derivatives match central differences
    at a random (delta, tau) point.
    """
    synthetic = {
        "name": "SyntheticPECoshSinh",
        "gas_constant": 8.314462618,
        "critical": {"T": 300.0, "rho": 1000.0, "p": 1.0e6},
        "ideal": [
            {"type": "log_delta", "a": 1.0},
            {"type": "a1",        "a": 1.23, "b": -0.45},
            {"type": "log_tau",   "a": 3.0},
            {"type": "PE_cosh",   "a": 0.876,  "b": 2.345},
            {"type": "PE_sinh",   "a": -0.432, "b": 1.678},
        ],
        "residual_polynomial": [],
        "residual_exponential": [],
        "residual_gaussian": [],
    }
    fl = Fluid.from_dict(synthetic)
    pack = fl.pack()
    codes, a_arr, b_arr = pack[-3], pack[-2], pack[-1]

    ok = True
    # Check at several (delta, tau) points
    for delta, tau in [(0.5, 1.5), (1.0, 0.8), (2.0, 2.5), (0.1, 0.3)]:
        A, A_d, A_t, A_dd, A_tt, A_dt = alpha_0_derivs(delta, tau, codes, a_arr, b_arr)
        hh = 1e-6 * max(1.0, abs(tau))
        A_p, _, _, _, _, _ = alpha_0_derivs(delta, tau + hh, codes, a_arr, b_arr)
        A_m, _, _, _, _, _ = alpha_0_derivs(delta, tau - hh, codes, a_arr, b_arr)
        A_t_fd = (A_p - A_m) / (2 * hh)
        ok &= _check(f"cosh/sinh A_t  at d={delta},t={tau}", A_t, A_t_fd, rtol=1e-6)

        _, _, A_t_p, _, _, _ = alpha_0_derivs(delta, tau + hh, codes, a_arr, b_arr)
        _, _, A_t_m, _, _, _ = alpha_0_derivs(delta, tau - hh, codes, a_arr, b_arr)
        A_tt_fd = (A_t_p - A_t_m) / (2 * hh)
        ok &= _check(f"cosh/sinh A_tt at d={delta},t={tau}", A_tt, A_tt_fd, rtol=1e-4)
    return ok


if __name__ == "__main__":
    print("=" * 72)
    print("Test 1: Nitrogen properties (Span 2000)")
    print("=" * 72)
    r1 = test_nitrogen_properties()

    print()
    print("=" * 72)
    print("Test 2: Nitrogen saturation curve")
    print("=" * 72)
    r2 = test_nitrogen_saturation()

    print()
    print("=" * 72)
    print("Test 3: Ideal PE_cosh and PE_sinh terms via finite differences")
    print("=" * 72)
    r3 = test_pe_cosh_sinh_terms()

    print()
    print("=" * 72)
    all_ok = all([r1, r2, r3])
    print(f"OVERALL: {'ALL TESTS PASSED' if all_ok else 'FAILURES DETECTED'}")
    print("=" * 72)
    sys.exit(0 if all_ok else 1)
