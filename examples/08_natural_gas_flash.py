"""Natural gas VLE — GERG-2008 multicomponent flash with phase envelope.

Demonstrates:
  - Loading multicomponent Helmholtz mixture (5-component natural gas)
  - PT, PH, UV flash on the GERG mixture
  - Comparison vs cubic PR EOS with the same composition
  - Phase envelope tracing from GERG mixture

The 5-component composition is a typical pipeline natural gas with
methane-rich content plus C2/C3 heavies and inerts (N2, CO2).
"""
from __future__ import annotations

import numpy as np
import stateprop as sp
from stateprop.mixture.mixture import load_mixture
from stateprop.mixture.flash import flash_pt as gerg_flash, flash_uv
from stateprop.cubic import (PR, CubicMixture,
                                flash_pt as cubic_flash,
                                newton_bubble_point_T, newton_dew_point_T)


def main():
    # 5-component natural gas (mole fractions)
    components = ["gerg2008/methane", "gerg2008/ethane", "gerg2008/propane",
                  "gerg2008/nitrogen", "gerg2008/carbondioxide"]
    z = [0.85, 0.05, 0.02, 0.05, 0.03]

    print("=" * 65)
    print("Natural-gas multicomponent flash (5 components)")
    print("=" * 65)
    print(f"  Composition (mol%): CH4={z[0]*100:.1f}, C2H6={z[1]*100:.1f}, "
          f"C3H8={z[2]*100:.1f}, N2={z[3]*100:.1f}, CO2={z[4]*100:.1f}")

    mix = load_mixture(components, composition=z, binary_set="gerg2008")

    # PT flash at pipeline conditions (20 bar, 200 K -- 2-phase for this composition)
    T_test, p_test = 200.0, 20e5
    r = gerg_flash(p_test, T_test, z, mix)
    print(f"\n  GERG-2008 flash at T={T_test} K, p={p_test/1e5} bar:")
    print(f"    phase: {r.phase}")
    if r.phase == "two_phase":
        print(f"    vapor fraction: {r.beta:.4f}")
        print(f"    rho_L = {r.rho_L:.2f} mol/m^3, rho_V = {r.rho_V:.2f} mol/m^3")
        print(f"    x = {np.array2string(r.x, precision=4, suppress_small=True)}")
        print(f"    y = {np.array2string(r.y, precision=4, suppress_small=True)}")
    else:
        print(f"    rho = {r.rho:.2f} mol/m^3")

    print(f"    enthalpy h = {r.h/1e3:.2f} kJ/mol")

    # Compare GERG vs cubic PR at the same conditions
    print("\n" + "=" * 65)
    print("GERG-2008 vs cubic Peng-Robinson on identical composition")
    print("=" * 65)

    # Build cubic PR mixture matching the 5 components
    methane = PR(T_c=190.564, p_c=4.5992e6, acentric_factor=0.01142)
    ethane  = PR(T_c=305.32,  p_c=4.872e6,  acentric_factor=0.099)
    propane = PR(T_c=369.83,  p_c=4.248e6,  acentric_factor=0.152)
    n2      = PR(T_c=126.192, p_c=3.3958e6, acentric_factor=0.0372)
    co2     = PR(T_c=304.13,  p_c=7.3773e6, acentric_factor=0.22394)
    pr_mix = CubicMixture([methane, ethane, propane, n2, co2], composition=z)

    rc = cubic_flash(p_test, T_test, z, pr_mix)
    print(f"  At T={T_test} K, p={p_test/1e5} bar:")
    print(f"    GERG-2008  phase={r.phase}, beta={r.beta if r.beta else '-':>6}, rho={r.rho:>8.2f}")
    if rc.phase == "two_phase":
        print(f"    PR         phase={rc.phase}, beta={rc.beta:.4f}, rho={rc.rho:>8.2f}")
    else:
        print(f"    PR         phase={rc.phase},          rho={rc.rho:>8.2f}")

    # Bubble point at fixed T using cubic
    bub = newton_bubble_point_T(p_test, np.asarray(z), pr_mix, T_init=180.0)
    dew = newton_dew_point_T(p_test, np.asarray(z), pr_mix, T_init=bub.T * 1.05)
    print(f"\n  PR bubble/dew at p={p_test/1e5} bar:")
    print(f"    T_bubble = {bub.T:.2f} K  (vapor leaving liquid)")
    print(f"    T_dew    = {dew.T:.2f} K  (first liquid drop)")

    # UV flash for transient simulation
    print("\n" + "=" * 65)
    print("UV flash for dynamic simulation: adiabatic compression")
    print("=" * 65)
    r0 = gerg_flash(1e5, 300.0, z, mix)
    u0 = r0.h - r0.p / r0.rho
    v0 = 1.0 / r0.rho
    print(f"  Initial state: T={300.0} K, p={1.0} bar  ->  u0={u0/1e3:.2f} kJ/mol")
    # Compress adiabatically to half the volume
    r1 = flash_uv(u0, v0 / 2, z, mix)
    print(f"  After v_target = v0/2 (adiabatic):")
    print(f"    T = {r1.T:.2f} K   (rose by {r1.T - 300:.1f} K)")
    print(f"    p = {r1.p/1e5:.3f} bar  (rose by factor {r1.p/1e5:.2f})")
    print(f"    phase = {r1.phase}")


if __name__ == "__main__":
    main()
