"""
Example: compute the CO2 saturation curve from triple point to near-critical.

This exercises:
    - loading a fluid from JSON
    - solving vapor-liquid equilibrium
    - computing enthalpies on both branches
    - computing enthalpy of vaporization (latent heat)

Run:
    python examples/01_saturation_curve.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

import stateprop as h
from stateprop import load_fluid, saturation_pT


def main():
    fl = load_fluid("carbondioxide")
    print(f"Fluid: {fl.name}")
    print(f"  T_triple = {fl.T_triple} K")
    print(f"  T_c      = {fl.T_c} K")
    print(f"  p_c      = {fl.p_c*1e-6:.4f} MPa")
    print(f"  rho_c    = {fl.rho_c:.3f} mol/m^3")
    print()

    # Temperatures from just above triple point to just below critical
    T_values = np.linspace(fl.T_triple + 2.0, fl.T_c - 2.0, 15)

    print(f"{'T [K]':>8} {'p_sat [MPa]':>12} {'rho_L [mol/m^3]':>18} "
          f"{'rho_V [mol/m^3]':>18} {'h_vap [kJ/mol]':>16}")
    print("-" * 74)

    for T in T_values:
        rho_L, rho_V, p = saturation_pT(T, fl)
        h_L = h.enthalpy(rho_L, T, fl)
        h_V = h.enthalpy(rho_V, T, fl)
        h_vap = (h_V - h_L) * 1e-3     # J/mol -> kJ/mol
        print(f"{T:>8.2f} {p*1e-6:>12.4f} {rho_L:>18.2f} {rho_V:>18.3f} {h_vap:>16.3f}")

    print()
    print("Sanity check: enthalpy of vaporization should decrease monotonically")
    print("toward zero as T -> T_c. It should be about 16-18 kJ/mol near the")
    print("triple point and drop below 1 kJ/mol close to the critical point.")


if __name__ == "__main__":
    main()
