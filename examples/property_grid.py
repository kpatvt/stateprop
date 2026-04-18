"""
Example: compute a table of thermodynamic properties on a (T, p) grid for CO2.

Demonstrates:
    - solving rho from (p, T) via the density solver
    - evaluating multiple properties at each state point
    - array-style evaluation

Run:
    python examples/property_grid.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

import stateprop as h
from stateprop import load_fluid
from stateprop.saturation import density_from_pressure


def main():
    fl = load_fluid("carbondioxide")

    # A grid of supercritical states
    T_values = [310.0, 350.0, 400.0, 500.0, 700.0]
    p_values_MPa = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]

    print(f"Thermodynamic properties of {fl.name}")
    print(f"(supercritical states; T > T_c = {fl.T_c} K)")
    print()

    for T in T_values:
        print(f"T = {T:6.1f} K")
        print(f"  {'p [MPa]':>8} {'rho [mol/m3]':>14} {'Z':>8} "
              f"{'cp [J/molK]':>14} {'w [m/s]':>10} {'h [J/mol]':>12}")
        for p_MPa in p_values_MPa:
            p = p_MPa * 1e6
            try:
                rho = density_from_pressure(p, T, fl, phase="vapor")
            except Exception as e:
                print(f"  {p_MPa:>8.3f}  <density solver failed: {e}>")
                continue

            Z = h.compressibility_factor(rho, T, fl)
            cp = h.cp(rho, T, fl)
            w = h.speed_of_sound(rho, T, fl)
            h_val = h.enthalpy(rho, T, fl)
            print(f"  {p_MPa:>8.3f} {rho:>14.3f} {Z:>8.4f} "
                  f"{cp:>14.3f} {w:>10.2f} {h_val:>12.2f}")
        print()

    # Demonstrate vector evaluation on a fine grid:
    print("Vector evaluation example:")
    print("  200 state points along the T=400 K isotherm, p from 1 to 40 MPa")
    p_grid = np.linspace(1e6, 40e6, 200)
    rho_grid = np.array([density_from_pressure(p, 400.0, fl, phase="vapor")
                          for p in p_grid])
    T_grid = np.full_like(rho_grid, 400.0)

    # Vector calls take rho and T arrays and return an array
    Z_arr = h.compressibility_factor(rho_grid, T_grid, fl)
    cp_arr = h.cp(rho_grid, T_grid, fl)

    print(f"    Z range along isotherm:   {Z_arr.min():.4f} to {Z_arr.max():.4f}")
    print(f"    cp range along isotherm:  {cp_arr.min():.3f} to {cp_arr.max():.3f} J/(mol K)")


if __name__ == "__main__":
    main()
