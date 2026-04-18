"""
Example: classic steam tables generated from IAPWS-95.

Prints saturated-water properties and a superheated-steam table in mass-based
(engineering) units, identical in spirit to the tables in most thermodynamics
textbooks.

Run:
    python examples/steam_tables.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

import stateprop as h
from stateprop import load_fluid, saturation_pT
from stateprop.saturation import density_from_pressure


def mass_props(rho_mol, T, fluid):
    """Return a dict of mass-based properties at (rho_mol [mol/m^3], T [K])."""
    M = fluid.molar_mass
    return {
        "rho": rho_mol * M,
        "u":  h.internal_energy(rho_mol, T, fluid) / M * 1e-3,
        "h":  h.enthalpy(rho_mol, T, fluid)        / M * 1e-3,
        "s":  h.entropy(rho_mol, T, fluid)         / M * 1e-3,
        "cv": h.cv(rho_mol, T, fluid)              / M * 1e-3,
        "cp": h.cp(rho_mol, T, fluid)              / M * 1e-3,
        "w":  h.speed_of_sound(rho_mol, T, fluid),
    }


def saturation_table(fluid):
    """Print a saturation table (p_sat, rho_L, rho_V, h_L, h_V, h_fg, s_L, s_V)."""
    print("=" * 96)
    print("SATURATED WATER - Temperature table")
    print("=" * 96)
    print(f"{'T':>6} {'p_sat':>10} | {'rho_L':>9} {'rho_V':>12} | "
          f"{'h_L':>8} {'h_V':>8} {'h_fg':>8} | {'s_L':>8} {'s_V':>8}")
    print(f"{'[K]':>6} {'[MPa]':>10} | {'[kg/m3]':>9} {'[kg/m3]':>12} | "
          f"{'[kJ/kg]':>8} {'[kJ/kg]':>8} {'[kJ/kg]':>8} | "
          f"{'[kJ/kgK]':>8} {'[kJ/kgK]':>8}")
    print("-" * 96)

    for T in [275.0, 300.0, 325.0, 350.0, 373.15, 400.0, 450.0, 500.0,
              550.0, 600.0, 625.0, 645.0]:
        rho_L, rho_V, p = saturation_pT(T, fluid)
        L = mass_props(rho_L, T, fluid)
        V = mass_props(rho_V, T, fluid)
        h_fg = V["h"] - L["h"]
        print(f"{T:>6.2f} {p*1e-6:>10.5f} | {L['rho']:>9.3f} {V['rho']:>12.4e} | "
              f"{L['h']:>8.2f} {V['h']:>8.2f} {h_fg:>8.2f} | "
              f"{L['s']:>8.4f} {V['s']:>8.4f}")


def superheated_table(fluid, p_MPa):
    """Print a superheated-steam table at fixed pressure."""
    p = p_MPa * 1e6
    print()
    print("=" * 80)
    print(f"SUPERHEATED STEAM at p = {p_MPa} MPa")
    print("=" * 80)
    print(f"{'T [K]':>8} {'rho [kg/m3]':>12} {'u [kJ/kg]':>11} {'h [kJ/kg]':>11} "
          f"{'s [kJ/kgK]':>12} {'cp [kJ/kgK]':>12} {'w [m/s]':>9}")
    print("-" * 80)

    # Determine saturation temperature at this pressure (if subcritical)
    if p < fluid.p_c:
        # Find T_sat by bisection on p_sat(T) = p
        T_lo, T_hi = 300.0, fluid.T_c - 0.5
        for _ in range(50):
            T_mid = 0.5 * (T_lo + T_hi)
            try:
                _, _, p_mid = saturation_pT(T_mid, fluid)
                if p_mid < p:
                    T_lo = T_mid
                else:
                    T_hi = T_mid
            except Exception:
                T_hi = T_mid
            if abs(T_hi - T_lo) < 0.01:
                break
        T_sat = 0.5 * (T_lo + T_hi)
        print(f"  (T_sat at {p_MPa} MPa ~ {T_sat:.2f} K)")

    # Tabulate at a range of temperatures above saturation (or for supercritical, above T_c)
    T_start = max(373.15, T_sat + 5.0) if p < fluid.p_c else fluid.T_c + 5.0
    T_list = np.linspace(T_start, 1100.0, 12)

    for T in T_list:
        try:
            rho_mol = density_from_pressure(p, T, fluid, phase="vapor")
            P = mass_props(rho_mol, T, fluid)
            print(f"{T:>8.2f} {P['rho']:>12.4e} {P['u']:>11.2f} {P['h']:>11.2f} "
                  f"{P['s']:>12.5f} {P['cp']:>12.5f} {P['w']:>9.2f}")
        except Exception as e:
            print(f"{T:>8.2f}  <solver failed: {e}>")


def main():
    fluid = load_fluid("water")
    print(f"IAPWS-95 Reference Formulation for Water")
    print(f"  M          = {fluid.molar_mass*1000:.6f} g/mol")
    print(f"  T_c        = {fluid.T_c} K")
    print(f"  p_c        = {fluid.p_c*1e-6:.3f} MPa")
    print(f"  rho_c      = {fluid.rho_c * fluid.molar_mass:.1f} kg/m^3")
    print(f"  T_triple   = {fluid.T_triple} K")
    print(f"  p_triple   = {fluid.p_triple:.3f} Pa")
    print()
    print("Note: The internal-energy/entropy reference state is set such that")
    print("      u' = s' = 0 for saturated liquid at the triple point.")
    print()

    saturation_table(fluid)
    superheated_table(fluid, 0.1)      # 0.1 MPa (1 atm) -- includes steam at 1 atm
    superheated_table(fluid, 1.0)
    superheated_table(fluid, 10.0)


if __name__ == "__main__":
    main()
