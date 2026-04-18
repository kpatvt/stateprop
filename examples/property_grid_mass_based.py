"""
Example: property table in engineering (mass-based) units.

Shows a complete set of thermodynamic properties for CO2 on a (T, p) grid:

    T    [K]            temperature
    rho  [kg/m^3]       density
    u    [kJ/kg]        specific internal energy
    h    [kJ/kg]        specific enthalpy
    s    [kJ/(kg K)]    specific entropy
    cv   [kJ/(kg K)]    specific isochoric heat capacity
    cp   [kJ/(kg K)]    specific isobaric heat capacity
    w    [m/s]          thermodynamic speed of sound

The library internally works on a per-mole basis (SI with amount-of-substance).
Conversion to per-mass is a simple divide-by-M using the molar_mass recorded
in the fluid's JSON. This script wraps that conversion in a small helper so
the output looks like a classic steam-table / thermophysical-property page.

Run:
    python examples/property_grid_mass_based.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

import stateprop as h
from stateprop import load_fluid
from stateprop.saturation import density_from_pressure


def mass_based_properties(rho_molar, T, fluid):
    """Return a dict of mass-based properties at (rho_molar [mol/m^3], T [K]).

    Units:
        rho:   kg/m^3
        u, h:  kJ/kg
        s, cv, cp: kJ/(kg K)
        w:     m/s
    """
    if fluid.molar_mass is None:
        raise ValueError(f"Fluid '{fluid.name}' has no molar_mass; cannot "
                         "convert to mass-based units. Add 'molar_mass' "
                         "(kg/mol) to its JSON.")
    M = fluid.molar_mass  # kg/mol

    # Molar quantities from the library (SI per mole)
    u_molar  = h.internal_energy(rho_molar, T, fluid)    # J/mol
    h_molar  = h.enthalpy(rho_molar, T, fluid)           # J/mol
    s_molar  = h.entropy(rho_molar, T, fluid)            # J/(mol K)
    cv_molar = h.cv(rho_molar, T, fluid)                 # J/(mol K)
    cp_molar = h.cp(rho_molar, T, fluid)                 # J/(mol K)
    w        = h.speed_of_sound(rho_molar, T, fluid)     # m/s

    # Convert: per-mole -> per-mass by dividing by M; J -> kJ by dividing by 1000
    return {
        "rho_mass": rho_molar * M,                       # kg/m^3
        "u":  u_molar  / M * 1e-3,                       # kJ/kg
        "h":  h_molar  / M * 1e-3,                       # kJ/kg
        "s":  s_molar  / M * 1e-3,                       # kJ/(kg K)
        "cv": cv_molar / M * 1e-3,                       # kJ/(kg K)
        "cp": cp_molar / M * 1e-3,                       # kJ/(kg K)
        "w":  w,                                         # m/s
    }


def print_table(fluid, T_values, p_values_MPa, phase="auto"):
    """Print a rectangular (T, p) property table in mass-based units."""
    print(f"Properties of {fluid.name}")
    print(f"  molar_mass = {fluid.molar_mass*1000:.4f} g/mol")
    print(f"  T_c = {fluid.T_c} K,  p_c = {fluid.p_c*1e-6:.4f} MPa,  "
          f"rho_c = {fluid.rho_c*fluid.molar_mass:.3f} kg/m^3")
    print()

    header = (f"{'T [K]':>8} {'p [MPa]':>8} {'rho [kg/m3]':>12} "
              f"{'u [kJ/kg]':>11} {'h [kJ/kg]':>11} {'s [kJ/kgK]':>12} "
              f"{'cv [kJ/kgK]':>12} {'cp [kJ/kgK]':>12} {'w [m/s]':>10}")
    sep = "-" * len(header)

    for T in T_values:
        print(header if T == T_values[0] else "")
        print(sep)
        for p_MPa in p_values_MPa:
            p = p_MPa * 1e6
            try:
                rho_mol = density_from_pressure(p, T, fluid, phase=phase)
                props = mass_based_properties(rho_mol, T, fluid)
                print(f"{T:>8.2f} {p_MPa:>8.3f} {props['rho_mass']:>12.4f} "
                      f"{props['u']:>11.3f} {props['h']:>11.3f} "
                      f"{props['s']:>12.5f} {props['cv']:>12.5f} "
                      f"{props['cp']:>12.5f} {props['w']:>10.3f}")
            except Exception as e:
                print(f"{T:>8.2f} {p_MPa:>8.3f}  <solver failed: {e}>")


def main():
    fluid = load_fluid("carbondioxide")

    # Supercritical grid: all points above T_c, so a single-phase density solver works.
    print("=" * 100)
    print("SUPERCRITICAL REGION  (T > T_c, single-phase fluid)")
    print("=" * 100)
    T_values = [310.0, 350.0, 400.0, 500.0, 700.0]
    p_values_MPa = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]
    print_table(fluid, T_values, p_values_MPa, phase="auto")

    # Subcritical vapor branch (gas phase below T_c)
    print()
    print("=" * 100)
    print("SUBCRITICAL VAPOR  (T < T_c,  p < p_sat(T))")
    print("=" * 100)
    # Pick pressures that are below the saturation pressure at each T.
    # CO2 p_sat at 250 K is ~1.8 MPa; at 280 K, ~4.2 MPa; at 300 K, ~6.7 MPa.
    print_table(fluid, [250.0], [0.1, 0.5, 1.0, 1.5], phase="vapor")
    print()
    print_table(fluid, [280.0], [0.1, 1.0, 2.0, 4.0], phase="vapor")
    print()
    print_table(fluid, [300.0], [0.1, 1.0, 3.0, 6.0], phase="vapor")

    # Subcritical liquid branch
    print()
    print("=" * 100)
    print("SUBCRITICAL LIQUID  (T < T_c,  p > p_sat(T))")
    print("=" * 100)
    print_table(fluid, [250.0], [5.0, 10.0, 20.0, 50.0], phase="liquid")
    print()
    print_table(fluid, [280.0], [10.0, 20.0, 50.0], phase="liquid")

    # Saturation states -- the classic "sat-liquid | sat-vapor" pair
    print()
    print("=" * 100)
    print("SATURATED STATES  (VLE coexistence)")
    print("=" * 100)
    print(f"{'T [K]':>8} {'p_sat [MPa]':>12} "
          f"{'rho_L [kg/m3]':>14} {'rho_V [kg/m3]':>14} "
          f"{'h_L [kJ/kg]':>12} {'h_V [kJ/kg]':>12} "
          f"{'s_L [kJ/kgK]':>14} {'s_V [kJ/kgK]':>14} "
          f"{'h_fg [kJ/kg]':>14}")
    print("-" * 128)
    for T in [220.0, 240.0, 260.0, 280.0, 300.0]:
        rho_L_mol, rho_V_mol, p_sat = h.saturation_pT(T, fluid)
        L = mass_based_properties(rho_L_mol, T, fluid)
        V = mass_based_properties(rho_V_mol, T, fluid)
        h_fg = V["h"] - L["h"]
        print(f"{T:>8.2f} {p_sat*1e-6:>12.4f} "
              f"{L['rho_mass']:>14.3f} {V['rho_mass']:>14.4f} "
              f"{L['h']:>12.3f} {V['h']:>12.3f} "
              f"{L['s']:>14.5f} {V['s']:>14.5f} "
              f"{h_fg:>14.3f}")

    print()
    print("Notes on reference state:")
    print("  The Span-Wagner formulation uses an arbitrary reference-state")
    print("  convention baked into the ideal-gas part's a1 and a2 lead constants.")
    print("  Absolute values of u, h, s are therefore only meaningful as differences;")
    print("  see the original reference for the convention used here.")


if __name__ == "__main__":
    main()
