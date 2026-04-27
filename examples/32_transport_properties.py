"""Transport property correlations: viscosity, thermal conductivity,
surface tension.

Demonstrates the v0.9.32 + v0.9.33 transport-property module:

  - Chung-Lee-Starling viscosity (pure + mixture, low-pressure and dense)
  - Chung thermal conductivity
  - Brock-Bird corresponding-states surface tension
  - Macleod-Sugden parachor surface tension (uses liquid + vapor density)

The transport module is EOS-independent — any component object with
the required fields (T_c, p_c, acentric_factor, molar_mass, optionally
dipole_moment) works. Examples below use components from the cubic and
SAFT modules interchangeably.
"""
from __future__ import annotations

import numpy as np
from stateprop.transport import (viscosity_chung, thermal_conductivity_chung,
                                    viscosity_mixture_chung,
                                    thermal_conductivity_mixture_chung,
                                    surface_tension_brock_bird,
                                    surface_tension_macleod_sugden,
                                    surface_tension_mixture_macleod_sugden)
from stateprop.saft import (METHANE, ETHANE, PROPANE, WATER, METHANOL,
                              SAFTMixture)


def main():
    print("=" * 70)
    print("Transport properties: viscosity, thermal conductivity, sigma")
    print("=" * 70)

    # 1. Pure-fluid viscosity (Chung)
    print("\nPure-fluid viscosity at 1 bar (dilute-gas Chung correlation):")
    print(f"\n  {'Component':<12s}  {'T (K)':>7s}  "
          f"{'eta (uPa.s)':>14s}  {'reference':>18s}")
    print("  " + "-" * 60)
    # Reference values from NIST WebBook at 1 bar
    refs = {
        ("methane", 300):  11.21,
        ("ethane", 300):   9.40,
        ("propane", 300):  8.20,
        ("water", 373.15): 12.17,   # steam at 100 C
        ("methanol", 350): 11.86,
    }
    for comp in [METHANE, ETHANE, PROPANE, WATER, METHANOL]:
        T = 373.15 if comp.name == "water" else (350.0 if comp.name == "methanol" else 300.0)
        try:
            eta = viscosity_chung(comp, T) * 1e6   # convert to uPa.s
            ref = refs.get((comp.name, int(T)), float('nan'))
            ref_str = f"{ref:.2f}" if not np.isnan(ref) else "-"
            print(f"  {comp.name:<12s}  {T:>7.2f}  {eta:>14.3f}  "
                  f"{ref_str:>18s}")
        except Exception as e:
            print(f"  {comp.name:<12s}  failed: {str(e)[:40]}")

    # 2. Dense-gas viscosity correction
    print("\n" + "=" * 70)
    print("Dense-gas viscosity: methane at 250 K, varying density")
    print("=" * 70)
    saft = SAFTMixture([METHANE], [1.0])
    print(f"\n  {'p (bar)':>9s}  {'rho (mol/m^3)':>14s}  "
          f"{'eta (uPa.s)':>14s}  {'enhancement':>13s}")
    print("  " + "-" * 60)
    eta_dilute = viscosity_chung(METHANE, 250.0) * 1e6
    for p_bar in [1.0, 50.0, 100.0, 200.0]:
        try:
            phase = 'liquid' if p_bar > 100 else 'vapor'
            rho = saft.density_from_pressure(p_bar * 1e5, 250.0,
                                                phase_hint=phase)
            eta = viscosity_chung(METHANE, 250.0, rho_mol=rho) * 1e6
            enhancement = eta / eta_dilute
            print(f"  {p_bar:>9.1f}  {rho:>14.2f}  {eta:>14.3f}  "
                  f"{enhancement:>13.3f}x")
        except Exception as e:
            print(f"  {p_bar:>9.1f}  failed: {str(e)[:40]}")

    # 3. Thermal conductivity
    print("\n" + "=" * 70)
    print("Thermal conductivity (Chung) — pure components at 300 K, 1 bar")
    print("=" * 70)
    print(f"\n  {'Component':<12s}  {'k (mW/m/K)':>12s}  {'reference':>14s}")
    print("  " + "-" * 50)
    k_refs = {"methane": 34.4, "ethane": 21.6, "propane": 18.2}
    for comp in [METHANE, ETHANE, PROPANE]:
        try:
            k = thermal_conductivity_chung(comp, 300.0) * 1000   # mW/m/K
            ref = k_refs.get(comp.name, float('nan'))
            ref_str = f"{ref:.1f}" if not np.isnan(ref) else "-"
            print(f"  {comp.name:<12s}  {k:>12.3f}  {ref_str:>14s}")
        except Exception as e:
            print(f"  {comp.name:<12s}  failed: {str(e)[:40]}")

    # 4. Mixture transport: Chung mixing rules
    print("\n" + "=" * 70)
    print("Mixture transport: methane(0.6) + ethane(0.3) + propane(0.1) at 300 K")
    print("=" * 70)
    z = [0.6, 0.3, 0.1]
    components = [METHANE, ETHANE, PROPANE]
    eta_mix = viscosity_mixture_chung(components, z, 300.0) * 1e6
    k_mix = thermal_conductivity_mixture_chung(components, z, 300.0) * 1000
    print(f"\n  Dilute-gas viscosity:        eta_mix = {eta_mix:.3f} uPa.s")
    print(f"  Dilute-gas thermal cond.:    k_mix   = {k_mix:.3f} mW/m/K")

    # 5. Surface tension: Brock-Bird (pure) and Macleod-Sugden (mixture)
    print("\n" + "=" * 70)
    print("Surface tension")
    print("=" * 70)

    # Brock-Bird pure (corresponding-states)
    print(f"\nBrock-Bird (pure, corresponding states) at 298.15 K:")
    print(f"  {'Component':<12s}  {'sigma (mN/m)':>13s}  "
          f"{'reference':>14s}")
    print("  " + "-" * 50)
    sigma_refs = {"water": 71.97, "methanol": 22.07, "ethane": 1.46}
    for comp in [WATER, METHANOL]:
        T_test = 298.15
        try:
            sigma = surface_tension_brock_bird(comp, T_test) * 1000   # mN/m
            ref = sigma_refs.get(comp.name, float('nan'))
            ref_str = f"{ref:.2f}" if not np.isnan(ref) else "-"
            print(f"  {comp.name:<12s}  {sigma:>13.3f}  {ref_str:>14s}")
        except Exception as e:
            print(f"  {comp.name:<12s}  failed: {str(e)[:40]}")

    # Macleod-Sugden using a SAFT density
    print(f"\nMacleod-Sugden (parachor-based) at 298.15 K:")
    print(f"  Uses liquid + vapor densities -- here from PC-SAFT")
    saft_water = SAFTMixture([WATER], [1.0])
    rho_L = saft_water.density_from_pressure(101325, 298.15, phase_hint='liquid')
    # For very low p estimate vapor density from ideal gas
    rho_V = 101325.0 / (8.314472 * 298.15)
    sigma_ms = surface_tension_macleod_sugden(WATER, rho_L, rho_V) * 1000
    print(f"  Water:   rho_L = {rho_L:>10.2f} mol/m^3, "
          f"rho_V = {rho_V:>8.2f} mol/m^3")
    print(f"           sigma = {sigma_ms:.2f} mN/m  (NIST: 71.97 mN/m)")

    # Mixture parachor surface tension: water-methanol
    saft_mix = SAFTMixture([WATER, METHANOL], [0.5, 0.5])
    rho_L = saft_mix.density_from_pressure(101325, 298.15, phase_hint='liquid')
    sigma_mix = surface_tension_mixture_macleod_sugden(
        comps=[WATER, METHANOL], x=[0.5, 0.5], y=[0.5, 0.5],
        rho_L_mol=rho_L, rho_V_mol=rho_V) * 1000
    print(f"\n  Water/methanol (50/50): sigma = {sigma_mix:.2f} mN/m")
    print(f"  (Experimental ~36 mN/m; Macleod-Sugden interpolation is")
    print(f"  approximate for strongly nonideal mixtures.)")

    print("\n" + "=" * 70)
    print("Notes")
    print("=" * 70)
    print("""
  - Chung correlations: 5-15% accuracy on pure fluids, somewhat worse
    for highly polar / hydrogen-bonding species. Density-dependent
    correction extends to dense gas / liquid regions.
  - Brock-Bird: corresponding-states sigma, accurate to ~5% for
    nonpolar fluids; less accurate for polar species.
  - Macleod-Sugden: needs a parachor (tabulated or estimated via
    group contribution); accurate to ~5% when parachor is known
    from the same temperature regime.
  - For high-pressure dense gas viscosity beyond Chung's range,
    the v0.9.38 Stiel-Thodos / Jossi correction is also available.
""")


if __name__ == "__main__":
    main()
