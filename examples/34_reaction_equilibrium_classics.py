"""Reaction equilibrium: water-gas shift, methanol synthesis, ammonia (v0.9.61).

Demonstrates the new ideal-gas reaction equilibrium module that combines
NIST Shomate thermochemistry with a single-reaction extent solver.

For each reaction Sum_i nu_i A_i = 0, the equilibrium constant is
    K_eq(T) = exp[-Sum_i nu_i Gf_i(T) / (RT)]

For ideal-gas mixtures, the equilibrium condition becomes
    K_eq = Prod_i (y_i * p / p_ref)^nu_i

solved for the reaction extent xi at given (T, p, n_initial).

Three industrially-important reactions are demonstrated:

  1. Water-gas shift:  CO + H2O = CO2 + H2     (mildly exothermic)
  2. Methanol synthesis: CO + 2 H2 = CH3OH     (highly exothermic)
  3. Haber-Bosch (NH3): N2 + 3 H2 = 2 NH3      (highly exothermic)
"""
from __future__ import annotations

import os
import numpy as np
from stateprop.reaction import Reaction


def main():
    # ---------------------------------------------------------------
    # 1. Water-gas shift CO + H2O = CO2 + H2
    # ---------------------------------------------------------------
    print("=" * 70)
    print("Water-gas shift: CO + H2O = CO2 + H2")
    print("=" * 70)
    rxn = Reaction.from_names(
        reactants={'CO': 1, 'H2O': 1},
        products={'CO2': 1, 'H2': 1})

    print(f"\n  {'T (K)':>6s}  {'dH (kJ/mol)':>12s}  {'dG (kJ/mol)':>12s}  "
          f"{'K_eq':>10s}")
    Ts = [600, 700, 800, 900, 1000]
    Ks_wgs = []
    for T in Ts:
        K = rxn.K_eq(T)
        Ks_wgs.append(K)
        print(f"  {T:>6.0f}  {rxn.dH_rxn(T)/1000:>12.3f}  "
              f"{rxn.dG_rxn(T)/1000:>12.3f}  {K:>10.3f}")
    print("  (Exothermic: K decreases with T; high-T shift favors reactants)")

    # Conversion at 800 K, 10 bar, equimolar feed
    print("\n  Equilibrium at 800 K, 10 bar, 1 mol CO + 1 mol H2O:")
    r = rxn.equilibrium_extent_ideal_gas(
        T=800.0, p=10e5, n_initial=[1.0, 1.0, 0.0, 0.0])
    print(f"    Extent = {r.xi:.4f}")
    print(f"    Equilibrium moles: CO={r.n_eq[0]:.4f}, H2O={r.n_eq[1]:.4f}, "
          f"CO2={r.n_eq[2]:.4f}, H2={r.n_eq[3]:.4f}")
    print(f"    CO conversion: {(1 - r.n_eq[0]/1.0)*100:.2f}%")

    # Excess steam: 1 mol CO + 3 mol H2O -> drives WGS forward
    r2 = rxn.equilibrium_extent_ideal_gas(
        T=800.0, p=10e5, n_initial=[1.0, 3.0, 0.0, 0.0])
    print(f"\n  With 3:1 excess steam (1 mol CO + 3 mol H2O):")
    print(f"    CO conversion: {(1 - r2.n_eq[0]/1.0)*100:.2f}%  "
          f"(higher: Le Chatelier)")

    # ---------------------------------------------------------------
    # 2. Methanol synthesis CO + 2 H2 = CH3OH
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Methanol synthesis: CO + 2 H2 = CH3OH")
    print("=" * 70)
    rxn = Reaction.from_names(
        reactants={'CO': 1, 'H2': 2}, products={'CH3OH': 1})
    print(f"\n  {'T (K)':>6s}  {'dH (kJ/mol)':>12s}  {'dG (kJ/mol)':>12s}  "
          f"{'K_eq':>12s}")
    Ts_meoh = [400, 450, 500, 550, 600]
    for T in Ts_meoh:
        print(f"  {T:>6.0f}  {rxn.dH_rxn(T)/1000:>12.3f}  "
              f"{rxn.dG_rxn(T)/1000:>12.3f}  {rxn.K_eq(T):>12.4e}")
    print("  (Strongly exothermic, dH ≈ -97 kJ/mol; pressure helps thermodynamics)")

    # Pressure scan at 500 K
    print("\n  Pressure dependence at 500 K, stoichiometric feed (1 mol CO + 2 mol H2):")
    print(f"    {'p (bar)':>9s}  {'CO conv (%)':>12s}  {'y_CH3OH':>10s}")
    for p_bar in [10, 50, 100, 200, 300]:
        r = rxn.equilibrium_extent_ideal_gas(
            T=500.0, p=p_bar*1e5, n_initial=[1.0, 2.0, 0.0])
        conv = (1 - r.n_eq[0]/1.0) * 100
        print(f"    {p_bar:>9.0f}  {conv:>12.2f}  {r.y_eq[2]:>10.4f}")
    print("  (Higher p increases extent: net mole reduction Δν = -2)")

    # ---------------------------------------------------------------
    # 3. Ammonia synthesis N2 + 3 H2 = 2 NH3
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Haber-Bosch ammonia synthesis: N2 + 3 H2 = 2 NH3")
    print("=" * 70)
    rxn = Reaction.from_names(
        reactants={'N2': 1, 'H2': 3}, products={'NH3': 2})
    print(f"\n  {'T (K)':>6s}  {'dH (kJ/mol)':>12s}  {'dG (kJ/mol)':>12s}  "
          f"{'K_eq':>12s}")
    Ts_nh3 = [400, 500, 600, 700, 800]
    for T in Ts_nh3:
        print(f"  {T:>6.0f}  {rxn.dH_rxn(T)/1000:>12.3f}  "
              f"{rxn.dG_rxn(T)/1000:>12.3f}  {rxn.K_eq(T):>12.4e}")
    print("  (Exothermic, dH ≈ -92 kJ/mol)")

    print("\n  Industrial conditions (T ≈ 700 K, p ≈ 200 bar, stoichiometric):")
    r = rxn.equilibrium_extent_ideal_gas(
        T=700.0, p=200e5, n_initial=[1.0, 3.0, 0.0])
    print(f"    Extent = {r.xi:.4f}")
    print(f"    y_N2 = {r.y_eq[0]:.4f}, y_H2 = {r.y_eq[1]:.4f}, "
          f"y_NH3 = {r.y_eq[2]:.4f}")
    print(f"    N2 conversion: {(1 - r.n_eq[0]/1.0)*100:.2f}%  "
          f"(real plants run lower with recycle)")

    # ---------------------------------------------------------------
    # 4. Plot K_eq vs T for the three reactions
    # ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        T_arr = np.linspace(400, 1200, 81)

        # WGS
        wgs = Reaction.from_names(reactants={'CO': 1, 'H2O': 1},
                                    products={'CO2': 1, 'H2': 1})
        K_wgs = [wgs.K_eq(float(T)) for T in T_arr]

        # Methanol
        meoh = Reaction.from_names(reactants={'CO': 1, 'H2': 2},
                                     products={'CH3OH': 1})
        K_meoh = [meoh.K_eq(float(T)) for T in T_arr]

        # Ammonia
        nh3 = Reaction.from_names(reactants={'N2': 1, 'H2': 3},
                                    products={'NH3': 2})
        K_nh3 = [nh3.K_eq(float(T)) for T in T_arr]

        ax.semilogy(T_arr, K_wgs, 'b-', lw=1.5, label='Water-gas shift')
        ax.semilogy(T_arr, K_meoh, 'g-', lw=1.5, label='Methanol synthesis')
        ax.semilogy(T_arr, K_nh3, 'r-', lw=1.5, label='Ammonia synthesis')
        ax.axhline(1.0, color='k', linestyle=':', alpha=0.4)
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel(r'$K_{eq}$ (dimensionless, $p_{ref}$ = 1 bar)')
        ax.set_title('Equilibrium constants for industrial reactions')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, which='both')
        out = "/mnt/user-data/outputs/reaction_K_eq_vs_T.png"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        print(f"\n  Plot saved to {out}")
    except ImportError:
        print("\n  (matplotlib not available; skipping plot)")


if __name__ == "__main__":
    main()
