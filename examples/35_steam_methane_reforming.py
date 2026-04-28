"""Steam methane reforming — multi-reaction equilibrium (v0.9.62).

Demonstrates the new MultiReaction class for solving simultaneously-
coupled reactions. Steam methane reforming is the canonical example
because it requires TWO linearly-independent reactions to span the
C-H-O reaction space:

  R1:  CH4 + H2O = CO + 3 H2     (steam reforming, endothermic, +226 kJ/mol)
  R2:  CO + H2O = CO2 + H2       (water-gas shift, exothermic, -34 kJ/mol)

The combined behavior is non-monotonic in T:
  - Low T: R1 reactant-favored, R2 product-favored. Limited reforming
    but high WGS conversion -> CO2-rich syngas.
  - High T: R1 product-favored, R2 less favorable. Complete reforming
    but more CO -> CO-rich syngas.

The H2/CO ratio is tunable via T, p, and steam-to-carbon (S/C) ratio,
which is the key engineering knob for matching downstream uses (CO-rich
for Fischer-Tropsch, H2-rich for ammonia, balanced for methanol).
"""
from __future__ import annotations

import os
import numpy as np
from stateprop.reaction import MultiReaction


def main():
    print("=" * 70)
    print("Steam methane reforming: 2-reaction equilibrium")
    print("=" * 70)

    system = MultiReaction.from_specs([
        # R1: steam reforming
        {'reactants': {'CH4': 1, 'H2O': 1}, 'products': {'CO': 1, 'H2': 3}},
        # R2: water-gas shift
        {'reactants': {'CO': 1, 'H2O': 1},  'products': {'CO2': 1, 'H2': 1}},
    ])
    print(f"\n  Species: {system.species_names}")
    print(f"  Stoichiometry matrix (rows = reactions, cols = species):")
    print(f"    {'':<12s} " + ' '.join(f"{s:>5s}" for s in system.species_names))
    for r in range(system.R):
        label = ['R1 reform', 'R2 WGS'][r]
        row = '  '.join(f"{int(system.nu[r, j]):>5d}" if abs(system.nu[r, j]) > 1e-9
                          else f"{0:>5d}" for j in range(system.N))
        print(f"    {label:<12s} {row}")

    # Temperature sweep at S/C = 3, p = 1 bar
    print("\n" + "=" * 70)
    print("Temperature sweep at p=1 bar, S/C=3")
    print("=" * 70)
    print(f"\n  {'T (K)':>6s}  {'CH4 conv':>9s}  "
          f"{'H2/CO':>7s}  {'H2 yield':>9s}  {'y_CO':>8s}  {'y_CO2':>8s}  "
          f"{'y_H2':>8s}")
    Ts = [700, 800, 900, 1000, 1100, 1200]
    Ts_arr, conv_arr, h2_arr, co_arr, co2_arr, h2_co_arr = [], [], [], [], [], []
    for T in Ts:
        r = system.equilibrium_ideal_gas(
            T=float(T), p=1e5, n_initial={'CH4': 1.0, 'H2O': 3.0})
        if not r.converged:
            print(f"  {T:>6.0f}  failed: {r.message[:50]}")
            continue
        n = {name: r.n_eq[r.species.index(name)] for name in r.species}
        conv = 1.0 - n['CH4'] / 1.0
        h2_co = n['H2'] / n['CO'] if n['CO'] > 1e-9 else float('inf')
        Ts_arr.append(T)
        conv_arr.append(conv)
        h2_arr.append(n['H2'])
        co_arr.append(r.y_eq[r.species.index('CO')])
        co2_arr.append(r.y_eq[r.species.index('CO2')])
        h2_co_arr.append(h2_co)
        y_H2 = r.y_eq[r.species.index('H2')]
        print(f"  {T:>6.0f}  {conv:>9.3%}  {h2_co:>7.2f}  {n['H2']:>9.3f}  "
              f"{co_arr[-1]:>8.4f}  {co2_arr[-1]:>8.4f}  {y_H2:>8.4f}")

    # S/C sweep at fixed T, p
    print("\n" + "=" * 70)
    print("Steam-to-carbon (S/C) sweep at T=1100 K, p=1 bar")
    print("=" * 70)
    print(f"\n  {'S/C':>5s}  {'CH4 conv':>9s}  {'H2 yield':>9s}  "
          f"{'H2/CO':>7s}  {'C in CO2':>9s}")
    for SC in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        r = system.equilibrium_ideal_gas(
            T=1100.0, p=1e5, n_initial={'CH4': 1.0, 'H2O': float(SC)})
        if not r.converged:
            print(f"  {SC:>5.1f}  failed: {r.message[:50]}")
            continue
        n = {name: r.n_eq[r.species.index(name)] for name in r.species}
        conv = 1.0 - n['CH4'] / 1.0
        h2_co = n['H2'] / n['CO'] if n['CO'] > 1e-9 else float('inf')
        c_co2 = n['CO2'] / (n['CO'] + n['CO2'])
        print(f"  {SC:>5.2f}  {conv:>9.3%}  {n['H2']:>9.3f}  {h2_co:>7.2f}  "
              f"{c_co2:>9.3f}")
    print("\n  Higher S/C drives more H2 (extra steam shifts WGS forward)")
    print("  but lowers thermal efficiency; industrial S/C ~ 2.5-3.5.")

    # Pressure sweep
    print("\n" + "=" * 70)
    print("Pressure sweep at T=1100 K, S/C=3")
    print("=" * 70)
    print(f"\n  {'p (bar)':>9s}  {'CH4 conv':>9s}  {'H2 yield':>9s}  "
          f"{'y_CH4 unconv':>13s}")
    for p_bar in [1, 5, 10, 30, 50]:
        r = system.equilibrium_ideal_gas(
            T=1100.0, p=p_bar*1e5, n_initial={'CH4': 1.0, 'H2O': 3.0})
        if not r.converged:
            print(f"  {p_bar:>9.0f}  failed: {r.message[:50]}")
            continue
        n = {name: r.n_eq[r.species.index(name)] for name in r.species}
        conv = 1.0 - n['CH4'] / 1.0
        y_ch4 = r.y_eq[r.species.index('CH4')]
        print(f"  {p_bar:>9.0f}  {conv:>9.3%}  {n['H2']:>9.3f}  {y_ch4:>13.5f}")
    print("\n  Higher p suppresses reforming (R1 has dn = +2; Le Chatelier)")
    print("  Industrial reformers run at 20-40 bar to balance equipment")
    print("  cost vs. thermodynamic penalty.")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Subplot 1: Mole fractions vs T
        T_dense = np.linspace(700, 1200, 51)
        comp = {name: [] for name in system.species_names}
        for T in T_dense:
            r = system.equilibrium_ideal_gas(
                T=float(T), p=1e5, n_initial={'CH4': 1.0, 'H2O': 3.0})
            if r.converged:
                for name in system.species_names:
                    comp[name].append(r.y_eq[r.species.index(name)])
            else:
                for name in system.species_names:
                    comp[name].append(np.nan)

        colors = {'CH4': 'tab:blue', 'H2O': 'tab:cyan',
                   'CO': 'tab:orange', 'H2': 'tab:green',
                   'CO2': 'tab:red'}
        for name in ['CH4', 'H2O', 'CO', 'H2', 'CO2']:
            axes[0].plot(T_dense, comp[name], '-', color=colors[name],
                          lw=1.5, label=name)
        axes[0].set_xlabel("Temperature (K)")
        axes[0].set_ylabel("Equilibrium mole fraction")
        axes[0].set_title("Steam reforming equilibrium\n(p=1 bar, S/C=3)")
        axes[0].legend(loc='center right', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Subplot 2: H2/CO ratio vs T
        H2_arr = np.array(comp['H2'])
        CO_arr = np.array(comp['CO'])
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = H2_arr / CO_arr
        axes[1].semilogy(T_dense, ratio, 'b-', lw=2)
        axes[1].axhline(2.0, color='gray', linestyle=':', alpha=0.5,
                         label='H2/CO=2 (methanol target)')
        axes[1].axhline(3.0, color='gray', linestyle='--', alpha=0.5,
                         label='H2/CO=3 (Fischer-Tropsch)')
        axes[1].set_xlabel("Temperature (K)")
        axes[1].set_ylabel("H2/CO mole ratio")
        axes[1].set_title("Syngas H2/CO ratio")
        axes[1].legend(loc='upper right', fontsize=9)
        axes[1].grid(True, alpha=0.3, which='both')

        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "steam_reforming_equilibrium.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        print(f"\n  Plot saved to {out}")
    except ImportError:
        print("\n  (matplotlib not available; skipping plot)")


if __name__ == "__main__":
    main()
