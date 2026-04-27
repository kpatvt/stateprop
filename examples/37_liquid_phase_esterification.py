"""Liquid-phase chemical equilibrium with activity coefficients (v0.9.64).

The canonical example: esterification of acetic acid with ethanol to
produce ethyl acetate and water, catalyzed by sulfuric acid:

    CH3COOH (l) + C2H5OH (l) = CH3COOC2H5 (l) + H2O (l)

K_eq for this reaction is well-tabulated in literature (~4 at 298 K)
and the system shows substantial non-ideality due to the highly polar
species. UNIFAC predicts γ_water ≈ 2.4 and γ_EtOAc ≈ 1.7 at typical
operating compositions, which significantly affects the equilibrium
extent vs. an ideal-solution estimate.

Three demonstrations:
  1. Single-reaction esterification: ideal-solution analytic, ideal-
     solution numerical, and UNIFAC-corrected solutions compared.
  2. Le Chatelier with excess ethanol (drives equilibrium forward).
  3. Multi-reaction: simultaneous AcOH+EtOH and AcOH+MeOH esterifications
     (competing for shared acetic acid).
"""
from __future__ import annotations

import os
import math
import numpy as np
from stateprop.reaction import LiquidPhaseReaction, MultiLiquidPhaseReaction
from stateprop.activity.compounds import make_unifac


class _IdealMix:
    """Trivial activity model for ideal-solution comparison."""
    def __init__(self, N): self.N = N
    def gammas(self, T, x):
        return np.ones(len(x))


def main():
    print("=" * 70)
    print("Liquid-phase esterification: AcOH + EtOH = EtOAc + H2O")
    print("=" * 70)

    rxn = LiquidPhaseReaction(
        species_names=['acetic_acid', 'ethanol', 'ethyl_acetate', 'water'],
        nu=[-1, -1, +1, +1],
        K_eq_298=4.0,         # well-tabulated value
        dH_rxn=-2.3e3,        # mildly exothermic; weak T-dependence
    )
    print(f"\n  K_eq(298 K) = {rxn.K_eq(298.15):.3f}")
    print(f"  K_eq(333 K) = {rxn.K_eq(333.15):.3f}")
    print(f"  K_eq(363 K) = {rxn.K_eq(363.15):.3f}")
    print("  (Mildly exothermic: K decreases as T increases)")

    # ---------------------------------------------------------------
    # 1. Single reaction: ideal vs UNIFAC at 333 K
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1) Equimolar feed at 333 K: ideal solution vs. UNIFAC")
    print("=" * 70)

    K_at_T = rxn.K_eq(333.15)
    # Analytic ideal-solution: xi/(1-xi) = sqrt(K)
    xi_anal = math.sqrt(K_at_T) / (1 + math.sqrt(K_at_T))

    r_id = rxn.equilibrium_extent(T=333.15,
                                     n_initial=[1.0, 1.0, 0.0, 0.0],
                                     activity_model=_IdealMix(4))
    uf = make_unifac(['acetic_acid', 'ethanol', 'ethyl_acetate', 'water'])
    r_uf = rxn.equilibrium_extent(T=333.15,
                                     n_initial=[1.0, 1.0, 0.0, 0.0],
                                     activity_model=uf)

    print(f"\n  K_eq(333 K) = {K_at_T:.3f}")
    print(f"  Ideal-solution analytic:  xi = {xi_anal:.4f}, "
          f"AcOH conversion = {xi_anal:.1%}")
    print(f"  Ideal-solution numerical: xi = {r_id.xi:.4f}, "
          f"AcOH conversion = {r_id.xi:.1%}")
    print(f"  UNIFAC corrected:         xi = {r_uf.xi:.4f}, "
          f"AcOH conversion = {r_uf.xi:.1%}")

    print(f"\n  At UNIFAC equilibrium:")
    for nm, x_i, gi in zip(r_uf.species, r_uf.x_eq, r_uf.gamma_eq):
        print(f"    {nm:<14s}  x = {x_i:.4f}, γ = {gi:.4f}")
    print(f"  K_a from solution = {r_uf.K_a:.4f} (should equal K_eq = "
          f"{r_uf.K_eq:.4f})")
    print(f"\n  UNIFAC predicts LOWER conversion than ideal because the")
    print(f"  products (water, ethyl acetate) have higher γ than the")
    print(f"  reactants -- shifts K_y backward to keep K_a = K_eq.")

    # ---------------------------------------------------------------
    # 2. Le Chatelier: excess EtOH
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2) Le Chatelier: AcOH conversion vs EtOH:AcOH ratio at 333 K")
    print("=" * 70)
    print(f"\n  {'EtOH:AcOH':>10s}  {'xi':>7s}  {'AcOH conv':>10s}  "
          f"{'y_EtOAc':>9s}")
    for ratio in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        r = rxn.equilibrium_extent(T=333.15,
                                      n_initial=[1.0, ratio, 0.0, 0.0],
                                      activity_model=uf)
        if not r.converged:
            print(f"  {ratio:>10.2f}  failed")
            continue
        conv = (1 - r.n_eq[0] / 1.0) * 100
        y_etoac = r.x_eq[2]
        print(f"  {ratio:>10.2f}  {r.xi:>7.4f}  {conv:>9.2f}%  "
              f"{y_etoac:>9.4f}")
    print("\n  Excess ethanol drives equilibrium forward (Le Chatelier).")
    print("  Industrial reactive distillation pushes conversion >95%")
    print("  by removing water as it forms; pure equilibrium (no removal)")
    print("  caps at the values shown here.")

    # ---------------------------------------------------------------
    # 3. Multi-reaction: competing esterifications
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3) Competing esterifications: AcOH+EtOH vs AcOH+MeOH at 333 K")
    print("=" * 70)
    r1 = LiquidPhaseReaction(['acetic_acid', 'ethanol',
                                'ethyl_acetate', 'water'],
                                [-1,-1,+1,+1], K_eq_298=4.0, dH_rxn=-2.3e3)
    r2 = LiquidPhaseReaction(['acetic_acid', 'methanol',
                                'methyl_acetate', 'water'],
                                [-1,-1,+1,+1], K_eq_298=5.0, dH_rxn=-3.5e3)
    system = MultiLiquidPhaseReaction([r1, r2])
    uf6 = make_unifac(list(system.species_names))

    print(f"\n  K_eq^EtOAc(333) = {r1.K_eq(333.15):.3f}")
    print(f"  K_eq^MeOAc(333) = {r2.K_eq(333.15):.3f}")
    print(f"  Methanol esterification is more favorable (smaller MeOH;")
    print(f"  less steric / less Hbond competition with water).")

    print(f"\n  Feed: 2 mol AcOH + 1 mol EtOH + 1 mol MeOH at 333 K")
    r = system.equilibrium(T=333.15,
                              n_initial={'acetic_acid': 2.0,
                                         'ethanol': 1.0,
                                         'methanol': 1.0},
                              activity_model=uf6)
    print(f"  Converged: {r.converged} in {r.iterations} iterations")
    print(f"  Equilibrium composition:")
    for nm, ni, gi in zip(r.species, r.n_eq, r.gamma_eq):
        x_i = ni / r.n_eq.sum()
        print(f"    {nm:<16s}  n = {ni:.4f}, x = {x_i:.4f}, γ = {gi:.4f}")

    n_etoac = r.n_eq[r.species.index('ethyl_acetate')]
    n_meoac = r.n_eq[r.species.index('methyl_acetate')]
    print(f"\n  EtOAc / MeOAc product ratio: {n_etoac/n_meoac:.3f}")
    print(f"  AcOH consumed: {2.0 - r.n_eq[r.species.index('acetic_acid')]:.4f} mol")
    print(f"  Water produced: {r.n_eq[r.species.index('water')]:.4f} mol")
    print(f"  (Water = sum of esters formed -> mass balance check)")

    # Plot 1: K_eq vs T van't Hoff curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        T_arr = np.linspace(298, 393, 30)

        # Subplot 1: conversion vs T at 1:1 feed (ideal vs UNIFAC)
        conv_id = []
        conv_uf = []
        for T in T_arr:
            r_i = rxn.equilibrium_extent(T=float(T),
                                            n_initial=[1.0, 1.0, 0.0, 0.0],
                                            activity_model=_IdealMix(4))
            r_u = rxn.equilibrium_extent(T=float(T),
                                            n_initial=[1.0, 1.0, 0.0, 0.0],
                                            activity_model=uf)
            conv_id.append(r_i.xi * 100 if r_i.converged else np.nan)
            conv_uf.append(r_u.xi * 100 if r_u.converged else np.nan)
        axes[0].plot(T_arr - 273.15, conv_id, 'b--', lw=1.5,
                      label='Ideal solution')
        axes[0].plot(T_arr - 273.15, conv_uf, 'b-', lw=2.0,
                      label='UNIFAC')
        axes[0].set_xlabel("Temperature (°C)")
        axes[0].set_ylabel("AcOH conversion (%)")
        axes[0].set_title("Esterification of AcOH + EtOH (1:1)\nIdeal vs UNIFAC")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 100)

        # Subplot 2: conversion vs feed ratio
        ratios = np.linspace(0.5, 5.0, 25)
        conv_ratio = []
        for r_in in ratios:
            r_u = rxn.equilibrium_extent(T=333.15,
                                            n_initial=[1.0, float(r_in), 0.0, 0.0],
                                            activity_model=uf)
            conv_ratio.append(r_u.xi * 100 if r_u.converged else np.nan)
        axes[1].plot(ratios, conv_ratio, 'g-', lw=2.0)
        axes[1].set_xlabel("EtOH:AcOH feed ratio")
        axes[1].set_ylabel("AcOH conversion (%)")
        axes[1].set_title("Le Chatelier on excess EtOH at 333 K (UNIFAC)")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)
        axes[1].axvline(1.0, color='gray', linestyle=':', alpha=0.5)

        out = "/mnt/user-data/outputs/liquid_phase_esterification.png"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        print(f"\n  Plot saved to {out}")
    except ImportError:
        print("\n  (matplotlib not available; skipping plot)")


if __name__ == "__main__":
    main()
