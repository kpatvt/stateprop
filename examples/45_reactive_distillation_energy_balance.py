"""Reactive distillation with energy balance (vs constant-molar-overflow).

Demonstrates v0.9.67's energy-balance N-S solver: drops CMO in favor of
per-stage enthalpy balances. Side-by-side comparison of the same
esterification column run both ways shows how much CMO can mis-state
internal flow rates when reaction heats and feed enthalpies matter.

System:
    AcOH + EtOH <=> EtOAc + H2O    in liquid phase
    UNIFAC for activity coefficients
    8 stages, feed at stage 4, reactive on stages 3-6
    R = 2, D = 0.5 mol/s, F = 1.0 mol/s, p = 1 atm

The column is fed slightly subcooled (335 K vs ~360 K column) so the
energy-balance effect on V/L profiles is measurable, not just symbolic.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stateprop.reaction import reactive_distillation_column
from stateprop.reaction.liquid_phase import LiquidPhaseReaction
from stateprop.activity.compounds import make_unifac


def antoine(A, B, C):
    """log10(p_mmHg) = A - B/(T+C); returns p in Pa."""
    return lambda T: 10.0 ** (A - B / (T - C)) * 133.322


def main():
    species = ["acetic_acid", "ethanol", "ethyl_acetate", "water"]
    psats = [
        antoine(7.55716, 1642.540, 39.764),
        antoine(8.20417, 1642.890, 42.85),
        antoine(7.10179, 1244.95, 55.84),
        antoine(8.07131, 1730.630, 39.574),
    ]
    rxn = LiquidPhaseReaction(
        species_names=species,
        nu=[-1, -1, +1, +1],
        K_eq_298=4.0,
        dH_rxn=-2300.0,
    )
    uf = make_unifac(species)

    # Constant-Cp + h_vap_ref enthalpy model.
    # Reference state: saturated liquid at 298.15 K, h_L^* = 0.
    T_REF = 298.15
    Cp_L = np.array([124.0, 113.0, 170.0, 75.3])             # J/(mol K)
    Cp_V = np.array([67.0, 73.0, 113.0, 34.0])               # J/(mol K)
    h_vap_298 = np.array([23700.0, 42000.0, 35000.0, 44000.0])
    h_L_funcs = [(lambda T, i=i: Cp_L[i] * (T - T_REF)) for i in range(4)]
    h_V_funcs = [(lambda T, i=i: h_vap_298[i] + Cp_V[i] * (T - T_REF))
                 for i in range(4)]

    cfg = dict(
        n_stages=8, feed_stage=4, feed_F=1.0,
        feed_z=[0.5, 0.5, 0.0, 0.0], feed_T=335.0,
        reflux_ratio=2.0, distillate_rate=0.5,
        pressure=101325.0,
        species_names=species,
        activity_model=uf, psat_funcs=psats,
        reactions=[rxn], reactive_stages=[3, 4, 5, 6],
    )

    print("Solving with CMO + bubble-point ...")
    r_cmo = reactive_distillation_column(
        **cfg, method="naphtali_sandholm",
        max_newton_iter=30, newton_tol=1e-7)
    print(f"  CMO converged in {r_cmo.iterations} iters: {r_cmo.message}")

    print("Solving with energy balance ...")
    r_eb = reactive_distillation_column(
        **cfg, method="naphtali_sandholm",
        energy_balance=True,
        h_V_funcs=h_V_funcs, h_L_funcs=h_L_funcs,
        max_newton_iter=40, newton_tol=1e-7)
    print(f"  EB  converged in {r_eb.iterations} iters: {r_eb.message}")

    # Print profiles
    print()
    print("Stage |  CMO T   EB T  | CMO V    EB V  | CMO L    EB L  "
          "|  CMO x_AcOEt  EB x_AcOEt")
    print("-" * 88)
    for j in range(r_cmo.n_stages):
        print(f"  {j+1:2d}  | {r_cmo.T[j]:6.1f}  {r_eb.T[j]:6.1f} "
              f"| {r_cmo.V[j]:6.3f}  {r_eb.V[j]:6.3f} "
              f"| {r_cmo.L[j]:6.3f}  {r_eb.L[j]:6.3f} "
              f"|   {r_cmo.x[j, 2]:6.3f}      {r_eb.x[j, 2]:6.3f}")

    print()
    print(f"Conversion (AcOH):  CMO = {r_cmo.conversion('acetic_acid'):.2%}, "
          f"EB = {r_eb.conversion('acetic_acid'):.2%}")
    print(f"Distillate purity:  CMO x_AcOEt = {r_cmo.x_D[2]:.4f}, "
          f"EB x_AcOEt = {r_eb.x_D[2]:.4f}")
    print(f"Bottoms x_water:    CMO = {r_cmo.x_B[3]:.4f}, "
          f"EB = {r_eb.x_B[3]:.4f}")

    # Plot
    stages = np.arange(1, r_cmo.n_stages + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(stages, r_cmo.T, "o-", label="CMO + bubble-pt")
    axes[0].plot(stages, r_eb.T, "s-", label="energy balance")
    axes[0].set_xlabel("stage")
    axes[0].set_ylabel("T (K)")
    axes[0].set_title("Temperature profile")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(stages, r_cmo.V, "o-", label="CMO V")
    axes[1].plot(stages, r_eb.V, "s-", label="EB V")
    axes[1].plot(stages, r_cmo.L, "o--", label="CMO L")
    axes[1].plot(stages, r_eb.L, "s--", label="EB L")
    axes[1].set_xlabel("stage")
    axes[1].set_ylabel("flow (mol/s)")
    axes[1].set_title("Flow profiles")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    for i, sp in enumerate(species):
        axes[2].plot(stages, r_eb.x[:, i], "-o",
                     label=sp.replace("_", " "), markersize=4)
    axes[2].set_xlabel("stage")
    axes[2].set_ylabel("x (mole fraction)")
    axes[2].set_title("Liquid composition (energy-balance solution)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "reactive_distillation_energy_balance.png")
    plt.savefig(out, dpi=110)
    print(f"\nPlot written to {out}")


if __name__ == "__main__":
    main()
