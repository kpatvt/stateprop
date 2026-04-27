"""Reactive extraction with energy balance — water/acetone/benzene
with a synthetic exothermic reaction in the extract phase.

This example uses a stoichiometrically balanced (chemically synthetic)
reaction so we can exercise the full reactive + energy-balance solver
in a system where the binodal is robust:

    water + acetone  <-->  2 benzene     (synthetic)
    K_eq = 2 * K_a(non-reactive)        (mild perturbation)
    dH_rxn = -10 kJ/mol                 (exothermic)

Real reactive-extraction systems (e.g. AcOH + EtOH -> EtOAc + H2O for
acetic-acid recovery) suffer from the co-solvent effect of ethanol
shrinking the binodal — solvable in principle but condition-dependent.
The synthetic reaction here illustrates the solver's behavior on a
clean LLE substrate.
"""

import numpy as np
import warnings
warnings.simplefilter("ignore", RuntimeWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stateprop.extraction import lle_extraction_column
from stateprop.activity.compounds import make_unifac_lle
from stateprop.reaction.liquid_phase import LiquidPhaseReaction


def main():
    species = ["water", "acetone", "benzene"]
    uf = make_unifac_lle(species)

    # Per-species liquid enthalpies, ref 298.15 K, h_L^* = 0
    T_REF = 298.15
    Cp_L = np.array([75.3, 124.0, 134.0])
    h_L_funcs = [(lambda T, i=i: Cp_L[i] * (T - T_REF)) for i in range(3)]

    base_cfg = dict(
        n_stages=5,
        feed_F=1.0, feed_z=[0.7, 0.3, 0.0],
        solvent_S=2.0, solvent_z=[0.0, 0.0, 1.0],
        species_names=species, activity_model=uf,
        max_newton_iter=60, tol=1e-8,
    )

    # ---- Reference: non-reactive isothermal ----
    print("=" * 60)
    print("Variant 1: non-reactive, isothermal at 298.15 K")
    print("=" * 60)
    r1 = lle_extraction_column(T=298.15, **base_cfg)
    print(f"  converged: {r1.converged} ({r1.iterations} iters)")
    print(f"  acetone recovery: {r1.recovery('acetone'):.3%}")

    # Compute K_a at the non-reactive solution to seed the reaction
    gE = np.asarray(uf.gammas(298.15, r1.x_E[2]))
    a = gE * r1.x_E[2]
    K_a_consistent = float((a[2] ** 2) / (a[0] * a[1] + 1e-30))
    print(f"  K_a (stage 3, extract): {K_a_consistent:.4e}")

    # ---- Energy balance only ----
    print()
    print("=" * 60)
    print("Variant 2: non-reactive + energy balance, "
          "feed_T=305 K, solvent_T=290 K")
    print("=" * 60)
    r2 = lle_extraction_column(
        energy_balance=True, feed_T=305.0, solvent_T=290.0,
        h_L_funcs=h_L_funcs, **base_cfg)
    print(f"  converged: {r2.converged} ({r2.iterations} iters)")
    print(f"  T profile: {np.round(r2.T, 2)}")

    # ---- Reactive only ----
    print()
    print("=" * 60)
    print("Variant 3: reactive in extract, isothermal at 298.15 K")
    print("=" * 60)
    K_eq = 2.0 * K_a_consistent
    rxn = LiquidPhaseReaction(species_names=species,
                              nu=[-1, -1, +2],
                              K_eq_298=K_eq, dH_rxn=-10000.0)
    r3 = lle_extraction_column(
        T=298.15, reactions=[rxn], reactive_stages=[2, 3, 4],
        reaction_phase="E", **base_cfg)
    print(f"  converged: {r3.converged} ({r3.iterations} iters)")
    print(f"  K_eq used: {K_eq:.4e}")
    print(f"  xi profile: {r3.xi.flatten()}")
    print(f"  acetone consumed: {r3.conversion('acetone'):.3%}")

    # ---- Reactive + energy balance ----
    print()
    print("=" * 60)
    print("Variant 4: reactive + energy balance (full)")
    print("=" * 60)
    r4 = lle_extraction_column(
        reactions=[rxn], reactive_stages=[2, 3, 4],
        reaction_phase="E",
        energy_balance=True, feed_T=298.15, solvent_T=298.15,
        h_L_funcs=h_L_funcs, **base_cfg)
    print(f"  converged: {r4.converged} ({r4.iterations} iters)")
    print(f"  T profile: {np.round(r4.T, 3)}")
    print(f"  T max - feed_T: {r4.T.max() - 298.15:.3f} K  "
          "(rxn heat raises stage T)")
    print(f"  xi profile: {r4.xi.flatten()}")
    print(f"  acetone consumed: {r4.conversion('acetone'):.3%}")

    # Per-stage K_a = K_eq(T_j) closure check (van't Hoff applies because
    # dH_rxn != 0)
    max_err = 0.0
    for stage_1idx in [2, 3, 4]:
        j = stage_1idx - 1
        gE = np.asarray(uf.gammas(float(r4.T[j]), r4.x_E[j]))
        a = gE * r4.x_E[j]
        K_a = float((a[2] ** 2) / (a[0] * a[1] + 1e-30))
        K_eq_T = float(np.exp(rxn.ln_K_eq(float(r4.T[j]))))
        rel_err = abs(K_a - K_eq_T) / K_eq_T
        max_err = max(max_err, rel_err)
    print(f"  max |K_a - K_eq(T_j)| / K_eq across reactive stages: "
          f"{max_err:.2e}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    stages = np.arange(1, base_cfg["n_stages"] + 1)

    axes[0].plot(stages, r1.T, "o-", label="iso (V1)")
    axes[0].plot(stages, r2.T, "s-", label="EB only (V2)")
    axes[0].plot(stages, r3.T, "^-", label="reactive iso (V3)")
    axes[0].plot(stages, r4.T, "d-", label="reactive + EB (V4)")
    axes[0].set_xlabel("stage (1 = feed end)")
    axes[0].set_ylabel("T [K]")
    axes[0].set_title("Stage temperatures")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(stages, r1.x_R[:, 1], "o-", label="V1 (iso)")
    axes[1].plot(stages, r3.x_R[:, 1], "^-", label="V3 (reactive iso)")
    axes[1].plot(stages, r4.x_R[:, 1], "d-", label="V4 (full)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("stage")
    axes[1].set_ylabel("x_acetone in raffinate")
    axes[1].set_title("Raffinate acetone profile")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    axes[2].plot(stages, r3.xi.flatten(), "^-", label="V3")
    axes[2].plot(stages, r4.xi.flatten(), "d-", label="V4")
    axes[2].set_xlabel("stage")
    axes[2].set_ylabel("reaction extent xi [mol]")
    axes[2].set_title("Reaction extent")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = "/tmp/reactive_extraction_with_energy_balance.png"
    plt.savefig(out, dpi=110)
    print(f"\nPlot written to {out}")


if __name__ == "__main__":
    main()
