"""Liquid-liquid extraction: water + acetone + benzene at 298 K.

Demonstrates v0.9.68's countercurrent extraction column solver.
Acetone partitions out of an aqueous feed (carrier=water) into a
benzene solvent stream.  At 298 K with UNIFAC-LLE,
K_acetone = x_E/x_R ~ 9 favors the benzene phase strongly.

Two studies:
  1. Recovery vs number of equilibrium stages (fixed S/F = 1).
  2. Acetone-stripping curve: raffinate purity vs solvent ratio (fixed
     n_stages = 5).
"""

import os
import numpy as np
import warnings
warnings.simplefilter("ignore", RuntimeWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stateprop.extraction import lle_extraction_column
from stateprop.activity.compounds import make_unifac_lle


def main():
    species = ["water", "acetone", "benzene"]
    uf = make_unifac_lle(species)

    # ---------- Study 1: recovery vs n_stages ----------
    n_list = [1, 2, 3, 4, 5, 6, 8]
    rec_n = []
    for n in n_list:
        res = lle_extraction_column(
            n_stages=n,
            feed_F=1.0, feed_z=[0.85, 0.15, 0.0],
            solvent_S=1.0, solvent_z=[0.0, 0.0, 1.0],
            T=298.15, species_names=species, activity_model=uf,
            max_newton_iter=50, tol=1e-7)
        rec_n.append((n, res.recovery("acetone"), res.x_R[-1, 1]))
    print("Study 1: recovery vs n_stages (S/F = 1, x_acetone^F = 0.15)")
    print(f"  n_stages  acetone recovery   x_acetone in raffinate")
    for n, R, xr in rec_n:
        print(f"     {n:2d}        {R:7.2%}            {xr:.4e}")

    # ---------- Study 2: raffinate purity vs S/F ----------
    SF_list = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    rec_S = []
    for SF in SF_list:
        res = lle_extraction_column(
            n_stages=5,
            feed_F=1.0, feed_z=[0.85, 0.15, 0.0],
            solvent_S=SF, solvent_z=[0.0, 0.0, 1.0],
            T=298.15, species_names=species, activity_model=uf,
            max_newton_iter=50, tol=1e-7)
        rec_S.append((SF, res.recovery("acetone"), res.x_R[-1, 1]))
    print("\nStudy 2: recovery vs S/F (n_stages = 5, x_acetone^F = 0.15)")
    print(f"  S/F     acetone recovery   x_acetone in raffinate")
    for SF, R, xr in rec_S:
        print(f"  {SF:4.2f}     {R:7.2%}            {xr:.4e}")

    # ---------- Composition profile for the headline case ----------
    res_main = lle_extraction_column(
        n_stages=6,
        feed_F=1.0, feed_z=[0.85, 0.15, 0.0],
        solvent_S=1.5, solvent_z=[0.0, 0.0, 1.0],
        T=298.15, species_names=species, activity_model=uf,
        max_newton_iter=50, tol=1e-7)
    print("\nHeadline case: 6-stage column, F=1, S=1.5")
    print(f"  message: {res_main.message}")
    print(f"  acetone recovery: {res_main.recovery('acetone'):.2%}")
    print()
    print("Stage |   R         E    | x_R [w, ac, bz]      "
          "| x_E [w, ac, bz]")
    print("------+------------------+----------------------"
          "+----------------------")
    for j in range(res_main.n_stages):
        print(f"  {j}   | {res_main.R[j]:7.4f}  {res_main.E[j]:7.4f} "
              f"| {res_main.x_R[j,0]:.4f} {res_main.x_R[j,1]:.4f} "
              f"{res_main.x_R[j,2]:.4f}  "
              f"| {res_main.x_E[j,0]:.4f} {res_main.x_E[j,1]:.4f} "
              f"{res_main.x_E[j,2]:.4f}")

    # ---------- Plots ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    n_arr = np.array([t[0] for t in rec_n])
    rec_arr = np.array([t[1] for t in rec_n])
    axes[0].plot(n_arr, 100 * rec_arr, "o-")
    axes[0].set_xlabel("number of equilibrium stages")
    axes[0].set_ylabel("acetone recovery (%)")
    axes[0].set_title("Recovery vs n_stages (S/F = 1)")
    axes[0].grid(alpha=0.3)

    sf_arr = np.array([t[0] for t in rec_S])
    rec_S_arr = np.array([t[1] for t in rec_S])
    axes[1].plot(sf_arr, 100 * rec_S_arr, "s-")
    axes[1].set_xlabel("solvent / feed (mol/mol)")
    axes[1].set_ylabel("acetone recovery (%)")
    axes[1].set_title("Recovery vs S/F (n_stages = 5)")
    axes[1].grid(alpha=0.3)

    stages = np.arange(1, res_main.n_stages + 1)
    axes[2].plot(stages, res_main.x_R[:, 1], "o-",
                 label="x_acetone (raffinate)")
    axes[2].plot(stages, res_main.x_E[:, 1], "s-",
                 label="x_acetone (extract)")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("stage (1 = feed end)")
    axes[2].set_ylabel("acetone mole fraction")
    axes[2].set_title("Acetone profile across the column")
    axes[2].legend()
    axes[2].grid(alpha=0.3, which="both")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "lle_extraction_acetone_recovery.png")
    plt.savefig(out, dpi=110)
    print(f"\nPlot written to {out}")


if __name__ == "__main__":
    main()
