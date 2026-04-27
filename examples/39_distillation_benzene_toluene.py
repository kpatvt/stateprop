"""Non-reactive benzene/toluene distillation at 1 atm.

Demonstrates the v0.9.70 distillation_column API on a textbook
binary case.  Studies:
  1. Distillate purity vs reflux ratio (R_min approach)
  2. Distillate purity vs number of stages
  3. Composition profile across the column at the headline operating
     point.
"""
from __future__ import annotations

import warnings
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stateprop.distillation import distillation_column
from stateprop.activity.compounds import make_unifac


def antoine(A, B, C):
    """Antoine equation P_sat(T_K) returning Pa."""
    def f(T):
        return 10 ** (A - B / ((T - 273.15) + C)) * 133.322
    return f


def main():
    species = ["benzene", "toluene"]
    uf = make_unifac(species)
    psats = [
        antoine(6.90565, 1211.033, 220.790),   # benzene
        antoine(6.95464, 1344.800, 219.482),   # toluene
    ]

    base_kwargs = dict(
        feed_F=100.0, feed_z=[0.5, 0.5], feed_T=355.0,
        distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf,
        psat_funcs=psats,
    )

    # ---------- Study 1: purity vs reflux ratio ----------
    print("Study 1: 12-stage column, feed at stage 6, distillate purity vs R")
    R_list = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0]
    purities_R = []
    for R in R_list:
        res = distillation_column(
            n_stages=12, feed_stage=6, reflux_ratio=R,
            **base_kwargs)
        purities_R.append(res.x_D[0])
        print(f"  R={R:5.2f}  x_D[benzene]={res.x_D[0]:.4f}  "
              f"recovery={res.recovery('benzene','distillate'):.4f}  "
              f"iters={res.iterations}")

    # ---------- Study 2: purity vs stages ----------
    print("\nStudy 2: R=2.0 fixed, distillate purity vs n_stages")
    n_list = [4, 6, 8, 10, 12, 16, 20, 30]
    purities_N = []
    for n in n_list:
        feed = max(2, n // 2)
        res = distillation_column(
            n_stages=n, feed_stage=feed, reflux_ratio=2.0,
            **base_kwargs)
        purities_N.append(res.x_D[0])
        print(f"  N={n:2d}  feed_stage={feed:2d}  "
              f"x_D[benzene]={res.x_D[0]:.5f}  iters={res.iterations}")

    # ---------- Headline case: composition + T profile ----------
    res = distillation_column(
        n_stages=15, feed_stage=8, reflux_ratio=2.5,
        **base_kwargs)
    print(f"\nHeadline case: 15 stages, feed at 8, R=2.5")
    print(f"  message:  {res.message}")
    print(f"  x_D[benzene]={res.x_D[0]:.5f}  "
          f"x_B[toluene]={res.x_B[1]:.5f}")
    print(f"  benzene recovery to D = "
          f"{res.recovery('benzene','distillate'):.4%}")
    print(f"  T[top]={res.T[0]:.2f} K   T[bottom]={res.T[-1]:.2f} K")
    print()
    print("  Stage |   T[K]   |  L       V       |  x[B]   x[T]   |  y[B]   y[T]")
    print("  ------+----------+------------------+----------------+----------------")
    for j in range(res.n_stages):
        print(f"   {j+1:3d}  | {res.T[j]:7.2f}  | "
              f"{res.L[j]:6.2f}  {res.V[j]:6.2f}  | "
              f"{res.x[j,0]:.4f}  {res.x[j,1]:.4f}  | "
              f"{res.y[j,0]:.4f}  {res.y[j,1]:.4f}")

    # ---------- Plots ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(R_list, [100 * p for p in purities_R], "o-")
    axes[0].set_xlabel("reflux ratio R = L/D")
    axes[0].set_ylabel("x_D[benzene] (mol%)")
    axes[0].set_title("Distillate purity vs R\n(N=12, feed_stage=6)")
    axes[0].set_xscale("log")
    axes[0].grid(alpha=0.3, which="both")

    axes[1].plot(n_list, [100 * p for p in purities_N], "s-")
    axes[1].set_xlabel("number of equilibrium stages")
    axes[1].set_ylabel("x_D[benzene] (mol%)")
    axes[1].set_title("Distillate purity vs N\n(R=2.0, feed at midpoint)")
    axes[1].grid(alpha=0.3)

    stages = np.arange(1, res.n_stages + 1)
    axes[2].plot(stages, res.x[:, 0], "o-",  label="x[benzene] (liquid)")
    axes[2].plot(stages, res.y[:, 0], "s-",  label="y[benzene] (vapor)")
    axes[2].set_xlabel("stage (1 = top, condenser side)")
    axes[2].set_ylabel("benzene mole fraction")
    axes[2].set_title("Composition profile\n(N=15, feed at 8, R=2.5)")
    axes[2].legend(loc="best", fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = "/tmp/distillation_benzene_toluene.png"
    plt.savefig(out, dpi=110)
    print(f"\nPlot written to {out}")


if __name__ == "__main__":
    main()
