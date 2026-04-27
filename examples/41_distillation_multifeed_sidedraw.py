"""Multi-feed distillation with a liquid side-draw.

Demonstrates v0.9.71 features on a benzene/toluene/cumene ternary:

  - Two feeds at different stages.  Stage 4 (high in the column)
    receives a benzene-rich stream; stage 10 (deeper into the
    stripping section) receives a toluene/cumene-rich stream.
  - A liquid side-draw at stage 8 pulls a toluene-rich intermediate
    cut between the rectifying and stripping sections.  The column
    therefore has THREE product outlets: benzene-rich distillate,
    toluene-rich side draw, and cumene-rich bottoms.

The column converges in a handful of Newton iterations and the
mass balance closes to numerical precision on every species.

Studies:
  1. Side-draw flow rate U sweep: how purity in each of the three
     outlets responds to U.
  2. Composition profile across the column at the headline operating
     point (R = 3, D = 28, U_8 = 22).
"""
from __future__ import annotations

import warnings
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stateprop.distillation import distillation_column, FeedSpec
from stateprop.activity.compounds import make_unifac


def antoine(A, B, C):
    """Antoine: P_sat(T_K) -> Pa."""
    def f(T):
        return 10 ** (A - B / ((T - 273.15) + C)) * 133.322
    return f


def main():
    species = ["benzene", "toluene", "cumene"]
    uf = make_unifac(species)
    psats = [
        antoine(6.90565, 1211.033, 220.790),   # benzene
        antoine(6.95464, 1344.800, 219.482),   # toluene
        antoine(6.96292, 1469.677, 207.806),   # cumene
    ]

    # Two feeds: lighter at stage 4, heavier at stage 10
    feed_A = FeedSpec(stage=4,  F=50.0, z=[0.60, 0.30, 0.10])
    feed_B = FeedSpec(stage=10, F=50.0, z=[0.05, 0.45, 0.50])

    # ---------- Study 1: sweep side-draw flow ----------
    print("Study 1: vary liquid side-draw at stage 8")
    print("  U_8    iters | x_D[B,T,C]                     | "
          "x_8[B,T,C]                     | x_B[B,T,C]")
    print("  -----  ----- + ------------------------------ + "
          "------------------------------ + ------------------------------")
    U_list = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
    profiles = []
    for U in U_list:
        # If U=0, omit the dict entry to test that path
        draws = {8: U} if U > 0 else None
        res = distillation_column(
            n_stages=15, feeds=[feed_A, feed_B],
            reflux_ratio=3.0, distillate_rate=28.0, pressure=101325.0,
            liquid_draws=draws,
            species_names=species, activity_model=uf, psat_funcs=psats,
            max_newton_iter=60, newton_tol=1e-7)
        x_8 = res.x[7]   # 1-indexed stage 8 -> Python idx 7
        print(f"  {U:5.1f}  {res.iterations:4d}  | "
              f"{res.x_D[0]:.3f} {res.x_D[1]:.3f} {res.x_D[2]:.3f}            | "
              f"{x_8[0]:.3f} {x_8[1]:.3f} {x_8[2]:.3f}            | "
              f"{res.x_B[0]:.3f} {res.x_B[1]:.3f} {res.x_B[2]:.3f}")
        profiles.append((U, res))

    # ---------- Study 2: headline operating point ----------
    print("\nStudy 2: headline case (R=3, D=28, U_8=22)")
    res = distillation_column(
        n_stages=15, feeds=[feed_A, feed_B],
        reflux_ratio=3.0, distillate_rate=28.0, pressure=101325.0,
        liquid_draws={8: 22.0},
        species_names=species, activity_model=uf, psat_funcs=psats,
        max_newton_iter=60, newton_tol=1e-7)
    print(f"  message: {res.message}")
    print(f"  feeds: F_A at stage {feed_A.stage} (F={feed_A.F}), "
          f"F_B at stage {feed_B.stage} (F={feed_B.F})")
    print(f"  D = {res.D}, B = {res.B}, U_8 = 22")
    print(f"  recovery (benzene -> distillate): "
          f"{res.recovery('benzene', 'distillate'):.4%}")
    print(f"  recovery (toluene -> side draw):  "
          f"{res.recovery('toluene', 'liquid_draw:8'):.4%}")
    print(f"  recovery (cumene  -> bottoms):    "
          f"{res.recovery('cumene', 'bottoms'):.4%}")
    # Total recovery for each species across all three outlets:
    for sp in species:
        rD = res.recovery(sp, "distillate")
        rB = res.recovery(sp, "bottoms")
        rU = res.recovery(sp, "liquid_draw:8")
        print(f"  {sp:7s}: D={rD:.4f}  B={rB:.4f}  U_8={rU:.4f}  "
              f"sum={rD+rB+rU:.10f}")

    # Stage-by-stage composition + L profile
    print()
    print("  Stage |   T[K]   |   L      V    |   x[B]    x[T]    x[C]")
    print("  ------+----------+---------------+-----------------------------")
    for j in range(res.n_stages):
        marker = ""
        if (j + 1) == feed_A.stage:
            marker = "  <- feed A"
        elif (j + 1) == feed_B.stage:
            marker = "  <- feed B"
        elif (j + 1) == 8:
            marker = "  <- U_8 draw"
        print(f"   {j+1:3d}  | {res.T[j]:7.2f}  | "
              f"{res.L[j]:6.2f} {res.V[j]:6.2f}  | "
              f"{res.x[j,0]:.4f}  {res.x[j,1]:.4f}  {res.x[j,2]:.4f}"
              f"{marker}")

    # ---------- Plots ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Plot 1: distillate / side-draw / bottoms purity vs U
    Us = np.array([p[0] for p in profiles])
    xD_T = np.array([p[1].x_D[0] for p in profiles])  # benzene in D
    x8_T = np.array([p[1].x[7, 1] for p in profiles])  # toluene in side draw
    xB_C = np.array([p[1].x_B[2] for p in profiles])  # cumene in bottoms
    axes[0].plot(Us, 100 * xD_T, "o-", label="x_D[benzene]")
    axes[0].plot(Us, 100 * x8_T, "s-", label="x_U[toluene]")
    axes[0].plot(Us, 100 * xB_C, "^-", label="x_B[cumene]")
    axes[0].set_xlabel("liquid side-draw rate U_8 (mol)")
    axes[0].set_ylabel("outlet purity (mol%)")
    axes[0].set_title("Outlet purity vs side-draw rate")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: composition profile at headline
    stages = np.arange(1, res.n_stages + 1)
    axes[1].plot(stages, res.x[:, 0], "o-", label="x[benzene]")
    axes[1].plot(stages, res.x[:, 1], "s-", label="x[toluene]")
    axes[1].plot(stages, res.x[:, 2], "^-", label="x[cumene]")
    axes[1].axvline(feed_A.stage, color="grey", linestyle=":", alpha=0.5)
    axes[1].axvline(feed_B.stage, color="grey", linestyle=":", alpha=0.5)
    axes[1].axvline(8, color="red",  linestyle="--", alpha=0.5)
    axes[1].set_xlabel("stage (1 = top)")
    axes[1].set_ylabel("liquid mole fraction")
    axes[1].set_title("Composition profile (headline)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # Plot 3: L profile showing the steps at feeds and draws
    axes[2].step(stages, res.L, where="mid", linewidth=2, label="L (liquid)")
    axes[2].step(stages, res.V, where="mid", linewidth=2, label="V (vapor)")
    axes[2].axvline(feed_A.stage, color="grey", linestyle=":", alpha=0.5,
                    label="feeds / draw")
    axes[2].axvline(feed_B.stage, color="grey", linestyle=":", alpha=0.5)
    axes[2].axvline(8, color="red", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("stage")
    axes[2].set_ylabel("flow (mol)")
    axes[2].set_title("Internal flow profile")
    axes[2].legend(fontsize=9, loc="best")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = "/tmp/distillation_multifeed_sidedraw.png"
    plt.savefig(out, dpi=110)
    print(f"\nPlot written to {out}")


if __name__ == "__main__":
    main()
