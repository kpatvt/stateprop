"""Q-fraction feeds and partial condensers — v0.9.72 demo.

Two studies on the benzene/toluene/cumene ternary:

  Study 1 — q-fraction sweep at fixed column geometry.
            Sweep feed_q from 0.0 (saturated vapor) to 1.0 (saturated
            liquid) and observe how the L/V profile and the distillate
            purity respond.

  Study 2 — Total vs partial condenser comparison.
            Compare a 13-stage total-condenser column (12 trays +
            reboiler) against a 14-stage partial-condenser column (1
            condenser + 12 trays + reboiler) at the same trays and
            operating conditions.  The partial column has an extra
            equilibrium stage and so achieves a marginally better
            separation; the vapor distillate composition equals
            y_0 = K_0 * x_0 in the partial case.
"""
from __future__ import annotations
import os

import warnings
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stateprop.distillation import distillation_column, FeedSpec
from stateprop.activity.compounds import make_unifac


def antoine(A, B, C):
    return lambda T: 10 ** (A - B / ((T - 273.15) + C)) * 133.322


def main():
    species = ["benzene", "toluene", "cumene"]
    uf = make_unifac(species)
    psats = [
        antoine(6.90565, 1211.033, 220.790),
        antoine(6.95464, 1344.800, 219.482),
        antoine(6.96292, 1469.677, 207.806),
    ]

    base = dict(
        n_stages=12, feed_stage=6,
        feed_F=100.0, feed_z=[0.6, 0.3, 0.1], feed_T=370.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)

    # ---------- Study 1: q-fraction sweep ----------
    print("Study 1: q-fraction sweep (q from 0 to 1)")
    print("  q     iters | x_D[B,T,C]                | L below feed | V above feed | V below feed")
    print("  ----  ----- + ------------------------- + ------------ + ------------ + ------------")
    q_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    profiles = []
    for q in q_vals:
        res = distillation_column(**base, feed_q=q)
        L_below = res.L[6]   # below the feed at stage 6
        V_above = res.V[4]   # above the feed
        V_below = res.V[6]   # below the feed
        print(f"  {q:.2f}  {res.iterations:4d}  | "
              f"{res.x_D[0]:.3f} {res.x_D[1]:.3f} {res.x_D[2]:.3f}        | "
              f"{L_below:6.1f}       | {V_above:6.1f}       | {V_below:6.1f}")
        profiles.append((q, res))

    # ---------- Study 2: total vs partial condenser ----------
    print()
    print("Study 2: total vs partial condenser (same physical column)")
    base2 = dict(
        feed_F=100.0, feed_z=[0.6, 0.3, 0.1], feed_T=370.0,
        reflux_ratio=2.0, distillate_rate=50.0, pressure=101325.0,
        species_names=species, activity_model=uf, psat_funcs=psats)
    # 13-stage total: 12 trays + reboiler (stage 1 = top tray, stage 13 = reboiler)
    res_total = distillation_column(
        n_stages=13, feed_stage=7, condenser="total", **base2)
    # 14-stage partial: 1 partial condenser + 12 trays + reboiler
    # (stage 1 = condenser, stage 8 = the same physical tray as stage 7 in total)
    res_partial = distillation_column(
        n_stages=14, feed_stage=8, condenser="partial", **base2)

    print(f"  total   (n=13, feed at 7):  iters={res_total.iterations}, "
          f"x_D[B]={res_total.x_D[0]:.5f},  T_top={res_total.T[0]:.2f} K  "
          f"(distillate is LIQUID)")
    print(f"  partial (n=14, feed at 8):  iters={res_partial.iterations}, "
          f"y_D[B]={res_partial.x_D[0]:.5f},  T_cond={res_partial.T[0]:.2f} K "
          f" (distillate is VAPOR)")
    diff = res_partial.x_D[0] - res_total.x_D[0]
    print(f"  difference: partial gives {diff:+.5f} ({diff*100:+.3f} pp) "
          f"in benzene purity")

    # ---------- Plots ----------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Plot 1: L profile vs stage for several q values
    stages = np.arange(1, 13)
    for q, res in profiles:
        axes[0, 0].step(stages, res.L, where="mid", label=f"q={q}")
    axes[0, 0].axvline(6, color="grey", linestyle="--", alpha=0.5,
                       label="feed stage")
    axes[0, 0].set_xlabel("stage")
    axes[0, 0].set_ylabel("L (mol)")
    axes[0, 0].set_title("Liquid flow profile vs q-fraction")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: V profile vs stage for several q values
    for q, res in profiles:
        axes[0, 1].step(stages, res.V, where="mid", label=f"q={q}")
    axes[0, 1].axvline(6, color="grey", linestyle="--", alpha=0.5,
                       label="feed stage")
    axes[0, 1].set_xlabel("stage")
    axes[0, 1].set_ylabel("V (mol)")
    axes[0, 1].set_title("Vapor flow profile vs q-fraction")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: distillate purities vs q
    qs = np.array([p[0] for p in profiles])
    xD_B = np.array([p[1].x_D[0] for p in profiles])
    xD_T = np.array([p[1].x_D[1] for p in profiles])
    axes[1, 0].plot(qs, xD_B, "o-", label="x_D[benzene]", linewidth=2)
    axes[1, 0].plot(qs, xD_T, "s-", label="x_D[toluene]", linewidth=2)
    axes[1, 0].set_xlabel("feed q-fraction (1=sat liquid, 0=sat vapor)")
    axes[1, 0].set_ylabel("distillate composition")
    axes[1, 0].set_title("Distillate purity vs q-fraction")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: composition profiles for total vs partial
    s_t = np.arange(1, res_total.n_stages + 1)
    s_p = np.arange(1, res_partial.n_stages + 1)
    axes[1, 1].plot(s_t, res_total.x[:, 0], "o-", label="total: x[benzene]")
    axes[1, 1].plot(s_p, res_partial.x[:, 0], "s--",
                    label="partial: x[benzene]", alpha=0.8)
    axes[1, 1].plot(s_t, res_total.x[:, 1], "^-", label="total: x[toluene]")
    axes[1, 1].plot(s_p, res_partial.x[:, 1], "v--",
                    label="partial: x[toluene]", alpha=0.8)
    axes[1, 1].set_xlabel("stage")
    axes[1, 1].set_ylabel("liquid mole fraction")
    axes[1, 1].set_title("Composition profile: total vs partial condenser")
    axes[1, 1].legend(fontsize=8, loc="best")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "distillation_q_fraction_partial_condenser.png")
    plt.savefig(out, dpi=110)
    print(f"\nPlot written to {out}")


if __name__ == "__main__":
    main()
