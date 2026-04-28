"""Specified vapor-fraction flash for flash-drum sizing (v0.9.56).

Demonstrates the new Pα and Tα flash modes that solve for the operating
T (or p) at which a mixture is exactly fraction α vaporized. These
generalize the bubble-point (α=0) and dew-point (α=1) calculations and
are the natural primitives for:
  - Sizing flash drums for partial vaporization
  - Constructing T-x-y and P-x-y diagrams (sweep α from 0 to 1)
  - Building constant-pressure phase envelope curves

System: methane + n-butane + n-decane ternary (lean gas with heavy
fraction; produces a wide phase envelope).
"""
from __future__ import annotations

import os
import numpy as np
from stateprop.cubic import (PR, CubicMixture,
                                flash_p_alpha, flash_t_alpha,
                                newton_bubble_point_p, newton_dew_point_p,
                                newton_bubble_point_T, newton_dew_point_T)


def main():
    methane = PR(T_c=190.564, p_c=4.5992e6, acentric_factor=0.01142)
    ethane  = PR(T_c=305.32,  p_c=4.872e6,  acentric_factor=0.099)
    propane = PR(T_c=369.83,  p_c=4.248e6,  acentric_factor=0.152)
    z = np.array([0.6, 0.3, 0.1])
    mx = CubicMixture([methane, ethane, propane], composition=z)

    print("=" * 70)
    print("Specified vapor-fraction flash (v0.9.56) for flash-drum sizing")
    print("=" * 70)
    print(f"  Composition: CH4={z[0]:.2f}, C2H6={z[1]:.2f}, C3H8={z[2]:.2f}")

    # ---------------------------------------------------------------
    # P-alpha sweep: at fixed pressure, find T for each alpha
    # ---------------------------------------------------------------
    p_op = 20e5
    print(f"\n  At fixed p = {p_op/1e5:.1f} bar, sweep alpha:\n")
    print(f"    {'alpha':>6s}  {'T (K)':>8s}  {'beta':>8s}  "
          f"{'phase':<12s}")
    print("    " + "-" * 40)
    Ts_pa = []
    alphas = np.linspace(0.0, 1.0, 11)
    for alpha in alphas:
        try:
            r = flash_p_alpha(p_op, float(alpha), z, mx)
            beta = r.beta if r.beta is not None else (
                1.0 if r.phase == "vapor" else 0.0)
            Ts_pa.append(r.T)
            print(f"    {alpha:>6.2f}  {r.T:>8.2f}  {beta:>8.4f}  {r.phase:<12s}")
        except Exception as e:
            Ts_pa.append(np.nan)
            print(f"    {alpha:>6.2f}  failed: {str(e)[:50]}")

    # ---------------------------------------------------------------
    # T-alpha sweep: at fixed temperature, find p for each alpha
    # ---------------------------------------------------------------
    T_op = 250.0
    print(f"\n  At fixed T = {T_op:.1f} K, sweep alpha:\n")
    print(f"    {'alpha':>6s}  {'p (bar)':>10s}  {'beta':>8s}  "
          f"{'phase':<12s}")
    print("    " + "-" * 42)
    ps_ta = []
    for alpha in alphas:
        try:
            r = flash_t_alpha(T_op, float(alpha), z, mx)
            beta = r.beta if r.beta is not None else (
                1.0 if r.phase == "vapor" else 0.0)
            ps_ta.append(r.p)
            print(f"    {alpha:>6.2f}  {r.p/1e5:>10.3f}  {beta:>8.4f}  "
                  f"{r.phase:<12s}")
        except Exception as e:
            ps_ta.append(np.nan)
            print(f"    {alpha:>6.2f}  failed: {str(e)[:50]}")

    # ---------------------------------------------------------------
    # Engineering example: design a 50%-vaporization flash drum
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Flash-drum design example: 50% vaporization at 20 bar")
    print("=" * 70)
    r = flash_p_alpha(p_op, 0.5, z, mx)
    print(f"\n  Operating temperature:  T = {r.T:.2f} K  ({r.T - 273.15:.2f} C)")
    print(f"  Vapor product (mole fraction):")
    for i, name in enumerate(["CH4", "C2H6", "C3H8"]):
        print(f"    y_{name:<5s} = {r.y[i]:.4f}")
    print(f"  Liquid product (mole fraction):")
    for i, name in enumerate(["CH4", "C2H6", "C3H8"]):
        print(f"    x_{name:<5s} = {r.x[i]:.4f}")

    # K-values
    K = r.y / r.x
    print(f"\n  K-values (y/x): "
          f"K_CH4={K[0]:.3f}, K_C2={K[1]:.3f}, K_C3={K[2]:.3f}")
    print(f"  Light component (CH4) goes to vapor; heavy (C3) to liquid.")

    # ---------------------------------------------------------------
    # Plot: phase envelope from alpha sweeps
    # ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Build envelope by sweeping (T, p_bub) and (T, p_dew)
        T_range = np.linspace(150.0, 360.0, 30)
        p_bub_arr = []
        p_dew_arr = []
        T_bub_arr = []
        T_dew_arr = []
        for T in T_range:
            try:
                rb = newton_bubble_point_p(float(T), z, mx)
                p_bub_arr.append(rb.p)
                T_bub_arr.append(T)
            except Exception:
                pass
            try:
                rd = newton_dew_point_p(float(T), z, mx, p_init=20e5)
                p_dew_arr.append(rd.p)
                T_dew_arr.append(T)
            except Exception:
                pass

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(T_bub_arr, np.array(p_bub_arr)/1e5, "b-",
                 lw=1.5, label="Bubble curve (alpha=0)")
        ax.plot(T_dew_arr, np.array(p_dew_arr)/1e5, "r-",
                 lw=1.5, label="Dew curve (alpha=1)")
        # Mark P-alpha sweep at p=20 bar
        valid = ~np.isnan(Ts_pa)
        ax.plot(np.asarray(Ts_pa)[valid],
                 np.full(np.sum(valid), p_op/1e5),
                 "go", markersize=6, label=f"P-alpha sweep at p={p_op/1e5:.0f} bar")
        # Mark T-alpha sweep at T=250 K
        valid = ~np.isnan(ps_ta)
        ax.plot(np.full(np.sum(valid), T_op),
                 np.asarray(ps_ta)[valid] / 1e5,
                 "ms", markersize=6, label=f"T-alpha sweep at T={T_op:.0f} K")

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Pressure (bar)")
        ax.set_title("CH4/C2H6/C3H8 (0.6/0.3/0.1) -- "
                       "P-alpha and T-alpha flash sweeps")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "partial_vaporization_envelope.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        print(f"\n  Plot saved to {out}")
    except ImportError:
        print("\n  (matplotlib not available; skipping plot)")


if __name__ == "__main__":
    main()
