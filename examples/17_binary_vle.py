"""Binary VLE T-x-y diagram via gamma-phi flash with multiple activity models.

Demonstrates:
  - Compound database (`make_unifac`, `make_uniquac`) by name
  - Building NRTL with hand-fitted parameters
  - Bubble-T scan to construct T-x-y diagram at fixed pressure
  - Comparing UNIFAC, NRTL, UNIQUAC predictions side by side

System: ethanol + water at atmospheric pressure (the canonical
strong-positive-deviation system with a homogeneous azeotrope).
"""
from __future__ import annotations

import os
import numpy as np
from stateprop.activity import (NRTL, AntoinePsat, GammaPhiFlash)
from stateprop.activity.compounds import make_unifac, make_uniquac


def main():
    # Antoine equations for ethanol and water (Reid-Prausnitz-Poling form,
    # log10(p_mmHg) = A - B/(T_C + C); convert to Pa internally via AntoinePsat).
    eth_psat = AntoinePsat(A=5.37229, B=1670.409, C=-40.191)
    wat_psat = AntoinePsat(A=4.6543,  B=1435.264, C=-64.848)
    p_total = 101325.0

    print("=" * 65)
    print("Ethanol-Water VLE diagram at 1 atm")
    print("=" * 65)

    # 1. UNIFAC via compound database
    uf = make_unifac(["ethanol", "water"])

    # 2. UNIQUAC -- compound DB auto-fills r/q from group sums; user supplies b
    #    Standard DECHEMA values for ethanol-water UNIQUAC: b12=42.5, b21=-148.4
    uq = make_uniquac(["ethanol", "water"],
                       b=np.array([[0.0, 42.51], [-148.42, 0.0]]))

    # 3. NRTL -- common literature values for ethanol(1)-water(2)
    alpha = np.array([[0.0, 0.3], [0.3, 0.0]])
    b_nrtl = np.array([[0.0, -55.17], [670.44, 0.0]])
    nrtl = NRTL(alpha=alpha, b=b_nrtl)

    models = [("UNIFAC", uf), ("UNIQUAC", uq), ("NRTL", nrtl)]

    # Sweep liquid compositions
    x_vals = np.linspace(0.01, 0.99, 25)
    print(f"\n  Computing bubble-T at {len(x_vals)} compositions for "
          f"each model...")

    results = {}
    for name, model in models:
        flash = GammaPhiFlash(activity_model=model,
                                psat_funcs=[eth_psat, wat_psat])
        Ts, ys = [], []
        for x_eth in x_vals:
            x = np.array([x_eth, 1.0 - x_eth])
            try:
                r = flash.bubble_t(p=p_total, x=x)
                Ts.append(r.T)
                ys.append(r.y[0])
            except Exception:
                Ts.append(np.nan)
                ys.append(np.nan)
        results[name] = (np.array(Ts), np.array(ys))
        valid = ~np.isnan(Ts)
        T_arr = np.array(Ts)
        print(f"    {name:>8s}: T range {T_arr[valid].min()-273.15:.2f} "
              f"to {T_arr[valid].max()-273.15:.2f} C "
              f"({np.sum(valid)}/{len(Ts)} converged)")

    # Print a comparison table at selected compositions
    print(f"\n{'x_EtOH':>8s}  {'UNIFAC':<22s} {'UNIQUAC':<22s} {'NRTL':<22s}")
    print(f"{'':>8s}  {'(T, y)':<22s} {'(T, y)':<22s} {'(T, y)':<22s}")
    print("-" * 80)
    for x_eth in [0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95]:
        idx = np.argmin(np.abs(x_vals - x_eth))
        line = f"  {x_vals[idx]:.3f}  "
        for name, _ in models:
            T, y = results[name][0][idx], results[name][1][idx]
            if np.isfinite(T):
                line += f"({T-273.15:5.1f} C, {y:.3f})    "
            else:
                line += "(    N/A      )       "
        print(line)

    # Locate the azeotrope (where x = y) for each model
    print(f"\nAzeotrope estimate (where x_EtOH = y_EtOH):")
    for name, _ in models:
        Ts, ys = results[name]
        # Find sign change of (y - x)
        d = ys - x_vals
        for i in range(len(d) - 1):
            if np.isfinite(d[i]) and np.isfinite(d[i+1]) and d[i] * d[i+1] < 0:
                # Linear interpolation for azeotrope
                w = -d[i] / (d[i+1] - d[i])
                x_az = x_vals[i] + w * (x_vals[i+1] - x_vals[i])
                T_az = Ts[i] + w * (Ts[i+1] - Ts[i])
                print(f"  {name:>8s}: x = {x_az:.4f}, T = {T_az-273.15:.2f} C")
                break
        else:
            print(f"  {name:>8s}: no azeotrope found in scanned range")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        colors = {"UNIFAC": "tab:blue", "UNIQUAC": "tab:orange",
                  "NRTL": "tab:green"}
        for name, _ in models:
            Ts, ys = results[name]
            T_C = Ts - 273.15
            axes[0].plot(x_vals, T_C, "-",
                          color=colors[name], label=f"{name} (bubble)")
            axes[0].plot(ys, T_C, "--",
                          color=colors[name], label=f"{name} (dew)")
            axes[1].plot(x_vals, ys, "-", color=colors[name], label=name)

        axes[0].set_xlabel("Mole fraction ethanol")
        axes[0].set_ylabel("Temperature (C)")
        axes[0].set_title("T-x-y at 1 atm")
        axes[0].legend(loc="lower right", fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot([0, 1], [0, 1], "k:", alpha=0.5)
        axes[1].set_xlabel("x_EtOH (liquid)")
        axes[1].set_ylabel("y_EtOH (vapor)")
        axes[1].set_title("y-x diagram at 1 atm")
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)

        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "binary_vle_ethanol_water.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        print(f"\n  Plot saved to {out}")
    except ImportError:
        print("\n  (matplotlib not available; skipping plot)")


if __name__ == "__main__":
    main()
