"""Cubic-EOS phase envelope and mixture critical point.

Demonstrates:
  - Building a multicomponent cubic mixture (PR with binary interaction params)
  - Tracing the full phase envelope from triple to critical
  - Computing the mixture critical point via Heidemann-Khalil
  - Comparing PR vs SRK on the same envelope
  - Plotting envelope in P-T coordinates with the critical point marked

System: methane + n-butane + n-decane ternary, a hydrocarbon mixture
that produces a wide envelope due to the heavy fraction.
"""
from __future__ import annotations

import numpy as np
import os
from stateprop.cubic import (PR, SRK, CubicMixture,
                                trace_envelope, envelope_point, critical_point)


def main():
    # 3-component mixture: lean gas + heavy condensate
    methane = PR(T_c=190.564, p_c=4.5992e6, acentric_factor=0.01142)
    butane  = PR(T_c=425.12,  p_c=3.796e6,  acentric_factor=0.2002)
    decane  = PR(T_c=617.7,   p_c=2.103e6,  acentric_factor=0.4884)
    z = np.array([0.7, 0.2, 0.1])
    mx_pr = CubicMixture([methane, butane, decane], composition=z)

    print("=" * 65)
    print("Cubic phase envelope for methane / n-butane / n-decane")
    print("=" * 65)
    print(f"  Composition: CH4={z[0]:.2f}, n-C4={z[1]:.2f}, n-C10={z[2]:.2f}")

    # Critical point (Heidemann-Khalil)
    print("\n  Computing mixture critical point (Heidemann-Khalil)...")
    crit = critical_point(z, mx_pr)
    print(f"    T_c = {crit["T_c"]:.2f} K")
    print(f"    p_c = {crit["p_c"]/1e5:.2f} bar")
    print(f"    rho_c = {crit["rho_c"]:.2f} mol/m^3")

    # Trace the envelope
    print("\n  Tracing phase envelope...")
    env = trace_envelope(z, mx_pr, max_points_per_branch=100)
    Ts = env["T"]
    ps = env["p"]
    print(f"    {env['n_points']} envelope points")
    print(f"    T range: {Ts.min():.1f} - {Ts.max():.1f} K")
    print(f"    p range: {ps.min()/1e5:.3f} - {ps.max()/1e5:.2f} bar")

    # Compare with SRK -- same composition, different EOS
    print("\n" + "=" * 65)
    print("Same composition with SRK EOS")
    print("=" * 65)
    methane_s = SRK(T_c=190.564, p_c=4.5992e6, acentric_factor=0.01142)
    butane_s  = SRK(T_c=425.12,  p_c=3.796e6,  acentric_factor=0.2002)
    decane_s  = SRK(T_c=617.7,   p_c=2.103e6,  acentric_factor=0.4884)
    mx_srk = CubicMixture([methane_s, butane_s, decane_s], composition=z)
    crit_srk = critical_point(z, mx_srk)
    env_srk = trace_envelope(z, mx_srk, max_points_per_branch=100)
    Ts_srk = env_srk["T"]
    ps_srk = env_srk["p"]
    print(f"  PR  critical: T={crit["T_c"]:.2f} K, p={crit["p_c"]/1e5:.2f} bar")
    print(f"  SRK critical: T={crit_srk["T_c"]:.2f} K, p={crit_srk["p_c"]/1e5:.2f} bar")
    print(f"  Difference: dT={crit_srk["T_c"] - crit["T_c"]:.2f} K, "
          f"dp={(crit_srk["p_c"] - crit["p_c"])/1e5:.2f} bar")

    # Plot envelope -- only if matplotlib is available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(Ts, ps / 1e5, "b-", lw=1.5, label="PR (Peng-Robinson)")
        ax.plot(Ts_srk, ps_srk / 1e5, "r--", lw=1.5, label="SRK")
        ax.plot([crit["T_c"]], [crit["p_c"] / 1e5], "bo", markersize=8,
                 label=f"PR critical ({crit["T_c"]:.0f} K, {crit["p_c"]/1e5:.0f} bar)")
        ax.plot([crit_srk["T_c"]], [crit_srk["p_c"] / 1e5], "rs", markersize=8,
                 label=f"SRK critical ({crit_srk["T_c"]:.0f} K, "
                       f"{crit_srk["p_c"]/1e5:.0f} bar)")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Pressure (bar)")
        ax.set_title(f"Phase envelope: CH4/n-C4/n-C10 = {z[0]}/{z[1]}/{z[2]}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "cubic_envelope_pr_vs_srk.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        print(f"\n  Plot saved to {out}")
    except ImportError:
        print("\n  (matplotlib not available; skipping plot)")


if __name__ == "__main__":
    main()
