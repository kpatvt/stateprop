"""
Example: plot the coexistence dome in multiple coordinate systems.

Generates three plots for water:
  1. log(p) - T  vapor-pressure curve
  2. T - s       dome (in mass-based engineering units)
  3. p - h       dome (Mollier-style, mass-based)
  4. T - rho     dome

Produces a PNG at /mnt/user-data/outputs/phase_envelope_water.png.

Run:
    python examples/plot_phase_envelope.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")         # non-interactive backend
import matplotlib.pyplot as plt

import stateprop as he


def main():
    fluid = he.load_fluid("water")
    env = he.trace_phase_envelope(fluid, n_points=120, critical_density=0.8)
    mb = env.as_mass_based()

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Saturation curve for water (IAPWS-95)   -- {len(env.T)} points",
                 fontsize=14, fontweight="bold")

    # ---- 1. p vs T (log-p) -------------------------------------------------
    ax = axs[0, 0]
    ax.semilogy(env.T, mb["p_MPa"], "b-", lw=2)
    ax.semilogy([env.T_c], [fluid.p_c * 1e-6], "r*", markersize=14,
                label=f"Critical point ({env.T_c:.2f} K, {fluid.p_c*1e-6:.3f} MPa)")
    ax.semilogy([env.T_triple], [fluid.p_triple * 1e-6], "g*", markersize=12,
                label=f"Triple point ({env.T_triple:.2f} K, {fluid.p_triple:.2e} Pa)")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Saturation pressure [MPa]")
    ax.set_title("Vapor-pressure curve")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # ---- 2. T-s dome -------------------------------------------------------
    ax = axs[0, 1]
    x, y = env.dome_coordinates("s_kg", "T")
    ax.plot(x, y, "b-", lw=2)
    ax.fill(x, y, alpha=0.15, color="blue")
    # Critical point in (s, T) coordinates
    idx_c = np.argmax(env.T)
    s_c_kg = mb["s_L"][idx_c]
    ax.plot([s_c_kg], [env.T_c], "r*", markersize=14,
            label=f"Critical ({s_c_kg:.3f} kJ/kg-K, {env.T_c:.1f} K)")
    ax.set_xlabel("Specific entropy [kJ/(kg K)]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("T - s coexistence dome")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # ---- 3. p-h dome (Mollier-style) ---------------------------------------
    ax = axs[1, 0]
    x, y = env.dome_coordinates("h_kg", "p_MPa")
    ax.plot(x, y, "b-", lw=2)
    ax.fill(x, y, alpha=0.15, color="blue")
    # Critical point in (h, p) coordinates
    idx_c = np.argmax(env.T)
    h_c = mb["h_L"][idx_c]
    ax.plot([h_c], [fluid.p_c * 1e-6], "r*", markersize=14,
            label=f"Critical ({h_c:.1f} kJ/kg, {fluid.p_c*1e-6:.2f} MPa)")
    ax.set_yscale("log")
    ax.set_xlabel("Specific enthalpy [kJ/kg]")
    ax.set_ylabel("Pressure [MPa]")
    ax.set_title("p - h coexistence dome (Mollier-style)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")

    # ---- 4. T-rho dome -----------------------------------------------------
    ax = axs[1, 1]
    x, y = env.dome_coordinates("rho_kg", "T")
    ax.plot(x, y, "b-", lw=2)
    ax.fill(x, y, alpha=0.15, color="blue")
    rho_c_kg = fluid.rho_c * fluid.molar_mass
    ax.plot([rho_c_kg], [env.T_c], "r*", markersize=14,
            label=f"Critical ({rho_c_kg:.1f} kg/m^3)")
    ax.set_xscale("log")
    ax.set_xlabel("Density [kg/m^3]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("T - rho coexistence dome")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()

    # Save figure
    out_path = "/mnt/user-data/outputs/phase_envelope_water.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Phase envelope plot saved to {out_path}")
    print(f"  {len(env.T)} points from {env.T[0]:.2f} K to {env.T[-1]:.2f} K")
    print(f"  Pressure range: {env.p[0]*1e-6:.2e} -- {env.p[-1]*1e-6:.3f} MPa")


if __name__ == "__main__":
    main()
