"""Real-gas K_eq corrections for high-pressure reactions (v0.9.63).

Demonstrates the new EOS-based fugacity-coefficient correction to
chemical reaction equilibrium calculations.

For ideal gas:
    K_eq(T) = Prod_i (y_i * p / p_ref)^nu_i

For real gas with fugacity coefficient phi_i:
    K_eq(T) = Prod_i (y_i * phi_i * p / p_ref)^nu_i

Equivalently:  K_y * (p/p_ref)^Δν * K_phi = K_eq, where
    K_phi = Prod_i phi_i^nu_i

The correction matters when:
  (a) Pressures are high (50+ bar)
  (b) Mixtures have polar/associating species (methanol, ammonia, water)
  (c) Both — like industrial methanol or ammonia synthesis

Two industrially-important reactions are demonstrated:

  1. Methanol synthesis (CO + 2 H2 = CH3OH) at 50-300 bar
  2. Ammonia synthesis (N2 + 3 H2 = 2 NH3) at 100-400 bar

For both, the real-gas correction shifts K_y substantially at industrial
pressures, leading to higher predicted equilibrium conversions than the
ideal-gas approximation suggests.
"""
from __future__ import annotations

import os
import numpy as np
from stateprop.reaction import Reaction, MultiReaction
from stateprop.cubic import PR, CubicMixture


def main():
    # ---------------------------------------------------------------
    # 1. Methanol synthesis at high pressure
    # ---------------------------------------------------------------
    print("=" * 70)
    print("Methanol synthesis: CO + 2 H2 = CH3OH at 500 K")
    print("=" * 70)
    print("\n  Comparing ideal-gas vs real-gas (Peng-Robinson) equilibrium\n")

    # PR mixture in same order as Reaction.from_names builds: [CO, H2, CH3OH]
    co     = PR(T_c=132.85, p_c=3.494e6,  acentric_factor=0.045)
    h2     = PR(T_c=33.145, p_c=1.296e6,  acentric_factor=-0.219)
    ch3oh  = PR(T_c=512.60, p_c=8.084e6,  acentric_factor=0.5625)
    pr_meoh = CubicMixture([co, h2, ch3oh], composition=[0.33, 0.66, 0.01])

    rxn = Reaction.from_names(reactants={'CO': 1, 'H2': 2},
                                products={'CH3OH': 1})
    print(f"  {'p (bar)':>9s}  {'IG conv':>9s}  {'RG conv':>9s}  "
          f"{'K_phi':>10s}  {'IG y_MeOH':>10s}  {'RG y_MeOH':>10s}")
    print("  " + "-" * 70)
    for p_bar in [1, 10, 50, 100, 200, 300]:
        p = p_bar * 1e5
        r_id = rxn.equilibrium_extent_ideal_gas(
            T=500.0, p=p, n_initial=[1.0, 2.0, 0.0])
        r_rg = rxn.equilibrium_extent_real_gas(
            T=500.0, p=p, n_initial=[1.0, 2.0, 0.0], eos=pr_meoh)

        # Compute K_phi at the real-gas solution
        rho = pr_meoh.density_from_pressure(p, 500.0, r_rg.y_eq,
                                              phase_hint='vapor')
        ln_phi = pr_meoh.ln_phi(rho, 500.0, r_rg.y_eq)
        K_phi = float(np.exp((rxn.nu * ln_phi).sum()))

        conv_id = (1 - r_id.n_eq[0] / 1.0) * 100
        conv_rg = (1 - r_rg.n_eq[0] / 1.0) * 100
        print(f"  {p_bar:>9.0f}  {conv_id:>8.2f}%  {conv_rg:>8.2f}%  "
              f"{K_phi:>10.4f}  {r_id.y_eq[2]:>10.4f}  {r_rg.y_eq[2]:>10.4f}")

    print("\n  Notes:")
    print("    - K_phi -> 1 at low p (ideal gas limit)")
    print("    - K_phi << 1 at high p means CH3OH has much smaller phi")
    print("      than reactants -> real K_y must be larger to compensate")
    print("    - Net effect: real-gas predicts ~10 percentage points higher")
    print("      methanol conversion at 100 bar vs ideal-gas")

    # ---------------------------------------------------------------
    # 2. Ammonia synthesis at industrial conditions
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Ammonia synthesis: N2 + 3 H2 = 2 NH3 at 700 K")
    print("=" * 70)
    print("\n  Industrial Haber-Bosch runs at 150-300 bar -- precisely where")
    print("  real-gas corrections are essential to match observed conversion.\n")

    # Reaction species order: [N2, H2, NH3]
    n2_pr  = PR(T_c=126.192, p_c=3.3958e6, acentric_factor=0.0372)
    h2_pr  = PR(T_c=33.145,  p_c=1.296e6,  acentric_factor=-0.219)
    nh3_pr = PR(T_c=405.65,  p_c=11.357e6, acentric_factor=0.252)
    pr_nh3 = CubicMixture([n2_pr, h2_pr, nh3_pr], composition=[0.25, 0.74, 0.01])

    rxn_nh3 = Reaction.from_names(reactants={'N2': 1, 'H2': 3},
                                     products={'NH3': 2})

    print(f"  {'p (bar)':>9s}  {'IG conv':>9s}  {'RG conv':>9s}  "
          f"{'K_phi':>10s}  {'IG y_NH3':>10s}  {'RG y_NH3':>10s}")
    print("  " + "-" * 70)
    for p_bar in [1, 50, 100, 200, 300, 400]:
        p = p_bar * 1e5
        r_id = rxn_nh3.equilibrium_extent_ideal_gas(
            T=700.0, p=p, n_initial=[1.0, 3.0, 0.0])
        try:
            r_rg = rxn_nh3.equilibrium_extent_real_gas(
                T=700.0, p=p, n_initial=[1.0, 3.0, 0.0], eos=pr_nh3)
        except Exception as e:
            print(f"  {p_bar:>9.0f}  RG failed: {str(e)[:50]}")
            continue

        rho = pr_nh3.density_from_pressure(p, 700.0, r_rg.y_eq,
                                              phase_hint='vapor')
        ln_phi = pr_nh3.ln_phi(rho, 700.0, r_rg.y_eq)
        K_phi = float(np.exp((rxn_nh3.nu * ln_phi).sum()))

        conv_id = (1 - r_id.n_eq[0] / 1.0) * 100
        conv_rg = (1 - r_rg.n_eq[0] / 1.0) * 100
        print(f"  {p_bar:>9.0f}  {conv_id:>8.2f}%  {conv_rg:>8.2f}%  "
              f"{K_phi:>10.4f}  {r_id.y_eq[2]:>10.4f}  {r_rg.y_eq[2]:>10.4f}")

    # ---------------------------------------------------------------
    # 3. Plot
    # ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        p_arr = np.logspace(0, np.log10(400), 30)   # 1 to 400 bar
        # Methanol
        conv_id_m = []
        conv_rg_m = []
        for p_bar in p_arr:
            p = p_bar * 1e5
            r_id = rxn.equilibrium_extent_ideal_gas(
                T=500.0, p=p, n_initial=[1.0, 2.0, 0.0])
            r_rg = rxn.equilibrium_extent_real_gas(
                T=500.0, p=p, n_initial=[1.0, 2.0, 0.0], eos=pr_meoh)
            conv_id_m.append((1 - r_id.n_eq[0]) * 100 if r_id.converged else np.nan)
            conv_rg_m.append((1 - r_rg.n_eq[0]) * 100 if r_rg.converged else np.nan)

        # Ammonia
        conv_id_a = []
        conv_rg_a = []
        for p_bar in p_arr:
            p = p_bar * 1e5
            r_id = rxn_nh3.equilibrium_extent_ideal_gas(
                T=700.0, p=p, n_initial=[1.0, 3.0, 0.0])
            try:
                r_rg = rxn_nh3.equilibrium_extent_real_gas(
                    T=700.0, p=p, n_initial=[1.0, 3.0, 0.0], eos=pr_nh3)
                conv_rg_a.append((1 - r_rg.n_eq[0]) * 100 if r_rg.converged
                                  else np.nan)
            except Exception:
                conv_rg_a.append(np.nan)
            conv_id_a.append((1 - r_id.n_eq[0]) * 100 if r_id.converged else np.nan)

        axes[0].semilogx(p_arr, conv_id_m, 'b--', lw=1.5,
                          label='Ideal gas')
        axes[0].semilogx(p_arr, conv_rg_m, 'b-', lw=2.0,
                          label='Real gas (PR)')
        axes[0].set_xlabel("Pressure (bar)")
        axes[0].set_ylabel("CO conversion (%)")
        axes[0].set_title("Methanol synthesis at 500 K\n(CO + 2H2 = CH3OH)")
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3, which='both')
        axes[0].set_ylim(0, 100)

        axes[1].semilogx(p_arr, conv_id_a, 'r--', lw=1.5,
                          label='Ideal gas')
        axes[1].semilogx(p_arr, conv_rg_a, 'r-', lw=2.0,
                          label='Real gas (PR)')
        axes[1].set_xlabel("Pressure (bar)")
        axes[1].set_ylabel("N2 conversion (%)")
        axes[1].set_title("Ammonia synthesis at 700 K\n(N2 + 3H2 = 2 NH3)")
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3, which='both')
        axes[1].set_ylim(0, 100)

        out = "/mnt/user-data/outputs/real_gas_K_eq_methanol_ammonia.png"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        print(f"\n  Plot saved to {out}")
    except ImportError:
        print("\n  (matplotlib not available; skipping plot)")


if __name__ == "__main__":
    main()
