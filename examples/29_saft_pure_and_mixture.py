"""PC-SAFT for pure and mixed fluids with hydrogen-bonded association.

Demonstrates the PC-SAFT (Gross & Sadowski 2001) framework with
explicit Wertheim TPT1 association for hydrogen-bonded molecules:

  - Pure-component parameters from the Gross-Sadowski database
  - Pressure / density evaluation for hard-chain + dispersion + association
  - Liquid density at compressed-liquid conditions
  - Fugacity coefficients for non-ideal vapor mixtures
  - Comparison: 4C scheme (water) vs 2B scheme (alcohols)

PC-SAFT explicitly resolves the physics of hydrogen bonding via the
association term, which makes it a natural fit for strongly polar /
self-associating systems where cubic EOS struggle. The trade-off is
computational cost: the Wertheim site-balance iteration adds 5-20x
to the per-evaluation time vs cubic EOS.
"""
from __future__ import annotations

import numpy as np
from stateprop.saft import (SAFTMixture, METHANE, ETHANE, PROPANE,
                              WATER, METHANOL, ETHANOL, N_PROPANOL)


def main():
    print("=" * 70)
    print("PC-SAFT pure and mixed associating fluids")
    print("=" * 70)

    # 1. Pure-component parameters
    print("\nPure-component PC-SAFT parameters:")
    print(f"  {'Component':<12s} {'m':>6s} {'sigma':>7s} {'eps/k':>8s}  "
          f"{'eps_AB/k':>9s} {'kappa_AB':>10s} {'scheme':>7s}")
    for c in [METHANE, ETHANE, WATER, METHANOL, ETHANOL, N_PROPANOL]:
        eps_AB = getattr(c, 'eps_AB_k', 0.0) or 0.0
        kappa = getattr(c, 'kappa_AB', 0.0) or 0.0
        scheme = (getattr(c, 'assoc_scheme', '-')
                   if eps_AB > 0 else '-')
        print(f"  {c.name:<12s} {c.m:>6.3f} {c.sigma:>7.3f} "
              f"{c.epsilon_k:>8.2f}  {eps_AB:>9.1f} {kappa:>10.4f} "
              f"{scheme:>7s}")

    print(f"\n  Note: 'eps_AB/k' and 'kappa_AB' are the Wertheim TPT1")
    print(f"  association parameters. 4C means 4 sites (e.g. water has")
    print(f"  2 H + 2 lone pairs); 2B means 2 sites (1 OH + 1 lone pair).")

    # 2. Direct pressure / density evaluation (fast, no iteration loops)
    print("\n" + "=" * 70)
    print("Pure water at T=298.15 K, varying density")
    print("=" * 70)
    saft_water = SAFTMixture([WATER], [1.0])
    print(f"\n  {'rho (mol/m^3)':>14s}  {'p (bar)':>10s}  "
          f"{'description':<20s}")
    print("  " + "-" * 50)
    # Use vapor-branch densities only (pressure = explicit, no iteration)
    for rho, label in [(1.0, "very dilute vapor"),
                         (10.0, "dilute vapor"),
                         (40.0, "ideal-gas regime"),
                         (100.0, "dense vapor"),
                         (1000.0, "near-saturation vapor")]:
        p = saft_water.pressure(rho, 298.15)
        # Compare to ideal gas
        p_ig = rho * 8.314472 * 298.15
        ratio = p / p_ig
        print(f"  {rho:>14.2f}  {p/1e5:>10.4f}  {label:<20s} (Z={ratio:.4f})")

    # 3. Compressed liquid density (one Newton call -- fast)
    print("\n" + "=" * 70)
    print("Compressed liquid water density vs pressure at 298 K")
    print("=" * 70)
    print(f"\n  {'p (bar)':>9s}  {'rho (mol/m^3)':>14s}  "
          f"{'rho (kg/m^3)':>14s}  {'NIST (kg/m^3)':>14s}")
    print("  " + "-" * 60)
    nist_water_298 = {1: 997.0, 10: 997.4, 50: 999.4, 100: 1001.9, 500: 1019.6}
    for p_bar in [1.0, 10.0, 50.0, 100.0, 500.0]:
        rho = saft_water.density_from_pressure(p_bar * 1e5, 298.15,
                                                  phase_hint='liquid')
        rho_mass = rho * WATER.molar_mass
        nist = nist_water_298.get(int(p_bar), float('nan'))
        print(f"  {p_bar:>9.1f}  {rho:>14.2f}  {rho_mass:>14.3f}  "
              f"{nist:>14.3f}")

    print(f"\n  PC-SAFT slightly under-predicts liquid water density")
    print(f"  (~1.3% low). Full re-fit of (m, sigma, eps/k, eps_AB, kappa_AB)")
    print(f"  against NIST IAPWS would close this gap; the v0.9.28 4C")
    print(f"  re-fit prioritized vapor-pressure accuracy.")

    # 4. Mixture: methane-ethane-propane (no association, fast)
    print("\n" + "=" * 70)
    print("Light hydrocarbon mixture: methane(0.6) + ethane(0.3) + propane(0.1)")
    print("=" * 70)
    z_hc = [0.6, 0.3, 0.1]
    saft_hc = SAFTMixture([METHANE, ETHANE, PROPANE], z_hc)
    print(f"\n  Density and fugacity coefficients at T=250 K, varying p:\n")
    print(f"  {'p (bar)':>9s}  {'rho (mol/m^3)':>14s}  "
          f"{'phi_CH4':>10s}  {'phi_C2':>10s}  {'phi_C3':>10s}")
    print("  " + "-" * 60)
    for p_bar in [10.0, 50.0, 100.0]:
        try:
            rho = saft_hc.density_from_pressure(p_bar * 1e5, 250.0,
                                                  phase_hint='vapor')
            ln_phi = saft_hc.ln_phi(rho, 250.0)
            phi = np.exp(ln_phi)
            print(f"  {p_bar:>9.1f}  {rho:>14.3f}  {phi[0]:>10.4f}  "
                  f"{phi[1]:>10.4f}  {phi[2]:>10.4f}")
        except Exception as e:
            print(f"  {p_bar:>9.1f}  failed: {str(e)[:40]}")

    # 5. Bubble pressure for the hydrocarbon mixture (no association = fast)
    print("\n" + "=" * 70)
    print("Bubble pressure curve (light hydrocarbon mixture)")
    print("=" * 70)
    from stateprop.cubic.flash import newton_bubble_point_p
    print(f"\n  {'T (K)':>6s}  {'p_bubble (bar)':>14s}  "
          f"{'y_CH4':>8s}  {'y_C2':>8s}  {'y_C3':>8s}")
    print("  " + "-" * 50)
    z_arr = np.array(z_hc)
    p_init = 30e5
    for T in [200.0, 220.0, 240.0, 260.0]:
        try:
            r = newton_bubble_point_p(T, z_arr, saft_hc, p_init=p_init, tol=1e-7)
            print(f"  {T:>6.1f}  {r.p/1e5:>14.3f}  {r.y[0]:>8.4f}  "
                  f"{r.y[1]:>8.4f}  {r.y[2]:>8.4f}")
            p_init = r.p   # warm-start next iteration
        except Exception as e:
            print(f"  {T:>6.1f}  failed: {str(e)[:50]}")

    print(f"\n  PC-SAFT for non-associating hydrocarbon mixtures gives")
    print(f"  predictions comparable to PR/SRK with the advantage that")
    print(f"  the association term is *available* if any component is")
    print(f"  added that does associate (e.g. methanol contamination).")


if __name__ == "__main__":
    main()
