"""Reactive flash: VLE + liquid-phase chemical equilibrium (v0.9.65).

Demonstrates the new `reactive_flash_TP` solver on the canonical
esterification reaction:

    CH3COOH (l) + C2H5OH (l) = CH3COOC2H5 (l) + H2O (l)

The four species span a wide range of volatility:

    Species          bp at 1 atm
    --------         -----------
    Ethyl acetate    350.3 K (most volatile)
    Ethanol          351.4 K
    Water            373.2 K
    Acetic acid      391.0 K (least volatile)

The boiling point ordering is what makes reactive distillation work:
ethyl acetate forms in the liquid as a product, then preferentially
vaporizes -- removing it from the reaction zone and shifting the
equilibrium FORWARD (Le Chatelier), well past the pure-liquid limit.

Three demonstrations:
  1. Below bubble point (T=320 K): reactive flash matches no-VLE
     equilibrium exactly (consistency check).
  2. Boiling point regime (T=355 K, 1 atm): VLE coupling boosts AcOH
     conversion from 50% (pure liquid) to 64% (reactive flash).
  3. Pressure scan at fixed T: lower p makes the system more vapor-rich,
     showing how a real RD column achieves >90% conversion via
     stage-wise vapor-product withdrawal.
"""
from __future__ import annotations

import os
import numpy as np
from stateprop.reaction import LiquidPhaseReaction, reactive_flash_TP
from stateprop.activity.compounds import make_unifac


def antoine(A, B, C):
    """Return p_sat(T) [Pa] from Antoine coefficients (mmHg form)."""
    def psat(T):
        return 133.322 * 10**(A - B/(T + C))
    return psat


def main():
    # Antoine coefficients (NIST WebBook / DECHEMA, T in K, log10 p/mmHg)
    psat_acoh   = antoine(7.55716, 1642.540, -39.764)   # acetic acid
    psat_etoh   = antoine(8.20417, 1642.890, -42.85)    # ethanol
    psat_etoac  = antoine(7.10179, 1244.95,  -55.84)    # ethyl acetate
    psat_water  = antoine(8.07131, 1730.630, -39.574)   # water

    species = ['acetic_acid', 'ethanol', 'ethyl_acetate', 'water']
    psats   = [psat_acoh, psat_etoh, psat_etoac, psat_water]

    uf = make_unifac(species)
    rxn = LiquidPhaseReaction(
        species_names=species, nu=[-1, -1, +1, +1],
        K_eq_298=4.0, dH_rxn=-2.3e3,
    )

    print("=" * 70)
    print("Esterification reactive flash: AcOH + EtOH = EtOAc + H2O")
    print("=" * 70)
    print(f"\n  Antoine sanity (1 atm = 101325 Pa):")
    bp = {'acetic_acid': 391.05, 'ethanol': 351.45,
          'ethyl_acetate': 350.30, 'water':  373.15}
    for nm, ps in zip(species, psats):
        print(f"    {nm:<14s} p_sat({bp[nm]:5.1f} K) = {ps(bp[nm]):8.0f} Pa")
    print(f"\n  Reaction K_eq(298 K) = {rxn.K_eq(298.15):.3f}, "
          f"K_eq(355 K) = {rxn.K_eq(355.0):.3f}")

    # ---------------------------------------------------------------
    # 1. Subcooled: must match no-VLE liquid equilibrium
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1) Subcooled liquid (T=320 K, 1 atm): reactive flash = no-VLE limit")
    print("=" * 70)
    r_flash = reactive_flash_TP(
        T=320.0, p=101325.0, F=2.0, z=[0.5, 0.5, 0.0, 0.0],
        activity_model=uf, psat_funcs=psats, reactions=[rxn],
        species_names=species, tol=1e-7, maxiter=80)
    r_liq = rxn.equilibrium_extent(T=320.0, n_initial=[1.0, 1.0, 0.0, 0.0],
                                      activity_model=uf)
    print(f"\n  Reactive flash: V/(V+L) = {r_flash.V/(r_flash.V+r_flash.L):.4f}")
    print(f"  Reactive flash xi: {r_flash.xi[0]:.4f}")
    print(f"  Pure-liquid xi:    {r_liq.xi:.4f}")
    print(f"  Match: {abs(r_flash.xi[0] - r_liq.xi) < 1e-4}")
    print(f"  AcOH conversion: {r_flash.xi[0]*100:.2f}%")

    # ---------------------------------------------------------------
    # 2. Boiling regime: Le Chatelier on volatile EtOAc removal
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2) Boiling regime (T=355 K, 1 atm): RD enhances conversion")
    print("=" * 70)
    r_flash = reactive_flash_TP(
        T=355.0, p=101325.0, F=2.0, z=[0.5, 0.5, 0.0, 0.0],
        activity_model=uf, psat_funcs=psats, reactions=[rxn],
        species_names=species, tol=1e-7, maxiter=80)
    r_liq = rxn.equilibrium_extent(T=355.0, n_initial=[1.0, 1.0, 0.0, 0.0],
                                      activity_model=uf)
    beta = r_flash.V / (r_flash.V + r_flash.L)
    print(f"\n  Reactive flash:   V/(V+L) = {beta:.4f}, xi = {r_flash.xi[0]:.4f}")
    print(f"  Pure-liquid:      xi = {r_liq.xi:.4f}")
    print(f"  Conversion gain:  +{(r_flash.xi[0]/r_liq.xi - 1)*100:.1f}%")
    print(f"  AcOH conversion:  {r_flash.xi[0]*100:.2f}% (vs {r_liq.xi*100:.2f}% pure liquid)")
    print(f"\n  Liquid composition (heavy in AcOH, water):")
    for nm, x_i, g_i in zip(r_flash.species if hasattr(r_flash, 'species')
                              else species, r_flash.x, r_flash.gamma):
        print(f"    {nm:<14s}  x = {x_i:.4f}, γ = {g_i:.3f}")
    print(f"  Vapor composition (rich in EtOH, EtOAc):")
    for nm, y_i in zip(species, r_flash.y):
        print(f"    {nm:<14s}  y = {y_i:.4f}")

    # ---------------------------------------------------------------
    # 3. Pressure scan: lower p drives more vapor → higher conversion
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3) Pressure scan at T=345 K, equimolar AcOH+EtOH feed")
    print("=" * 70)
    print(f"\n  {'p (kPa)':>9s}  {'V/(V+L)':>9s}  {'xi':>7s}  "
          f"{'AcOH conv':>10s}  {'y_EtOAc':>9s}")
    p_kpa_arr = [10, 30, 50, 80, 100, 150, 200]
    for p_kpa in p_kpa_arr:
        p_pa = p_kpa * 1000.0
        r = reactive_flash_TP(
            T=345.0, p=p_pa, F=2.0, z=[0.5, 0.5, 0.0, 0.0],
            activity_model=uf, psat_funcs=psats, reactions=[rxn],
            species_names=species, tol=1e-7, maxiter=80)
        if not r.converged:
            print(f"  {p_kpa:>9.0f}  failed")
            continue
        beta = r.V / (r.V + r.L) if (r.V + r.L) > 0 else 0
        print(f"  {p_kpa:>9.0f}  {beta:>9.4f}  {r.xi[0]:>7.4f}  "
              f"{r.xi[0]*100:>9.2f}%  {r.y[2]:>9.4f}")
    print("\n  At low p (10-50 kPa), all-vapor with high reaction extent driven")
    print("  by the dew-point pseudo-liquid composition.")
    print("  At high p (>=200 kPa), all-liquid; reduces to pure-liquid equilibrium.")
    print("  Industrial RD columns operate near 1 atm in the two-phase 'sweet spot'.")

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Subplot 1: conversion vs T at 1 atm (flash vs liquid-only)
        T_arr = np.linspace(310, 380, 30)
        xi_flash = []
        xi_liq = []
        beta_arr = []
        for T in T_arr:
            r_f = reactive_flash_TP(
                T=float(T), p=101325.0, F=2.0, z=[0.5, 0.5, 0.0, 0.0],
                activity_model=uf, psat_funcs=psats, reactions=[rxn],
                species_names=species, tol=1e-7, maxiter=80)
            r_l = rxn.equilibrium_extent(T=float(T),
                                            n_initial=[1.0, 1.0, 0.0, 0.0],
                                            activity_model=uf)
            xi_flash.append(r_f.xi[0] if r_f.converged else np.nan)
            xi_liq.append(r_l.xi if r_l.converged else np.nan)
            beta_arr.append((r_f.V/(r_f.V+r_f.L)) if r_f.converged else np.nan)
        T_C = T_arr - 273.15
        axes[0].plot(T_C, [x * 100 for x in xi_liq], 'b--', lw=1.5,
                      label='Pure liquid (no VLE)')
        axes[0].plot(T_C, [x * 100 for x in xi_flash], 'b-', lw=2,
                      label='Reactive flash (with VLE)')
        ax2 = axes[0].twinx()
        ax2.plot(T_C, [b * 100 for b in beta_arr], 'g:', lw=1.5,
                   label='Vapor fraction')
        axes[0].set_xlabel("Temperature (°C)")
        axes[0].set_ylabel("AcOH conversion (%)", color='b')
        ax2.set_ylabel("V/(V+L) [%]", color='g')
        axes[0].set_title("Reactive flash vs pure-liquid equilibrium\n(1 atm, 1:1 feed)")
        axes[0].legend(loc='upper left')
        ax2.legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 100)

        # Subplot 2: pressure scan at fixed T
        p_arr = np.logspace(np.log10(5), np.log10(300), 30)  # 5-300 kPa
        xi_p = []
        beta_p = []
        for p_kpa in p_arr:
            r = reactive_flash_TP(
                T=345.0, p=float(p_kpa) * 1000.0, F=2.0,
                z=[0.5, 0.5, 0.0, 0.0],
                activity_model=uf, psat_funcs=psats, reactions=[rxn],
                species_names=species, tol=1e-7, maxiter=80)
            xi_p.append(r.xi[0] if r.converged else np.nan)
            beta_p.append(r.V/(r.V+r.L) if r.converged else np.nan)
        axes[1].semilogx(p_arr, [x * 100 for x in xi_p], 'r-', lw=2,
                          label='AcOH conversion')
        ax2b = axes[1].twinx()
        ax2b.semilogx(p_arr, [b * 100 for b in beta_p], 'g:', lw=1.5,
                        label='Vapor fraction')
        axes[1].set_xlabel("Pressure (kPa)")
        axes[1].set_ylabel("AcOH conversion (%)", color='r')
        ax2b.set_ylabel("V/(V+L) [%]", color='g')
        axes[1].set_title("Pressure dependence at T=345 K\n(1:1 AcOH+EtOH feed)")
        axes[1].legend(loc='upper right')
        ax2b.legend(loc='center right')
        axes[1].grid(True, alpha=0.3, which='both')
        axes[1].set_ylim(0, 100)

        out = "/mnt/user-data/outputs/reactive_flash_esterification.png"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        print(f"\n  Plot saved to {out}")
    except ImportError:
        print("\n  (matplotlib not available; skipping plot)")


if __name__ == "__main__":
    main()
