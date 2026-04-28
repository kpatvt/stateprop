"""Reactive distillation column for ethyl acetate production.

Demonstrates the canonical reactive-distillation advantage: continuous
removal of one product (water as overhead vapor) shifts the chemical
equilibrium forward via Le Chatelier's principle, raising overall
conversion of acetic acid + ethanol = ethyl acetate + water beyond
what a closed-batch reactor at the same temperature can achieve.

Background
----------
Pure-component normal boiling points at 1 atm:
    EtOAc  77.1 C   <-- desired ester product
    EtOH   78.4 C
    H2O   100.0 C
    AcOH  118.0 C

Phase-behavior characteristics (UNIFAC):
    EtOAc/EtOH  azeotrope ~71.8 C, ~54 mol% EtOAc
    EtOAc/H2O   heterogeneous azeotrope ~70.4 C, ~88 mol% EtOAc
    EtOH/H2O    azeotrope ~78.2 C, ~96 mol% EtOH

Reaction:
    AcOH + EtOH = EtOAc + H2O,  K_eq(298 K) ~ 4.0,  Delta H_rxn ~ -2.3 kJ/mol

Kinetic-equilibrium hierarchy: K_eq is mild (~4) so a batch reactor at
the column's operating temperature cannot exceed ~50% conversion at
equimolar feed. Reactive distillation can drive much higher conversion
because the lower-boiling species are continuously stripped.

Configuration
-------------
An 8-stage column with the reactive zone in the middle (stages 3-6).
Feed: equimolar AcOH + EtOH at stage 4. Reflux ratio is varied to
study the trade-off between separation (high R, more recycle of
reactants back to the reactive zone) and column duty.
"""
from __future__ import annotations
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stateprop.reaction import (LiquidPhaseReaction,
                                  reactive_distillation_column)
from stateprop.activity.compounds import make_unifac


def antoine_pa(A, B, C):
    """Antoine eqn: log10(P[mmHg]) = A - B/(T - C); returns Pa."""
    return lambda T: 133.322 * 10.0**(A - B/(T - C))


def main():
    # Antoine constants (NIST format, T in K, P in mmHg)
    psat_acoh  = antoine_pa(7.55716, 1642.540, 39.764)   # acetic acid
    psat_etoh  = antoine_pa(8.20417, 1642.890, 42.85)    # ethanol
    psat_etoac = antoine_pa(7.10179, 1244.95,  55.84)    # ethyl acetate
    psat_h2o   = antoine_pa(8.07131, 1730.630, 39.574)   # water

    species = ['acetic_acid', 'ethanol', 'ethyl_acetate', 'water']
    psat_funcs = [psat_acoh, psat_etoh, psat_etoac, psat_h2o]
    activity = make_unifac(species)

    rxn = LiquidPhaseReaction(
        species_names=species,
        nu=[-1, -1, +1, +1],
        K_eq_298=4.0,
        dH_rxn=-2.3e3,
    )

    # ------------------------------------------------------------------
    # Reference: equilibrium AcOH conversion in a CLOSED batch liquid
    # reactor over a range of temperatures.  This is the upper bound for
    # a reactor *without* product separation -- shows what RD must beat.
    # ------------------------------------------------------------------
    T_batch = np.linspace(330, 380, 11)
    conv_batch = []
    for T in T_batch:
        try:
            r = rxn.equilibrium_extent(
                T=float(T), n_initial=[0.5, 0.5, 0., 0.],
                activity_model=activity)
            conv_batch.append(r.xi / 0.5)
        except Exception:
            conv_batch.append(np.nan)
    conv_batch = np.array(conv_batch)

    print("=" * 64)
    print("Reactive distillation: AcOH + EtOH -> EtOAc + H2O")
    print("=" * 64)
    print(f"\n{'-'*60}")
    print("Reference: closed-batch liquid equilibrium (no separation)")
    print(f"{'-'*60}")
    print(f"  {'T [K]':>7s}  {'K_eq':>7s}  {'gamma_w':>8s}  {'AcOH conv':>10s}")
    for T, x in zip(T_batch[::2], conv_batch[::2]):
        K = rxn.K_eq(float(T))
        gammas = activity.gammas(float(T), [0.25, 0.25, 0.25, 0.25])
        print(f"  {T:7.1f}  {K:7.3f}  {gammas[3]:8.3f}  {x:10.2%}")

    # ------------------------------------------------------------------
    # Reflux-ratio sweep at fixed feed/distillate
    # ------------------------------------------------------------------
    print(f"\n{'-'*60}")
    print("RD column sweep: 8 stages, feed @ stage 4, reactive 3-6")
    print(f"{'-'*60}")
    print(f"  {'R':>5s}  {'D':>5s}  {'iter':>5s}  {'conv':>8s}  "
          f"{'x_D EtOAc':>10s}  {'x_B EtOAc':>10s}  {'T_top':>7s}  {'T_reb':>7s}")
    R_values = [1.5, 2.5, 4.0]
    D_value = 0.5
    sweep_results = []
    for R in R_values:
        result = reactive_distillation_column(
            n_stages=8, feed_stage=4, feed_F=1.0,
            feed_z=[0.5, 0.5, 0., 0.], feed_T=350.,
            reflux_ratio=R, distillate_rate=D_value,
            pressure=101325., species_names=species,
            activity_model=activity, psat_funcs=psat_funcs,
            reactions=[rxn], reactive_stages=[3, 4, 5, 6],
            max_outer_iter=100, tol=2e-3, damping=0.3)
        sweep_results.append(result)
        if result.converged:
            print(f"  {R:5.1f}  {D_value:5.2f}  {result.iterations:>5d}  "
                  f"{result.conversion('acetic_acid'):8.2%}  "
                  f"{result.x_D[2]:10.4f}  {result.x_B[2]:10.4f}  "
                  f"{result.T[0]:7.2f}  {result.T[-1]:7.2f}")
        else:
            print(f"  {R:5.1f}  {D_value:5.2f}  ----  did not converge")

    # ------------------------------------------------------------------
    # Profile plot for the most interesting case
    # ------------------------------------------------------------------
    best = sweep_results[1]  # R = 2.5
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    stages = np.arange(1, best.n_stages + 1)
    labels = ['AcOH', 'EtOH', 'EtOAc', 'H2O']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (lbl, c) in enumerate(zip(labels, colors)):
        ax.plot(best.x[:, i], stages, '-o', color=c, label=f'x ({lbl})')
        ax.plot(best.y[:, i], stages, '--^', color=c, alpha=0.4,
                label=f'y ({lbl})')
    for s in best.reactive_stages:
        ax.axhspan(s - 0.5, s + 0.5, color='lightyellow', alpha=0.5,
                   zorder=-1)
    ax.set_xlabel('mole fraction')
    ax.set_ylabel('stage (1 = top)')
    ax.invert_yaxis()
    ax.legend(loc='center right', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(f'Composition profile (R = 2.5, '
                 f'X = {best.conversion("acetic_acid"):.1%})')

    ax = axes[1]
    ax.plot(best.T, stages, '-o', color='red', label='T (column)')
    for s in best.reactive_stages:
        ax.axhspan(s - 0.5, s + 0.5, color='lightyellow', alpha=0.5,
                   zorder=-1)
    ax.set_xlabel('temperature [K]')
    ax.set_ylabel('stage (1 = top)')
    ax.invert_yaxis()
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_title('Temperature profile')

    fig.suptitle('Reactive distillation: AcOH + EtOH = EtOAc + H2O',
                  fontsize=13)
    fig.tight_layout()
    _outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(_outdir, exist_ok=True)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "reactive_distillation_profile.png"), dpi=120)
    print(f"\nSaved: examples/outputs/reactive_distillation_profile.png")

    # ------------------------------------------------------------------
    # Conversion-vs-reflux comparison plot
    # ------------------------------------------------------------------
    fig2, ax = plt.subplots(figsize=(8, 5))
    convs_rd = [r.conversion('acetic_acid') if r.converged else np.nan
                for r in sweep_results]
    ax.plot(R_values, convs_rd, '-s', color='C2', label='RD column',
            markersize=8)
    # Reference: batch equilibrium at average column T (~355 K)
    batch_at_355 = np.interp(355., T_batch, conv_batch)
    ax.axhline(batch_at_355, color='gray', ls='--',
                label=f'batch eq @ 355 K = {batch_at_355:.2%}')
    ax.set_xlabel('reflux ratio R')
    ax.set_ylabel('overall AcOH conversion')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('RD-column conversion vs reflux ratio (10 stages, D=0.5)')
    fig2.tight_layout()
    fig2.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "reactive_distillation_rd_vs_batch.png"), dpi=120)
    print(f"Saved: examples/outputs/reactive_distillation_rd_vs_batch.png")

    print("\n" + "=" * 64)
    print("Take-aways")
    print("=" * 64)
    converged = [(R, c) for R, c, r in zip(R_values, convs_rd, sweep_results)
                 if r.converged]
    print(f"  1. Batch equilibrium AcOH conversion at 350-380 K: ~50%")
    if converged:
        Rs, cs = zip(*converged)
        print(f"  2. RD column at R={list(Rs)} reaches "
              f"{[f'{c:.1%}' for c in cs]} conversion")
    print( "  3. The RD advantage at this configuration is modest (a few")
    print( "     percent above batch) because: (a) K_eq ~ 4 is mild;")
    print( "     (b) modified Raoult misses the heterogeneous EtOAc/water")
    print( "     azeotrope that real industrial designs exploit;")
    print( "     (c) only 4 reactive stages.")
    print( "  4. Higher reflux feeds more reactant back to the reactive")
    print( "     zone but at increased reboiler duty.")


if __name__ == '__main__':
    main()
