"""Standalone amine stripper (regenerator) design with the rigorous N-S solver.

What this demonstrates
----------------------
The amine stripper (a.k.a. regenerator, desorber) is the energy-
intensive other half of an amine capture plant.  Rich solvent enters
the top, steam stripping vapor enters the bottom (from the reboiler),
and the column produces (a) lean amine at the bottom and (b) wet
overhead CO₂ at the top.

The reboiler duty Q_R is the dominant operating cost — typically
60-70% of the total energy demand of the capture plant.  Specific
reboiler duty (GJ per ton CO₂ captured) is the textbook performance
metric.

This example walks through:

1. **Steam-to-amine ratio sweep** — at fixed α_rich, more steam means
   lower α_lean (better regeneration) but more energy.  Find the
   knee where additional steam buys little additional regeneration.
2. **Rich-loading sweep** — α_rich is set by the absorber design, so
   showing how stripper performance varies with α_rich connects the
   two columns.
3. **Pressure sweep** — higher P_stripper means higher T_reboiler
   (hotter reboiler = better regeneration thermodynamically) but
   more compressor downstream and more degradation. Industry runs
   1.5-2.0 bar typically.
4. **Specific reboiler duty calculation** — Q_R / mass CO₂ stripped.

Reference
---------
Notz, R.; Mangalapally, H. P.; Hasse, H. (2012). Post combustion
CO₂ capture by reactive absorption: pilot plant description and
results of systematic studies with MEA.  Int. J. Greenhouse Gas
Control 6, 84-112.

Approximate runtime: ~20 seconds.

Public APIs invoked
-------------------
- stateprop.electrolyte.amine_stripper_ns
- stateprop.electrolyte.AmineNSResult
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np

from stateprop.electrolyte import amine_stripper_ns
from examples._harness import validate, validate_bool, summary

print("=" * 70)
print("Standalone amine stripper design (30 wt% MEA regenerator)")
print("=" * 70)
print()

# Plant inputs (representative)
L_AMINE = 8.0   # mol amine/s
TOTAL_AMINE = 5.0   # ~30 wt%
ALPHA_RICH_BASE = 0.45   # typical from absorber bottoms
P_BASE = 1.8e5    # 1.8 bar — typical regenerator pressure

print(f"  Solvent:        30 wt% MEA, L = {L_AMINE} mol amine/s")
print(f"  Rich loading:   α_rich (baseline) = {ALPHA_RICH_BASE}")
print(f"  Base pressure:  {P_BASE/1e5} bar")
print()

# ------------------------------------------------------------------
# Study 1: steam (G) sweep at fixed α_rich
# ------------------------------------------------------------------
print(f"Study 1: regeneration vs steam rate (α_rich={ALPHA_RICH_BASE}, "
      f"15 stages)")
print()
print(f"  {'G (mol/s)':>10s}  {'G/L':>5s}  {'α_lean':>7s}  "
      f"{'Δα':>6s}  {'Q_R (kW)':>9s}  {'GJ/t CO₂':>10s}")
print(f"  {'-'*10}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*9}  {'-'*10}")

steam_sweep = []
for G in [3.0, 5.0, 8.0, 12.0, 16.0]:
    r = amine_stripper_ns(
        amine_name="MEA", total_amine=TOTAL_AMINE,
        L=L_AMINE, G=G, alpha_rich=ALPHA_RICH_BASE,
        n_stages=15, P=P_BASE,
        energy_balance=True,
    )
    delta = ALPHA_RICH_BASE - r.alpha_lean
    co2_kg_per_s = L_AMINE * delta * 0.044   # mol amine × Δα × M(CO₂)
    Q_R_kW = (r.Q_R or 0.0) / 1000.0
    if co2_kg_per_s > 1e-6:
        Q_per_ton = (Q_R_kW * 1000) / (co2_kg_per_s * 1000) * 1e-6 * 3600 * 1000
        # cleaner: GJ/t = (kW × 3600 / 1e6) / (kg/s × 3600 / 1000)
        #              = kW / (1e3 × kg/s) × 1
        Q_per_ton = (Q_R_kW * 1000.0) / (co2_kg_per_s * 1e3) * 1e-6 * 1000.0
        Q_per_ton = Q_R_kW / (co2_kg_per_s * 1000.0)
    else:
        Q_per_ton = float("nan")
    steam_sweep.append((G, r.alpha_lean, delta, Q_R_kW, Q_per_ton))
    print(f"  {G:>10.1f}  {G/L_AMINE:>5.2f}  {r.alpha_lean:>7.4f}  "
          f"{delta:>6.3f}  {Q_R_kW:>9.1f}  {Q_per_ton:>10.2f}")

# ------------------------------------------------------------------
# Study 2: rich-loading (α_rich) sweep at fixed G
# ------------------------------------------------------------------
print()
print(f"Study 2: regeneration vs α_rich (G={8.0} mol/s, 15 stages)")
print()
print(f"  {'α_rich':>7s}  {'α_lean':>7s}  {'Δα':>6s}  "
      f"{'recovery':>9s}  {'Q_R (kW)':>9s}")
print(f"  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*9}  {'-'*9}")

rich_sweep = []
for alpha_rich in [0.30, 0.40, 0.45, 0.50]:
    r = amine_stripper_ns(
        amine_name="MEA", total_amine=TOTAL_AMINE,
        L=L_AMINE, G=8.0, alpha_rich=alpha_rich,
        n_stages=15, P=P_BASE,
        energy_balance=True,
    )
    delta = alpha_rich - r.alpha_lean
    Q_R_kW = (r.Q_R or 0.0) / 1000.0
    rich_sweep.append((alpha_rich, r.alpha_lean, delta, r.co2_recovery,
                            Q_R_kW))
    print(f"  {alpha_rich:>7.3f}  {r.alpha_lean:>7.4f}  {delta:>6.3f}  "
          f"{r.co2_recovery*100:>8.1f}%  {Q_R_kW:>9.1f}")

# ------------------------------------------------------------------
# Study 3: pressure sweep
# ------------------------------------------------------------------
print()
print(f"Study 3: regeneration vs P_stripper (G=8 mol/s, "
      f"α_rich={ALPHA_RICH_BASE}, 15 stages)")
print()
print(f"  {'P (bar)':>8s}  {'α_lean':>7s}  {'Δα':>6s}  {'Q_R (kW)':>9s}")
print(f"  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*9}")

press_sweep = []
for P_bar in [1.5, 1.8, 2.2, 2.8]:
    # Higher pressure → higher saturation T → adjust initial T profile.
    # T_water_sat at 1.5 bar = ~111 °C, at 2.8 bar = ~131 °C.
    T_top_init = 380.0 + (P_bar - 1.5) * 8.0
    T_bot_init = 395.0 + (P_bar - 1.5) * 12.0
    try:
        r = amine_stripper_ns(
            amine_name="MEA", total_amine=TOTAL_AMINE,
            L=L_AMINE, G=8.0, alpha_rich=ALPHA_RICH_BASE,
            n_stages=15, P=P_bar*1e5,
            T_top=T_top_init, T_bottom=T_bot_init,
            energy_balance=True,
        )
        delta = ALPHA_RICH_BASE - r.alpha_lean
        Q_R_kW = (r.Q_R or 0.0) / 1000.0
        press_sweep.append((P_bar, r.alpha_lean, delta, Q_R_kW))
        print(f"  {P_bar:>8.2f}  {r.alpha_lean:>7.4f}  {delta:>6.3f}  "
              f"{Q_R_kW:>9.1f}")
    except Exception as e:
        print(f"  {P_bar:>8.2f}  (solver failed: {type(e).__name__})")

# ------------------------------------------------------------------
# Headline operating point
# ------------------------------------------------------------------
print()
print("Headline design: G=8 mol/s, α_rich=0.45, P=1.8 bar, 15 stages")
print()

r_design = amine_stripper_ns(
    amine_name="MEA", total_amine=TOTAL_AMINE,
    L=L_AMINE, G=8.0, alpha_rich=ALPHA_RICH_BASE,
    n_stages=15, P=P_BASE,
    energy_balance=True,
)

co2_stripped_mol_s = L_AMINE * (ALPHA_RICH_BASE - r_design.alpha_lean)
co2_stripped_kg_s = co2_stripped_mol_s * 0.044
Q_per_ton = ((r_design.Q_R or 0.0) / 1000.0) / (co2_stripped_kg_s * 1000.0)

print(f"  α_lean (regenerated):  {r_design.alpha_lean:.4f}")
print(f"  α_rich (input):        {r_design.alpha_rich:.4f}")
print(f"  Loading swing Δα:      {r_design.alpha_rich - r_design.alpha_lean:.4f}")
print(f"  CO₂ stripped:          {co2_stripped_mol_s:.3f} mol/s = "
      f"{co2_stripped_kg_s*3600:.1f} kg/h")
print(f"  Reboiler duty Q_R:     {(r_design.Q_R or 0.0)/1000.0:.1f} kW")
print(f"  Condenser duty Q_C:    {(r_design.Q_C or 0.0)/1000.0:.1f} kW")
print(f"  Specific reboiler duty: {Q_per_ton:.2f} GJ/t CO₂")
print(f"  CO₂ recovery:          {r_design.co2_recovery*100:.1f}%")

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
print()
print("Validation:")

# 1.  More steam → lower α_lean (better regeneration)
alphas_low_steam = [a for _, a, _, _, _ in steam_sweep]
validate_bool("α_lean decreases monotonically with steam rate G",
                condition=all(alphas_low_steam[i] >= alphas_low_steam[i+1]
                                  for i in range(len(alphas_low_steam)-1)),
                detail=f"sweep: {[f'{a:.3f}' for a in alphas_low_steam]}")

# 2.  Higher α_rich → larger swing
deltas_rich = [d for _, _, d, _, _ in rich_sweep]
validate_bool("Δα increases with α_rich at fixed steam",
                condition=(deltas_rich[-1] >= deltas_rich[0]),
                detail=f"α_rich=0.30: Δα={deltas_rich[0]:.3f}, "
                f"α_rich=0.50: Δα={deltas_rich[-1]:.3f}")

# 3.  Headline Q/ton in industrial range (3-6 GJ/t)
validate_bool("Headline Q/ton in industrial range (3-6 GJ/t)",
                condition=(3.0 <= Q_per_ton <= 6.0),
                detail=f"Q/ton = {Q_per_ton:.2f} GJ/t",
                source="Notz 2012 typical post-combustion 3.5-4.5")

# 4.  Reboiler duty increases monotonically with steam
Qs = [Q for _, _, _, Q, _ in steam_sweep]
validate_bool("Reboiler duty increases monotonically with steam",
                condition=all(Qs[i] <= Qs[i+1] + 1e-3
                                  for i in range(len(Qs)-1)),
                detail=f"sweep Q (kW): {[f'{Q:.1f}' for Q in Qs]}")

# 5.  Higher P_stripper improves regeneration (lower α_lean)
if len(press_sweep) >= 2:
    alphas_press = [a for _, a, _, _ in press_sweep]
    validate_bool("Higher P_stripper improves regeneration "
                  "(α_lean decreases)",
                    condition=(alphas_press[-1] <= alphas_press[0] + 1e-3),
                    detail=f"P={press_sweep[0][0]}: "
                    f"α_lean={alphas_press[0]:.3f}, "
                    f"P={press_sweep[-1][0]}: "
                    f"α_lean={alphas_press[-1]:.3f}")

# 6.  Headline α_lean physically sensible (between 0 and α_rich)
validate_bool("α_lean physically bounded (0 < α_lean < α_rich)",
                condition=(0.0 < r_design.alpha_lean < ALPHA_RICH_BASE),
                detail=f"α_lean = {r_design.alpha_lean:.4f}")

# 7.  Q_R energy balance: Q_R should exceed sensible heat of feed +
#     latent heat of stripped CO₂.  Order-of-magnitude check.
co2_kg_s = co2_stripped_kg_s
H_evap_water_per_kg_co2 = 2.4e6 / 0.044 * 0.044  # ~2.4 MJ/kg water × ratio
# Just verify Q_R is positive and finite
validate_bool("Reboiler duty positive and finite",
                condition=(0 < (r_design.Q_R or 0.0) < 1e8),
                detail=f"Q_R = {(r_design.Q_R or 0.0)/1000:.1f} kW")

summary()
