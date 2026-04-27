"""Standalone amine absorber design with the rigorous N-S solver.

What this demonstrates
----------------------
Before tackling the full integrated capture flowsheet (example 60),
it's instructive to see the absorber column on its own.  The
v0.9.114 ``amine_absorber_ns()`` solves a multi-stage absorber as
a Naphtali-Sandholm column with the full activity-coefficient model,
returning per-stage compositions, T profile, and CO₂ recovery.

This example walks through the central design tradeoffs:

1. **L/G ratio sweep** — the solvent-to-gas molar ratio sets how much
   CO₂ the column can absorb.  Below the minimum L/G the column
   pinches and approaches an equilibrium limit far from the spec.
2. **Lean loading α_lean sweep** — fresher solvent (lower α_lean)
   gives more driving force at the top of the column.  α_lean is
   set by the regenerator design (example 59) so this connects
   absorber and stripper.
3. **Stage count sweep** — diminishing returns above ~12 stages for
   typical industrial designs.

Reference
---------
Cousins, A.; Wardhaugh, L. T.; Feron, P. H. M. (2011). Survey of
process flow sheet modifications for energy efficient CO₂ capture
from flue gases using chemical absorption.  Int. J. Greenhouse Gas
Control 5, 605-619.

Approximate runtime: ~15 seconds.

Public APIs invoked
-------------------
- stateprop.electrolyte.amine_absorber_ns
- stateprop.electrolyte.AmineNSResult
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np

from stateprop.electrolyte import amine_absorber_ns
from examples._harness import validate, validate_bool, summary

print("=" * 70)
print("Standalone amine absorber design (30 wt% MEA)")
print("=" * 70)
print()

# Plant inputs (representative coal-flue gas)
G_FLUE = 15.0    # mol/s
Y_CO2 = 0.12     # 12 % post-combustion coal
TOTAL_AMINE = 5.0  # ~30 wt% MEA → 5 mol amine/kg solvent
ALPHA_LEAN_BASE = 0.20

print(f"  Flue gas:  G = {G_FLUE} mol/s, y_CO2 = {Y_CO2*100:.0f}%")
print(f"  Solvent:   30 wt% MEA, α_lean baseline = {ALPHA_LEAN_BASE}")
print()

# ------------------------------------------------------------------
# Study 1: L/G ratio sweep
# ------------------------------------------------------------------
print("Study 1: CO₂ recovery vs L/G ratio (10 stages, α_lean=0.20)")
print()
print(f"  {'L (mol/s)':>10s}  {'L/G':>6s}  {'recovery':>9s}  {'α_rich':>7s}  "
      f"{'P_CO2_eq @ rich':>15s}")
print(f"  {'-'*10}  {'-'*6}  {'-'*9}  {'-'*7}  {'-'*15}")

LG_sweep = []
for L in [4.0, 6.0, 8.0, 10.0, 12.0, 15.0]:
    r = amine_absorber_ns(
        amine_name="MEA", total_amine=TOTAL_AMINE,
        L=L, G=G_FLUE, alpha_lean=ALPHA_LEAN_BASE, y_in_CO2=Y_CO2,
        n_stages=10, P=1.013e5,
    )
    LG_sweep.append((L, L/G_FLUE, r.co2_recovery, r.alpha_rich,
                     r.P_CO2_eq[-1]))
    print(f"  {L:>10.1f}  {L/G_FLUE:>6.3f}  {r.co2_recovery*100:>8.1f}%  "
          f"{r.alpha_rich:>7.4f}  {r.P_CO2_eq[-1]:>13.3f} bar")

# Locate the recovery knee — typically between L/G = 0.4 and 1.0
# for 12 % CO₂ feed.
print()

# ------------------------------------------------------------------
# Study 2: lean loading sweep at fixed L/G
# ------------------------------------------------------------------
print("Study 2: CO₂ recovery vs lean loading α_lean (L=8, G=15, "
      "10 stages)")
print()
print(f"  {'α_lean':>7s}  {'recovery':>9s}  {'α_rich':>7s}  "
      f"{'Δα = α_rich-α_lean':>20s}")
print(f"  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*20}")

alpha_sweep = []
for alpha_lean in [0.05, 0.10, 0.20, 0.30, 0.40]:
    r = amine_absorber_ns(
        amine_name="MEA", total_amine=TOTAL_AMINE,
        L=8.0, G=G_FLUE, alpha_lean=alpha_lean, y_in_CO2=Y_CO2,
        n_stages=10, P=1.013e5,
    )
    delta = r.alpha_rich - alpha_lean
    alpha_sweep.append((alpha_lean, r.co2_recovery, r.alpha_rich, delta))
    print(f"  {alpha_lean:>7.3f}  {r.co2_recovery*100:>8.1f}%  "
          f"{r.alpha_rich:>7.4f}  {delta:>20.4f}")

# ------------------------------------------------------------------
# Study 3: stage count sweep (diminishing returns)
# ------------------------------------------------------------------
print()
print("Study 3: CO₂ recovery vs n_stages (L=8, G=15, α_lean=0.20)")
print()
print(f"  {'N_stages':>8s}  {'recovery':>9s}  {'α_rich':>7s}")
print(f"  {'-'*8}  {'-'*9}  {'-'*7}")

ns_sweep = []
for N in [4, 6, 8, 10, 14, 20]:
    r = amine_absorber_ns(
        amine_name="MEA", total_amine=TOTAL_AMINE,
        L=8.0, G=G_FLUE, alpha_lean=0.20, y_in_CO2=Y_CO2,
        n_stages=N, P=1.013e5,
    )
    ns_sweep.append((N, r.co2_recovery, r.alpha_rich))
    print(f"  {N:>8d}  {r.co2_recovery*100:>8.1f}%  {r.alpha_rich:>7.4f}")

# ------------------------------------------------------------------
# Headline operating point: full column profile
# ------------------------------------------------------------------
print()
print("Headline design: L=8 mol/s, α_lean=0.10, 12 stages")
print()

r_design = amine_absorber_ns(
    amine_name="MEA", total_amine=TOTAL_AMINE,
    L=8.0, G=G_FLUE, alpha_lean=0.10, y_in_CO2=Y_CO2,
    n_stages=12, P=1.013e5,
)

print(f"  CO₂ recovery:       {r_design.co2_recovery*100:.1f}%")
print(f"  α_rich (bottoms):   {r_design.alpha_rich:.4f}")
print(f"  α_lean (top, given): {r_design.alpha_lean:.4f}")
print(f"  Loading swing Δα:   {r_design.alpha_rich - r_design.alpha_lean:.4f}")

print()
print("  Stage-by-stage loading α and equilibrium P_CO2:")
print(f"  {'stage':>5s}  {'α':>7s}  {'P_CO2_eq (bar)':>15s}")
print(f"  {'-'*5}  {'-'*7}  {'-'*15}")
for j, (a, p) in enumerate(zip(r_design.alpha, r_design.P_CO2_eq)):
    print(f"  {j+1:>5d}  {a:>7.4f}  {p:>13.4f}")

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
print()
print("Validation:")

# 1.  Recovery is monotonically increasing in L/G (more solvent → more
#     absorption)
recoveries = [r for _, _, r, _, _ in LG_sweep]
validate_bool("Recovery increases monotonically with L/G",
                condition=all(recoveries[i] <= recoveries[i+1] + 0.01
                                  for i in range(len(recoveries)-1)),
                detail=f"sweep recovery: "
                f"{[f'{r*100:.1f}%' for r in recoveries]}")

# 2.  Recovery is monotonically *decreasing* in α_lean (richer solvent
#     → less driving force at top of column)
recs_a = [r for _, r, _, _ in alpha_sweep]
validate_bool("Recovery decreases monotonically with α_lean",
                condition=all(recs_a[i] >= recs_a[i+1] - 0.01
                                  for i in range(len(recs_a)-1)),
                detail=f"sweep recovery: "
                f"{[f'{r*100:.1f}%' for r in recs_a]}")

# 3.  More stages → more recovery (with diminishing returns)
recs_n = [r for _, r, _ in ns_sweep]
validate_bool("Recovery increases with stage count",
                condition=(recs_n[-1] >= recs_n[0]),
                detail=f"N=4: {recs_n[0]*100:.1f}%, "
                f"N={ns_sweep[-1][0]}: {recs_n[-1]*100:.1f}%")

# 4.  Diminishing returns: doubling from 10 to 20 gives less than
#     doubling effect on recovery
N10 = next(r for n, r, _ in ns_sweep if n == 10)
N20 = next(r for n, r, _ in ns_sweep if n == 20)
gain_10_to_20 = N20 - N10
N4 = next(r for n, r, _ in ns_sweep if n == 4)
gain_4_to_10 = N10 - N4
validate_bool("Diminishing returns: 4→10 gain > 10→20 gain",
                condition=(gain_4_to_10 > gain_10_to_20),
                detail=f"4→10: +{gain_4_to_10*100:.1f}%, "
                f"10→20: +{gain_10_to_20*100:.1f}%")

# 5.  Headline design hits industrial target (CO₂ recovery > 85 %)
validate_bool("Headline design recovery > 85 % (industrial target)",
                condition=(r_design.co2_recovery > 0.85),
                detail=f"recovery = {r_design.co2_recovery*100:.1f}%",
                source="Cousins 2011 typical post-combustion target")

# 6.  α_rich physically bounded (must be > α_lean and < ~0.55 for MEA)
validate_bool("α_rich physically bounded (α_lean < α_rich < 0.6)",
                condition=(r_design.alpha_lean < r_design.alpha_rich < 0.6),
                detail=f"α_lean={r_design.alpha_lean:.3f} < "
                f"α_rich={r_design.alpha_rich:.3f} < 0.6")

# 7.  Loading swing Δα in industrial range (0.15-0.45)
delta_alpha = r_design.alpha_rich - r_design.alpha_lean
validate_bool("Headline Δα in expected range (0.15-0.45)",
                condition=(0.15 <= delta_alpha <= 0.45),
                detail=f"Δα = {delta_alpha:.3f}",
                source="Cousins 2011 typical 0.15-0.25")

summary()
