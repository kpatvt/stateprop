"""Methanol synthesis equilibrium via Gibbs minimization.

What this demonstrates
----------------------
Methanol is synthesized industrially from syngas via two coupled
reactions:

    CO  + 2 H₂ ⇌ CH₃OH                    (methanolation, exothermic)
    CO₂ + 3 H₂ ⇌ CH₃OH + H₂O              (carbon-dioxide hydrogenation)
    CO  +   H₂O ⇌ CO₂ + H₂                 (water-gas shift, fast)

These three are linearly dependent — only two extents are independent.
Rather than choose a basis, **Gibbs minimization** treats all five
species as unknowns and finds the composition that minimizes total
Gibbs energy subject to atom balance (C, H, O conservation).  This
is the most robust formulation because it doesn't require choosing
which reactions to include or worrying about reaction-path
constraints.

The approach: provide standard chemical potentials μ°(T) for each
species (from JANAF tables or DIPPR-style enthalpy/entropy
correlations), atomic formulas, and initial moles.  The solver
returns equilibrium moles satisfying both the atom balance and the
non-linear chemical-potential equality conditions.

This example sweeps T (200-350 °C) and p (10-150 bar) for typical
syngas compositions, recovering the well-known industrial result:
**low T and high p favor methanol**.  We compare against published
equilibrium curves (Bissett 1977, Skrzypek 1991).

Reference
---------
Bissett, L. A. (1977). Equilibrium constants for shift reactions
and methanol synthesis.  Chem. Eng. 84, 155.

Skrzypek, J.; Lachowska, M.; Moroz, H. (1991). Kinetics of methanol
synthesis over commercial copper/zinc oxide/alumina catalysts.
Chem. Eng. Sci. 46, 2809.

Approximate runtime: ~5 seconds.

Public APIs invoked
-------------------
- stateprop.reaction.gibbs_minimize_from_thermo
- stateprop.reaction.thermo.BUILTIN_SPECIES
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np

from stateprop.reaction.thermo import BUILTIN_SPECIES
from stateprop.reaction import gibbs_minimize_from_thermo
from examples._harness import validate, validate_bool, summary

print("=" * 70)
print("Methanol synthesis: Gibbs minimization")
print("=" * 70)
print()

# Five-species system
species_names = ["CO", "CO2", "H2", "CH3OH", "H2O"]
labels       = ["CO", "CO₂", "H₂", "CH₃OH", "H₂O"]
formulas = [
    {"C": 1, "O": 1},
    {"C": 1, "O": 2},
    {"H": 2},
    {"C": 1, "H": 4, "O": 1},
    {"H": 2, "O": 1},
]
species_thermo = [BUILTIN_SPECIES[s] for s in species_names]

# Initial composition: typical industrial syngas (after WGS adjustment)
# H2:CO = 2.6, with some CO2 from steam reforming
N_TOTAL_FEED = 100.0
n_feed = {
    "CO":    25.0,
    "CO2":    5.0,
    "H2":   65.0,
    "CH3OH": 1e-6,    # tiny but nonzero (solver requires > 0)
    "H2O":    5.0,
}
n_init = [n_feed[s] for s in species_names]

print(f"  Feed (per 100 mol total):")
for lbl, ni in zip(labels, n_init):
    print(f"    {lbl:>6s}: {ni:>6.2f} mol")
print(f"  H₂/CO ratio: {n_init[2]/n_init[0]:.2f}")
print()

# ------------------------------------------------------------------
# Study 1: T sweep at 50 bar
# ------------------------------------------------------------------
print("Study 1: equilibrium vs T at p = 50 bar")
print()
print(f"  {'T (°C)':>7s}  {'CH₃OH (mol)':>12s}  {'X_CO':>6s}  "
      f"{'X_CO₂':>7s}  {'H₂O (mol)':>10s}")
print(f"  {'-'*7}  {'-'*12}  {'-'*6}  {'-'*7}  {'-'*10}")

T_sweep = []
for T_C in [200, 225, 250, 275, 300, 325, 350]:
    T = T_C + 273.15
    r = gibbs_minimize_from_thermo(T, 50e5, species_thermo, formulas,
                                            n_init)
    if not r.converged:
        print(f"  {T_C:>7d}  did not converge")
        continue
    n_meoh = float(r.n[3])
    X_CO = (n_init[0] - r.n[0]) / n_init[0]
    X_CO2 = (n_init[1] - r.n[1]) / n_init[1]
    n_h2o = float(r.n[4])
    T_sweep.append((T_C, n_meoh, X_CO, X_CO2, n_h2o))
    print(f"  {T_C:>7d}  {n_meoh:>12.3f}  {X_CO*100:>5.1f}%  "
          f"{X_CO2*100:>6.1f}%  {n_h2o:>10.3f}")

# ------------------------------------------------------------------
# Study 2: p sweep at 250 °C
# ------------------------------------------------------------------
print()
print("Study 2: equilibrium vs p at T = 250 °C")
print()
print(f"  {'p (bar)':>8s}  {'CH₃OH (mol)':>12s}  {'X_CO':>6s}")
print(f"  {'-'*8}  {'-'*12}  {'-'*6}")

p_sweep = []
T_design = 523.15
for p_bar in [5, 20, 50, 100, 150, 200]:
    r = gibbs_minimize_from_thermo(T_design, p_bar*1e5, species_thermo,
                                            formulas, n_init)
    if not r.converged:
        continue
    n_meoh = float(r.n[3])
    X_CO = (n_init[0] - r.n[0]) / n_init[0]
    p_sweep.append((p_bar, n_meoh, X_CO))
    print(f"  {p_bar:>8.0f}  {n_meoh:>12.3f}  {X_CO*100:>5.1f}%")

# ------------------------------------------------------------------
# Study 3: H2/CO ratio sweep at design conditions (250 °C, 50 bar)
# ------------------------------------------------------------------
print()
print("Study 3: equilibrium vs H₂/CO ratio at T=250°C, p=50 bar")
print()
print(f"  {'H₂/CO':>6s}  {'CH₃OH (mol)':>12s}  {'X_CO':>6s}")
print(f"  {'-'*6}  {'-'*12}  {'-'*6}")

ratio_sweep = []
for h2_co in [1.5, 2.0, 2.5, 3.0, 4.0]:
    n_co = 25.0
    n_h2 = h2_co * n_co
    n_init_v = [n_co, 5.0, n_h2, 1e-6, 5.0]
    r = gibbs_minimize_from_thermo(523.15, 50e5, species_thermo,
                                            formulas, n_init_v)
    if not r.converged:
        continue
    n_meoh = float(r.n[3])
    X_CO = (n_init_v[0] - r.n[0]) / n_init_v[0]
    ratio_sweep.append((h2_co, n_meoh, X_CO))
    print(f"  {h2_co:>6.1f}  {n_meoh:>12.3f}  {X_CO*100:>5.1f}%")

# ------------------------------------------------------------------
# Headline industrial design point
# ------------------------------------------------------------------
print()
print("Industrial design point: T=250°C, p=80 bar, H₂/CO=2.6")
print()

T_d, p_d = 523.15, 80e5
r_design = gibbs_minimize_from_thermo(T_d, p_d, species_thermo, formulas,
                                                n_init)
total_n = float(np.sum(r_design.n))
print(f"  Final equilibrium composition (mol%):")
for lbl, ni in zip(labels, r_design.n):
    print(f"    {lbl:>6s}: {float(ni)/total_n*100:>5.2f}%")
print(f"  CH₃OH yield: {float(r_design.n[3]):.2f} mol per 100 mol feed")
print(f"  CO conversion: {(n_init[0]-r_design.n[0])/n_init[0]*100:.1f}%")

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
print()
print("Validation:")

# 1.  Le Chatelier: increasing p favors methanol (reduces total moles)
mehoh_p = [m for _, m, _ in p_sweep]
validate_bool("Increasing p increases CH₃OH yield (Le Chatelier)",
                condition=all(mehoh_p[i] <= mehoh_p[i+1]
                                  for i in range(len(mehoh_p)-1)),
                detail=f"sweep: "
                f"{[f'{m:.2f}' for m in mehoh_p]} mol")

# 2.  Le Chatelier: decreasing T favors methanol (exothermic reaction)
mehoh_t = [m for _, m, _, _, _ in T_sweep]
validate_bool("Decreasing T increases CH₃OH yield (exothermic)",
                condition=(mehoh_t[0] > mehoh_t[-1]),
                detail=f"T=200°C: {mehoh_t[0]:.2f} mol vs "
                f"T=350°C: {mehoh_t[-1]:.2f} mol")

# 3.  At 250 °C, 50 bar with 2.6:1 H2:CO, expect ~12-15 mol MeOH per 100 mol
#     (Skrzypek 1991 Figure 2 baseline)
T_design_pt = next((m for T_C, m, _, _, _ in T_sweep if T_C == 250), None)
if T_design_pt is not None:
    validate("CH₃OH yield at 250°C, 50 bar, H₂:CO=2.6",
              reference=13.0, computed=T_design_pt,
              units="mol per 100 mol feed",
              tol_rel=0.30,
              source="Skrzypek 1991; industry-baseline equilibrium")

# 4.  Atom balance: total C, H, O must be conserved
n_arr = np.asarray(r_design.n)
C_initial = sum(n_init[i] * formulas[i].get("C", 0)
                 for i in range(len(species_names)))
C_final = sum(float(n_arr[i]) * formulas[i].get("C", 0)
               for i in range(len(species_names)))
validate("Carbon atom balance preserved",
          reference=C_initial, computed=C_final,
          units="mol C", tol_rel=1e-6,
          source="Theoretical: atom balance is a hard constraint")

H_initial = sum(n_init[i] * formulas[i].get("H", 0)
                 for i in range(len(species_names)))
H_final = sum(float(n_arr[i]) * formulas[i].get("H", 0)
               for i in range(len(species_names)))
validate("Hydrogen atom balance preserved",
          reference=H_initial, computed=H_final,
          units="mol H", tol_rel=1e-6,
          source="Theoretical: atom balance is a hard constraint")

O_initial = sum(n_init[i] * formulas[i].get("O", 0)
                 for i in range(len(species_names)))
O_final = sum(float(n_arr[i]) * formulas[i].get("O", 0)
               for i in range(len(species_names)))
validate("Oxygen atom balance preserved",
          reference=O_initial, computed=O_final,
          units="mol O", tol_rel=1e-6,
          source="Theoretical: atom balance is a hard constraint")

# 5.  All n_i ≥ 0 (no negative moles)
validate_bool("All equilibrium mole numbers non-negative",
                condition=all(float(n_i) >= -1e-12 for n_i in r_design.n),
                detail=f"min n_i = {float(min(r_design.n)):.3e}")

# 6.  Higher H2/CO favors more CH3OH (more H2 = more reactant)
mehoh_r = [m for _, m, _ in ratio_sweep]
validate_bool("Higher H₂/CO ratio increases CH₃OH",
                condition=(mehoh_r[-1] > mehoh_r[0]),
                detail=f"H₂/CO=1.5: {mehoh_r[0]:.2f}, "
                f"H₂/CO=4.0: {mehoh_r[-1]:.2f} mol")

summary()
