"""Solvent screening: MEA vs MDEA vs DEA at the same capture duty.

What this demonstrates
----------------------
Solvent selection is the first decision in CO₂ capture plant design.
The three workhorse alkanolamines have very different chemistry:

- **MEA** (monoethanolamine) — primary amine, fast kinetics, high
  enthalpy of absorption (-85 kJ/mol). Industry standard since the
  1930s. Downsides: degradation, corrosion, high regeneration energy.

- **DEA** (diethanolamine) — secondary amine, intermediate kinetics
  (-65 kJ/mol). Used in refinery sweetening. Slower than MEA but more
  resistant to degradation.

- **MDEA** (methyldiethanolamine) — tertiary amine, no carbamate
  formation (instead bicarbonate route via base catalysis), much
  lower ΔH_abs (-45 kJ/mol). Selectively absorbs H₂S over CO₂. The
  modern choice for low-energy designs and selective gas treating,
  often used in blends with PZ or MEA as activator.

The rule of thumb: lower ΔH_abs → lower regenerator duty BUT slower
kinetics, requires more stages or activators.

This example runs the full CaptureFlowsheet integrator for all three
amines at the same flue gas duty and compares Q/ton, capture
recovery, loading swing, and solvent inventory.

Reference
---------
Kohl, A. L.; Nielsen, R. (1997). Gas Purification (5th ed.). Gulf
Publishing.  Chapter 2 — Alkanolamines for Hydrogen Sulfide and
Carbon Dioxide Removal.

Approximate runtime: ~30 seconds.

Public APIs invoked
-------------------
- stateprop.electrolyte.CaptureFlowsheet
- stateprop.electrolyte.lookup_amine
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np

from stateprop.electrolyte import CaptureFlowsheet, lookup_amine
from examples._harness import validate, validate_bool, summary

print("=" * 70)
print("Solvent screening: MEA vs MDEA vs DEA")
print("=" * 70)
print()

# Plant duty: same coal flue gas, same recovery target, vary solvent
G_FLUE = 28.0
Y_CO2 = 0.12

# Operating conditions chosen at typical industrial baseline for each
# amine.  Baseline solvent flow was tuned to give roughly comparable
# duties.
amines = [
    ("MEA",  5.0, 8.0,  0.30, "primary, fast, ΔH_abs=-85 kJ/mol"),
    ("DEA",  4.5, 9.0,  0.30, "secondary, ΔH_abs=-65 kJ/mol"),
    ("MDEA", 4.0, 14.0, 0.50, "tertiary, slow but ΔH_abs=-45 kJ/mol"),
]

# Print physical-property table first
print("Physical properties of the three amines:")
print()
print(f"  {'amine':>5s}  {'MW':>5s}  {'pKa(25)':>7s}  {'ΔH_abs':>8s}  "
      f"{'tertiary':>9s}  {'description':>40s}")
print(f"  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*40}")
for name, _, _, _, desc in amines:
    a = lookup_amine(name)
    tag = "yes" if a.is_tertiary else "no"
    print(f"  {name:>5s}  {a.MW:>5.1f}  {a.pKa_25:>7.2f}  "
          f"{a.delta_H_abs/1000:>+7.0f}  {tag:>9s}  {desc:>40s}")

# ------------------------------------------------------------------
# Run the full flowsheet for each amine
# ------------------------------------------------------------------
print()
print(f"  Common spec: flue gas G={G_FLUE} mol/s, y_CO2={Y_CO2*100:.0f}%")
print(f"  Operating: T_abs_feed=40°C, P_abs=1 atm, P_strip=1.8 bar")
print()

print(f"  {'amine':>5s}  {'L_amine':>8s}  {'wt%':>4s}  {'recovery':>9s}  "
      f"{'α_lean':>7s}  {'α_rich':>7s}  {'Δα':>5s}  {'Q/ton':>7s}  "
      f"{'iter':>4s}")
print(f"  {'':>5s}  {'mol/s':>8s}  {'':>4s}  {'%':>9s}  "
      f"{'':>7s}  {'':>7s}  {'':>5s}  {'GJ/t':>7s}  {'':>4s}")
print(f"  {'-'*5}  {'-'*8}  {'-'*4}  {'-'*9}  "
      f"{'-'*7}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*4}")

results = {}
for name, total_amine, L, wt_frac, _ in amines:
    fs = CaptureFlowsheet(name, total_amine,
                                  n_stages_absorber=20,
                                  n_stages_stripper=15)
    try:
        r = fs.solve(
            G_flue=G_FLUE, y_in_CO2=Y_CO2, L_amine=L,
            T_absorber_feed=313.15, P_absorber=1.013,
            T_strip_top=378.15, T_strip_bottom=393.15,
            P_stripper=1.8, T_cond=313.15,
            delta_T_min_HX=5.0,
            wt_frac_amine=wt_frac,
            alpha_lean_init=0.05 if name == "MDEA" else 0.20,
            damp=0.5, max_outer=40, tol=1e-3,
        )
        results[name] = r
        delta = r.alpha_rich - r.alpha_lean
        print(f"  {name:>5s}  {L:>8.1f}  {wt_frac*100:>4.0f}  "
              f"{r.co2_recovery*100:>8.1f}%  {r.alpha_lean:>7.4f}  "
              f"{r.alpha_rich:>7.4f}  {delta:>5.3f}  "
              f"{r.Q_per_ton_CO2:>7.2f}  {r.iterations:>4d}")
    except Exception as e:
        print(f"  {name:>5s}  solver failed: {type(e).__name__}: {e}")
        results[name] = None

# ------------------------------------------------------------------
# Comparison and analysis
# ------------------------------------------------------------------
print()
print("Analysis:")
print()

successful = [(n, r) for n, r in results.items() if r is not None
              and r.converged]
if len(successful) >= 2:
    # Find lowest Q/ton among successful runs
    best = min(successful, key=lambda nr: nr[1].Q_per_ton_CO2)
    print(f"  Lowest Q/ton: {best[0]} at {best[1].Q_per_ton_CO2:.2f} GJ/t")
    # Highest recovery
    best_r = max(successful, key=lambda nr: nr[1].co2_recovery)
    print(f"  Highest recovery: {best_r[0]} at "
          f"{best_r[1].co2_recovery*100:.1f}%")

print()
print("Engineering takeaway:")
print(f"  - MEA: high recovery + high Q/ton (industry baseline)")
print(f"  - DEA: lower kinetic activity but moderate Q/ton")
print(f"  - MDEA: lowest ΔH_abs → lowest theoretical Q/ton, but")
print(f"    requires more solvent / stages / time at this column geometry")
print(f"    (real MDEA designs typically use PZ activator for kinetics)")

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
print()
print("Validation:")

# 1.  All three amines should converge for a reasonable spec
n_conv = sum(1 for r in results.values() if r is not None
             and r.converged)
validate_bool("≥2 of 3 amines converged",
                condition=(n_conv >= 2),
                detail=f"{n_conv}/3 converged")

# 2.  MEA should hit high recovery at the chosen L_amine
if results.get("MEA") and results["MEA"].converged:
    validate_bool("MEA recovery > 85 % at industrial-scale L_amine",
                    condition=(results["MEA"].co2_recovery > 0.85),
                    detail=f"MEA recovery = "
                    f"{results['MEA'].co2_recovery*100:.1f}%",
                    source="Industry baseline post-combustion")

# 3.  MEA Q/ton should land in industrial envelope
if results.get("MEA") and results["MEA"].converged:
    validate_bool("MEA Q/ton in 3-7 GJ/t envelope",
                    condition=(3.0 <= results["MEA"].Q_per_ton_CO2 <= 7.0),
                    detail=f"MEA Q/ton = "
                    f"{results['MEA'].Q_per_ton_CO2:.2f} GJ/t",
                    source="Notz 2012 / Cousins 2011")

# 4.  Loading swing positive for all converged amines
for name, r in results.items():
    if r is not None and r.converged:
        delta = r.alpha_rich - r.alpha_lean
        validate_bool(f"{name} α_rich > α_lean (positive swing)",
                        condition=(delta > 0),
                        detail=f"Δα = {delta:.3f}")

# 5.  MDEA, if converged, should have lower α_rich than MEA at same
#     conditions (lower equilibrium loading capacity for tertiary amines)
if (results.get("MEA") and results["MEA"].converged
        and results.get("MDEA") and results["MDEA"].converged):
    validate_bool("MDEA α_rich < MEA α_rich (tertiary lower capacity)",
                    condition=(results["MDEA"].alpha_rich
                                  < results["MEA"].alpha_rich),
                    detail=f"MDEA: {results['MDEA'].alpha_rich:.3f}, "
                    f"MEA: {results['MEA'].alpha_rich:.3f}",
                    source="Carbamate vs bicarbonate stoichiometry")

summary()
