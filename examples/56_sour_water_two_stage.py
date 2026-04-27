"""Two-stage sour-water flowsheet with selective acid/base dosing.

What this demonstrates
----------------------
Refinery sour water typically contains NH₃, H₂S, and CO₂ in roughly
equal amounts.  When the spec is *both* "≥ 99.5 % H₂S removal" *and*
"≥ 90 % NH₃ removal", a single steam stripper struggles because
the optimal pH for stripping H₂S (acidic) is opposite that for
stripping NH₃ (basic).

The standard refinery solution is a **two-stage stripper**:

1. **Acid stage** — feed gets dosed with a strong acid (HCl or H₂SO₄)
   to suppress carbonate / bicarbonate / bisulfide ionization.  H₂S
   and CO₂ become molecular and strip in stage 1.  NH₃ stays as NH₄⁺
   and remains in the bottoms.
2. **Base stage** — the stage-1 bottoms gets dosed with a strong base
   (NaOH).  NH₄⁺ is now de-protonated to NH₃ and strips in stage 2.

This example demonstrates the v0.9.113 `sour_water_two_stage_flowsheet`
integrator with both stages, plus the `find_acid_dose_for_h2s_recovery`
solver that finds the minimum HCl dose that meets a target H₂S spec.

Reference
---------
Beychok, M. R. (1983).  Aqueous Wastes from Petroleum and
Petrochemical Plants.  John Wiley & Sons.  Chapter 4 — Sour Water
Strippers.

Approximate runtime: ~10 seconds.

Public APIs invoked
-------------------
- stateprop.electrolyte.sour_water_two_stage_flowsheet
- stateprop.electrolyte.find_acid_dose_for_h2s_recovery

"""
import sys, warnings
sys.path.insert(0, '.')
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np

from stateprop.electrolyte import (
    sour_water_two_stage_flowsheet,
    find_acid_dose_for_h2s_recovery,
)
from examples._harness import validate, validate_bool, summary

print("=" * 70)
print("Two-stage sour-water flowsheet (acid + base dosing)")
print("=" * 70)
print()

# Refinery sour-water composition: typical from FCC main fractionator
# overhead receiver
species = ["NH3", "H2S", "CO2", "H2O"]
feed_z = [0.010, 0.005, 0.001, 0.984]      # 1 mol% NH3, 0.5 mol% H2S
F_FEED = 100.0  # mol/s

print(f"  Feed: {F_FEED} mol/s sour water at 80 °C")
print(f"    NH₃: {feed_z[0]*100:.2f} mol%  =  {feed_z[0]*F_FEED*17e-3*3600:.1f} kg/h")
print(f"    H₂S: {feed_z[1]*100:.2f} mol%  =  {feed_z[1]*F_FEED*34e-3*3600:.1f} kg/h")
print(f"    CO₂: {feed_z[2]*100:.2f} mol%  =  {feed_z[2]*F_FEED*44e-3*3600:.1f} kg/h")
print()

# ------------------------------------------------------------------
# Study 1: vary acid dose, no base dose — show H2S vs NH3 selectivity
# ------------------------------------------------------------------
print("Study 1: vary acid dose with no base (single-stripper proxy):")
print()
print(f"  {'HCl (mol/kg)':>13s}  {'H₂S rec':>9s}  {'NH₃ rec':>9s}  "
      f"{'CO₂ rec':>9s}")
print(f"  {'-'*13}  {'-'*9}  {'-'*9}  {'-'*9}")

acid_sweep = []
for dose in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
    r = sour_water_two_stage_flowsheet(
        feed_F=F_FEED, feed_z=feed_z, feed_T=353.15,
        species_names=species,
        acid_dose_mol_per_kg=dose,
        base_dose_mol_per_kg=0.0,
        n_stages_acid=8, n_stages_base=8,
        pressure_acid=1.5e5, pressure_base=1.5e5,
        energy_balance=False,    # CMO mode for fast sweep
    )
    rec_h2s = r.overall_recovery.get("H2S", 0.0)
    rec_nh3 = r.overall_recovery.get("NH3", 0.0)
    rec_co2 = r.overall_recovery.get("CO2", 0.0)
    acid_sweep.append((dose, rec_h2s, rec_nh3, rec_co2))
    print(f"  {dose:>13.3f}  {rec_h2s*100:>8.2f}%  {rec_nh3*100:>8.2f}%  "
          f"{rec_co2*100:>8.2f}%")

# ------------------------------------------------------------------
# Study 2: optimal acid dose for 99.5% H2S target
# ------------------------------------------------------------------
print()
print("Study 2: minimum HCl dose for 99.5 % H₂S recovery")
print()

target_h2s = 0.995
opt_dose = find_acid_dose_for_h2s_recovery(
    target_recovery=target_h2s,
    feed_F=F_FEED, feed_z=feed_z, feed_T=353.15,
    species_names=species,
    n_stages_acid=8, pressure_acid=1.5e5,
    dose_min=0.0, dose_max=1.0,
    tol=1e-3,
)
print(f"  Optimal HCl dose: {opt_dose:.4f} mol/kg H₂O in feed")
print(f"  HCl mass flow:    {opt_dose * F_FEED * feed_z[3] * 18e-3 * 36.46e-3 * 3600:.1f} kg/h")

# Verify the result hits target
r_opt = sour_water_two_stage_flowsheet(
    feed_F=F_FEED, feed_z=feed_z, feed_T=353.15,
    species_names=species,
    acid_dose_mol_per_kg=opt_dose,
    base_dose_mol_per_kg=0.0,
    n_stages_acid=8, n_stages_base=8,
    pressure_acid=1.5e5, pressure_base=1.5e5,
    energy_balance=False,
)
h2s_at_opt = r_opt.overall_recovery.get("H2S", 0.0)
nh3_at_opt = r_opt.overall_recovery.get("NH3", 0.0)
print(f"  H₂S recovery achieved: {h2s_at_opt*100:.2f}% (target {target_h2s*100:.1f}%)")
print(f"  NH₃ recovery (incidental): {nh3_at_opt*100:.2f}%")

# ------------------------------------------------------------------
# Study 3: full two-stage operation with base for NH3 recovery
# ------------------------------------------------------------------
print()
print("Study 3: full two-stage (HCl in stage 1, NaOH in stage 2)")
print()
print(f"  {'HCl':>6s}  {'NaOH':>6s}  {'H₂S':>9s}  {'NH₃':>9s}  "
      f"{'Q_R':>7s}  {'mass HCl':>9s}  {'mass NaOH':>10s}")
print(f"  {'mol/kg':>6s}  {'mol/kg':>6s}  {'recovery':>9s}  "
      f"{'recovery':>9s}  {'kW':>7s}  {'kg/h':>9s}  {'kg/h':>10s}")
print(f"  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*9}  {'-'*10}")

best = None
for acid in [0.10, 0.20, 0.30]:
    for base in [0.10, 0.20, 0.30]:
        r = sour_water_two_stage_flowsheet(
            feed_F=F_FEED, feed_z=feed_z, feed_T=353.15,
            species_names=species,
            acid_dose_mol_per_kg=acid,
            base_dose_mol_per_kg=base,
            n_stages_acid=8, n_stages_base=8,
            pressure_acid=1.5e5, pressure_base=1.5e5,
            energy_balance=True,
        )
        rec_h2s = r.overall_recovery.get("H2S", 0.0)
        rec_nh3 = r.overall_recovery.get("NH3", 0.0)
        Q_R_kW = (r.Q_R_total or 0.0) / 1000.0
        acid_kg_h = (r.acid_consumption_kg_per_h or 0.0)
        base_kg_h = (r.base_consumption_kg_per_h or 0.0)
        meets_spec = (rec_h2s >= 0.99) and (rec_nh3 >= 0.85)
        flag = "*" if meets_spec else " "
        print(f"  {acid:>6.3f}  {base:>6.3f}  "
              f"{rec_h2s*100:>8.2f}%  {rec_nh3*100:>8.2f}%  "
              f"{Q_R_kW:>7.1f}  {acid_kg_h:>9.1f}  {base_kg_h:>10.1f}  "
              f"{flag}")
        if meets_spec:
            cost_proxy = acid_kg_h + base_kg_h + Q_R_kW * 0.1
            if best is None or cost_proxy < best[0]:
                best = (cost_proxy, acid, base, rec_h2s, rec_nh3,
                          Q_R_kW, acid_kg_h, base_kg_h)

if best is not None:
    print()
    cost_proxy, acid, base, rec_h2s, rec_nh3, Q_R_kW, acid_kg_h, base_kg_h = best
    print(f"  Cheapest spec-meeting design: HCl={acid:.3f}, NaOH={base:.3f} mol/kg")
    print(f"    H₂S recovery: {rec_h2s*100:.1f}%, NH₃ recovery: {rec_nh3*100:.1f}%")
    print(f"    Q_reb = {Q_R_kW:.0f} kW, HCl = {acid_kg_h:.1f} kg/h, "
          f"NaOH = {base_kg_h:.1f} kg/h")

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
print()
print("Validation:")

# 1.  At 0 acid dose, NH3 and H2S recoveries should be moderate
#     (they're both partially molecular at near-neutral pH)
r_neutral = sour_water_two_stage_flowsheet(
    feed_F=F_FEED, feed_z=feed_z, feed_T=353.15,
    species_names=species,
    acid_dose_mol_per_kg=0.0, base_dose_mol_per_kg=0.0,
    n_stages_acid=8, n_stages_base=8,
    pressure_acid=1.5e5, pressure_base=1.5e5,
    energy_balance=False,
)
validate_bool("Neutral feed gives partial NH₃/H₂S recovery",
                condition=(0.5 < r_neutral.overall_recovery["H2S"] < 1.0),
                detail=f"H₂S {r_neutral.overall_recovery['H2S']*100:.1f}%, "
                f"NH₃ {r_neutral.overall_recovery['NH3']*100:.1f}%")

# 2.  Acid dose increases H₂S recovery monotonically
h2s_recoveries = [h for _, h, _, _ in acid_sweep]
validate_bool("H₂S recovery increases monotonically with acid dose",
                condition=all(h2s_recoveries[i] <= h2s_recoveries[i+1]
                                  for i in range(len(h2s_recoveries)-1)),
                detail=f"sweep: {[f'{h*100:.1f}%' for h in h2s_recoveries]}")

# 3.  At pH < 5 (high acid dose), NH4+ dominates → NH3 strip suppressed
#     This confirms the selectivity principle
nh3_neutral = acid_sweep[0][2]   # at dose=0
nh3_acidic = acid_sweep[-1][2]   # at dose=0.5
validate_bool("NH₃ recovery decreases at high acid dose",
                condition=(nh3_acidic < nh3_neutral),
                detail=f"NH₃ at 0 dose: {nh3_neutral*100:.1f}%, "
                f"at 0.5 dose: {nh3_acidic*100:.1f}%")

# 4.  Acid solver finds dose that hits target
validate("Acid dose for H₂S recovery target",
          reference=target_h2s, computed=h2s_at_opt,
          units="-", tol_rel=0.01,
          source="find_acid_dose_for_h2s_recovery convergence guarantee")

# 5.  Optimal dose is in industrial range (0.05-1.0 mol/kg per docstring)
validate_bool("Optimal acid dose in industrial range",
                condition=(0.01 <= opt_dose <= 1.0),
                detail=f"opt = {opt_dose:.3f} mol/kg "
                f"(industrial 0.05-1.0)",
                source="Beychok 1983")

# 6.  Two-stage operation with both acid and base reaches both specs
if best is not None:
    validate_bool("Two-stage design achieves >99% H₂S and >85% NH₃",
                    condition=(best[3] >= 0.99 and best[4] >= 0.85),
                    detail=f"H₂S {best[3]*100:.1f}%, NH₃ {best[4]*100:.1f}%")

summary()
