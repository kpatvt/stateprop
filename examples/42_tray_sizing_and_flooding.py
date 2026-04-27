"""Tray sizing and flooding analysis for a benzene/toluene column.

What this demonstrates
----------------------
After Naphtali-Sandholm has converged the column profile (V, L, T, x,
y per stage), the tray-sizing module
(`stateprop.distillation.tray_hydraulics`) finds the column diameter
that keeps every stage below a chosen flooding fraction (typically
75-85% of flood velocity).

This is the gap between a *converged separation* and a *real piece of
equipment* — sizing turns the column profile into a diameter, plate
spacing, weir geometry, downcomer area, etc.

The workflow:
1. Solve the separation with `distillation_column()`
2. Pull the V, L, T, x, y profiles from the result
3. Call `size_tray_diameter()` for the minimum diameter at 75% flood
4. Call `tray_hydraulics()` once with that diameter for the full
   stage-by-stage hydraulic check

We size a 12-stage benzene/toluene column at three feed rates and
three flooding-fraction targets to show how diameter scales.

Reference
---------
Kister, H. Z. (1992). Distillation Design.  McGraw-Hill.
Chapter 6 — Tray Capacity and Hydraulics; Chapter 14 — Tray Sizing.

Approximate runtime: ~10 seconds.

Public APIs invoked
-------------------
- stateprop.distillation.distillation_column
- stateprop.distillation.tray_hydraulics.tray_hydraulics
- stateprop.distillation.tray_hydraulics.size_tray_diameter
- stateprop.distillation.tray_hydraulics.TrayDesign
- stateprop.activity.compounds.make_unifac

"""
import sys, warnings
sys.path.insert(0, '.')
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np

from stateprop.distillation import distillation_column
from stateprop.distillation.tray_hydraulics import (
    TrayDesign, tray_hydraulics, size_tray_diameter,
)
from stateprop.activity.compounds import make_unifac
from examples._harness import validate, validate_bool, summary


def antoine(A, B, C):
    """Antoine equation P_sat(T_K) returning Pa."""
    def f(T):
        return 10 ** (A - B / ((T - 273.15) + C)) * 133.322
    return f


# ------------------------------------------------------------------
# Build and solve the benzene/toluene column
# ------------------------------------------------------------------
print("=" * 70)
print("Tray sizing for a benzene/toluene distillation column")
print("=" * 70)
print()

species = ["benzene", "toluene"]
uf = make_unifac(species)
psats = [
    antoine(6.90565, 1211.033, 220.790),   # benzene
    antoine(6.95464, 1344.800, 219.482),   # toluene
]

# Headline operating point: 12 stages, R=2.0, equimolar feed
def solve_column(F):
    """Solve the column at feed rate F [mol/s].  Recovers all profiles."""
    return distillation_column(
        n_stages=12, feed_stage=6,
        feed_F=F, feed_z=[0.5, 0.5], feed_T=355.0,
        reflux_ratio=2.0, distillate_rate=0.5 * F,
        pressure=101325.0,
        species_names=species, activity_model=uf,
        psat_funcs=psats,
    )


# Compose the V_profile, L_profile, T_profile, x_profile, y_profile
# inputs to the hydraulics module from a column result.
def get_profiles(res):
    """Extract per-stage flows + compositions from a column result.

    The N-S framework returns interface flows (V[j] is the vapor
    leaving stage j upward, L[j] is the liquid leaving stage j
    downward).  For tray hydraulics we want stage-traffic flows; for
    a CMO column V[j] is constant in each section so the distinction
    is immaterial above a few-percent level.
    """
    return dict(
        V_profile=list(res.V),
        L_profile=list(res.L),
        T_profile=list(res.T),
        x_profile=np.asarray(res.x),
        y_profile=np.asarray(res.y),
    )


# ------------------------------------------------------------------
# Study 1: scan feed rate, find required diameter at 75% flood
# ------------------------------------------------------------------
print("Study 1: required tower diameter vs feed rate (75% flooding target)")
print()
print(f"  {'Feed (mol/s)':>13s}  {'Diameter (m)':>13s}  {'Throughput (kg/h)':>18s}")
print(f"  {'-'*13}  {'-'*13}  {'-'*18}")

# Average MW for the throughput display
mw_avg = 0.5 * 78.11 + 0.5 * 92.14   # benzene + toluene g/mol

diameters_at_75 = []
for F in [10.0, 30.0, 100.0]:
    res = solve_column(F)
    if not res.converged:
        print(f"  WARNING: column did not converge at F={F}")
        continue
    profiles = get_profiles(res)
    D = size_tray_diameter(
        **profiles, P=101325.0, species_names=species,
        spacing=0.6, weir_height=0.05, hole_area_frac=0.10,
        target_flood_frac=0.75,
    )
    throughput_kg_h = F * mw_avg / 1000.0 * 3600.0
    diameters_at_75.append((F, D, throughput_kg_h))
    print(f"  {F:>13.1f}  {D:>13.3f}  {throughput_kg_h:>18.0f}")

# ------------------------------------------------------------------
# Study 2: same column at fixed feed (30 mol/s), vary flooding target
# ------------------------------------------------------------------
print()
print("Study 2: required diameter vs target flooding fraction "
      "(F=30 mol/s, 12 stages)")
print()
print(f"  {'Target flood %':>14s}  {'Diameter (m)':>13s}")
print(f"  {'-'*14}  {'-'*13}")

res = solve_column(30.0)
profiles = get_profiles(res)

diameters_vs_flood = []
for flood_frac in [0.50, 0.60, 0.75, 0.85]:
    D = size_tray_diameter(
        **profiles, P=101325.0, species_names=species,
        spacing=0.6, weir_height=0.05, hole_area_frac=0.10,
        target_flood_frac=flood_frac,
    )
    diameters_vs_flood.append((flood_frac, D))
    print(f"  {flood_frac*100:>13.0f}%  {D:>13.3f}")

# ------------------------------------------------------------------
# Study 3: full hydraulic profile at the chosen design diameter
# ------------------------------------------------------------------
print()
print("Study 3: stage-by-stage hydraulic profile at design diameter")

# Pick the 75% flood design from Study 2
F = 30.0
target_flood = 0.75
res = solve_column(F)
profiles = get_profiles(res)
D_design = size_tray_diameter(
    **profiles, P=101325.0, species_names=species,
    spacing=0.6, weir_height=0.05, hole_area_frac=0.10,
    target_flood_frac=target_flood,
)
td = TrayDesign(
    diameter=D_design, spacing=0.6,
    weir_height=0.05, hole_area_frac=0.10,
)
hyd = tray_hydraulics(
    **profiles, P=101325.0, species_names=species,
    tray_design=td,
)

print(f"  Design: D = {D_design:.3f} m, spacing = 0.60 m, weir_h = 0.05 m")
print(f"  Total pressure drop across {len(profiles['V_profile'])} stages: "
      f"{hyd.total_pressure_drop/1000:.2f} kPa")
print(f"  Maximum % flood: {hyd.max_pct_flood:.1f}% at stage {hyd.max_pct_flood_stage}")
print(f"  Stages flagged for flooding (>80%): "
      f"{[s for s in hyd.flooding_stages]}")
print(f"  Stages flagged for weeping (<50% min vel): "
      f"{[s for s in hyd.weeping_stages]}")

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
print()
print("Validation:")

# 1.  Tower diameter at F=30 mol/s, 75% flood, should be in the
#     industrial envelope.  At 9200 kg/h benzene/toluene throughput,
#     a 0.5-1.5 m tower is the typical range.  We just check it lands
#     in this range (validates the calculation is in bounds, not a
#     specific Kister number — which would require a specific reference
#     case from the textbook).
F30_diameter = next(d for f, d, _ in diameters_at_75 if f == 30.0)
validate_bool("Diameter at F=30 mol/s, 75% flood in industrial range",
                condition=(0.5 <= F30_diameter <= 1.5),
                detail=f"D = {F30_diameter:.3f} m (envelope 0.5-1.5 m)",
                source="Kister 1992 industrial distillation envelope")

# 2.  Smaller flood-frac target → larger diameter (monotonic)
flood_50 = next(d for f, d in diameters_vs_flood if f == 0.50)
flood_85 = next(d for f, d in diameters_vs_flood if f == 0.85)
validate_bool("Diameter monotonic in flood target "
              "(50% target > 85% target)",
                condition=(flood_50 > flood_85),
                detail=f"D(50%)={flood_50:.3f} m, D(85%)={flood_85:.3f} m",
                source="Theoretical: lower flood frac → larger diameter")

# 3.  Throughput scales with diameter² (mass-balance / cross-sectional area)
#     The diameter for 100 mol/s should be ~sqrt(100/30) ≈ 1.83× that
#     of 30 mol/s, in the absence of liquid-loading effects.
F30_d = next(d for f, d, _ in diameters_at_75 if f == 30.0)
F100_d = next(d for f, d, _ in diameters_at_75 if f == 100.0)
ratio = F100_d / F30_d
expected = (100.0 / 30.0) ** 0.5
validate("Throughput scaling: D ∝ sqrt(F) at fixed flooding target",
          reference=expected, computed=ratio,
          units="-", tol_rel=0.20,
          source="Mass balance: A_active ∝ V → D ∝ sqrt(V)")

# 4.  At the design diameter, max % flood ≤ target
validate_bool(f"Hydraulic check: max %flood ≤ design target "
              f"({target_flood*100:.0f}%)",
                condition=(hyd.max_pct_flood <= target_flood * 100 + 1.0),
                detail=f"actual max = {hyd.max_pct_flood:.1f}%",
                source="size_tray_diameter() bisection guarantee")

# 5.  No stages flooding at design conditions
validate_bool("No stages flooding at design diameter",
                condition=(len(hyd.flooding_stages) == 0),
                detail=f"flagged stages: {hyd.flooding_stages}")

summary()
