"""TBP curve discretization: refinery feed characterization.

Demonstrates v0.9.91's TBP discretization tools — converting laboratory
distillation data (true-boiling-point or ASTM D86/D2887 simulated
distillation) into a discrete list of pseudo-components ready for
column / flash / EOS work.

Three discretization strategies are compared:

    * equal_volume    — refinery default; cuts span equal cum-vol % slices
    * equal_NBP       — Whitson recommendation; cuts span equal NBP intervals
    * gauss_laguerre  — fewer-cut approximation with heavier-tail emphasis

ASTM standard distillations (D86 atmospheric, D2887 simulated by GC)
need to be converted to TBP first via Daubert (1994) corrections
before discretization. This example shows the full chain.
"""
from __future__ import annotations

import warnings
warnings.simplefilter("ignore", RuntimeWarning)

from stateprop.tbp import (
    discretize_TBP, discretize_from_D86, discretize_from_D2887,
    interpolate_TBP, D86_to_TBP, D2887_to_TBP,
    API_to_SG, SG_to_API,
)


# =====================================================================
# Part 1: A typical diesel TBP curve from the lab
# =====================================================================

# Volume %, Temperature [°C → K]
TBP_volume = [0,    10,   30,   50,   70,   90,   100]
TBP_temp_C = [180,  220,  260,  290,  315,  345,  385]
TBP_temp_K = [t + 273.15 for t in TBP_temp_C]

print("=" * 70)
print("Part 1: Laboratory TBP for a diesel cut")
print("=" * 70)
print(f"\nLab data (cum-vol %, T [°C]):")
for v, t in zip(TBP_volume, TBP_temp_C):
    print(f"  {v:>4d}%   {t:>4d} °C  ({t + 273.15:.1f} K)")

# Quick interpolation at a non-tabulated point
T_at_25 = interpolate_TBP(25, TBP_volume, TBP_temp_K)
print(f"\nInterpolated at 25%: T = {T_at_25:.1f} K = {T_at_25 - 273.15:.1f} °C")


# =====================================================================
# Part 2: Compare three discretization methods
# =====================================================================

print("\n" + "=" * 70)
print("Part 2: Three discretization methods, 5 cuts each, API = 38°")
print("=" * 70)

for method in ["equal_volume", "equal_NBP", "gauss_laguerre"]:
    print(f"\n--- {method} ---")
    res = discretize_TBP(
        NBP_table=TBP_temp_K,
        volume_table=TBP_volume,
        n_cuts=5,
        API_gravity=38.0,
        method=method,
        name_prefix=f"D_{method[:3]}",
    )
    print(res.summary())


# =====================================================================
# Part 3: Cross-check that volume / mass / mole fractions sum to 1
# =====================================================================

print("\n" + "=" * 70)
print("Part 3: Numerical hygiene — fraction normalization")
print("=" * 70)

res = discretize_TBP(
    NBP_table=TBP_temp_K,
    volume_table=TBP_volume,
    n_cuts=8,
    API_gravity=38.0,
)
print(f"\n8 equal-volume cuts of the diesel TBP:")
print(f"  Σ volume_fractions = {res.volume_fractions.sum():.15f}")
print(f"  Σ mass_fractions   = {res.mass_fractions.sum():.15f}")
print(f"  Σ mole_fractions   = {res.mole_fractions.sum():.15f}")
print(f"\nAll three sums equal 1 to machine precision ✓")

# Volume continuity: NBP at the upper edge of cut i must equal the
# NBP at the lower edge of cut i+1.
print(f"\nVolume continuity at every internal cut boundary:")
for i in range(7):
    diff = abs(res.NBP_upper[i] - res.NBP_lower[i + 1])
    print(f"  cut{i+1}_hi - cut{i+2}_lo = {diff:.2e} K")


# =====================================================================
# Part 4: Watson K vs API gravity — which SG strategy?
# =====================================================================

print("\n" + "=" * 70)
print("Part 4: SG strategies — Watson K vs API gravity")
print("=" * 70)

# Watson K assigns each cut its own SG based on (1.8·NBP)^(1/3) / K_W;
# heavier cuts get higher SGs naturally, which is more realistic.
# API gravity gives every cut the same SG (the one corresponding to
# the overall stream API).
print(f"\nUsing API_gravity=38° (everything has same SG):")
res_api = discretize_TBP(NBP_table=TBP_temp_K, volume_table=TBP_volume,
                          n_cuts=5, API_gravity=38.0)
for cut in res_api.cuts:
    print(f"  {cut.name}: NBP={cut.NBP:.0f}K, SG={cut.SG:.4f}")

print(f"\nUsing Watson_K=11.8 (each cut gets its own SG):")
res_kw = discretize_TBP(NBP_table=TBP_temp_K, volume_table=TBP_volume,
                          n_cuts=5, Watson_K=11.8)
for cut in res_kw.cuts:
    print(f"  {cut.name}: NBP={cut.NBP:.0f}K, SG={cut.SG:.4f}")


# =====================================================================
# Part 5: ASTM D86 simulated distillation conversion
# =====================================================================

print("\n" + "=" * 70)
print("Part 5: ASTM D86 → TBP (Daubert 1994)")
print("=" * 70)

# A typical D86 distillation report (refinery practice — quicker
# and cheaper than full TBP).
D86_volume = [10, 30, 50, 70, 90]
D86_temp_C = [195, 235, 270, 310, 355]
D86_temp_K = [t + 273.15 for t in D86_temp_C]

print(f"\nLab D86 distillation (vol%, T_C):")
for v, t in zip(D86_volume, D86_temp_C):
    print(f"  {v:>3d}%   {t:>4d} °C")

# Convert to TBP first
TBP_from_D86 = D86_to_TBP(D86_volume, D86_temp_K)
print(f"\nDaubert (1994) D86 → TBP correction (in K):")
for v, T_d86, T_tbp in zip(D86_volume, D86_temp_K, TBP_from_D86):
    delta = T_tbp - T_d86
    print(f"  {v:>3d}%:  D86={T_d86:.1f} K  →  TBP={T_tbp:.1f} K  (Δ={delta:+.1f} K)")

# One-call wrapper: D86 directly to discretized cuts
res_D86 = discretize_from_D86(D86_volume, D86_temp_K,
                                n_cuts=4, API_gravity=37.0)
print(f"\n4 cuts directly from D86 data:")
print(res_D86.summary())


# =====================================================================
# Part 6: API ↔ SG conversions
# =====================================================================

print("\n" + "=" * 70)
print("Part 6: API ↔ SG round-trip")
print("=" * 70)

print(f"\n{'API °':>8s} {'SG':>10s} {'API back':>10s}")
for API in [10, 20, 30, 35, 40, 50]:
    SG = API_to_SG(API)
    API_back = SG_to_API(SG)
    print(f"{API:>8.1f} {SG:>10.4f} {API_back:>10.2f}")
print(f"\nWater (API=10° benchmark) → SG = {API_to_SG(10):.4f}")
print(f"Heavy crude (API=20°)      → SG = {API_to_SG(20):.4f}")
print(f"Light crude (API=40°)      → SG = {API_to_SG(40):.4f}")


# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
TBP discretization workflow:

  Lab measures TBP / D86 / D2887 distillation
        ↓ D86_to_TBP() or D2887_to_TBP() if needed
  Calibrated TBP table (cum-vol %, T)
        ↓ discretize_TBP(method=..., SG_strategy=...)
  TBPDiscretization object exposing:
    * res.cuts             — list of PseudoComponent
    * res.volume_fractions — per-cut volume fraction
    * res.mass_fractions   — per-cut mass fraction
    * res.mole_fractions   — per-cut mole fraction (use as feed_z)
    * res.NBP_lower / NBP_upper — cut boundaries

The output is directly usable as feed composition in any column
or flash. See ``crude_distillation_with_side_strippers.py`` for
the end-to-end refinery workflow.

Method selection guidance:
    equal_volume   — default; works well for N >= 6 cuts
    equal_NBP      — better for crudes with prominent heavy ends
    gauss_laguerre — better when N <= 5 cuts and accuracy matters

SG strategy selection:
    SG_table   — best when per-cut SG is measured
    Watson_K   — physically meaningful; each cut gets its own SG
    API_gravity / SG_avg — uniform SG across cuts; simplest
""")
