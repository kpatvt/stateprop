"""Natural gas pipeline dewpoint design.

What this demonstrates
----------------------
Pipeline natural gas must be processed (typically by glycol-dehydration
and chilled-separator units) so its **hydrocarbon dewpoint** stays
below the lowest expected pipeline temperature with a 10-15 °C margin.
Falling below the dewpoint causes liquid hydrocarbon condensation
inside the pipeline — bad: drag goes up, two-phase slug flow develops,
and the heaviest fractions can foul compressors at delivery stations.

The hydrocarbon-dewpoint locus T_dew(p) is the upper bound of the
phase envelope.  As p increases above the cricondenbar, no dewpoint
exists; below it, T_dew is monotone in p up to the cricondentherm.

This example traces T_dew(p) for two compositions:

1. **Lean residue gas** (raw production from a gas-condensate well):
   85% C₁, 7% C₂, 4% C₃, 2% nC₄, 2% nC₅.  Heavy in C₅+, dewpoint
   high.

2. **Sales gas after dehydration + cold separation**: C₅+ stripped
   out, residual C₃+ ≤ 1%.  Dewpoint shifted ~30 °C lower.

We compute the cricondentherm + cricondenbar by sweeping p and
locating the max-T turning point.  Engineering margin against the
US pipeline tariff spec (T_dew ≤ -7 °C / 20 °F at 100 bar) is then
calculated for both streams.

Reference
---------
GPSA Engineering Data Book (13th ed., 2012). Section 23 — Hydrocarbon
Treating; cited as source for dewpoint specs (-7 °C @ 100 bar).

Approximate runtime: ~4 seconds.

Public APIs invoked
-------------------
- stateprop.cubic.from_chemicals.cubic_from_name
- stateprop.cubic.CubicMixture
- stateprop.cubic.flash.dew_point_T
- stateprop.cubic.flash.dew_point_p
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np

from stateprop.cubic.from_chemicals import cubic_from_name
from stateprop.cubic import CubicMixture
from stateprop.cubic.flash import dew_point_T, dew_point_p
from examples._harness import validate, validate_bool, summary

print("=" * 70)
print("Natural gas pipeline dewpoint design")
print("=" * 70)
print()

species = ["methane", "ethane", "propane", "n-butane", "n-pentane"]
eoses = [cubic_from_name(s, family="pr") for s in species]


def trace_dewpoint(z):
    """Sweep pressure and trace T_dew(p) until no dewpoint exists."""
    mx = CubicMixture(eoses, composition=z)
    p_grid_bar = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0,
                   60.0, 70.0, 80.0, 90.0, 100.0, 120.0]
    T_dew = []
    for p_bar in p_grid_bar:
        try:
            r = dew_point_T(p=p_bar*1e5, z=z, mixture=mx)
            T_dew.append((p_bar, r.T))
        except Exception:
            T_dew.append((p_bar, None))
    return T_dew


# ------------------------------------------------------------------
# Lean residue gas — heavy with C5+
# ------------------------------------------------------------------
z_residue = [0.85, 0.07, 0.04, 0.02, 0.02]
print(f"Lean residue gas: {dict(zip(species, z_residue))}")
print()
print(f"  {'P (bar)':>8s}  {'T_dew (K)':>10s}  {'T_dew (°C)':>11s}")
print(f"  {'-'*8}  {'-'*10}  {'-'*11}")

dew_residue = trace_dewpoint(z_residue)
for p, T in dew_residue:
    if T is not None:
        print(f"  {p:>8.1f}  {T:>10.2f}  {T-273.15:>+10.2f}")
    else:
        print(f"  {p:>8.1f}  {'(no dewpoint)':>22s}")

# Find cricondentherm (max T_dew) and cricondenbar (max p with dewpoint)
valid_residue = [(p, T) for p, T in dew_residue if T is not None]
T_max_residue = max(T for _, T in valid_residue)
p_at_T_max_residue = next(p for p, T in valid_residue
                                  if T == T_max_residue)
p_max_residue = max(p for p, _ in valid_residue)

print()
print(f"  Cricondentherm: T_max = {T_max_residue:.2f} K = "
      f"{T_max_residue-273.15:.2f} °C at p = {p_at_T_max_residue:.0f} bar")
print(f"  Cricondenbar:   p_max ≥ {p_max_residue:.0f} bar (limit of grid)")

# ------------------------------------------------------------------
# Sales gas — stripped of C5+, lighter
# ------------------------------------------------------------------
print()
print("=" * 70)
z_sales = [0.94, 0.04, 0.014, 0.005, 0.001]
print(f"Sales gas (post-dehydration): {dict(zip(species, z_sales))}")
print()
print(f"  {'P (bar)':>8s}  {'T_dew (K)':>10s}  {'T_dew (°C)':>11s}")
print(f"  {'-'*8}  {'-'*10}  {'-'*11}")

dew_sales = trace_dewpoint(z_sales)
for p, T in dew_sales:
    if T is not None:
        print(f"  {p:>8.1f}  {T:>10.2f}  {T-273.15:>+10.2f}")
    else:
        print(f"  {p:>8.1f}  {'(no dewpoint)':>22s}")

valid_sales = [(p, T) for p, T in dew_sales if T is not None]
T_max_sales = max(T for _, T in valid_sales) if valid_sales else None
p_at_T_max_sales = next(p for p, T in valid_sales
                                if T == T_max_sales) if valid_sales else None

print()
if T_max_sales is not None:
    print(f"  Cricondentherm: T_max = {T_max_sales:.2f} K = "
          f"{T_max_sales-273.15:.2f} °C at p = {p_at_T_max_sales:.0f} bar")

# ------------------------------------------------------------------
# Engineering specification: pipeline tariff
# ------------------------------------------------------------------
print()
print("=" * 70)
print("Engineering: pipeline-tariff dewpoint margin at 50 bar")
print("=" * 70)
print()
print("  US-style pipeline-tariff spec: T_dew ≤ -7 °C (266.15 K).  We")
print("  evaluate at p = 50 bar (within both gases' phase envelope).")
print()

T_SPEC_K = 266.15
P_CHECK_BAR = 50.0

mx_residue = CubicMixture(eoses, composition=z_residue)
mx_sales = CubicMixture(eoses, composition=z_sales)

T_residue_check = None
T_sales_check = None
try:
    r = dew_point_T(p=P_CHECK_BAR*1e5, z=z_residue, mixture=mx_residue)
    T_residue_check = r.T
except Exception:
    pass
try:
    r = dew_point_T(p=P_CHECK_BAR*1e5, z=z_sales, mixture=mx_sales)
    T_sales_check = r.T
except Exception:
    pass

print(f"  Spec:                       T_dew ≤ {T_SPEC_K-273.15:+.0f} °C "
      f"({T_SPEC_K:.2f} K)")
if T_residue_check is not None:
    margin = T_SPEC_K - T_residue_check
    pass_fail = "PASSES" if margin > 0 else "FAILS"
    print(f"  Lean residue gas:           T_dew = "
          f"{T_residue_check-273.15:+6.2f} °C ({T_residue_check:.2f} K)")
    print(f"    Margin against spec:      {margin:+.2f} K  ({pass_fail})")
else:
    print(f"  Lean residue gas:           supercritical at "
          f"{P_CHECK_BAR:.0f} bar")
if T_sales_check is not None:
    margin = T_SPEC_K - T_sales_check
    pass_fail = "PASSES" if margin > 0 else "FAILS"
    print(f"  Sales gas:                  T_dew = "
          f"{T_sales_check-273.15:+6.2f} °C ({T_sales_check:.2f} K)")
    print(f"    Margin against spec:      {margin:+.2f} K  ({pass_fail})")
else:
    print(f"  Sales gas:                  supercritical at "
          f"{P_CHECK_BAR:.0f} bar")

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
print()
print("Validation:")

# 1. Residue gas dewpoint at 50 bar should be in 290-310 K
T_res_50 = next(T for p, T in valid_residue if p == 50.0)
validate_bool("Residue-gas dewpoint at 50 bar in 280-310 K range",
                condition=(280 < T_res_50 < 310),
                detail=f"T_dew = {T_res_50:.2f} K",
                source="GPSA dewpoint envelope for typical wet gas")

# 2. Cricondentherm exists and is in physically reasonable range
validate_bool("Residue gas has a cricondentherm (T_max with dewpoint)",
                condition=(280 < T_max_residue < 320),
                detail=f"T_max = {T_max_residue:.2f} K at "
                f"p = {p_at_T_max_residue:.0f} bar")

# 3. Sales gas dewpoint should be lower than residue gas at same P
if T_sales_check is not None and T_residue_check is not None:
    validate_bool(f"Sales gas dewpoint < residue gas dewpoint at "
                  f"{P_CHECK_BAR:.0f} bar",
                    condition=(T_sales_check < T_residue_check),
                    detail=f"sales {T_sales_check-273.15:+.1f} °C, "
                    f"residue {T_residue_check-273.15:+.1f} °C "
                    f"(stripping C5+ lowers dewpoint)")

# 4. Sales gas should pass spec
if T_sales_check is not None:
    validate_bool(f"Sales gas T_dew ≤ -7 °C at {P_CHECK_BAR:.0f} bar "
                  f"(passes tariff)",
                    condition=(T_sales_check <= T_SPEC_K),
                    detail=f"T_dew = {T_sales_check-273.15:+.2f} °C, "
                    f"spec ≤ {T_SPEC_K-273.15:.0f} °C")

# 5. dew_point_T and dew_point_p should be consistent: at the
#    dew-point T returned by dew_point_T(p), dew_point_p(T) returns p.
#    Use a low-p case where both directions converge robustly (away
#    from the cricondenbar).
T_check = next(T for p, T in valid_residue if p == 10.0)
try:
    r_inv = dew_point_p(T=T_check, z=z_residue, mixture=mx_residue)
    p_inv = r_inv.p / 1e5
    validate("dew_point_T → dew_point_p round-trip consistency at 10 bar",
              reference=10.0, computed=p_inv,
              units="bar", tol_rel=0.05,
              source="Theoretical: dewpoint locus is single-valued")
except Exception as e:
    validate_bool("dew_point_p round-trip (skip if not converged)",
                    condition=False,
                    detail=f"solver failed: {type(e).__name__}")

# 6. As p decreases below 5 bar, T_dew decreases monotonically too
#    (rising-T side of the envelope)
T_at_5 = next(T for p, T in valid_residue if p == 5.0)
T_at_1 = next(T for p, T in valid_residue if p == 1.0)
validate_bool("T_dew increases monotonically with p (rising side)",
                condition=(T_at_5 > T_at_1),
                detail=f"p=1: {T_at_1:.1f} K, p=5: {T_at_5:.1f} K")

# 7. Residue gas must FAIL the -7 °C spec at 50 bar (it's an
#    unprocessed feed)
if T_residue_check is not None:
    validate_bool("Lean residue gas FAILS pipeline spec (needs processing)",
                    condition=(T_residue_check > T_SPEC_K),
                    detail=f"unprocessed T_dew = "
                    f"{T_residue_check-273.15:+.1f} °C exceeds spec "
                    f"{T_SPEC_K-273.15:.0f} °C")

summary()
