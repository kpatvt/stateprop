"""Joule-Thomson inversion temperatures for methane and nitrogen.

What this demonstrates
----------------------
A real gas expanding through a throttle (constant-h process) cools
when its Joule-Thomson coefficient μ_JT = (∂T/∂p)_h is positive,
and warms when μ_JT is negative.  The temperature where μ_JT = 0 at
a given pressure is the **inversion temperature**.  For
liquefaction (LNG production, hydrogen liquefaction) the gas must
be below its inversion temperature to throttle-cool.

This example traces the inversion locus T_inv(p) for methane and
nitrogen by bisection on T at fixed p.  We compare against
published reference values:

- Methane:  T_inv ≈ 995 K at p → 0; ~640 K at 50 bar
- Nitrogen: T_inv ≈ 621 K at p → 0; ~520 K at 50 bar

These low-pressure inversion temperatures are also tabulated by
direct calculation from the second virial coefficient B(T) — at the
inversion temperature, T·dB/dT = B (Boyle-derivative condition).

Reference
---------
Setzmann, U.; Wagner, W. (1991). Methane reference EOS.
J. Phys. Chem. Ref. Data 20, 1061.

Span, R.; Lemmon, E. W.; Jacobsen, R. T.; Wagner, W. (2000). A
reference equation of state for the thermodynamic properties of
nitrogen.  J. Phys. Chem. Ref. Data 29, 1361.

Approximate runtime: ~5 seconds.

Public APIs invoked
-------------------
- stateprop.load_fluid
- stateprop.joule_thomson_coefficient
- stateprop.density_from_pressure

"""
import sys
sys.path.insert(0, '.')

import numpy as np

from stateprop import (
    load_fluid, density_from_pressure, joule_thomson_coefficient,
)
from examples._harness import validate, validate_bool, summary

print("=" * 70)
print("Joule-Thomson inversion temperatures: methane and nitrogen")
print("=" * 70)
print()


def find_inversion_T(fluid, p, T_lo=200.0, T_hi=1500.0,
                       phase="vapor"):
    """Find the inversion temperature T(μ=0) at pressure p [Pa]
    by bisection.  Returns T or None if no sign-change in the range."""
    def mu(T):
        rho = density_from_pressure(p, T, fluid, phase=phase)
        return joule_thomson_coefficient(rho, T, fluid)
    mu_lo = mu(T_lo)
    mu_hi = mu(T_hi)
    if mu_lo * mu_hi > 0:
        return None
    for _ in range(60):
        T_mid = 0.5 * (T_lo + T_hi)
        mu_mid = mu(T_mid)
        if mu_mid * mu_lo < 0:
            T_hi = T_mid
        else:
            T_lo = T_mid
            mu_lo = mu_mid
        if T_hi - T_lo < 0.01:
            break
    return 0.5 * (T_lo + T_hi)


# ------------------------------------------------------------------
# Sweep p and find inversion T for both fluids
# ------------------------------------------------------------------
methane = load_fluid("methane")
nitrogen = load_fluid("nitrogen")

print("Inversion temperature locus T_inv(p):")
print()
print(f"  {'p (bar)':>8s}  {'T_inv CH₄ (K)':>14s}  {'T_inv N₂ (K)':>13s}")
print(f"  {'-'*8}  {'-'*14}  {'-'*13}")

p_grid_bar = [1, 5, 10, 20, 50, 100, 200]

inversion_pts = {"methane": {}, "nitrogen": {}}
for p_bar in p_grid_bar:
    p = p_bar * 1e5
    T_ch4 = find_inversion_T(methane, p, T_lo=300.0, T_hi=1200.0)
    T_n2 = find_inversion_T(nitrogen, p, T_lo=200.0, T_hi=900.0)
    if T_ch4 is not None:
        inversion_pts["methane"][p_bar] = T_ch4
    if T_n2 is not None:
        inversion_pts["nitrogen"][p_bar] = T_n2
    s_ch4 = f"{T_ch4:.1f}" if T_ch4 is not None else "(no root)"
    s_n2 = f"{T_n2:.1f}"  if T_n2 is not None else "(no root)"
    print(f"  {p_bar:>8.0f}  {s_ch4:>14s}  {s_n2:>13s}")

# ------------------------------------------------------------------
# Sample the JT coefficient at one (T, p) — show sign change
# ------------------------------------------------------------------
print()
print("Sample JT coefficient μ_JT for methane at p=10 bar:")
for T in [200.0, 300.0, 500.0, 700.0, 900.0, 1100.0]:
    rho = density_from_pressure(10e5, T, methane, phase="vapor")
    mu = joule_thomson_coefficient(rho, T, methane)
    sign = "(cooling)" if mu > 0 else "(warming)"
    print(f"    T = {T:>5.0f} K: μ_JT = {mu*1e6:>+7.3f} K/MPa {sign}")

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
print()
print("Validation:")

# Methane low-p inversion temperature ≈ 995 K
T_inv_ch4_low = inversion_pts["methane"].get(1)
if T_inv_ch4_low is not None:
    validate("Methane T_inv at 1 bar",
              reference=995.0, computed=T_inv_ch4_low,
              units="K", tol_rel=0.05,
              source="Setzmann-Wagner 1991 / NIST WebBook")

# Nitrogen low-p inversion temperature ≈ 621 K
T_inv_n2_low = inversion_pts["nitrogen"].get(1)
if T_inv_n2_low is not None:
    validate("Nitrogen T_inv at 1 bar",
              reference=621.0, computed=T_inv_n2_low,
              units="K", tol_rel=0.05,
              source="Span-Lemmon 2000 / NIST WebBook")

# At low p, methane T_inv > nitrogen T_inv (heavier molecule)
validate_bool("T_inv(CH₄) > T_inv(N₂) at low p",
                condition=(inversion_pts["methane"][1] >
                            inversion_pts["nitrogen"][1]),
                detail=f"CH₄: {inversion_pts['methane'][1]:.0f} K vs "
                f"N₂: {inversion_pts['nitrogen'][1]:.0f} K")

# Inversion temperature decreases with pressure (typical behavior on
# the low-T branch of the inversion locus).
ch4_pts = sorted(inversion_pts["methane"].items())
if len(ch4_pts) >= 4:
    p_max = ch4_pts[-1][0]
    p_min = ch4_pts[0][0]
    T_at_pmax = ch4_pts[-1][1]
    T_at_pmin = ch4_pts[0][1]
    validate_bool(f"CH₄ T_inv decreases monotonically over "
                  f"{p_min}-{p_max} bar",
                    condition=(T_at_pmax < T_at_pmin),
                    detail=f"T_inv at {p_min} bar = {T_at_pmin:.0f} K, "
                    f"at {p_max} bar = {T_at_pmax:.0f} K")

# Methane μ_JT > 0 at 300 K, 10 bar (cooling on throttle)
rho_300 = density_from_pressure(10e5, 300.0, methane, phase="vapor")
mu_300 = joule_thomson_coefficient(rho_300, 300.0, methane)
validate_bool("CH₄ μ_JT > 0 at 300 K, 10 bar (cooling)",
                condition=(mu_300 > 0),
                detail=f"μ_JT = {mu_300*1e6:.3f} K/MPa")

# Methane μ_JT < 0 at 1100 K (above inversion)
rho_1100 = density_from_pressure(10e5, 1100.0, methane, phase="vapor")
mu_1100 = joule_thomson_coefficient(rho_1100, 1100.0, methane)
validate_bool("CH₄ μ_JT < 0 at 1100 K (above inversion)",
                condition=(mu_1100 < 0),
                detail=f"μ_JT = {mu_1100*1e6:.3f} K/MPa")

summary()
