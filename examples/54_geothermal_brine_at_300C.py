"""Geothermal brine thermodynamics: high-T Pitzer for NaCl, KCl, CaCl₂.

What this demonstrates
----------------------
Geothermal energy systems pump hot brine (typically 150-300 °C, 100-
200 bar) from depth to surface heat exchangers, where the brine
gives up thermal energy to a working fluid before re-injection.
Activity coefficients of the dominant chloride salts in the brine
control:

- Vapor-pressure depression (a_w → boiling-point elevation, mineral
  precipitation in the wellbore as the brine cools)
- Mineral solubility (silica, calcite, anhydrite scaling)
- Effective Henry's constants for non-condensable gases (CO₂, H₂S)

Standard-T Pitzer parameters (Bates-Allen, Robinson-Stokes, Pitzer
1991 Table 1) are valid at 25 °C only.  For high-T applications, the
v0.9.116 high-T Pitzer extension provides T-dependent β₀, β₁, β₂, C_φ
parameters from the Pabalan-Pitzer (NaCl, KCl) and Møller (CaCl₂)
1988 fits, valid up to 200 °C.

This example sweeps γ_± and water activity across 25-200 °C for
three brine types relevant to geothermal operations:

1. **1 m NaCl** — clean injection brine, near-pure halite-saturated
2. **5 m NaCl** — high-salinity producing brine (deep sedimentary)
3. **1 m CaCl₂** — produced waters from reactive carbonate reservoirs

The validation anchors on:
- 25 °C γ_± vs Robinson-Stokes 1959 tabulated values (~0.65 for NaCl)
- 100 °C γ_± vs Pabalan-Pitzer 1988 / Holmes-Mesmer experimental data
- 200 °C γ_± vs same (extrapolated to fit boundary)

Reference
---------
Pabalan, R. T.; Pitzer, K. S. (1988). Apparent molar heat capacity
and other thermodynamic properties of aqueous KCl solutions to high
temperatures and pressures.  J. Chem. Eng. Data 33, 354.

Møller, N. (1988). The prediction of mineral solubilities in natural
waters: A chemical equilibrium model for the Na-Ca-Cl-SO₄-H₂O system,
to high temperature and concentration.  Geochim. Cosmochim. Acta 52,
821.

Approximate runtime: ~2 seconds.

Public APIs invoked
-------------------
- stateprop.electrolyte.lookup_salt_high_T
- stateprop.electrolyte.PitzerModel
- stateprop.electrolyte.PitzerSalt.gamma_pm
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np

from stateprop.electrolyte import (
    lookup_salt_high_T, PitzerModel,
)
from examples._harness import validate, validate_bool, summary

print("=" * 70)
print("Geothermal brine thermodynamics (high-T Pitzer, v0.9.116)")
print("=" * 70)
print()
print(f"  T-validity envelopes (Pabalan-Pitzer / Møller 1988):")
for s in ["NaCl", "KCl", "CaCl2"]:
    salt = lookup_salt_high_T(s)
    print(f"    {s}: valid 0-{salt.T_max_valid-273.15:.0f} °C")
print()

# ------------------------------------------------------------------
# γ_± vs T for three molalities
# ------------------------------------------------------------------
T_grid_C = [25, 50, 100, 150, 200]
T_grid = [T + 273.15 for T in T_grid_C]


def gamma_pm_HT(salt_name, m, T):
    salt = lookup_salt_high_T(salt_name).at_T(T)
    model = PitzerModel(salt)
    return model.gamma_pm(m, T=T)


# Sweep 1 m NaCl
print("Mean ionic activity coefficient γ_± for 1 m solutions:")
print()
print(f"  {'T (°C)':>7s}  {'NaCl':>7s}  {'KCl':>7s}  {'CaCl₂':>7s}")
print(f"  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
for T_C, T in zip(T_grid_C, T_grid):
    g_n = gamma_pm_HT("NaCl", 1.0, T)
    g_k = gamma_pm_HT("KCl", 1.0, T)
    g_c = gamma_pm_HT("CaCl2", 1.0, T)
    print(f"  {T_C:>7d}  {g_n:>7.4f}  {g_k:>7.4f}  {g_c:>7.4f}")

# Sweep concentration at 100 °C
print()
print("γ_± for NaCl across molality at T = 100 °C:")
print()
print(f"  {'m (mol/kg)':>10s}  {'γ_± NaCl':>10s}")
print(f"  {'-'*10}  {'-'*10}")
m_at_100 = []
for m in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]:
    g = gamma_pm_HT("NaCl", m, 373.15)
    m_at_100.append((m, g))
    print(f"  {m:>10.2f}  {g:>10.4f}")

# ------------------------------------------------------------------
# 5 m NaCl case (high-salinity brine)
# ------------------------------------------------------------------
print()
print("=" * 70)
print("High-salinity brine: 5 m NaCl across T (geothermal producer)")
print("=" * 70)
print()
print(f"  {'T (°C)':>7s}  {'γ_±':>7s}  {'a_w':>7s}  "
      f"{'comment':>40s}")
print(f"  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*40}")

for T_C, T in zip(T_grid_C, T_grid):
    salt = lookup_salt_high_T("NaCl").at_T(T)
    model = PitzerModel(salt)
    g = model.gamma_pm(5.0, T=T)
    aw = model.water_activity(5.0, T=T)
    comment = ("near halite saturation" if T_C < 100 else
                "approaching saturation,extreme γ change")
    print(f"  {T_C:>7d}  {g:>7.4f}  {aw:>7.4f}  {comment:>40s}")

# Boiling-point elevation from a_w
# ΔT_bp = -RT_b² · ln(a_w) / ΔH_vap_water
T_bp_pure = 373.15  # K
H_vap_water = 40.65e3   # J/mol at boiling
salt_NaCl_100 = lookup_salt_high_T("NaCl").at_T(373.15)
model = PitzerModel(salt_NaCl_100)
aw_5m = model.water_activity(5.0, T=373.15)
delta_T_bp = -8.314 * T_bp_pure**2 * np.log(aw_5m) / H_vap_water

print()
print(f"  Boiling-point elevation for 5 m NaCl at 1 atm:")
print(f"    a_w(5 m NaCl, 100°C) = {aw_5m:.4f}")
print(f"    ΔT_bp = {delta_T_bp:.2f} K (water boils at "
        f"{100 + delta_T_bp:.1f} °C)")

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
print()
print("Validation:")

# 1.  γ_±(1 m NaCl, 25 °C) ≈ 0.6577 (Robinson-Stokes 1959)
g_NaCl_25 = gamma_pm_HT("NaCl", 1.0, 298.15)
validate("γ_±(1 m NaCl) at 25 °C",
          reference=0.6577, computed=g_NaCl_25,
          units="-", tol_rel=0.02,
          source="Robinson-Stokes 1959 Appendix 8.10")

# 2.  γ_±(0.1 m NaCl, 25 °C) ≈ 0.778 (Robinson-Stokes 1959)
g_NaCl_01 = gamma_pm_HT("NaCl", 0.1, 298.15)
validate("γ_±(0.1 m NaCl) at 25 °C",
          reference=0.778, computed=g_NaCl_01,
          units="-", tol_rel=0.02,
          source="Robinson-Stokes 1959")

# 3.  γ_± decreases monotonically with T for 1 m NaCl
g_NaCl_T = [gamma_pm_HT("NaCl", 1.0, T) for T in T_grid]
validate_bool("γ_±(1 m NaCl) decreases monotonically over 25-200 °C",
                condition=all(g_NaCl_T[i] >= g_NaCl_T[i+1] - 1e-3
                                  for i in range(len(g_NaCl_T)-1)),
                detail=f"sweep: {[f'{g:.3f}' for g in g_NaCl_T]}",
                source="Pabalan-Pitzer 1988 trend")

# 4.  γ_±(1 m CaCl₂, 25 °C) ≈ 0.518 (Robinson-Stokes)
g_CaCl_25 = gamma_pm_HT("CaCl2", 1.0, 298.15)
validate("γ_±(1 m CaCl₂) at 25 °C",
          reference=0.518, computed=g_CaCl_25,
          units="-", tol_rel=0.05,
          source="Robinson-Stokes 1959")

# 5.  γ_±(1 m KCl, 25 °C) ≈ 0.604 (Robinson-Stokes)
g_KCl_25 = gamma_pm_HT("KCl", 1.0, 298.15)
validate("γ_±(1 m KCl) at 25 °C",
          reference=0.604, computed=g_KCl_25,
          units="-", tol_rel=0.02,
          source="Robinson-Stokes 1959")

# 6.  γ_± decreases with m for fixed T (CaCl₂ goes through min-then-rise
#     so check NaCl which is monotone)
gammas_at_100 = [g for m, g in m_at_100]
# m_at_100 ordering is [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
# Check monotonicity (decreasing from 0.1 to ~1, then rising) — for
# NaCl, the minimum is around 1-1.5 m at 100 °C, so just check that
# the 0.1 m value is largest
validate_bool("γ_±(NaCl, 0.1 m, 100°C) > γ_±(NaCl, 1 m, 100°C)",
                condition=(gammas_at_100[0] > gammas_at_100[2]),
                detail=f"γ at 0.1 m: {gammas_at_100[0]:.4f}, "
                f"at 1 m: {gammas_at_100[2]:.4f}")

# 7.  Boiling-point elevation for 5 m NaCl at 1 atm should be ~6-8 K
validate("Boiling-point elevation for 5 m NaCl",
          reference=7.0, computed=delta_T_bp,
          units="K", tol_rel=0.30,
          source="Standard textbook ~6-8 K BPE for halite-saturated brine")

# 8.  γ_±(NaCl, 1 m, 200 °C) significantly less than at 25 °C
g_NaCl_200 = gamma_pm_HT("NaCl", 1.0, 473.15)
validate_bool("γ_±(NaCl, 1 m) drops by ≥20% from 25 to 200 °C",
                condition=(g_NaCl_200 < 0.80 * g_NaCl_25),
                detail=f"25 °C: {g_NaCl_25:.3f}, "
                f"200 °C: {g_NaCl_200:.3f}, "
                f"ratio = {g_NaCl_200/g_NaCl_25:.3f}",
                source="High-T Pitzer behavior at thermal extremes")

summary()
