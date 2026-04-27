"""CO₂ sequestration in a saline aquifer: dissolution and acidification.

What this demonstrates
----------------------
Geological CO₂ storage works in three sequential trapping mechanisms:
(1) structural trapping under impermeable cap rock; (2) dissolution
into resident brine ("solubility trapping"); and (3) mineralization
through reaction with carbonate / silicate rock.  The library's
sour-water speciation framework, combined with the mineral-solubility
module, lets us model the first two stages quantitatively.

This example:

1. Computes CO₂ dissolved in pure water at 60 °C across pressure
   (Henry's law via `sour_water.henry_constant`).  This is the
   maximum trapping capacity per unit volume of pore space.
2. Sweeps total dissolved CO₂ in pure water at fixed T, tracking pH
   and the carbonate distribution m_CO2(aq), m_HCO3⁻, m_CO3²⁻.  As
   pH drops below 4, essentially all dissolved C is molecular CO₂.
3. Computes the calcite saturation index SI as CO₂ dissolves into a
   bicarbonate-bearing brine.  This is the geochemistry that
   determines whether mineral trapping accelerates or stalls.

The classic finding: CO₂-charged brine near the injection well is
*undersaturated* with respect to calcite (acidified), so calcite
*dissolves* near the wellbore.  Far from the well, where pH has
recovered, calcite is *supersaturated* and precipitates — locking the
CO₂ in a solid mineral.  Mineralization timescales are decades to
centuries.

Reference
---------
Duan, Z.; Sun, R. (2003). An improved model calculating CO₂
solubility in pure water and aqueous NaCl solutions from 273 to 533 K
and from 0 to 2000 bar.  Chem. Geol. 193, 257-271.

Gilfillan, S. M. V.; Lollar, B. S.; Holland, G.; et al. (2009).
Solubility trapping in formation water as dominant CO₂ sink in
natural gas fields.  Nature 458, 614-618.

Approximate runtime: ~3 seconds.

Public APIs invoked
-------------------
- stateprop.electrolyte.sour_water.speciate
- stateprop.electrolyte.sour_water.henry_constant
- stateprop.electrolyte.MultiPitzerSystem.from_salts
- stateprop.electrolyte.lookup_mineral
- stateprop.electrolyte.saturation_index

"""
import sys, warnings
sys.path.insert(0, '.')
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np

from stateprop.electrolyte import (
    sour_water, MultiPitzerSystem,
    lookup_mineral, saturation_index,
)
from examples._harness import validate, validate_bool, summary

print("=" * 70)
print("CO₂ sequestration in saline aquifer (T = 60 °C)")
print("=" * 70)
print()

T_RES = 333.15  # 60 °C — typical reservoir T at 1500 m depth

# ------------------------------------------------------------------
# Part 1: Henry's law — CO₂ solubility in pure water vs P
# ------------------------------------------------------------------
print("Part 1: CO₂ dissolution in pure water vs partial pressure")
print()

# Get the bundled Henry constant for CO2 at T_RES.  Note the library's
# convention: H_CO2 [Pa·kg/mol], so m_CO2 = P_CO2 / H.
H_CO2 = sour_water.henry_constant("CO2", T_RES)
print(f"  H(CO₂, 60 °C) = {H_CO2:.3e} Pa·kg/mol  "
        f"(library convention: P = H·m)")
print(f"  → KH = 1/H = {1.0/H_CO2*1e5:.4f} mol/(kg·bar)")
print(f"  (Duan-Sun 2003 ref: ~0.030 mol/(kg·bar) at 60 °C)")
print()

print(f"  {'P_CO2 (bar)':>12s}  {'m_CO2 (mol/kg)':>16s}  {'pH':>5s}  "
      f"{'m_HCO3⁻':>11s}")
print(f"  {'-'*12}  {'-'*16}  {'-'*5}  {'-'*11}")

P_sweep = [1, 10, 50, 100, 150, 200]
m_CO2_at_P = []
for P_CO2 in P_sweep:
    # m_CO2 = P_partial [Pa] / H [Pa·kg/mol]
    m_CO2_total = (P_CO2 * 1e5) / H_CO2
    r = sour_water.speciate(T=T_RES, m_NH3_total=0.0,
                                  m_H2S_total=0.0, m_CO2_total=m_CO2_total)
    m_HCO3 = r.species_molalities["HCO3-"]
    m_CO2_at_P.append((P_CO2, m_CO2_total, r.pH, m_HCO3))
    print(f"  {P_CO2:>12d}  {m_CO2_total:>16.4f}  {r.pH:>5.2f}  "
            f"{m_HCO3:>11.3e}")

# ------------------------------------------------------------------
# Part 2: pH and carbonate speciation as CO₂ dissolves
# ------------------------------------------------------------------
print()
print("Part 2: pH and carbonate distribution vs total dissolved CO₂")
print()
print(f"  {'m_CO2_tot':>10s}  {'pH':>5s}  {'α_CO2':>7s}  "
      f"{'m_CO2(aq)':>11s}  {'m_HCO3⁻':>11s}")
print(f"  {'(mol/kg)':>10s}  {'':>5s}  {'':>7s}  "
      f"{'(mol/kg)':>11s}  {'(mol/kg)':>11s}")
print(f"  {'-'*10}  {'-'*5}  {'-'*7}  {'-'*11}  {'-'*11}")

m_sweep = [1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0]
for m in m_sweep:
    r = sour_water.speciate(T=T_RES, m_NH3_total=0.0,
                                  m_H2S_total=0.0, m_CO2_total=m)
    m_co2_aq = r.species_molalities["CO2"]
    m_hco3 = r.species_molalities["HCO3-"]
    print(f"  {m:>10.3f}  {r.pH:>5.2f}  {r.alpha_CO2:>7.4f}  "
            f"{m_co2_aq:>11.3e}  {m_hco3:>11.3e}")

# ------------------------------------------------------------------
# Part 3: Calcite saturation and the dissolution-precipitation cycle
# ------------------------------------------------------------------
print()
print("Part 3: Calcite stability vs P_CO₂ in a Na-Ca brine")
print()
print("  Brine: 1 m NaCl + 10 mM CaCl₂ (representative formation water)")
print()
print("  Note: In a CLOSED Na-Ca-Cl-CO₂ system, the IAP m_Ca·m_CO3 stays")
print("  constant as CO₂ dissolves — H⁺ released by CO₂ shifts CO₃²⁻ to")
print("  HCO3⁻ proportionally to its own increase.  The SI changes only")
print("  if (a) Ca²⁺ also reacts (calcite dissolves), or (b) external")
print("  buffering / mineral surfaces participate.  Below we vary the")
print("  Ca²⁺ molality to show how an open or partially-dissolved system")
print("  behaves.")
print()

# Pitzer system for the brine activity coefficients
mp = MultiPitzerSystem.from_salts(["NaCl", "CaCl2"])

# Carbonate equilibrium constants at 60 °C (van't Hoff form, anchored
# to standard 25 °C values).
log_Ka2_25 = -10.33
log_Ksp_calcite_25 = -8.48
delta_H_Ka2 = 14.85e3
delta_H_Ksp = -10.0e3

R = 8.314462618
T_25 = 298.15
log_Ka2 = log_Ka2_25 - delta_H_Ka2 / (np.log(10.0) * R) * (1/T_RES - 1/T_25)
log_Ksp = log_Ksp_calcite_25 - delta_H_Ksp / (np.log(10.0) * R) * (1/T_RES - 1/T_25)
Ka2 = 10 ** log_Ka2
Ksp = 10 ** log_Ksp

print(f"  At {T_RES-273.15:.0f} °C: log K_a2 = {log_Ka2:.3f}, "
        f"log K_sp(calcite) = {log_Ksp:.3f}")
print()

# Fix P_CO2 = 200 bar (deep injection), vary Ca²⁺ molality:
P_CO2_inj = 200.0
m_CO2_total = (P_CO2_inj * 1e5) / H_CO2
r_inj = sour_water.speciate(T=T_RES, m_NH3_total=0.0,
                                   m_H2S_total=0.0,
                                   m_CO2_total=m_CO2_total)
m_HCO3_inj = r_inj.species_molalities["HCO3-"]
m_H_inj = r_inj.species_molalities["H+"]
m_CO3_inj = Ka2 * m_HCO3_inj / m_H_inj

print(f"  Injection conditions (P_CO₂ = {P_CO2_inj} bar):")
print(f"    pH = {r_inj.pH:.2f}, "
        f"m_HCO3⁻ = {m_HCO3_inj:.3e}, m_CO3²⁻ = {m_CO3_inj:.3e}")
print()
print(f"  {'m_Ca²⁺':>9s}  {'γ_Ca':>5s}  {'IAP/Ksp':>9s}  "
      f"{'SI_calcite':>10s}  {'state':>15s}")
print(f"  {'(mol/kg)':>9s}  {'':>5s}  {'':>9s}  {'':>10s}  {'':>15s}")
print(f"  {'-'*9}  {'-'*5}  {'-'*9}  {'-'*10}  {'-'*15}")

SI_at_Ca = []
for m_Ca in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
    mol = {"Na+": 1.0, "Cl-": 1.0 + 2.0*m_Ca, "Ca++": m_Ca}
    gammas = mp.gammas(mol, T=T_RES)
    g_Ca = gammas["Ca++"]
    g_CO3 = g_Ca   # approximation
    IAP = m_Ca * m_CO3_inj * g_Ca * g_CO3
    SI = np.log10(IAP / Ksp)
    state = ("dissolves" if SI < -0.1 else
             "precipitates" if SI > 0.1 else "near eq.")
    SI_at_Ca.append((m_Ca, SI))
    print(f"  {m_Ca:>9.1e}  {g_Ca:>5.3f}  "
            f"{IAP/Ksp:>9.2e}  {SI:>+10.2f}  {state:>15s}")

print()
print("  Interpretation:")
print("  - At low Ca²⁺ (near a fresh injection well), SI < 0 → calcite")
print("    DISSOLVES, releasing more Ca²⁺ that buffers the acidity.")
print("  - As Ca²⁺ accumulates from dissolution, SI rises until")
print("    saturation is reached (near-equilibrium far from the well).")
print("  - On geological timescales, this leads to MINERAL TRAPPING:")
print("    silicate weathering generates more Ca²⁺/Mg²⁺, eventually")
print("    precipitating CO₂ as solid carbonate minerals.")

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
print()
print("Validation:")

# 1.  CO2 solubility at 60 °C, 200 bar via simple Henry's law.  The
#     library uses linear Henry's law (m = P/H), which over-predicts
#     solubility at high P relative to Duan-Sun's nonlinear EOS.  At
#     200 bar, linear Henry gives ~3 mol/kg; Duan-Sun gives ~1.1 mol/kg.
#     We validate that the value lands in a plausible range rather than
#     match Duan-Sun exactly — using Duan-Sun would require running an
#     external EOS for the gas-phase fugacity.
m_CO2_200 = (200.0 * 1e5) / H_CO2
validate_bool("CO₂ Henry's law solubility at 200 bar in 1-5 mol/kg "
              "(linear Henry envelope)",
                condition=(1.0 <= m_CO2_200 <= 5.0),
                detail=f"m_CO2 = {m_CO2_200:.2f} mol/kg "
                f"(linear Henry over-predicts vs Duan-Sun ~1.1)",
                source="Library: linear Henry's law; Duan-Sun 2003 "
                       "nonlinear gives ~1.1")

# 2.  pH at "saturation" should be acidic (pH 2.5-4 for CO2-water at
#     these conditions; lower than 3 reflects the high m_CO2 from
#     linear-Henry over-prediction)
r_sat = sour_water.speciate(T=T_RES, m_NH3_total=0.0,
                                   m_H2S_total=0.0,
                                   m_CO2_total=m_CO2_200)
validate_bool("CO₂-charged brine pH < 4 (acidic)",
                condition=(r_sat.pH < 4.0),
                detail=f"pH = {r_sat.pH:.2f}",
                source="Carbonate-system acid-base equilibrium")

# 3.  α_CO2 (molecular fraction) should approach 1.0 at high P
#     (low pH suppresses bicarbonate formation)
validate("α_CO2(aq) at saturation",
          reference=1.0, computed=r_sat.alpha_CO2,
          units="-", tol_rel=0.05,
          source="Theoretical: low pH → all C is molecular CO₂")

# 4.  pH decreases monotonically with dissolved CO₂
pHs = [pH for _, _, pH, _ in m_CO2_at_P]
validate_bool("pH decreases monotonically with dissolved CO₂",
                condition=all(pHs[i] >= pHs[i+1] - 0.01
                                  for i in range(len(pHs)-1)),
                detail=f"pHs: {[f'{p:.2f}' for p in pHs]}")

# 5.  Calcite SI increases with Ca²⁺ molality (more Ca → more
#     supersaturation potential)
SIs = [si for _, si in SI_at_Ca]
validate_bool("Calcite SI increases monotonically with Ca²⁺",
                condition=all(SIs[i] <= SIs[i+1] + 0.01
                                  for i in range(len(SIs)-1)),
                detail=f"SIs: {[f'{s:+.2f}' for s in SIs]}")

# 6.  At low Ca (1e-5 mol/kg), calcite is highly undersaturated (SI << 0)
SI_low_Ca = SIs[0]
validate_bool("Calcite undersaturated at low Ca²⁺ (SI < 0)",
                condition=(SI_low_Ca < 0),
                detail=f"SI = {SI_low_Ca:+.2f} at m_Ca=1e-5",
                source="Calcite dissolves near a CO₂ injection well")

# 7.  Ca activity coefficient < 1 in brine (electrostatic shielding)
mol = {"Na+": 1.0, "Cl-": 1.002, "Ca++": 1e-3}
gammas = mp.gammas(mol, T=T_RES)
validate_bool("γ(Ca²⁺) < 1 in 1 m NaCl brine",
                condition=(gammas["Ca++"] < 1.0),
                detail=f"γ(Ca²⁺) = {gammas['Ca++']:.3f}",
                source="Pitzer / Davies for 2-1 electrolyte at I~1 mol/kg")

summary()
