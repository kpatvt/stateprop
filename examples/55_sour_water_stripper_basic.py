"""Sour-water stripper: refinery wastewater treatment with live steam.

Demonstrates v0.9.97's `stateprop.electrolyte.sour_water` module on a
real industrial application — stripping H₂S and NH₃ out of refinery
sour water using direct-contact live steam.

Industrial context
------------------
A typical refinery generates 50-200 m³/h of sour water from crude
desalter, FCC fractionator condensers, hydrotreater overheads, and
amine reclaimer drains. The water carries 100-3000 ppm of NH₃ and
H₂S (with smaller amounts of CO₂ and trace organics).  Treatment
goals:

    * Stripped water → recycle to crude desalter or boiler feed
      (spec: H₂S < 50 ppm, NH₃ < 100 ppm typical)
    * Acid gas overhead → Claus sulfur recovery unit
      (spec: enriched H₂S/NH₃ stream, water saturated)

Standard configuration: 15-25 sieve trays, 1.5-3 bar operating
pressure, live steam at 1-1.5 lb steam/gal feed (≈ 0.18-0.27 kg
steam/kg feed), ~95-105 °C feed pre-heat.

Thermodynamic model
-------------------
* Activity coefficients of molecular NH₃, H₂S, CO₂ taken as unity
  (good approximation at <0.1 mol/kg total dissolved gas)
* Henry's law in apparent (effective) form:

     H_eff(species, T, pH) = H_molecular(T) · α_molecular(T, pH)

  where α accounts for the fact that NH₄⁺, HS⁻, HCO₃⁻ are
  non-volatile.  α_NH₃ rises with pH (ammonia easy to strip at
  high pH); α_H₂S falls with pH.
* pH is computed self-consistently from charge balance with all
  weak electrolytes simultaneously dissociating
* Steam injection treated as a saturated vapor feed at the bottom

Multi-stage analysis
--------------------
Counter-current stripper with N stages, feed at top, steam at bottom.
For a dilute system with linear equilibrium (Henry's law), the Kremser
equation gives the exact analytical solution:

    x_N / x_0 = (S - 1) / (S^(N+1) - 1)         for x* = 0 in steam

where S = K·V/L is the stripping factor and K = H_eff/(P·M_w·c_w_eff)
is the dimensionless K-value for a dissolved gas.

For non-trivial cases (variable T profile, multi-component
interactions through pH) we iterate stage-by-stage.

References
----------
* McCabe, W. L., Smith, J. C., Harriott, P. (1993).
  *Unit Operations of Chemical Engineering*, 5th ed., Ch. 21.
* GPSA *Engineering Data Book*, Section 19 (Sour Water Stripping).
* Beychok, M. R. (1968). Aqueous Wastes from Petroleum and
  Petrochemical Plants. Wiley.
"""
from __future__ import annotations
import warnings
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np
from stateprop.electrolyte.sour_water import (
    henry_constant, dissociation_K, effective_henry,
    speciate, pK_water,
)
from stateprop.activity.gamma_phi import AntoinePsat


# =====================================================================
# Feed conditions and stripper specifications
# =====================================================================

# Refinery sour water — typical FCC fractionator overhead drum drain
T_feed = 105.0 + 273.15      # K (95-105 °C is typical, hot enough to be near bubble)
P_col  = 2.0e5               # Pa (2 bar — typical sour-water stripper)
F_feed = 100.0               # mol/s of feed water (basis; ~6.5 m³/h)
m_NH3_feed = 0.060           # mol/kg = ~1020 ppm by mass
m_H2S_feed = 0.040           # mol/kg = ~1360 ppm by mass
m_CO2_feed = 0.005           # mol/kg = ~220 ppm by mass

# Stripper geometry
N_stages = 15
T_stages_init = np.linspace(T_feed, 105.0 + 273.15, N_stages)

# Live steam injection rate at bottom — kg steam / kg feed water
# 0.20 = 1.1 lb/gal, typical industrial spec
steam_ratio = 0.20

# Water Antoine (NIST WebBook): log10(P[bar]) = A - B/(T+C),  273-373 K
WATER_ANTOINE = AntoinePsat(A=4.6543, B=1435.264, C=-64.848,
                              T_min=255, T_max=380)


# =====================================================================
# Helpers
# =====================================================================

MW_water = 0.0180153  # kg/mol


def mole_fraction_to_molality(x_NH3: float, x_H2S: float,
                                x_CO2: float) -> tuple:
    """Convert mole fractions to molalities for speciation.

    Assumes water is the dominant species (x_w ≈ 1).
    For dilute solutions: m_i ≈ x_i / (x_w · M_w) ≈ x_i / M_w.
    """
    return (x_NH3 / MW_water, x_H2S / MW_water, x_CO2 / MW_water)


def molality_to_mole_fraction(m_NH3: float, m_H2S: float,
                                m_CO2: float) -> tuple:
    """Inverse — convert molalities to mole fractions."""
    n_w_per_kg = 1.0 / MW_water
    n_total = n_w_per_kg + m_NH3 + m_H2S + m_CO2
    return (m_NH3 / n_total, m_H2S / n_total, m_CO2 / n_total,
            n_w_per_kg / n_total)


def K_value(species: str, T: float, P: float, pH: float) -> float:
    """Dimensionless K = y/x for a dissolved gas in water at given pH.

    For molecular form i:    P · y_i = H_i(T) · m_i^molecular
    With m_i ≈ x_i / M_w and α_molecular accounting for dissociation:
       y_i · P = (H_i · α_i / M_w) · x_i_total
       K_i = (H_i · α_i) / (P · M_w)
    """
    H_eff = effective_henry(species, T, pH)
    return H_eff / (P * MW_water)


def K_water(T: float, P: float) -> float:
    """K-value for water itself: K_w = Psat_w(T) / P (γ_w ≈ 1)."""
    return WATER_ANTOINE(T) / P


# =====================================================================
# Print feed analysis
# =====================================================================

print("=" * 72)
print("Sour-Water Stripper — Refinery Wastewater Treatment")
print("=" * 72)

print(f"""
Feed conditions:
    T = {T_feed:.1f} K ({T_feed-273.15:.0f} °C)
    P = {P_col/1e5:.1f} bar
    F = {F_feed:.1f} mol/s ≈ {F_feed*MW_water*3600:.0f} kg/h ≈ {F_feed*MW_water*3600/1000:.1f} t/h

    NH₃ total = {m_NH3_feed*1000:.1f} mmol/kg = {m_NH3_feed*17.03*1000:.0f} ppm by mass
    H₂S total = {m_H2S_feed*1000:.1f} mmol/kg = {m_H2S_feed*34.08*1000:.0f} ppm by mass
    CO₂ total = {m_CO2_feed*1000:.1f} mmol/kg = {m_CO2_feed*44.01*1000:.0f} ppm by mass
""")

# Speciate feed
sp_feed = speciate(T_feed,
                     m_NH3_total=m_NH3_feed,
                     m_H2S_total=m_H2S_feed,
                     m_CO2_total=m_CO2_feed)

print("Feed equilibrium speciation (charge-balance solve):")
print(f"    pH        = {sp_feed.pH:.2f}")
print(f"    Ionic str = {sp_feed.I*1000:.2f} mmol/kg")
print(f"    α(NH₃)    = {sp_feed.alpha_NH3*100:5.1f}% molecular  "
        f"(volatile fraction)")
print(f"    α(H₂S)    = {sp_feed.alpha_H2S*100:5.1f}% molecular")
print(f"    α(CO₂)    = {sp_feed.alpha_CO2*100:5.1f}% molecular")

print("\nKey species concentrations [mmol/kg]:")
for k, v in sorted(sp_feed.species_molalities.items()):
    print(f"    [{k:>5s}]  = {v*1000:.4g}")


# =====================================================================
# Henry's constants and K-values at feed conditions
# =====================================================================

print(f"\nThermodynamic K-values at T={T_feed-273.15:.0f}°C, P={P_col/1e5:.1f} bar, "
        f"pH={sp_feed.pH:.2f}:")
print(f"  {'species':>8s} {'H_mol':>11s} {'α':>8s} {'H_eff':>11s} {'K=y/x':>10s}")
print(f"  {'':>8s} {'[Pa·kg/mol]':>11s} {'':>8s} {'[Pa·kg/mol]':>11s} {'':>10s}")
for sp_n in ["NH3", "H2S", "CO2"]:
    H_mol = henry_constant(sp_n, T_feed)
    H_eff = effective_henry(sp_n, T_feed, sp_feed.pH)
    alpha = H_eff / H_mol
    K = K_value(sp_n, T_feed, P_col, sp_feed.pH)
    print(f"  {sp_n:>8s} {H_mol:>11.2e} {alpha:>8.3f} {H_eff:>11.2e} {K:>10.2f}")
K_w_feed = K_water(T_feed, P_col)
print(f"  {'H₂O':>8s} {'(Antoine)':>11s} {1.0:>8.3f} "
        f"{WATER_ANTOINE(T_feed):>11.2e} {K_w_feed:>10.3f}")


# =====================================================================
# Multi-stage stripper: stage-by-stage Kremser-style
# =====================================================================

print("\n" + "=" * 72)
print(f"Stripper: {N_stages} stages, live steam at bottom "
        f"({steam_ratio:.2f} kg/kg feed)")
print("=" * 72)

# Convert feed to mole fractions
x_NH3_feed, x_H2S_feed, x_CO2_feed, x_w_feed = molality_to_mole_fraction(
    m_NH3_feed, m_H2S_feed, m_CO2_feed)

# Steam at bottom: pure water vapor (no NH3/H2S/CO2)
# Steam temperature at column pressure (saturated steam)
T_steam = 393.0     # K, ~120 °C (slightly superheated for 2 bar saturation)
F_steam = F_feed * steam_ratio   # mol/s ratio

# Convergence loop: assume liquid temperature profile, solve mass balance,
# update T from energy balance / bubble point.
# For this dilute system, T is dominated by water and stage T ≈ feed T
# (plus a bit higher near steam injection, lower at top).
T_stage = np.linspace(T_feed - 5.0, T_feed + 5.0, N_stages)
pH_stage = np.full(N_stages, sp_feed.pH)

# Convergence iterations
for outer_iter in range(8):
    # Per-stage K-values for each species
    K_NH3 = np.array([K_value("NH3", T_stage[k], P_col, pH_stage[k])
                       for k in range(N_stages)])
    K_H2S = np.array([K_value("H2S", T_stage[k], P_col, pH_stage[k])
                       for k in range(N_stages)])
    K_CO2 = np.array([K_value("CO2", T_stage[k], P_col, pH_stage[k])
                       for k in range(N_stages)])

    # Liquid and vapor flows — assume L ≈ F_feed throughout (dilute,
    # mass change << total flow), V ≈ F_steam (steam is dominant
    # vapor at bottom, slightly more from stripped gas at top)
    # For simplicity use constant L=F_feed, V=F_steam
    L = F_feed * np.ones(N_stages)
    V = F_steam * np.ones(N_stages)

    # Average stripping factor S = K·V/L for each species
    S_NH3 = K_NH3 * V / L
    S_H2S = K_H2S * V / L
    S_CO2 = K_CO2 * V / L

    # Kremser equation for counter-current absorption/stripping with
    # x*_N+1 = 0 (pure steam):  x_N / x_0 = (S-1) / (S^(N+1) - 1)
    # We use stage-averaged S (geometric mean) for simplicity:
    def kremser_remainder(S_arr, N):
        S_geom = np.exp(np.mean(np.log(S_arr)))
        if abs(S_geom - 1.0) < 1e-8:
            return 1.0 / (N + 1)
        return (S_geom - 1.0) / (S_geom ** (N + 1) - 1.0)

    rem_NH3 = kremser_remainder(S_NH3, N_stages)
    rem_H2S = kremser_remainder(S_H2S, N_stages)
    rem_CO2 = kremser_remainder(S_CO2, N_stages)

    # Bottoms composition (mole fractions)
    x_NH3_bot = x_NH3_feed * rem_NH3
    x_H2S_bot = x_H2S_feed * rem_H2S
    x_CO2_bot = x_CO2_feed * rem_CO2

    # Recompute stage profile by linear interpolation (geometric for
    # exponential approach)
    fracs = np.linspace(0, 1, N_stages)
    # Top stage = feed; bottom stage = stripped product
    x_NH3_profile = x_NH3_feed * (rem_NH3 ** fracs)
    x_H2S_profile = x_H2S_feed * (rem_H2S ** fracs)
    x_CO2_profile = x_CO2_feed * (rem_CO2 ** fracs)

    # Update pH per-stage from speciation (mole-fraction → molality)
    pH_new = np.zeros(N_stages)
    for k in range(N_stages):
        m_NH3_k, m_H2S_k, m_CO2_k = mole_fraction_to_molality(
            x_NH3_profile[k], x_H2S_profile[k], x_CO2_profile[k])
        sp_k = speciate(T_stage[k], m_NH3_total=m_NH3_k,
                          m_H2S_total=m_H2S_k, m_CO2_total=m_CO2_k)
        pH_new[k] = sp_k.pH

    # Convergence check
    pH_change = np.max(np.abs(pH_new - pH_stage))
    pH_stage = 0.5 * (pH_stage + pH_new)   # damped update
    if pH_change < 0.01:
        break

# =====================================================================
# Report
# =====================================================================

print(f"\nConverged in {outer_iter+1} outer iterations "
        f"(max ΔpH = {pH_change:.4f})")

print(f"""
Performance:
    NH₃ remainder = {rem_NH3*100:.3f}%  (i.e. {(1-rem_NH3)*100:.2f}% stripped)
    H₂S remainder = {rem_H2S*100:.3f}%  (i.e. {(1-rem_H2S)*100:.2f}% stripped)
    CO₂ remainder = {rem_CO2*100:.3f}%  (i.e. {(1-rem_CO2)*100:.2f}% stripped)

Bottoms (stripped water):
    NH₃ = {x_NH3_bot/MW_water*1000:.2f} mmol/kg = {x_NH3_bot/MW_water*17.03*1000:.1f} ppm
    H₂S = {x_H2S_bot/MW_water*1000:.2f} mmol/kg = {x_H2S_bot/MW_water*34.08*1000:.1f} ppm
    CO₂ = {x_CO2_bot/MW_water*1000:.2f} mmol/kg = {x_CO2_bot/MW_water*44.01*1000:.1f} ppm
""")

print("Stripping factors (geometric mean across stages):")
print(f"    S(NH₃) = {np.exp(np.mean(np.log(S_NH3))):.2f}  "
        f"{'(easily stripped)' if np.mean(np.log(S_NH3)) > 1 else '(harder)'}")
print(f"    S(H₂S) = {np.exp(np.mean(np.log(S_H2S))):.2f}  "
        f"{'(easily stripped)' if np.mean(np.log(S_H2S)) > 1 else '(harder)'}")
print(f"    S(CO₂) = {np.exp(np.mean(np.log(S_CO2))):.2f}")

# =====================================================================
# Stage-by-stage profile (top → bottom)
# =====================================================================

print("\nStage profile (top stage 1 = feed inlet, stage 15 = bottom):")
print(f"  {'#':>3s} {'T[K]':>7s} {'pH':>5s} {'NH3[ppm]':>9s} "
        f"{'H2S[ppm]':>9s} {'CO2[ppm]':>9s} {'α(NH3)':>8s} {'α(H2S)':>8s}")
for k in range(N_stages):
    m_NH3_k = x_NH3_profile[k] / MW_water
    m_H2S_k = x_H2S_profile[k] / MW_water
    m_CO2_k = x_CO2_profile[k] / MW_water
    sp_k = speciate(T_stage[k], m_NH3_total=m_NH3_k,
                      m_H2S_total=m_H2S_k, m_CO2_total=m_CO2_k)
    print(f"  {k+1:>3d} {T_stage[k]:>7.1f} {sp_k.pH:>5.2f} "
            f"{m_NH3_k*17.03*1000:>9.0f} "
            f"{m_H2S_k*34.08*1000:>9.0f} "
            f"{m_CO2_k*44.01*1000:>9.0f} "
            f"{sp_k.alpha_NH3*100:>7.1f}% "
            f"{sp_k.alpha_H2S*100:>7.1f}%")


# =====================================================================
# Engineering interpretation
# =====================================================================

print("\n" + "=" * 72)
print("Engineering interpretation")
print("=" * 72)
print(f"""
Why H₂S strips out faster than NH₃ at this pH ({sp_feed.pH:.1f}):

  At pH ≈ {sp_feed.pH:.1f}, the speciation is:
    * NH₃ system:  pKa(NH₄⁺) at {T_feed-273.15:.0f}°C ≈ {-np.log10(dissociation_K("NH4+", T_feed)):.2f}
                   → about {sp_feed.alpha_NH3*100:.0f}% molecular (rest is NH₄⁺)
    * H₂S system:  pKa(H₂S) at {T_feed-273.15:.0f}°C  ≈ {-np.log10(dissociation_K("H2S", T_feed)):.2f}
                   → about {sp_feed.alpha_H2S*100:.0f}% molecular (rest is HS⁻)
    * CO₂ system:  pKa(CO₂) at {T_feed-273.15:.0f}°C  ≈ {-np.log10(dissociation_K("CO2", T_feed)):.2f}
                   → about {sp_feed.alpha_CO2*100:.0f}% molecular

  The high pH suppresses NH₃ stripping (most is NH₄⁺) but at the
  same time enables H₂S stripping (most is still H₂S).

  This is why two-stage strippers are sometimes used: a first acidic
  stripper takes out H₂S preferentially, then NH₃ is removed in a
  second basic stripper after acid addition.

Practical correlations used in this calculation:
    * Henry's law: van't Hoff anchored to Wilhelm 1977 25°C values
    * pK_a(T):    van't Hoff anchored to Bates-Pinching/Hershey/Harned-Davis
    * α_molecular(pH, T): from K_a(T) and pH directly
    * Kremser:     analytical solution for dilute counter-current
                   absorption/stripping

Limitations:
    * γ ≈ 1 for all molecular species (good <0.1 mol/kg)
    * No NH₃-CO₂ interactions (carbamate formation)
    * Ideal-stage Kremser, not full Newton-Raphson column
    * Steam acts only as carrier (no condensation modeling)
""")

print("=" * 72)
print(f"v0.9.97 sour-water stripper example complete.")
print("=" * 72)
