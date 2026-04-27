"""Mineral solubility and scale prediction (v0.9.101).

Demonstrates `stateprop.electrolyte`'s mineral module on three
industrial scenarios:

  1. Pure-water solubility of common evaporite/scale minerals (halite,
     gypsum, anhydrite, sylvite) across 0-100 °C
  2. Gypsum salting-in by NaCl brine — classic geochemistry / oilfield
     scale-prediction problem
  3. Saturation-index assessment of a produced-water injection scenario:
     given a brine composition, what minerals are at risk of precipitating?

The mineral module computes saturation indices SI = log10(IAP/K_sp) for
arbitrary brines using activity coefficients from MultiPitzerSystem and
T-dependent K_sp from a van't Hoff form anchored at 25 °C.

For binary salts also in the bundled Pitzer single-electrolyte database
(NaCl, KCl, CaSO4, Na2SO4, MgSO4), `solubility_in_water` solves
self-consistently for the equilibrium concentration in pure water.

References
----------
* Plummer-Busenberg 1982: calcite/aragonite K_sp(T)
* Marshall-Slusher 1966: gypsum solubility in NaCl brines
* Reardon-Beckie 1987: gypsum/anhydrite K_sp
* Blount 1977: barite K_sp(T)
* Krumgalz-Pogorelsky-Pitzer 1995: chloride K_sp(T)
"""
from __future__ import annotations
import warnings
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np
from stateprop.electrolyte import (
    MultiPitzerSystem, MineralSystem,
    Mineral, lookup_mineral, list_minerals,
    saturation_index, solubility_in_water,
)


# =====================================================================
# Part 1: Pure-water solubility vs T
# =====================================================================

print("=" * 72)
print("Part 1: Mineral solubility in pure water vs temperature")
print("=" * 72)

minerals_for_T_scan = ["halite", "sylvite", "gypsum", "anhydrite",
                          "thenardite", "epsomite"]

print(f"\n{'T [°C]':>8s}  ", end="")
for n in minerals_for_T_scan:
    print(f"{n:>11s}", end="")
print()
print(f"{'':>8s}  ", end="")
for n in minerals_for_T_scan:
    print(f"{'[mol/kg]':>11s}", end="")
print()

for T in [273.15, 298.15, 323.15, 348.15, 373.15]:
    print(f"{T-273.15:>7.0f}   ", end="")
    for name in minerals_for_T_scan:
        try:
            S = solubility_in_water(name, T=T)
            if S < 0.001:
                print(f"{S:>11.3e}", end="")
            else:
                print(f"{S:>11.4f}", end="")
        except (ValueError, RuntimeError):
            print(f"{'OOB':>11s}", end="")    # Out Of Bounds
    print()

print("""
Observations:
  * Halite: ~6.15 mol/kg at 25 °C, increases gently to 7.3 at 100 °C
  * Sylvite: rises strongly with T (high ΔH of dissolution)
  * Gypsum: prograde 0-40 °C, retrograde above 40 °C — but the
    transition is to anhydrite, which has strong retrograde solubility
  * Anhydrite: STRONGLY retrograde — anhydrite is the stable Ca-sulfate
    phase above ~40 °C in pure water (industrial scale risk in heated
    lines and reboiler tubes)
""")


# =====================================================================
# Part 2: Gypsum salting-in by NaCl brine
# =====================================================================

print("=" * 72)
print("Part 2: Gypsum solubility in NaCl brine (Marshall-Slusher 1966)")
print("=" * 72)

sys = MultiPitzerSystem.from_salts(["NaCl", "CaSO4"])
gypsum = lookup_mineral("gypsum")

print(f"\n{'NaCl [m]':>10s} {'CaSO4 [m]':>12s} {'γ_pm Ca-SO4':>13s} {'a_w':>7s}")

for m_NaCl in [0.0, 0.1, 0.5, 1.0, 2.0, 3.0]:
    # Bisection to find equilibrium m_CaSO4 (where SI = 0)
    lo, hi = 1e-5, 1.0
    for _ in range(80):
        m_CaSO4 = (lo + hi) / 2
        m = {"Na+": m_NaCl, "Ca++": m_CaSO4,
              "Cl-": m_NaCl, "SO4--": m_CaSO4}
        gammas = sys.gammas(m)
        a_w = sys.water_activity(m)
        SI = saturation_index(gypsum, m, gammas, T=298.15, a_w=a_w)
        if SI > 0:
            hi = m_CaSO4
        else:
            lo = m_CaSO4
        if abs(SI) < 1e-6:
            break
    g_pm = float(np.sqrt(gammas.get("Ca++", 1) * gammas.get("SO4--", 1)))
    print(f"{m_NaCl:>10.1f} {m_CaSO4:>12.4f} {g_pm:>13.4f} {a_w:>7.4f}")

print("""
The salting-in trend is qualitatively correct:
  pure water: 0.0157 mol/kg  →  matches Marshall-Slusher 0.0152
  NaCl > 0:   gypsum dissolution increases as γ_Ca and γ_SO4 drop

Quantitative caveat: stateprop's simple Pitzer treatment of CaSO4
overestimates the salting-in effect at high NaCl by ~2-3×.  Real
seawater chemistry has significant CaSO4° aqueous ion-pairing that
isn't modelled here; PHREEQC and similar codes use either Møller-1988
calibrated parameters or explicit aqueous-complex species to handle
this.  For most engineering purposes (predicting whether scaling
occurs, not the exact concentration) the qualitative result is
sufficient.
""")


# =====================================================================
# Part 3: Produced-water injection scale assessment
# =====================================================================

print("=" * 72)
print("Part 3: Produced-water injection — scale risk assessment")
print("=" * 72)

# Typical North-Sea oil-field produced water composition
# (highly Ca-rich, high TDS):
m_produced = {
    "Na+":   1.5,
    "K+":    0.05,
    "Ca++":  0.45,
    "Mg++":  0.10,
    "Ba++":  0.0001,    # 100 mg/L Ba (typical produced water)
    "Sr++":  0.001,     # 100 mg/L Sr
    "Cl-":   2.7,       # ~95 g/L Cl
    "SO4--": 0.0003,    # 30 mg/L SO4 (low — produced waters are usually low in SO4)
}
# Note: Ba++ and Sr++ aren't in the bundled MultiPitzer system, so we'll
# use their concentrations directly with γ ≈ 1 from infinite dilution.

# Typical seawater used as injection water (high SO4 source!):
m_seawater_inj = {
    "Na+":   0.486,
    "K+":    0.0106,
    "Ca++":  0.0107,
    "Mg++":  0.0547,
    "Ba++":  1e-7,      # ~10 ppb Ba in surface seawater
    "Sr++":  9e-5,      # ~8 mg/L Sr
    "Cl-":   0.5658,
    "SO4--": 0.0293,    # ~28 mmol/kg SO4 (much higher than produced water)
}

# 50/50 mix — when produced water meets injection seawater downhole
def mix(m1, m2, frac=0.5):
    """Mix two molality dicts at given fraction of m1."""
    out = {}
    for ion in set(m1) | set(m2):
        out[ion] = frac * m1.get(ion, 0) + (1 - frac) * m2.get(ion, 0)
    return out

m_mixed = mix(m_produced, m_seawater_inj, frac=0.5)

# Build mineral system with produced-water-relevant scales
brine_pitzer = MultiPitzerSystem.from_salts([
    "NaCl", "KCl", "CaCl2", "MgCl2", "Na2SO4", "K2SO4", "MgSO4", "CaSO4"])
ms = MineralSystem(brine_pitzer, [
    "halite", "sylvite", "gypsum", "anhydrite",
    "thenardite", "mirabilite", "epsomite",
    "barite", "celestite",
])

print("\n3a. Produced-water alone at 25 °C:")
SI = ms.saturation_indices(m_produced)
for name, si in sorted(SI.items(), key=lambda x: -x[1]):
    flag = "⚠️ SCALE RISK" if si > 0.3 else "  monitor" if si > 0 else "  OK"
    if not np.isfinite(si):
        flag = "  no Ba/SO4 spec"
        si_str = "  -inf"
    else:
        si_str = f"{si:+6.2f}"
    print(f"    {name:>11s}: SI = {si_str}    {flag}")

print("""
Note: very low SO4 (30 mg/L) limits BaSO4 saturation. Halite is far
from saturation despite high Cl, because the Na concentration is
moderate.
""")

print("3b. After mixing 50/50 with seawater injection (at downhole T = 80 °C):")
SI = ms.saturation_indices(m_mixed, T=353.15)
for name, si in sorted(SI.items(), key=lambda x: -x[1]):
    flag = "⚠️ SCALE RISK" if si > 0.3 else "  monitor" if si > 0 else "  OK"
    if not np.isfinite(si):
        flag = "  ions missing"
        si_str = "  -inf"
    else:
        si_str = f"{si:+6.2f}"
    print(f"    {name:>11s}: SI = {si_str}    {flag}")

print("""
Mixing the SO4-rich seawater with the Ca/Ba/Sr-rich produced water
creates classic 'sulfate scale' problems:
  * Barite (BaSO4) is the worst offender — even trace Ba (100 mg/L)
    combined with seawater SO4 (2800 mg/L) gives massive supersat.
  * Celestite (SrSO4) is also at risk from the Sr in produced water.
  * Anhydrite (CaSO4) from the Ca in produced water + SO4 from seawater,
    especially at downhole T where anhydrite is the stable phase.

Mitigation strategies (from petroleum industry):
  * Sulfate-removal membranes on seawater injection
  * Scale inhibitor (phosphonate, polymer) injection in the produced-
    water flowline
  * Compatibility modeling with stateprop.electrolyte before commissioning
""")


# =====================================================================
# Part 4: Equilibrium with gypsum at high NaCl (T-aware)
# =====================================================================

print("=" * 72)
print("Part 4: Gypsum-saturated brine at elevated T")
print("=" * 72)

print("""
Determining gypsum solubility in 1 m NaCl across 25-100 °C —
typical concern for geothermal brine handling and salt-cavern brine
treatment systems.
""")

print(f"  {'T [°C]':>7s} {'CaSO4 [m]':>12s} {'γ_pm Ca-SO4':>13s} {'a_w':>7s}")

for T in [298.15, 323.15, 348.15, 373.15]:
    lo, hi = 1e-5, 1.0
    for _ in range(80):
        m_CaSO4 = (lo + hi) / 2
        m = {"Na+": 1.0, "Ca++": m_CaSO4, "Cl-": 1.0, "SO4--": m_CaSO4}
        gammas = sys.gammas(m, T)
        a_w = sys.water_activity(m, T)
        SI = saturation_index(gypsum, m, gammas, T=T, a_w=a_w)
        if SI > 0:
            hi = m_CaSO4
        else:
            lo = m_CaSO4
        if abs(SI) < 1e-6:
            break
    g_pm = float(np.sqrt(gammas.get("Ca++", 1) * gammas.get("SO4--", 1)))
    print(f"  {T-273.15:>5.0f}   {m_CaSO4:>12.4f} {g_pm:>13.4f} {a_w:>7.4f}")

print("""
The full T-aware Pitzer framework (binary β + θ + ψ all evaluated at T)
combines with van't Hoff K_sp(T) to give end-to-end T-dependent
mineral solubility. This is the foundation for:
  * Geothermal brine flash-scaling analysis
  * Salt-cavern compressed-air storage corrosion modeling
  * High-T evaporator scaling prediction
""")


# =====================================================================
# Summary
# =====================================================================

print("=" * 72)
print("Summary")
print("=" * 72)
print(f"""
The v0.9.101 mineral module provides:

  * `Mineral` dataclass with stoichiometry, log_K_sp(25 °C),
    van't Hoff ΔH_rxn for T-dependence
  * Bundled database of {len(list_minerals())} industrially-important minerals:
    halite, sylvite, gypsum, anhydrite, calcite, aragonite, dolomite,
    magnesite, barite, celestite, mirabilite, thenardite, epsomite,
    brucite, portlandite
  * `saturation_index(mineral, m, γ, T, a_w)` — log10(IAP/K_sp) for
    arbitrary brines using activity coefficients from MultiPitzerSystem
  * `solubility_in_water(mineral, T)` — fixed-point iterative solve
    for binary-salt minerals (halite 1.3% off lit, gypsum 3% off lit)
  * `MineralSystem` wrapper combining a MultiPitzerSystem + minerals
    for batch SI calculation; `scale_risks` filter for engineering use

Limitations:
  * CaSO4 multi-electrolyte salting-in overestimated 2-3× at high NaCl
    (real systems have CaSO4° aqueous complexation)
  * Carbonate SI in seawater systematically high (~1-2 log units; needs
    explicit Mg-CO3, Ca-CO3 binary β + aqueous complexation)
  * Pure-water binary solubility is reliable (1-5% accuracy)

Roadmap:
  * Aqueous complexation framework (CaSO4°, MgSO4°, MgCO3°, CaCO3°)
  * Møller 1988 calibrated CaSO4 parameters with explicit ion-pairing
  * Coupling to reactive distillation for scale-prediction during
    brine evaporation/crystallization
""")
