"""Aqueous electrolyte thermodynamics: Pitzer ion-interaction model.

Demonstrates v0.9.96's `stateprop.electrolyte` module — activity
coefficients, osmotic coefficients, and water activity for aqueous
salt solutions using the Pitzer (1973, 1991) ion-interaction
framework.

Pitzer's model is the de-facto standard for aqueous electrolyte
thermodynamics: it composes a long-range Debye-Hückel electrostatic
term with short-range ion-ion virial corrections (β⁰, β¹, β², C^φ),
and reproduces published γ_± and φ data to better than 0.5% for
1:1 and 2:1 salts up to several mol/kg.

This example covers:

    * γ_± for NaCl, KCl, HCl, CaCl2 across dilute → concentrated
    * Comparison of Pitzer / Davies / pure Debye-Hückel limiting law
    * Water activity vs molality (used for relative-humidity over
      salt solutions, food-science applications)
    * Salt mixtures with γ > 1 (HCl at high m, salting-in)
    * Custom Pitzer parameters for a hypothetical salt
"""
from __future__ import annotations
import warnings
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np
from stateprop.electrolyte import (
    PitzerModel, PitzerSalt, list_salts,
    debye_huckel_A, davies_log_gamma_pm, debye_huckel_log_gamma_pm,
    water_density, water_dielectric,
)


# =====================================================================
# Part 1: Constants and the Debye-Hückel A coefficient
# =====================================================================

print("=" * 72)
print("Part 1: Debye-Hückel A coefficient and water properties")
print("=" * 72)

print(f"\nWater properties (Bradley-Pitzer 1979 fit):")
print(f"  ρ_w(25°C) = {water_density(298.15):.2f} kg/m³  (NIST 997.05)")
print(f"  ε_r(25°C) = {water_dielectric(298.15):.2f}      (CRC 78.38)")

print(f"\nA_φ (Pitzer 1991 Eq. 1.14) at various T:")
for T in [273.15, 298.15, 323.15, 348.15, 373.15]:
    A = debye_huckel_A(T)
    print(f"  T = {T:.2f} K ({T-273.15:.0f}°C):  A_φ = {A:.4f}")

print("\nA_φ rises with T because ε_r drops faster than √T grows;")
print("at higher T the Debye-Hückel screening becomes less effective.")


# =====================================================================
# Part 2: Bundled salts
# =====================================================================

print("\n" + "=" * 72)
print(f"Part 2: {len(list_salts())} bundled Pitzer salts at 25°C")
print("=" * 72)

print()
print(", ".join(list_salts()))


# =====================================================================
# Part 3: γ_± for several salts vs Robinson-Stokes 1959
# =====================================================================

print("\n" + "=" * 72)
print("Part 3: Mean ionic activity coefficient γ_± vs Robinson-Stokes 1959")
print("=" * 72)

# (Salt, list of (m, γ_lit) tuples)
data = [
    ("NaCl",  [(0.001, 0.965), (0.01, 0.902), (0.1, 0.778),
                (0.5, 0.681), (1.0, 0.657), (2.0, 0.668),
                (3.0, 0.714), (5.0, 0.874)]),
    ("KCl",   [(0.1, 0.770), (0.5, 0.649), (1.0, 0.604),
                (2.0, 0.573), (4.0, 0.582)]),
    ("HCl",   [(0.1, 0.796), (0.5, 0.757), (1.0, 0.809),
                (2.0, 1.009), (3.0, 1.316)]),
    ("CaCl2", [(0.1, 0.518), (0.5, 0.448), (1.0, 0.500),
                (2.0, 0.792)]),
]

for salt, entries in data:
    p = PitzerModel(salt)
    print(f"\n  {salt}:")
    print(f"    {'m':>7s} {'γ±_Pitzer':>11s} {'γ±_RS_1959':>12s} {'err%':>8s}")
    for m, lit in entries:
        g = p.gamma_pm(m)
        err = (g - lit) / lit * 100
        print(f"    {m:>7.3f} {g:>11.4f} {lit:>12.4f} {err:>+7.2f}%")


# =====================================================================
# Part 4: Pitzer vs Davies vs DH limiting law
# =====================================================================

print("\n" + "=" * 72)
print("Part 4: Three model regimes for NaCl")
print("=" * 72)

p = PitzerModel("NaCl")
print(f"\n  {'m':>9s} {'DH-limit':>10s} {'Davies':>10s} {'Pitzer':>10s} {'RS_1959':>10s}")
for m in [1e-5, 1e-3, 1e-2, 0.1, 1.0, 5.0]:
    log_dh = debye_huckel_log_gamma_pm(1, -1, m)
    log_dav = davies_log_gamma_pm(1, -1, m)
    g_pitz = p.gamma_pm(m)
    g_dh = 10 ** log_dh
    g_dav = 10 ** log_dav
    rs = {1e-5: "—", 1e-3: 0.965, 1e-2: 0.902, 0.1: 0.778,
          1.0: 0.657, 5.0: 0.874}.get(m, "—")
    rs_str = f"{rs:.4f}" if isinstance(rs, float) else f"{rs:>10s}"
    print(f"  {m:>9.1e} {g_dh:>10.4f} {g_dav:>10.4f} "
          f"{g_pitz:>10.4f} {rs_str:>10s}")

print("\nObservations:")
print("  * DH limiting law breaks down above I ≈ 0.001 mol/kg")
print("  * Davies equation works to ~0.5 mol/kg, then drifts")
print("  * Pitzer matches data across 5 orders of magnitude in m")


# =====================================================================
# Part 5: Water activity (relative humidity over salt solutions)
# =====================================================================

print("\n" + "=" * 72)
print("Part 5: Water activity a_w over saturated salt solutions")
print("=" * 72)
print("\n(a_w gives the equilibrium relative humidity in % when air is")
print(" in contact with a salt solution at the given molality.)")

print(f"\n  {'salt':>8s} {'m':>5s} {'a_w':>8s} {'RH%':>7s}")
for salt, m in [("NaCl", 1.0), ("NaCl", 4.0), ("KCl", 1.0),
                  ("KCl", 4.0), ("CaCl2", 2.0), ("CaCl2", 5.0),
                  ("MgCl2", 2.0)]:
    try:
        p = PitzerModel(salt)
        a_w = p.water_activity(m)
        print(f"  {salt:>8s} {m:>5.1f} {a_w:>8.5f} {a_w*100:>6.2f}%")
    except KeyError:
        print(f"  {salt:>8s} not in DB")

print("\nThese values are the 'salt-induced humidity reduction' used")
print("for moisture control in food packaging, electronics drying, and")
print("the standard tables for relative-humidity calibration of sensors.")


# =====================================================================
# Part 6: A salt with γ > 1 (HCl)
# =====================================================================

print("\n" + "=" * 72)
print("Part 6: HCl shows γ_± > 1 at high m (salting-in)")
print("=" * 72)

p = PitzerModel("HCl")
print(f"\n  {'m':>4s} {'γ±':>9s} {'(φ-1)':>10s}")
for m in [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]:
    g = p.gamma_pm(m)
    phi = p.osmotic_coefficient(m)
    print(f"  {m:>4.2f} {g:>9.4f} {phi-1:>+10.4f}")

print("\nHCl γ_± dips below 1 at low m (DH dominates), then rises above 1")
print("at moderate m as ion-water interactions dominate.  This 'salting-in'")
print("behavior is characteristic of strong acids.")


# =====================================================================
# Part 7: Custom Pitzer parameters
# =====================================================================

print("\n" + "=" * 72)
print("Part 7: Custom Pitzer parameters for a hypothetical salt")
print("=" * 72)

# Imagine you have lab data for some salt — perhaps a nitrate or
# acetate not in the bundled set.  Create a PitzerSalt directly:
custom = PitzerSalt(
    name="HypotheticalSalt",
    z_M=1, z_X=-1, nu_M=1, nu_X=1,
    beta_0=0.040,
    beta_1=0.220,
    C_phi=0.0008,
)

p = PitzerModel(custom)
print(f"\n  Custom 1:1 salt with β⁰=0.040, β¹=0.220, C^φ=0.0008:")
print(f"  {'m':>4s} {'γ±':>9s} {'φ':>9s}")
for m in [0.1, 0.5, 1.0, 2.0]:
    g = p.gamma_pm(m)
    phi = p.osmotic_coefficient(m)
    print(f"  {m:>4.1f} {g:>9.4f} {phi:>9.4f}")


# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 72)
print("Summary")
print("=" * 72)
print("""
The Pitzer model is the workhorse for aqueous electrolyte thermo:
    * Validated against Robinson-Stokes 1959 reference data
    * 18 salts bundled at 25 °C from Pitzer 1991 / Kim-Frederick 1988
    * Custom parameter sets are a one-line PitzerSalt(...) call

When to use each model:
    * Pure Debye-Hückel:  I < 0.001 mol/kg only — limit law
    * Davies equation:     I < 0.5 mol/kg; convenient closed form
    * Pitzer:              everything else (the production tool)

Roadmap for v0.9.97+:
    * Multi-electrolyte mixing terms (NaCl + KCl + ...)
    * eNRTL refinement (full Chen-Song 2004 form)
    * Multi-solvent (water + methanol, etc.)
    * T-dependence of Pitzer parameters
    * Coupling to distillation: sour-water stripper example
""")
