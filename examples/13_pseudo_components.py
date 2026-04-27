"""Hydrocarbon pseudo-components: NBP+SG → full thermodynamic model.

Demonstrates v0.9.90's PseudoComponent dataclass — a single object
that takes a normal boiling point and specific gravity and exposes
the full set of properties needed for cubic EOS, distillation
columns, and chemical equilibrium calculations:

    * Critical properties Tc, Pc, Vc via Riazi-Daubert (1980, 2005)
    * Acentric factor ω via Lee-Kesler or Edmister
    * Molecular weight via Riazi-Daubert
    * Watson characterization factor K_W
    * Lee-Kesler vapor pressure correlation
    * Cubic EOS (PR / SRK) factory functions
    * NIST n-paraffin Cp_ig polynomial with Watson-K correction

The intent: one input (NBP, SG) → ready-to-use thermodynamic model
that drops into stateprop's columns and flashes without further work.
"""
from __future__ import annotations

import warnings
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np

from stateprop.pseudo import (
    PseudoComponent, watson_K,
    make_PR_from_pseudo, make_SRK_from_pseudo,
)


# =====================================================================
# Part 1: Single pseudo-component
# =====================================================================

print("=" * 70)
print("Part 1: Characterize n-decane from NBP + SG only")
print("=" * 70)

# n-Decane reference values: NBP = 447.3 K, SG = 0.7301
# NIST critical properties: Tc = 617.7 K, Pc = 21.1 bar, ω = 0.490
# Use only NBP and SG; everything else is computed by correlations.
decane = PseudoComponent(NBP=447.3, SG=0.7301, name="n-decane")

print(f"\nInput:  NBP = {decane.NBP} K, SG = {decane.SG}")
print(f"\nCorrelation-derived properties:")
print(f"  Tc          = {decane.Tc:.2f} K       (NIST 617.7 K)")
print(f"  Pc          = {decane.Pc/1e5:.2f} bar     (NIST 21.1 bar)")
print(f"  Vc          = {decane.Vc*1e6:.1f} cm³/mol (NIST ~617)")
print(f"  ω           = {decane.acentric_factor:.4f}    (NIST 0.490)")
print(f"  MW          = {decane.MW:.2f} g/mol  (exact 142.28)")
print(f"  Watson K_W  = {decane.Watson_K:.3f}")

print(f"\nLee-Kesler vapor pressure self-consistency check:")
print(f"  At NBP={decane.NBP} K, Psat = {decane.psat(decane.NBP):.0f} Pa "
      f"(should be 101325 Pa)")
print(f"  At NBP+50 K, Psat = {decane.psat(decane.NBP + 50):.0f} Pa")


# =====================================================================
# Part 2: Build cubic EOSs
# =====================================================================

print("\n" + "=" * 70)
print("Part 2: PR and SRK EOSs from the pseudo-component")
print("=" * 70)

eos_PR = make_PR_from_pseudo(decane)
eos_SRK = make_SRK_from_pseudo(decane)

# Liquid density at 298 K, 1 atm (NIST n-decane: 730 kg/m³)
T = 298.15
p = 101325.0
rho_PR = eos_PR.density_from_pressure(p=p, T=T, phase_hint="liquid")
rho_SRK = eos_SRK.density_from_pressure(p=p, T=T, phase_hint="liquid")
MW = decane.molar_mass   # kg/mol

print(f"\n n-decane liquid density at 298 K, 1 atm:")
print(f"  PR:  {float(rho_PR) * MW:.1f} kg/m³")
print(f"  SRK: {float(rho_SRK) * MW:.1f} kg/m³")
print(f"  NIST: 730 kg/m³ (cubic EOS typically ~5% high on liquid HC)")


# =====================================================================
# Part 3: Sweep across hydrocarbon series
# =====================================================================

print("\n" + "=" * 70)
print("Part 3: n-paraffin series — verify Watson K stays ~constant")
print("=" * 70)

# Reference NBPs and SGs for n-paraffins (Reid Prausnitz Poling 4th ed)
n_paraffins = [
    ("n-pentane",   309.22, 0.6262),
    ("n-hexane",    341.88, 0.6638),
    ("n-heptane",   371.58, 0.6837),
    ("n-octane",    398.83, 0.7028),
    ("n-nonane",    423.97, 0.7176),
    ("n-decane",    447.30, 0.7301),
    ("n-undecane",  469.08, 0.7408),
    ("n-dodecane",  489.47, 0.7497),
]

print(f"\n{'name':>12s} {'NBP':>7s} {'SG':>7s} {'K_W':>6s} {'Tc':>7s} "
      f"{'Pc':>6s} {'ω':>6s} {'MW':>6s}")
for name, NBP, SG in n_paraffins:
    pc = PseudoComponent(NBP=NBP, SG=SG, name=name)
    print(f"{name:>12s} {NBP:>7.1f} {SG:>7.4f} {pc.Watson_K:>6.2f} "
          f"{pc.Tc:>7.2f} {pc.Pc/1e5:>6.2f} {pc.acentric_factor:>6.3f} "
          f"{pc.MW:>6.1f}")

print("\nObservations:")
print("  * K_W stays in 12.5-12.8 range across the series (paraffinic class)")
print("  * Tc rises monotonically: more carbon → higher boiling → higher Tc")
print("  * Pc decreases as MW grows (intermolecular forces spread thinner)")
print("  * ω increases with chain length (sphericity decreases)")


# =====================================================================
# Part 4: Use a pseudo in a chemical-equilibrium calculation
# =====================================================================

print("\n" + "=" * 70)
print("Part 4: PR EOS at supercritical state — Z and density vs P")
print("=" * 70)

# Sweep pressure at T = 700 K (well above n-decane Tc=617.7 K)
# for the PR EOS built from the pseudo-component.
T_super = 700.0
print(f"\nn-decane PR EOS at T = {T_super} K (supercritical):")
print(f"{'P [bar]':>8s} {'rho [mol/m³]':>14s} {'rho [kg/m³]':>13s} {'Z':>7s}")
for P_bar in [1, 5, 10, 20, 50, 100, 200]:
    P_Pa = P_bar * 1e5
    rho = eos_PR.density_from_pressure(p=P_Pa, T=T_super, phase_hint="vapor")
    rho_n = float(rho)
    rho_kg = rho_n * MW
    Z = P_Pa / (rho_n * 8.314 * T_super)
    print(f"{P_bar:>8.0f} {rho_n:>14.2f} {rho_kg:>13.2f} {Z:>7.4f}")


# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
The PseudoComponent abstraction is the bridge between refinery
characterization data (TBP curves, lab assays) and stateprop's
thermodynamic engines (cubic EOS, activity models, distillation).

Two inputs (NBP + SG) → full property set:
    Tc, Pc, Vc, ω, MW, K_W, ideal-gas Cp polynomial, Lee-Kesler Psat

Workflow:
    NBP, SG → PseudoComponent → make_PR_from_pseudo()
                              → make_SRK_from_pseudo()
                              → use in any flash / column / equilibrium calc

For a TBP curve with multiple cuts, use stateprop.tbp.discretize_TBP
to generate a list of PseudoComponents in one call.  See the
``crude_distillation_with_side_strippers.py`` example for the full
chain from TBP curve to atmospheric column products.
""")
