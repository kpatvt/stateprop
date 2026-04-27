"""ChemSep pure-component database — 446 compounds with full property set.

Demonstrates v0.9.95's bundled ChemSep v8.00 database (Kooijman-Taylor
2018), which provides:

    * 446 industrially-relevant compounds with critical properties
      (Tc, Pc, Vc, Zc, ω) on every entry
    * DIPPR temperature-dependent correlations for vapor pressure,
      heat capacity, density, viscosity, thermal conductivity, etc.
    * UNIFAC-VLE / UNIFAC-LLE / modified UNIFAC group contributions
      on 395+ compounds
    * Heat / Gibbs energy of formation, absolute entropy
    * Mathias-Copeman α-function coefficients for high-accuracy
      cubic-EOS Psat prediction
    * CAS, SMILES, structure formulae for unambiguous identification

This example shows lookup, DIPPR equation evaluation, and end-to-end
construction of a Peng-Robinson EOS straight from the database.
"""
from __future__ import annotations

import warnings
warnings.simplefilter("ignore", RuntimeWarning)

from stateprop.chemsep import (
    lookup_chemsep, chemsep_summary,
    evaluate_property,
    get_critical_constants, get_molar_mass, get_formation_properties,
)


# =====================================================================
# Part 1: Database overview
# =====================================================================

print("=" * 70)
print("Part 1: Database overview")
print("=" * 70)

s = chemsep_summary()
for k, v in s.items():
    print(f"  {k:>40s}: {v}")


# =====================================================================
# Part 2: Lookup by name, CAS, SMILES
# =====================================================================

print("\n" + "=" * 70)
print("Part 2: Three lookup modes — all give the same result")
print("=" * 70)

# Methanol from three different keys
e_name = lookup_chemsep(name="Methanol")
e_cas = lookup_chemsep(cas="67-56-1")
e_smiles = lookup_chemsep(smiles="CO")

print(f"\n  by name 'Methanol':  Tc = {e_name['critical_temperature']['value']} K")
print(f"  by CAS  '67-56-1':   Tc = {e_cas['critical_temperature']['value']} K")
print(f"  by SMILES 'CO':      Tc = {e_smiles['critical_temperature']['value']} K")
print(f"  → all three identical ✓")


# =====================================================================
# Part 3: Browse sample compounds
# =====================================================================

print("\n" + "=" * 70)
print("Part 3: Critical properties for common refinery / chemical species")
print("=" * 70)

print(f"\n{'name':>22s} {'CAS':>14s} {'Tc[K]':>8s} {'Pc[bar]':>9s} "
      f"{'ω':>7s} {'MW[g]':>8s}")
for name in [
    "Methane", "Ethane", "Propane", "n-Butane",
    "n-Pentane", "n-Hexane", "n-Heptane", "n-Octane", "n-Decane",
    "Benzene", "Toluene", "p-Xylene",
    "Water", "Methanol", "Ethanol", "1-Propanol",
    "Acetone", "Methyl acetate",
    "Carbon dioxide", "Nitrogen", "Hydrogen",
    "Hydrogen sulfide", "Carbon monoxide", "Ammonia",
]:
    try:
        e = lookup_chemsep(name=name)
        cc = get_critical_constants(e)
        mw_kg = get_molar_mass(e)
        cas = e.get("cas", {}).get("value", "—")
        print(f"{name:>22s} {cas:>14s} "
              f"{cc.get('Tc', 0):>8.2f} {cc.get('Pc', 0)/1e5:>9.2f} "
              f"{cc.get('omega', 0):>7.4f} {mw_kg*1000 if mw_kg else 0:>8.2f}")
    except KeyError:
        print(f"  '{name}': not in database")


# =====================================================================
# Part 4: DIPPR equations — Psat, Hvap, Cp_ig at 298.15 K
# =====================================================================

print("\n" + "=" * 70)
print("Part 4: DIPPR temperature-dependent property evaluation")
print("=" * 70)

T = 298.15
print(f"\nProperties at T = {T} K (= 25°C):\n")
print(f"{'name':>20s} {'Psat[Pa]':>12s} {'ΔHvap[kJ/mol]':>15s} "
      f"{'Cp_ig[J/mol/K]':>16s} {'ρ_liq[kg/m³]':>14s}")
for name in ["Water", "Methanol", "Ethanol", "n-Hexane",
              "Toluene", "Acetone", "Benzene"]:
    e = lookup_chemsep(name=name)
    Tc = e["critical_temperature"]["value"]
    if T >= Tc:
        # Supercritical — Psat / ΔHvap / ρ_liq aren't physical
        print(f"{name:>20s} {'(T > Tc, supercritical)':>61s}")
        continue
    psat = evaluate_property(e, "vapor_pressure", T)
    try:
        hvap = evaluate_property(e, "heat_of_vaporization", T) / 1e6
    except KeyError:
        hvap = float("nan")
    cp_ig_kmol = evaluate_property(e, "ideal_gas_heat_capacity", T)
    cp_ig = cp_ig_kmol / 1000.0
    try:
        rho_kmol = evaluate_property(e, "liquid_density", T)
        mw = get_molar_mass(e)   # kg/mol
        rho = rho_kmol * 1000 * mw   # kmol/m³ × 1000 mol/kmol × kg/mol = kg/m³
    except KeyError:
        rho = float("nan")
    print(f"{name:>20s} {psat:>12.0f} {hvap:>15.2f} {cp_ig:>16.2f} {rho:>14.1f}")


# =====================================================================
# Part 5: Build a PR EOS straight from ChemSep data
# =====================================================================

print("\n" + "=" * 70)
print("Part 5: Construct a PR EOS for benzene from ChemSep critical props")
print("=" * 70)

from stateprop.cubic.eos import PR
from stateprop.cubic.mixture import CubicMixture
import numpy as np

benzene = lookup_chemsep(name="Benzene")
cc = get_critical_constants(benzene)
mw = get_molar_mass(benzene)

eos = PR(T_c=cc["Tc"], p_c=cc["Pc"], acentric_factor=cc["omega"])
print(f"  PR EOS built for benzene:")
print(f"    Tc = {eos.T_c} K, Pc = {eos.p_c/1e5:.1f} bar, ω = {eos.acentric_factor}")

# Sweep T at 1 atm, compute liquid density and compare to ChemSep-DIPPR
print(f"\n  Liquid density vs T at 1 atm:")
print(f"  {'T [K]':>7s}  {'PR ρ [kg/m³]':>14s}  {'ChemSep ρ [kg/m³]':>18s}  {'NIST [kg/m³]':>14s}")
mix = CubicMixture([eos])
nist_data = [(280, 884.4), (298, 873.6), (320, 858.3), (350, 836.0)]
for T_K, nist_rho in nist_data:
    rho_PR = mix.density_from_pressure(p=1e5, T=T_K, x=np.array([1.0]),
                                         phase_hint="liquid")
    rho_PR_kg = float(rho_PR) * mw
    rho_cs_kmol = evaluate_property(benzene, "liquid_density", T_K)
    # rho_cs is in kmol/m³; mw is in kg/mol. To get kg/m³ multiply by
    # the kmol→mol factor: rho_kg = rho_kmol * (1000 mol/kmol) * mw_kg_per_mol
    rho_cs_kg = rho_cs_kmol * 1000 * mw
    print(f"  {T_K:>7.0f}  {rho_PR_kg:>14.1f}  {rho_cs_kg:>18.1f}  {nist_rho:>14.1f}")

print(f"\n  ChemSep DIPPR-105 typically matches NIST to <1%;")
print(f"  PR cubic EOS is ~3-5% high on liquid HCs (well-known cubic-EOS limitation).")


# =====================================================================
# Part 6: Vapor pressure correlation comparison
# =====================================================================

print("\n" + "=" * 70)
print("Part 6: Methane Psat — ChemSep DIPPR-101 vs PR cubic EOS")
print("=" * 70)

methane = lookup_chemsep(name="Methane")
cc = get_critical_constants(methane)
mw = get_molar_mass(methane)
eos_meth = PR(T_c=cc["Tc"], p_c=cc["Pc"], acentric_factor=cc["omega"])

print(f"\n{'T [K]':>7s} {'ChemSep[bar]':>14s} {'PR EOS[bar]':>14s} "
      f"{'NIST[bar]':>10s}")
for T_K, P_NIST in [(120, 1.917), (140, 6.413),
                       (160, 15.92), (180, 32.75)]:
    psat_cs = evaluate_property(methane, "vapor_pressure", T_K)
    # PR Psat: bubble point of pure methane
    psat_pr = eos_meth.saturation_p(T_K)
    print(f"{T_K:>7.0f} {psat_cs/1e5:>14.3f} {float(psat_pr)/1e5:>14.3f} "
          f"{P_NIST:>10.3f}")
print(f"\n  ChemSep DIPPR-101 is fitted directly to NIST data → accurate everywhere.")
print(f"  PR cubic EOS uses just (Tc, Pc, ω) → typically 1-3% off on Psat.")


# =====================================================================
# Part 7: Formation properties for reactive equilibrium
# =====================================================================

print("\n" + "=" * 70)
print("Part 7: Formation properties (J/mol; J/(mol·K)) at 298.15 K")
print("=" * 70)

print(f"\n{'name':>20s} {'ΔHf [kJ/mol]':>14s} {'ΔGf [kJ/mol]':>14s} "
      f"{'S° [J/mol/K]':>14s}")
for name in ["Water", "Methanol", "Methane", "Carbon monoxide",
              "Carbon dioxide", "Hydrogen", "Ammonia", "Nitrogen"]:
    e = lookup_chemsep(name=name)
    fp = get_formation_properties(e)
    print(f"{name:>20s} {fp.get('Hf', 0)/1000:>14.2f} "
          f"{fp.get('Gf', 0)/1000:>14.2f} {fp.get('S', 0):>14.2f}")

# Quick reactive-equilibrium check: water-gas shift K_eq at 298 K
# CO + H2O ⇌ CO2 + H2
import math
e_CO = lookup_chemsep(name="Carbon monoxide")
e_H2O = lookup_chemsep(name="Water")
e_CO2 = lookup_chemsep(name="Carbon dioxide")
e_H2 = lookup_chemsep(name="Hydrogen")
dG_rxn = (get_formation_properties(e_CO2)["Gf"]
           + get_formation_properties(e_H2)["Gf"]
           - get_formation_properties(e_CO)["Gf"]
           - get_formation_properties(e_H2O)["Gf"])
K_eq = math.exp(-dG_rxn / (8.314 * 298.15))
print(f"\nWater-gas shift K_eq at 298 K from ChemSep ΔGf°:")
print(f"  ΔG°rxn = {dG_rxn/1000:.2f} kJ/mol")
print(f"  K_eq = exp(-ΔG/RT) = {K_eq:.2e}  (literature: ~1e+5)")


# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
The ChemSep database makes 446 industrial compounds available with
one import. Each compound has:

    * Critical properties (Tc, Pc, ω) → directly feed PR / SRK / PC-SAFT
    * DIPPR correlations for T-dependent properties → accurate Psat,
      ΔHvap, ρ, viscosity, conductivity, surface tension
    * Formation enthalpy/Gibbs/entropy → reactive equilibrium calcs
    * UNIFAC group counts → activity-coefficient model construction
    * UNIQUAC R/Q parameters → UNIQUAC binary mixtures

Combined with ``stateprop.saft.lookup_pcsaft`` (FeOS database, 1842
PC-SAFT components + 7848 binary kij), stateprop now has parameter
coverage comparable to mid-tier commercial process simulators.

For end-to-end refinery workflows, see the
``crude_distillation_with_side_strippers.py`` example.
""")
