"""End-to-end refinery feed analysis: ChemSep + PC-SAFT databases.

Demonstrates how the v0.9.93 (PC-SAFT databases) and v0.9.95 (ChemSep
database) integrations work together for a realistic process-design
task: characterize and flash a natural-gas-condensate stream using
parameters pulled directly from both databases.

The workflow is the kind of thing a process engineer does at the start
of every project:

    1. Look up critical properties for every species (ChemSep)
    2. Build a cubic EOS for vapor-phase work (PR)
    3. Look up PC-SAFT parameters and binary kij for the same species
    4. Compare PT-flash predictions from the two EOSs
    5. Use ChemSep DIPPR correlations as the reference for pure-fluid Psat

Bridging refinery-spec characterization to molecular-thermodynamics
parameters is what makes a process simulator credible.
"""
from __future__ import annotations

import warnings
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np

from stateprop.chemsep import (
    lookup_chemsep, evaluate_property,
    get_critical_constants, get_molar_mass,
)
from stateprop.saft import lookup_pcsaft, lookup_kij, make_saft_mixture


# =====================================================================
# Step 1 — Define the natural-gas-condensate stream
# =====================================================================

print("=" * 70)
print("Step 1: Natural-gas condensate composition")
print("=" * 70)

species = [
    "Methane",
    "Ethane",
    "Propane",
    "n-Butane",
    "n-Pentane",
    "n-Hexane",
    "Carbon dioxide",
    "Nitrogen",
]
composition = np.array([
    0.78,   # methane (lean gas with NGL content)
    0.09,   # ethane
    0.05,   # propane
    0.03,   # butane
    0.015,  # pentane
    0.005,  # hexane
    0.020,  # CO2
    0.010,  # N2
])
composition = composition / composition.sum()

print(f"\nFeed: {len(species)} components, total = 1 mol\n")
print(f"  {'species':>16s} {'mol frac':>10s}")
for n, x in zip(species, composition):
    print(f"  {n:>16s} {x:>10.4f}")


# =====================================================================
# Step 2 — Pull critical properties from ChemSep
# =====================================================================

print("\n" + "=" * 70)
print("Step 2: Critical properties from ChemSep (446-compound database)")
print("=" * 70)

print(f"\n  {'species':>16s} {'Tc[K]':>8s} {'Pc[bar]':>9s} {'ω':>7s} "
      f"{'MW[g/mol]':>11s}")
chemsep_records = []
critical_props = []
for n in species:
    e = lookup_chemsep(name=n)
    cc = get_critical_constants(e)
    mw = get_molar_mass(e)
    chemsep_records.append(e)
    critical_props.append(cc)
    print(f"  {n:>16s} {cc['Tc']:>8.2f} {cc['Pc']/1e5:>9.2f} "
          f"{cc['omega']:>7.4f} {mw*1000:>11.2f}")


# =====================================================================
# Step 3 — Look up PC-SAFT parameters from FeOS database
# =====================================================================

print("\n" + "=" * 70)
print("Step 3: PC-SAFT parameters from Esper 2023 (1842-compound DB)")
print("=" * 70)

# Map ChemSep names → PC-SAFT database names. Names differ slightly
# between databases (the FeOS PC-SAFT db uses "pentane" rather than
# "n-pentane", etc.).  CAS numbers are universally consistent so we
# use those as the cross-database key.
saft_cas = ["74-82-8", "74-84-0", "74-98-6", "106-97-8", "109-66-0",
            "110-54-3", "124-38-9", "7727-37-9"]

print(f"\n  {'species':>16s} {'m':>6s} {'σ[Å]':>8s} {'ε/k[K]':>9s} {'mu[D]':>7s}")
for n, cas in zip(species, saft_cas):
    c = lookup_pcsaft(cas=cas)
    print(f"  {n:>16s} {c.m:>6.3f} {c.sigma:>8.4f} {c.epsilon_k:>9.2f} "
          f"{c.dipole_moment:>7.3f}")


# =====================================================================
# Step 4 — Pull binary kij from Rehner 2023
# =====================================================================

print("\n" + "=" * 70)
print("Step 4: Binary interactions from Rehner 2023 (7848-pair DB)")
print("=" * 70)

# Show all pairs involving methane (since methane is dominant in the
# feed, methane-containing kij values matter most for PT-flash accuracy)
print(f"\n  Methane-binary kij values:")
print(f"  {'pair':>30s} {'kij':>10s}")
for n2, cas2 in zip(species[1:], saft_cas[1:]):
    kij = lookup_kij(cas1="74-82-8", cas2=cas2)
    if kij is None:
        print(f"  {'methane/'+n2:>30s} {'(missing)':>10s}")
    else:
        print(f"  {'methane/'+n2:>30s} {kij:>10.4f}")


# =====================================================================
# Step 5 — Build PC-SAFT mixture with auto-populated kij
# =====================================================================

print("\n" + "=" * 70)
print("Step 5: One-call PC-SAFT mixture construction")
print("=" * 70)

# make_saft_mixture pulls all components and all binary kij in one call.
# We pass CAS to ensure the lookup is unambiguous across DBs.
saft_mix = make_saft_mixture(
    names=[n.lower() for n in species],
    cas_list=saft_cas,
    composition=composition.tolist(),
)
print(f"\n  PC-SAFT mixture: {saft_mix.N} components, "
      f"{len(saft_mix._k_ij)} non-zero kij values from database")


# =====================================================================
# Step 6 — Compare PT-flash predictions: PR vs PC-SAFT
# =====================================================================

print("\n" + "=" * 70)
print("Step 6: PR vs PC-SAFT density at pipeline conditions")
print("=" * 70)

from stateprop.cubic.eos import PR
from stateprop.cubic.mixture import CubicMixture

# Build PR EOS list using ChemSep critical properties
pr_eoss = [PR(T_c=cc["Tc"], p_c=cc["Pc"], acentric_factor=cc["omega"])
            for cc in critical_props]
pr_mix = CubicMixture(pr_eoss)

# Average MW of the feed
MWs = np.array([cc.get("Tc", 0) * 0 + get_molar_mass(e)
                 for e, cc in zip(chemsep_records, critical_props)])
MW_avg = float(np.dot(composition, MWs))   # kg/mol

print(f"\nFeed average MW: {MW_avg*1000:.2f} g/mol")
print(f"\n{'T [K]':>7s} {'P [bar]':>9s} {'PR ρ_v [kg/m³]':>17s} "
      f"{'SAFT ρ_v [kg/m³]':>19s} {'PR Z':>7s} {'SAFT Z':>8s}")

for T, P_bar in [(283.15, 70), (300.0, 100), (350.0, 100), (400.0, 200)]:
    P_Pa = P_bar * 1e5
    rho_pr = pr_mix.density_from_pressure(p=P_Pa, T=T, x=composition,
                                            phase_hint="vapor")
    rho_pr_kg = float(rho_pr) * MW_avg
    Z_pr = P_Pa / (float(rho_pr) * 8.314 * T)
    rho_saft = saft_mix.density_from_pressure(p=P_Pa, T=T,
                                                phase_hint="vapor")
    rho_saft_kg = float(rho_saft) * MW_avg
    Z_saft = P_Pa / (float(rho_saft) * 8.314 * T)
    print(f"{T:>7.1f} {P_bar:>9.0f} {rho_pr_kg:>17.2f} {rho_saft_kg:>19.2f} "
          f"{Z_pr:>7.4f} {Z_saft:>8.4f}")


# =====================================================================
# Step 7 — Use ChemSep DIPPR-101 as Psat reference for distillation
# =====================================================================

print("\n" + "=" * 70)
print("Step 7: ChemSep DIPPR-101 → Psat functions ready for distillation")
print("=" * 70)

# This is exactly the form needed by stateprop.distillation.distillation_column:
# psat_funcs is a list of T -> Pa callables, one per species
psat_funcs = []
for e in chemsep_records:
    # Closure captures `e` by default-arg trick (avoids late-binding bug)
    psat_funcs.append(
        lambda T, e=e: evaluate_property(e, "vapor_pressure", T))

# Verify at atmospheric boiling: each pure species at its NBP should
# give P ≈ 1 atm
print(f"\n  Verification: pure-species Psat at NBP should give ~101325 Pa")
print(f"  {'species':>16s} {'NBP [K]':>9s} {'Psat [Pa]':>12s} {'err':>8s}")
for n, e, psat_fn in zip(species, chemsep_records, psat_funcs):
    if "normal_boiling_point" not in e:
        # Permanent gases (CO2, N2) don't have a 1-atm NBP
        print(f"  {n:>16s} {'(no NBP — permanent gas at 1 atm)':>43s}")
        continue
    NBP = e["normal_boiling_point"]["value"]
    psat = psat_fn(NBP)
    err = (psat - 101325) / 101325 * 100
    print(f"  {n:>16s} {NBP:>9.2f} {psat:>12.0f} {err:>+7.2f}%")


# =====================================================================
# Summary
# =====================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
The two databases compose naturally:

    ChemSep (Kooijman-Taylor 2018)      PC-SAFT (Esper-Rehner 2023)
    ─────────────────────────────       ──────────────────────────
    Tc, Pc, ω, MW                  →    PR EOS construction
    DIPPR-101 Psat                 →    Distillation column psat_funcs
    DIPPR-100 Cp_ig                →    Reaction equilibrium / EB
    UNIFAC group counts            →    Activity model construction
    Mathias-Copeman α-coefficients →    High-accuracy cubic EOS Psat
    Hf, Gf, S° formation           →    Gibbs minimizer reference state

                                        m, σ, ε/k                   →
                                        SAFTMixture parameters
                                        kij from binary database     →
                                        Auto-populated mixture kij

For a refinery feed of N species the entire characterization is
~30 lines of Python. The remaining work is just feeding those
parameters into stateprop's columns, flashes, or reactive solvers.
""")
