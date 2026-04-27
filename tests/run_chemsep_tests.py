"""ChemSep pure-component database tests.

Validates the bundled ChemSep XML→JSON conversion (446 compounds,
Kooijman-Taylor 2018):
  - Database loads, has expected size, and is cached
  - Lookup by CAS, name, SMILES; case-insensitive name matching
  - Critical-property scalars look correct vs published references
  - DIPPR equation evaluation (eqno 100, 101, 105, 106) reproduces
    NIST values for common species
  - Convenience helpers: get_critical_constants, get_molar_mass,
    get_formation_properties
"""
from __future__ import annotations
import sys, os, warnings
import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from stateprop.chemsep import (
    load_chemsep_database, chemsep_summary,
    lookup_chemsep, evaluate_dippr, evaluate_property,
    get_critical_constants, get_molar_mass, get_formation_properties,
)

_PASS = 0
_FAIL = 0


def section(n): print(f"\n[{n}]")
def check(label, ok):
    global _PASS, _FAIL
    if ok: _PASS += 1; print(f"  PASS  {label}")
    else: _FAIL += 1; print(f"  FAIL  {label}")


# =====================================================================
# Database loading
# =====================================================================

def test_database_summary_counts():
    section("test_database_summary_counts")
    s = chemsep_summary()
    check(f"446 compounds ({s['n_compounds']})", s["n_compounds"] == 446)
    check(f"vapor_pressure on >430 ({s['n_with_vapor_pressure']})",
          s["n_with_vapor_pressure"] >= 430)
    check(f"ideal_gas_Cp on all 446 ({s['n_with_ideal_gas_heat_capacity']})",
          s["n_with_ideal_gas_heat_capacity"] == 446)
    check(f"CAS on all 446 ({s['n_with_cas']})", s["n_with_cas"] == 446)
    check(f"SMILES on >440 ({s['n_with_smiles']})",
          s["n_with_smiles"] >= 440)
    check(f"UNIFAC-VLE on >390 ({s['n_with_unifac_vle_groups']})",
          s["n_with_unifac_vle_groups"] >= 390)


def test_database_caching():
    section("test_database_caching")
    a = load_chemsep_database()
    b = load_chemsep_database()
    check("repeated load returns cached object", a is b)


# =====================================================================
# Lookup
# =====================================================================

def test_lookup_by_name():
    section("test_lookup_by_name")
    for name, expected_Tc in [
        ("Methane", 190.56),
        ("Water", 647.14),
        ("Ethanol", 514.0),
        ("Benzene", 562.05),
        ("n-Hexane", 507.6),
    ]:
        try:
            e = lookup_chemsep(name=name)
            Tc = e["critical_temperature"]["value"]
            check(f"{name}: Tc={Tc} K (expect ≈{expected_Tc})",
                  abs(Tc - expected_Tc) < 1.5)
        except KeyError:
            check(f"{name} not found", False)


def test_lookup_by_cas():
    section("test_lookup_by_cas")
    cases = [
        ("Methane", "74-82-8"),
        ("Water", "7732-18-5"),
        ("Ethanol", "64-17-5"),
        ("Benzene", "71-43-2"),
        ("Carbon dioxide", "124-38-9"),
    ]
    for name, cas in cases:
        try:
            e_name = lookup_chemsep(name=name)
            e_cas = lookup_chemsep(cas=cas)
            check(f"{name}: CAS lookup matches name lookup",
                  e_name["library_index"]["value"]
                  == e_cas["library_index"]["value"])
        except KeyError:
            check(f"{name}/{cas} round-trip", False)


def test_lookup_by_smiles():
    section("test_lookup_by_smiles")
    for smiles, expected in [("C", "Methane"), ("O", "Water"),
                              ("CCO", "Ethanol"), ("c1ccccc1", "Benzene")]:
        try:
            e = lookup_chemsep(smiles=smiles)
            actual = e["name"]["value"]
            check(f"SMILES {smiles!r} → {actual} (expect {expected})",
                  actual.lower() == expected.lower())
        except KeyError:
            check(f"SMILES {smiles!r} not found", False)


def test_lookup_case_insensitive():
    section("test_lookup_case_insensitive")
    e1 = lookup_chemsep(name="Methane")
    e2 = lookup_chemsep(name="methane")
    e3 = lookup_chemsep(name="METHANE")
    check("lowercase / mixed / upper give same record",
          e1["library_index"] == e2["library_index"]
          == e3["library_index"])


def test_lookup_missing_raises():
    section("test_lookup_missing_raises")
    raised = False
    try:
        lookup_chemsep(name="unobtanium-xyz-doesnotexist")
    except KeyError:
        raised = True
    check("missing compound raises KeyError", raised)
    raised = False
    try:
        lookup_chemsep()
    except ValueError:
        raised = True
    check("no-identifier call raises ValueError", raised)


# =====================================================================
# Property values vs published references
# =====================================================================

def test_critical_properties_methane():
    """ChemSep methane critical properties should match NIST."""
    section("test_critical_properties_methane")
    e = lookup_chemsep(name="Methane")
    cc = get_critical_constants(e)
    # NIST methane: Tc=190.564 K, Pc=4.5992e6 Pa, omega=0.0114, Vc=9.86e-5 m³/mol
    check(f"Tc = {cc['Tc']:.2f} K (NIST 190.56)",
          abs(cc["Tc"] - 190.56) < 0.05)
    check(f"Pc = {cc['Pc']/1e5:.2f} bar (NIST 45.99)",
          abs(cc["Pc"] - 4.599e6) < 5e3)
    check(f"omega = {cc['omega']} (NIST 0.011)",
          abs(cc["omega"] - 0.011) < 0.01)
    check(f"Vc = {cc['Vc']*1e6:.2f} mL/mol (NIST 98.6)",
          abs(cc["Vc"] - 9.86e-5) < 1e-6)


def test_critical_properties_water():
    section("test_critical_properties_water")
    e = lookup_chemsep(name="Water")
    cc = get_critical_constants(e)
    check(f"Tc = {cc['Tc']} K (NIST 647.14)",
          abs(cc["Tc"] - 647.14) < 0.5)
    check(f"Pc = {cc['Pc']/1e5:.2f} bar (NIST 220.64)",
          abs(cc["Pc"] - 220.64e5) < 0.5e5)
    check(f"omega = {cc['omega']} (NIST 0.345)",
          abs(cc["omega"] - 0.345) < 0.02)


def test_molar_mass_conversions():
    section("test_molar_mass_conversions")
    # Water: 18.0153 g/mol = 0.0180153 kg/mol
    e = lookup_chemsep(name="Water")
    mw = get_molar_mass(e)
    check(f"water MW = {mw} kg/mol (expect 0.01801)",
          abs(mw - 0.01801) < 1e-4)
    # Methane: 16.043 g/mol = 0.016043 kg/mol
    e = lookup_chemsep(name="Methane")
    mw = get_molar_mass(e)
    check(f"methane MW = {mw} kg/mol (expect 0.01604)",
          abs(mw - 0.01604) < 1e-4)


def test_formation_properties():
    """Heat of formation, Gibbs energy of formation, entropy in J/mol."""
    section("test_formation_properties")
    # Water: Hf = -241.83 kJ/mol (gas), or -285.83 (liq); ChemSep gives gas
    # Gf = -228.6 kJ/mol (gas)
    # S° = 188.84 J/(mol·K)
    e = lookup_chemsep(name="Water")
    fp = get_formation_properties(e)
    check(f"water Hf = {fp['Hf']/1000:.1f} kJ/mol (gas, expect -241.83)",
          abs(fp["Hf"] - (-241830)) < 5e3)
    check(f"water Gf = {fp['Gf']/1000:.1f} kJ/mol (gas, expect -228.57)",
          abs(fp["Gf"] - (-228570)) < 5e3)
    check(f"water S° = {fp['S']:.2f} J/mol/K (expect 188.84)",
          abs(fp["S"] - 188.84) < 1.0)


# =====================================================================
# DIPPR equations
# =====================================================================

def test_DIPPR_101_methane_psat():
    """DIPPR-101 vapor pressure of methane at NBP must give 1 atm."""
    section("test_DIPPR_101_methane_psat")
    e = lookup_chemsep(name="Methane")
    NBP = e["normal_boiling_point"]["value"]
    psat = evaluate_property(e, "vapor_pressure", NBP)
    check(f"methane Psat({NBP} K) = {psat:.0f} Pa (expect 101325)",
          abs(psat - 101325) / 101325 < 0.01)


def test_DIPPR_101_water_psat():
    """Water at 373.15 K should give 1 atm."""
    section("test_DIPPR_101_water_psat")
    e = lookup_chemsep(name="Water")
    psat = evaluate_property(e, "vapor_pressure", 373.15)
    check(f"water Psat(373.15) = {psat:.0f} Pa (expect 101325)",
          abs(psat - 101325) / 101325 < 0.005)
    psat298 = evaluate_property(e, "vapor_pressure", 298.15)
    # NIST water saturation at 298.15 K = 3169.9 Pa
    check(f"water Psat(298.15) = {psat298:.0f} Pa (NIST 3170)",
          abs(psat298 - 3170) / 3170 < 0.02)


def test_DIPPR_106_water_hvap():
    """DIPPR-106 heat of vaporization at NBP for water = 40.66 kJ/mol."""
    section("test_DIPPR_106_water_hvap")
    e = lookup_chemsep(name="Water")
    hvap = evaluate_property(e, "heat_of_vaporization", 373.15)
    # ChemSep gives J/kmol; expected 40.66e6
    check(f"water ΔHvap(373.15) = {hvap/1e6:.2f} MJ/kmol (NIST 40.66)",
          abs(hvap - 40.66e6) / 40.66e6 < 0.02)


def test_DIPPR_100_water_cp_ig():
    """Ideal-gas Cp of water at 298.15 K = 33.6 J/(mol·K) = 33600 J/(kmol·K)."""
    section("test_DIPPR_100_water_cp_ig")
    e = lookup_chemsep(name="Water")
    cp = evaluate_property(e, "ideal_gas_heat_capacity", 298.15)
    check(f"water Cp_ig(298.15) = {cp:.1f} J/kmol/K (NIST 33600)",
          abs(cp - 33600) < 200)


def test_DIPPR_105_water_density():
    """Liquid density of water at 298.15 K = 55.34 kmol/m³ ≈ 997 kg/m³."""
    section("test_DIPPR_105_water_density")
    e = lookup_chemsep(name="Water")
    rho_kmol = evaluate_property(e, "liquid_density", 298.15)
    rho_kg = rho_kmol * 18.0153   # kmol/m³ × kg/kmol = kg/m³
    check(f"water ρ_liq(298.15) = {rho_kg:.0f} kg/m³ (NIST 997)",
          abs(rho_kg - 997) / 997 < 0.01)


def test_DIPPR_polynomial_form():
    """DIPPR eqno 100 (polynomial in T) — sanity check the implementation."""
    section("test_DIPPR_polynomial_form")
    eq = {"eqno": 100,
          "coefficients": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}}
    # A + B*T + C*T² + D*T³ + E*T⁴
    T = 2.0
    expected = 1 + 2*2 + 3*4 + 4*8 + 5*16
    val = evaluate_dippr(eq, T)
    check(f"polynomial evaluates correctly: {val} == {expected}",
          val == expected)


def test_DIPPR_unknown_eqno():
    """An unsupported eqno must raise ValueError."""
    section("test_DIPPR_unknown_eqno")
    eq = {"eqno": 99999, "coefficients": {"A": 1.0}}
    raised = False
    try:
        evaluate_dippr(eq, 300.0)
    except ValueError:
        raised = True
    check("unknown eqno raises ValueError", raised)


# =====================================================================
# Group contributions
# =====================================================================

def test_unifac_groups_present():
    """Common compounds must have UNIFAC-VLE groups parsed correctly."""
    section("test_unifac_groups_present")
    # Water: UNIFAC group 17 (H2O), value=1
    e = lookup_chemsep(name="Water")
    g = e.get("unifac_vle", {})
    check(f"water has UNIFAC groups ({g})", "17" in g and g["17"] == 1)
    # n-Butane: 2 CH3 (group 1) + 2 CH2 (group 2)
    e = lookup_chemsep(name="n-Butane")
    g = e.get("unifac_vle", {})
    check(f"n-butane has CH3 + CH2 ({g})",
          g.get("1") == 2 and g.get("2") == 2)


# =====================================================================
# Cross-check vs stateprop's own constants where possible
# =====================================================================

def test_consistency_with_PR_methane():
    """ChemSep methane Tc/Pc/omega should match the PR EOS constants
    in stateprop's saft.METHANE within published precision."""
    section("test_consistency_with_PR_methane")
    e = lookup_chemsep(name="Methane")
    cc = get_critical_constants(e)
    from stateprop.saft import METHANE
    check(f"Tc match: ChemSep {cc['Tc']} vs SAFT {METHANE.T_c}",
          abs(cc["Tc"] - METHANE.T_c) < 0.5)
    check(f"Pc match: ChemSep {cc['Pc']} vs SAFT {METHANE.p_c}",
          abs(cc["Pc"] - METHANE.p_c) / METHANE.p_c < 0.005)


def test_data_completeness():
    """At least N% of common refinery-relevant compounds must have all
    properties needed for cubic EOS construction (Tc, Pc, omega).
    """
    section("test_data_completeness")
    db = load_chemsep_database()
    n_complete = 0
    for entry in db:
        cc = get_critical_constants(entry)
        if "Tc" in cc and "Pc" in cc and "omega" in cc:
            n_complete += 1
    pct = n_complete / len(db) * 100
    check(f"{n_complete}/{len(db)} compounds with full Tc/Pc/omega "
          f"({pct:.0f}%)", pct >= 99)


def main():
    print("=" * 60)
    print("stateprop ChemSep database tests")
    print("=" * 60)
    tests = [
        test_database_summary_counts,
        test_database_caching,
        test_lookup_by_name,
        test_lookup_by_cas,
        test_lookup_by_smiles,
        test_lookup_case_insensitive,
        test_lookup_missing_raises,
        test_critical_properties_methane,
        test_critical_properties_water,
        test_molar_mass_conversions,
        test_formation_properties,
        test_DIPPR_101_methane_psat,
        test_DIPPR_101_water_psat,
        test_DIPPR_106_water_hvap,
        test_DIPPR_100_water_cp_ig,
        test_DIPPR_105_water_density,
        test_DIPPR_polynomial_form,
        test_DIPPR_unknown_eqno,
        test_unifac_groups_present,
        test_consistency_with_PR_methane,
        test_data_completeness,
    ]
    for t in tests:
        t()
    print("\n" + "=" * 60)
    print(f"RESULT: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)
    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()
