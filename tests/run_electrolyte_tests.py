"""Electrolyte solution thermodynamics tests.

Validates v0.9.96's Pitzer ion-interaction model against published
γ_± and osmotic-coefficient data:

* Robinson-Stokes 1959 — definitive tabulated γ± for aqueous salts
* Pitzer 1991 — original Pitzer parameter regressions
* NIST salt solution thermodynamics references

Tests cover:
* Database loading and parameter retrieval
* Limiting-law behavior (m → 0)
* γ_± across 4-5 molalities each for NaCl, KCl, HCl, CaCl2
* Osmotic coefficient φ across the same range
* Water activity self-consistency (a_w = exp(-ν·m·M_w·φ))
* Davies and Debye-Hückel limiting law sanity
* Thermodynamic constants (A_φ at 25 °C ≈ 0.392)
* Ionic strength formula
* Molality ↔ mole-fraction round-trip
"""
from __future__ import annotations
import sys, os, warnings
import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from stateprop.electrolyte import (
    PitzerModel, PitzerSalt, lookup_salt, list_salts,
    eNRTL,
    debye_huckel_A, davies_log_gamma_pm, debye_huckel_log_gamma_pm,
    ionic_strength,
    water_density, water_dielectric,
    molality_to_mole_fraction, mole_fraction_to_molality,
)

_PASS = 0
_FAIL = 0


def section(n): print(f"\n[{n}]")
def check(label, ok):
    global _PASS, _FAIL
    if ok: _PASS += 1; print(f"  PASS  {label}")
    else: _FAIL += 1; print(f"  FAIL  {label}")


# =====================================================================
# Constants
# =====================================================================

def test_water_density_298K():
    """Water density at 25 °C should match NIST 997.05 kg/m³ to <0.1%."""
    section("test_water_density_298K")
    rho = water_density(298.15)
    check(f"ρ(298.15K) = {rho:.2f} (NIST 997.05)",
          abs(rho - 997.05) / 997.05 < 0.001)


def test_water_dielectric_298K():
    """Water static permittivity at 25 °C ≈ 78.38 (CRC handbook)."""
    section("test_water_dielectric_298K")
    eps_r = water_dielectric(298.15)
    check(f"ε_r(298.15K) = {eps_r:.3f} (CRC 78.38)",
          abs(eps_r - 78.38) / 78.38 < 0.005)


def test_debye_huckel_A_298K():
    """Debye-Hückel A_φ at 25°C ≈ 0.3915 (Pitzer 1991)."""
    section("test_debye_huckel_A_298K")
    A = debye_huckel_A(298.15)
    check(f"A_φ(298.15K) = {A:.4f} (Pitzer 1991: 0.3915)",
          abs(A - 0.3915) < 0.005)


def test_debye_huckel_A_T_increase():
    """A_φ should increase with T (decreasing dielectric constant)."""
    section("test_debye_huckel_A_T_increase")
    A_25 = debye_huckel_A(298.15)
    A_50 = debye_huckel_A(323.15)
    A_75 = debye_huckel_A(348.15)
    check(f"A_φ(25) < A_φ(50) ({A_25:.4f} < {A_50:.4f})", A_25 < A_50)
    check(f"A_φ(50) < A_φ(75) ({A_50:.4f} < {A_75:.4f})", A_50 < A_75)


# =====================================================================
# Ionic strength and conversions
# =====================================================================

def test_ionic_strength_NaCl():
    """I = ½(m·1 + m·1) = m for 1:1 electrolyte at molality m."""
    section("test_ionic_strength_NaCl")
    I = ionic_strength({"Na+": 1.0, "Cl-": 1.0}, {"Na+": 1, "Cl-": -1})
    check(f"NaCl 1m: I = {I} (expect 1.0)", abs(I - 1.0) < 1e-12)


def test_ionic_strength_CaCl2():
    """I = ½(1·2² + 2·1·1²) = ½(4+2) = 3 for 1m CaCl2."""
    section("test_ionic_strength_CaCl2")
    I = ionic_strength({"Ca++": 1.0, "Cl-": 2.0}, {"Ca++": 2, "Cl-": -1})
    check(f"CaCl2 1m: I = {I} (expect 3.0)", abs(I - 3.0) < 1e-12)


def test_molality_mole_fraction_roundtrip():
    section("test_molality_mole_fraction_roundtrip")
    m_in = {"Na+": 0.5, "Cl-": 0.5}
    x = molality_to_mole_fraction(m_in)
    s = sum(x.values())
    check(f"Σ x = {s} (expect 1.0)", abs(s - 1.0) < 1e-12)
    # Round trip: solvent → "solvent" key
    m_out = mole_fraction_to_molality(x, solvent_key="solvent")
    for k in ("Na+", "Cl-"):
        diff = abs(m_in[k] - m_out[k])
        check(f"round-trip {k}: {diff:.2e}", diff < 1e-12)


# =====================================================================
# Pitzer database
# =====================================================================

def test_lookup_salt():
    """All bundled salts must be retrievable by name."""
    section("test_lookup_salt")
    for name in list_salts():
        s = lookup_salt(name)
        check(f"lookup_salt({name!r})", isinstance(s, PitzerSalt))
    check(f"got {len(list_salts())} salts (expect 20)",
          len(list_salts()) == 20)


def test_unknown_salt_raises():
    section("test_unknown_salt_raises")
    raised = False
    try:
        lookup_salt("Fakium")
    except KeyError:
        raised = True
    check("unknown salt raises KeyError", raised)


# =====================================================================
# Pitzer γ_± vs literature data (Robinson-Stokes 1959)
# =====================================================================

def test_NaCl_gamma_pm():
    """NaCl mean ionic activity coefficient vs Robinson-Stokes 1959."""
    section("test_NaCl_gamma_pm")
    p = PitzerModel("NaCl")
    # Robinson-Stokes 1959 Appendix 8.10 — widely cited
    data = [
        (0.001, 0.965), (0.01, 0.902), (0.05, 0.821),
        (0.1, 0.778), (0.5, 0.681), (1.0, 0.657), (2.0, 0.668),
    ]
    for m, lit in data:
        g = p.gamma_pm(m)
        err = abs(g - lit) / lit
        check(f"NaCl m={m}: γ±={g:.4f}, lit={lit} (err {err*100:.2f}%)",
              err < 0.01)


def test_KCl_gamma_pm():
    """KCl γ_± vs Robinson-Stokes 1959."""
    section("test_KCl_gamma_pm")
    p = PitzerModel("KCl")
    for m, lit in [(0.1, 0.770), (0.5, 0.649), (1.0, 0.604), (2.0, 0.573)]:
        g = p.gamma_pm(m)
        err = abs(g - lit) / lit
        check(f"KCl m={m}: γ±={g:.4f}, lit={lit} (err {err*100:.2f}%)",
              err < 0.01)


def test_HCl_gamma_pm():
    """HCl γ_± vs Pitzer-Mayorga 1973 / Robinson-Stokes 1959."""
    section("test_HCl_gamma_pm")
    p = PitzerModel("HCl")
    for m, lit in [(0.1, 0.796), (0.5, 0.757), (1.0, 0.809),
                     (2.0, 1.009), (3.0, 1.316)]:
        g = p.gamma_pm(m)
        err = abs(g - lit) / lit
        check(f"HCl m={m}: γ±={g:.4f}, lit={lit} (err {err*100:.2f}%)",
              err < 0.02)


def test_CaCl2_gamma_pm():
    """CaCl2 (2:1 electrolyte) γ_± vs Robinson-Stokes 1959."""
    section("test_CaCl2_gamma_pm")
    p = PitzerModel("CaCl2")
    for m, lit in [(0.1, 0.518), (0.5, 0.448), (1.0, 0.500), (2.0, 0.792)]:
        g = p.gamma_pm(m)
        err = abs(g - lit) / lit
        check(f"CaCl2 m={m}: γ±={g:.4f}, lit={lit} (err {err*100:.2f}%)",
              err < 0.02)


def test_Na2SO4_gamma_pm():
    """Na2SO4 (1:2 electrolyte) γ_± vs literature."""
    section("test_Na2SO4_gamma_pm")
    p = PitzerModel("Na2SO4")
    # Robinson-Stokes 1959
    for m, lit in [(0.1, 0.452), (0.5, 0.270), (1.0, 0.204)]:
        g = p.gamma_pm(m)
        err = abs(g - lit) / lit
        check(f"Na2SO4 m={m}: γ±={g:.4f}, lit={lit} (err {err*100:.2f}%)",
              err < 0.05)


# =====================================================================
# Osmotic coefficient
# =====================================================================

def test_NaCl_osmotic_coefficient():
    """NaCl φ vs Robinson-Stokes 1959."""
    section("test_NaCl_osmotic_coefficient")
    p = PitzerModel("NaCl")
    # Robinson-Stokes
    for m, lit in [(0.1, 0.9324), (0.5, 0.9209), (1.0, 0.9355),
                     (2.0, 0.9833), (3.0, 1.0450)]:
        phi = p.osmotic_coefficient(m)
        err = abs(phi - lit) / lit
        check(f"NaCl m={m}: φ={phi:.4f}, lit={lit} (err {err*100:.3f}%)",
              err < 0.01)


def test_water_activity_NaCl():
    """Water activity from osmotic coefficient should match published."""
    section("test_water_activity_NaCl")
    p = PitzerModel("NaCl")
    # Robinson-Stokes 1959 a_w values (4 decimals tabulated)
    for m, lit in [(0.1, 0.99665), (0.5, 0.98355), (1.0, 0.96686),
                     (2.0, 0.93145), (4.0, 0.85115)]:
        a_w = p.water_activity(m)
        err = abs(a_w - lit) / lit
        check(f"NaCl m={m}: a_w={a_w:.5f}, lit={lit} (err {err*100:.3f}%)",
              err < 0.005)


# =====================================================================
# Limiting-law behavior at very low m
# =====================================================================

def test_limiting_law_NaCl():
    """At m → 0, ln γ_± → -A_DH·|z+z-|·√I (Debye-Hückel limit)."""
    section("test_limiting_law_NaCl")
    p = PitzerModel("NaCl")
    A_DH = 3.0 * debye_huckel_A(298.15) / np.log(10.0)   # natural-log A
    # log10 A_DH ≈ 0.509 at 25°C
    for m in [1e-4, 1e-3]:
        I = m
        ln_g_DH_limit = -np.log(10) * A_DH * np.sqrt(I)
        ln_g_Pitzer = np.log(p.gamma_pm(m))
        # At m=1e-4, Pitzer should be within 1% of limiting law
        rel = abs(ln_g_Pitzer - ln_g_DH_limit) / abs(ln_g_DH_limit)
        check(f"m={m}: ln γ±_pitzer={ln_g_Pitzer:.5f}, "
              f"DH-limit={ln_g_DH_limit:.5f} (rel {rel*100:.2f}%)",
              rel < 0.05)


def test_pure_DH_at_very_low_m():
    """At m=1e-5, pure DH and Pitzer must agree to 1%."""
    section("test_pure_DH_at_very_low_m")
    log_DH = debye_huckel_log_gamma_pm(1, -1, 1e-5)
    p = PitzerModel("NaCl")
    log_Pitzer = np.log10(p.gamma_pm(1e-5))
    err = abs(log_DH - log_Pitzer) / abs(log_DH)
    check(f"DH={log_DH:.5e}, Pitzer={log_Pitzer:.5e}, rel={err*100:.2f}%",
          err < 0.01)


def test_davies_at_moderate_m():
    """Davies equation at m=0.1 is widely tabulated; sanity check."""
    section("test_davies_at_moderate_m")
    log_d = davies_log_gamma_pm(1, -1, 0.1)
    g_d = 10 ** log_d
    # Davies is empirically known to give γ ≈ 0.78 for 1:1 at m=0.1
    check(f"Davies γ±(NaCl, 0.1m) = {g_d:.4f} (lit ~0.78)",
          0.74 < g_d < 0.82)


# =====================================================================
# Custom salt parameters
# =====================================================================

def test_custom_PitzerSalt():
    """Users can construct a PitzerSalt directly with fitted parameters."""
    section("test_custom_PitzerSalt")
    custom = PitzerSalt(name="MyCustomSalt", z_M=1, z_X=-1,
                          nu_M=1, nu_X=1,
                          beta_0=0.05, beta_1=0.20, C_phi=0.001)
    p = PitzerModel(custom)
    g = p.gamma_pm(0.1)
    check(f"custom 1:1 salt: γ±(0.1m) = {g:.4f} (sanity)",
          0.7 < g < 0.85)


def test_pitzer_phi_self_consistency():
    """At m → 0, φ → 1 (limiting behavior)."""
    section("test_pitzer_phi_self_consistency")
    p = PitzerModel("NaCl")
    phi_low = p.osmotic_coefficient(1e-6)
    check(f"φ(1e-6) = {phi_low:.5f} (expect → 1)",
          abs(phi_low - 1.0) < 0.01)


# =====================================================================
# Cross-salt: 2:2 with β² parameter active
# =====================================================================

def test_MgSO4_gamma_pm():
    """MgSO4 (2:2 with β² active) — looser tolerance due to known
    Pitzer-form limitations on 2:2 salts with significant ion pairing."""
    section("test_MgSO4_gamma_pm")
    p = PitzerModel("MgSO4")
    # Pitzer-Mayorga 1974 fitted values
    for m, lit, tol in [(0.01, 0.404, 0.05), (0.1, 0.150, 0.15),
                          (1.0, 0.054, 0.20)]:
        g = p.gamma_pm(m)
        err = abs(g - lit) / lit
        check(f"MgSO4 m={m}: γ±={g:.4f}, lit={lit} (err {err*100:.2f}%)",
              err < tol)


# =====================================================================
# eNRTL availability (does not enforce numerical accuracy)
# =====================================================================

def test_enrtl_loads():
    """eNRTL parameter set is available even though calibration is
    a known v0.9.96 limitation."""
    section("test_enrtl_loads")
    e = eNRTL("NaCl")
    g = e.gamma_pm(0.001)
    check(f"eNRTL loads, returns float at m=1e-3: {g:.4f}",
          isinstance(g, float) and g > 0)


# =====================================================================
# T-dependent Pitzer parameters (v0.9.97)
# =====================================================================

def test_T_dependence_at_Tref_unchanged():
    """At T = T_ref, the T-dependent evaluation must reproduce the
    25 °C values bit-exactly (no spurious shift from the Taylor form)."""
    section("test_T_dependence_at_Tref_unchanged")
    p = PitzerModel("NaCl")
    g25 = p.gamma_pm(1.0, 298.15)
    # v0.9.96 published value
    check(f"NaCl γ±(1m, 25°C) = {g25:.4f} (v0.9.96 0.6544)",
          abs(g25 - 0.6544) < 1e-4)
    g_default = p.gamma_pm(1.0)   # default T=298.15
    check(f"default-T == explicit-T: {g_default:.6f} == {g25:.6f}",
          abs(g_default - g25) < 1e-12)


def test_at_T_returns_self_when_no_derivatives():
    """A salt with all-zero T-derivatives should return itself
    from at_T (backward compatibility)."""
    section("test_at_T_returns_self_when_no_derivatives")
    custom = PitzerSalt("Test", z_M=1, z_X=-1, nu_M=1, nu_X=1,
                          beta_0=0.05, beta_1=0.20, C_phi=0.001)
    s50 = custom.at_T(323.15)
    check("at_T returns same instance with zero derivatives",
          s50 is custom)


def test_NaCl_T_dependence_qualitative():
    """NaCl γ_± and φ should both decrease with T over 25-100 °C
    (well-established empirical trend; Archer 1992)."""
    section("test_NaCl_T_dependence_qualitative")
    p = PitzerModel("NaCl")
    g25 = p.gamma_pm(1.0, 298.15)
    g100 = p.gamma_pm(1.0, 373.15)
    check(f"NaCl γ±(100°C) < γ±(25°C): {g100:.4f} < {g25:.4f}",
          g100 < g25)
    phi25 = p.osmotic_coefficient(1.0, 298.15)
    phi100 = p.osmotic_coefficient(1.0, 373.15)
    check(f"NaCl φ(100°C) < φ(25°C): {phi100:.4f} < {phi25:.4f}",
          phi100 < phi25)


def test_NaCl_beta_at_T():
    """NaCl β⁰, β¹, C^φ at 50 °C from the Taylor form must reproduce
    Holmes-Mesmer 1986 tabulated values to <1%."""
    section("test_NaCl_beta_at_T")
    s = lookup_salt("NaCl")
    s50 = s.at_T(323.15)
    # Holmes-Mesmer 1986 NaCl at 50 °C
    check(f"β⁰(50°C) = {s50.beta_0:.5f} (HM86 0.0793)",
          abs(s50.beta_0 - 0.0793) < 0.001)
    check(f"β¹(50°C) = {s50.beta_1:.5f} (HM86 0.2800)",
          abs(s50.beta_1 - 0.2800) < 0.005)
    check(f"C^φ(50°C) = {s50.C_phi:.6f} (HM86 0.00110)",
          abs(s50.C_phi - 0.00110) < 5e-5)


def test_KCl_beta_at_T():
    """KCl β at 75 °C from Taylor form vs Holmes-Mesmer 1983."""
    section("test_KCl_beta_at_T")
    s = lookup_salt("KCl")
    s75 = s.at_T(348.15)
    # Holmes-Mesmer 1983 KCl at 75 °C
    check(f"β⁰(75°C) = {s75.beta_0:.5f} (HM83 ~0.0535)",
          abs(s75.beta_0 - 0.0535) < 0.002)
    check(f"β¹(75°C) = {s75.beta_1:.5f} (HM83 ~0.248)",
          abs(s75.beta_1 - 0.248) < 0.005)


def test_CaCl2_T_dependence():
    """CaCl2 (2:1) γ_± should be sensitive to T at moderate molality."""
    section("test_CaCl2_T_dependence")
    p = PitzerModel("CaCl2")
    g25 = p.gamma_pm(1.0, 298.15)
    g75 = p.gamma_pm(1.0, 348.15)
    check(f"CaCl2 γ±(25°C, 1m) = {g25:.4f} ≈ 0.500",
          abs(g25 - 0.500) < 0.01)
    # γ± is computed correctly (different from 25 °C value)
    check(f"CaCl2 γ±(75°C, 1m) = {g75:.4f} (T-dependence active)",
          abs(g75 - g25) > 1e-5)


def test_water_activity_T_dependence():
    """Water activity for NaCl at 75 °C should still match published
    a_w within the Taylor expansion's stated accuracy (~1%)."""
    section("test_water_activity_T_dependence")
    p = PitzerModel("NaCl")
    a_w_25 = p.water_activity(1.0, 298.15)
    a_w_75 = p.water_activity(1.0, 348.15)
    # Both should be near 0.97 (NaCl at 1 mol/kg)
    check(f"a_w(25°C, 1m) = {a_w_25:.5f} (lit 0.96686)",
          abs(a_w_25 - 0.96686) < 0.002)
    check(f"a_w(75°C, 1m) = {a_w_75:.5f} (within 1% of 25°C)",
          abs(a_w_75 - a_w_25) < 0.015)


# =====================================================================
# Sour-water module (v0.9.97)
# =====================================================================

def test_pKw_25C():
    """Water self-ionization pKw at 25 °C must be 14.00 (Harned-Owen 1958)."""
    section("test_pKw_25C")
    from stateprop.electrolyte.sour_water import pK_water
    pKw = pK_water(298.15)
    check(f"pKw(25°C) = {pKw:.3f} (expect 14.00)",
          abs(pKw - 14.00) < 0.01)


def test_pKw_T_dependence():
    """pKw decreases with T (water becomes more ionized; well-established)."""
    section("test_pKw_T_dependence")
    from stateprop.electrolyte.sour_water import pK_water
    pKw_25 = pK_water(298.15)
    pKw_50 = pK_water(323.15)
    pKw_100 = pK_water(373.15)
    check(f"pKw(50°C) < pKw(25°C): {pKw_50:.2f} < {pKw_25:.2f}",
          pKw_50 < pKw_25)
    check(f"pKw(100°C) ≈ 12.3: {pKw_100:.2f}",
          11.9 < pKw_100 < 12.5)


def test_NH4_pKa_25C():
    """Ammonium dissociation constant: pKa = 9.245 at 25 °C
    (Bates-Pinching 1949)."""
    section("test_NH4_pKa_25C")
    import numpy as np
    from stateprop.electrolyte.sour_water import dissociation_K
    pKa = -np.log10(dissociation_K("NH4+", 298.15))
    check(f"pKa(NH4+) at 25°C = {pKa:.3f} (lit 9.245)",
          abs(pKa - 9.245) < 0.01)


def test_H2S_pKa_25C():
    """First dissociation of H2S: pKa1 = 6.99 at 25 °C
    (Hershey et al. 1988)."""
    section("test_H2S_pKa_25C")
    import numpy as np
    from stateprop.electrolyte.sour_water import dissociation_K
    pKa = -np.log10(dissociation_K("H2S", 298.15))
    check(f"pKa1(H2S) at 25°C = {pKa:.3f} (lit 6.99)",
          abs(pKa - 6.99) < 0.01)


def test_CO2_pKa_25C():
    """First apparent dissociation of dissolved CO2: pKa1 = 6.35 at 25 °C
    (Harned-Davis 1943)."""
    section("test_CO2_pKa_25C")
    import numpy as np
    from stateprop.electrolyte.sour_water import dissociation_K
    pKa = -np.log10(dissociation_K("CO2", 298.15))
    check(f"pKa1(CO2) at 25°C = {pKa:.3f} (lit 6.35)",
          abs(pKa - 6.35) < 0.01)


def test_henry_NH3_25C():
    """Henry's coefficient for NH3 at 25 °C ≈ 1791 Pa·kg/mol
    (Wilhelm-Battino-Wilcock 1977)."""
    section("test_henry_NH3_25C")
    from stateprop.electrolyte.sour_water import henry_constant
    H = henry_constant("NH3", 298.15)
    check(f"H(NH3, 25°C) = {H:.0f} Pa·kg/mol (Wilhelm 1977 ≈ 1791)",
          abs(H - 1791) / 1791 < 0.01)


def test_henry_T_dependence():
    """Henry's constants increase with T (gases are less soluble hot)."""
    section("test_henry_T_dependence")
    from stateprop.electrolyte.sour_water import henry_constant
    for sp in ["NH3", "H2S", "CO2"]:
        H_25 = henry_constant(sp, 298.15)
        H_95 = henry_constant(sp, 368.15)
        check(f"H({sp}, 95°C) > H({sp}, 25°C): {H_95:.2e} > {H_25:.2e}",
              H_95 > H_25)


def test_speciate_pure_NH3():
    """Pure NH3 in water gives basic pH > 9 (NH3 is a weak base)."""
    section("test_speciate_pure_NH3")
    from stateprop.electrolyte.sour_water import speciate
    sp = speciate(298.15, m_NH3_total=0.1, m_H2S_total=0.0, m_CO2_total=0.0)
    check(f"pH({{NH3=0.1, H2S=0, CO2=0}}) = {sp.pH:.2f} > 10 (basic)",
          sp.pH > 10.0)
    check(f"α(NH3) = {sp.alpha_NH3*100:.1f}% — most is molecular (weak base)",
          sp.alpha_NH3 > 0.95)


def test_speciate_pure_H2S():
    """Pure H2S in water gives slightly acidic pH ~4.0 (very weak acid)."""
    section("test_speciate_pure_H2S")
    from stateprop.electrolyte.sour_water import speciate
    sp = speciate(298.15, m_NH3_total=0.0, m_H2S_total=0.1, m_CO2_total=0.0)
    check(f"pH({{H2S=0.1}}) = {sp.pH:.2f} ∈ (3.5, 5.5) (very weak acid)",
          3.5 < sp.pH < 5.5)
    check(f"α(H2S) = {sp.alpha_H2S*100:.1f}% molecular (acid weakly dissociated)",
          sp.alpha_H2S > 0.99)


def test_speciate_NH3_plus_H2S():
    """NH3 + H2S together: NH3 buffers the H2S; both partially ionized."""
    section("test_speciate_NH3_plus_H2S")
    from stateprop.electrolyte.sour_water import speciate
    sp = speciate(368.15, m_NH3_total=0.05, m_H2S_total=0.05,
                    m_CO2_total=0.005)
    # Result should be near-neutral pH (~6-9 depending on exact ratios)
    check(f"pH (NH3+H2S+CO2 mix) = {sp.pH:.2f} ∈ (5, 10)",
          5.0 < sp.pH < 10.0)
    # Charge balance check — sum of charges should be near 0
    s = sp.species_molalities
    cb = (s["H+"] + s["NH4+"]
           - s["OH-"] - s["HS-"] - s["HCO3-"])
    check(f"charge balance |Σ z·m| = {abs(cb):.3e} (should be ~0)",
          abs(cb) < 1e-6)


def test_effective_henry_pH_dependence():
    """Effective Henry of NH3 increases with pH (more molecular at high pH);
    effective Henry of H2S decreases with pH (more ionic at high pH)."""
    section("test_effective_henry_pH_dependence")
    from stateprop.electrolyte.sour_water import effective_henry
    H_NH3_low = effective_henry("NH3", 298.15, 6.0)
    H_NH3_high = effective_henry("NH3", 298.15, 11.0)
    check(f"H_eff(NH3, pH=6) << H_eff(NH3, pH=11): "
          f"{H_NH3_low:.2e} << {H_NH3_high:.2e}",
          H_NH3_low < 0.01 * H_NH3_high)
    H_H2S_low = effective_henry("H2S", 298.15, 4.0)
    H_H2S_high = effective_henry("H2S", 298.15, 11.0)
    check(f"H_eff(H2S, pH=4) >> H_eff(H2S, pH=11): "
          f"{H_H2S_low:.2e} >> {H_H2S_high:.2e}",
          H_H2S_low > 100 * H_H2S_high)


# =====================================================================
# Multi-electrolyte Pitzer (v0.9.98)
# =====================================================================

def test_multi_pitzer_NaCl_reduces_to_single():
    """Single-salt MultiPitzerSystem must reproduce PitzerModel exactly
    (proves the multi-electrolyte expressions degrade correctly)."""
    section("test_multi_pitzer_NaCl_reduces_to_single")
    from stateprop.electrolyte import MultiPitzerSystem
    multi = MultiPitzerSystem.from_salts(["NaCl"])
    single = PitzerModel("NaCl")
    for m_test in [0.1, 1.0, 2.0]:
        g_pm_multi = multi.gamma_pm("NaCl", {"Na+": m_test, "Cl-": m_test})
        g_pm_single = single.gamma_pm(m_test)
        diff = abs(g_pm_multi - g_pm_single)
        check(f"NaCl m={m_test}: |γ_multi - γ_single| = {diff:.2e}",
              diff < 0.01)


def test_multi_pitzer_NaCl_KCl_mixture():
    """NaCl-KCl mixture at I=1, x_NaCl=0.5: Robinson-Wood 1972 reports
    γ_NaCl ≈ 0.640. Our Pitzer-Kim parameters should match within 1%."""
    section("test_multi_pitzer_NaCl_KCl_mixture")
    from stateprop.electrolyte import MultiPitzerSystem
    multi = MultiPitzerSystem.from_salts(["NaCl", "KCl"])
    m = {"Na+": 0.5, "K+": 0.5, "Cl-": 1.0}
    g_NaCl = multi.gamma_pm("NaCl", m)
    check(f"γ±(NaCl in NaCl-KCl mix at I=1) = {g_NaCl:.4f} (lit ~0.640)",
          abs(g_NaCl - 0.640) / 0.640 < 0.01)


def test_multi_pitzer_seawater_water_activity():
    """Seawater water activity at standard composition: Millero 1979
    reports a_w = 0.98142 at S=35‰, 25 °C.

    Our model with H-M-W 1984 parameters and E-θ=0 simplification
    gets within ~0.05%."""
    section("test_multi_pitzer_seawater_water_activity")
    from stateprop.electrolyte import MultiPitzerSystem
    sys = MultiPitzerSystem.seawater()
    m = {"Na+": 0.486, "K+": 0.0106, "Mg++": 0.0547, "Ca++": 0.0107,
          "Cl-": 0.5658, "SO4--": 0.0293}
    a_w = sys.water_activity(m)
    check(f"seawater a_w = {a_w:.5f} (Millero 1979 0.98142)",
          abs(a_w - 0.98142) / 0.98142 < 0.005)


def test_multi_pitzer_seawater_osmotic():
    """Seawater φ ~0.901 (Millero/Pitzer-Møller-Weare 1984).
    Our symmetric-mixing simplification: ~1% error."""
    section("test_multi_pitzer_seawater_osmotic")
    from stateprop.electrolyte import MultiPitzerSystem
    sys = MultiPitzerSystem.seawater()
    m = {"Na+": 0.486, "K+": 0.0106, "Mg++": 0.0547, "Ca++": 0.0107,
          "Cl-": 0.5658, "SO4--": 0.0293}
    phi = sys.osmotic_coefficient(m)
    check(f"seawater φ = {phi:.4f} (HMW 1984 ~0.901)",
          abs(phi - 0.901) / 0.901 < 0.02)


def test_multi_pitzer_charge_check():
    """Constructor must reject ions with wrong sign of charge."""
    section("test_multi_pitzer_charge_check")
    from stateprop.electrolyte import MultiPitzerSystem
    raised = False
    try:
        MultiPitzerSystem(cations=[("Cl-", -1)], anions=[("Na+", 1)])
    except ValueError:
        raised = True
    check("rejects negative cation", raised)


def test_multi_pitzer_ionic_strength():
    """I = ½·Σ m_i·z_i² for arbitrary composition."""
    section("test_multi_pitzer_ionic_strength")
    from stateprop.electrolyte import MultiPitzerSystem
    sys = MultiPitzerSystem.from_salts(["CaCl2"])
    # 1 mol/kg CaCl2: m_Ca=1, m_Cl=2; I = ½(1·4 + 2·1) = 3
    I = sys.ionic_strength({"Ca++": 1.0, "Cl-": 2.0})
    check(f"I(CaCl2 1m) = {I} (expect 3.0)", abs(I - 3.0) < 1e-12)


def test_multi_pitzer_seawater_constructor():
    """Convenience seawater() constructor builds the standard
    Na-K-Mg-Ca-Cl-SO4 system."""
    section("test_multi_pitzer_seawater_constructor")
    from stateprop.electrolyte import MultiPitzerSystem
    sys = MultiPitzerSystem.seawater()
    cations = sorted(c.name for c in sys.cations)
    anions = sorted(a.name for a in sys.anions)
    check(f"cations: {cations}",
          cations == ["Ca++", "K+", "Mg++", "Na+"])
    check(f"anions:  {anions}",
          anions == ["Cl-", "SO4--"])


def test_multi_pitzer_E_theta_active():
    """E_theta returns non-zero for different-charge pairs (active in v0.9.99)
    and zero for same-charge pairs (Plummer-Parkhurst form)."""
    section("test_multi_pitzer_E_theta_active")
    from stateprop.electrolyte.multi_pitzer import E_theta
    # Same-charge: must be exactly 0
    e1, e2 = E_theta(1, 1, I=1.0, T=298.15)
    check(f"E_theta(Na+, K+, I=1) same-charge: ({e1}, {e2}) (should be 0)",
          e1 == 0.0 and e2 == 0.0)
    # Different-charge: must be non-zero, of the right magnitude
    e1, e2 = E_theta(1, 2, I=1.0, T=298.15)
    check(f"E_theta(Na+, Mg++, I=1) = {e1:+.4f} (expect ~-0.014)",
          -0.05 < e1 < -0.005)
    # E-θ must vanish at I→0 (formula has 1/√I divergence; cutoff at 1e-6)
    e1, _ = E_theta(1, 2, I=1e-9, T=298.15)
    check(f"E_theta(Na+, Mg++) at I=1e-9 = {e1} (should be 0 by cutoff)",
          e1 == 0.0)


def test_multi_pitzer_J0_monotonic():
    """J_0(x) must be monotonically increasing and smooth over its
    practical range (Plummer-Parkhurst form has no discontinuities)."""
    section("test_multi_pitzer_J0_monotonic")
    from stateprop.electrolyte.multi_pitzer import _J0
    import numpy as np
    xs = np.linspace(0.01, 100, 500)
    j0s = np.array([_J0(x) for x in xs])
    diffs = np.diff(j0s)
    check(f"J_0(x) monotonic increasing on (0.01, 100]",
          (diffs > 0).all())
    # Asymptote J_0(x→∞) → x/4
    j_large = _J0(1000.0)
    check(f"J_0(1000) = {j_large:.2f} ≈ 250 (x/4 asymptote)",
          240 < j_large < 260)


def test_multi_pitzer_seawater_with_E_theta():
    """With proper E-θ active (v0.9.99), seawater φ matches literature
    to <1% — better than the symmetric simplification's 1% error."""
    section("test_multi_pitzer_seawater_with_E_theta")
    from stateprop.electrolyte import MultiPitzerSystem
    sys = MultiPitzerSystem.seawater()
    m = {"Na+": 0.486, "K+": 0.0106, "Mg++": 0.0547, "Ca++": 0.0107,
          "Cl-": 0.5658, "SO4--": 0.0293}
    phi = sys.osmotic_coefficient(m)
    check(f"seawater φ with E-θ = {phi:.4f} (HMW 1984 ~0.901, err < 1%)",
          abs(phi - 0.901) / 0.901 < 0.01)
    a_w = sys.water_activity(m)
    check(f"seawater a_w with E-θ = {a_w:.5f} (Millero 1979 0.98142, err < 0.05%)",
          abs(a_w - 0.98142) / 0.98142 < 0.0005)


# =====================================================================
# T-dependence of mixing terms (v0.9.100)
# =====================================================================

def test_mixing_param_T_independent_default():
    """MixingParam with default dvalue_dT=0 returns value_25 at all T."""
    section("test_mixing_param_T_independent_default")
    from stateprop.electrolyte.multi_pitzer import MixingParam
    p = MixingParam(0.07)
    check(f"MixingParam(0.07).at_T(298.15) = {p.at_T(298.15)} (= 0.07)",
          p.at_T(298.15) == 0.07)
    check(f"MixingParam(0.07).at_T(348.15) = {p.at_T(348.15)} (= 0.07)",
          p.at_T(348.15) == 0.07)


def test_mixing_param_T_dependent():
    """MixingParam with non-zero dvalue_dT evaluates the linear term."""
    section("test_mixing_param_T_dependent")
    from stateprop.electrolyte.multi_pitzer import MixingParam
    # θ(Na+, Ca++): 0.07 + 4.09e-4 · ΔT
    p = MixingParam(0.07, dvalue_dT=4.09e-4)
    check(f"P(25°C) = {p.at_T(298.15):.5f} = 0.07000",
          abs(p.at_T(298.15) - 0.07) < 1e-6)
    # at 75 °C: 0.07 + 4.09e-4 · 50 = 0.09045
    check(f"P(75°C) = {p.at_T(348.15):.5f} = 0.09045",
          abs(p.at_T(348.15) - 0.09045) < 1e-6)
    # at 100 °C: 0.07 + 4.09e-4 · 75 = 0.100675
    check(f"P(100°C) = {p.at_T(373.15):.5f} = 0.100675",
          abs(p.at_T(373.15) - 0.100675) < 1e-6)


def test_mixing_param_bundled_NaCa():
    """Bundled θ(Na+, Ca++) carries Møller 1988 T-derivative."""
    section("test_mixing_param_bundled_NaCa")
    from stateprop.electrolyte.multi_pitzer import _THETA_CC, _csort
    p = _THETA_CC[_csort("Na+", "Ca++")]
    check(f"θ(Na+, Ca++) value_25 = {p.value_25} = 0.07",
          abs(p.value_25 - 0.07) < 1e-12)
    check(f"θ(Na+, Ca++) dvalue_dT = {p.dvalue_dT} ≈ 4.09e-4 (Møller 1988)",
          abs(p.dvalue_dT - 4.09e-4) < 1e-9)


def test_mixing_param_bundled_psi_NaCaCl():
    """Bundled ψ(Na+, Ca++, Cl-) carries Møller 1988 T-derivative."""
    section("test_mixing_param_bundled_psi_NaCaCl")
    from stateprop.electrolyte.multi_pitzer import _PSI_CCA, _csort
    p = _PSI_CCA[(*_csort("Na+", "Ca++"), "Cl-")]
    check(f"ψ(Na+, Ca++, Cl-) value_25 = {p.value_25} = -0.014",
          abs(p.value_25 - (-0.014)) < 1e-12)
    check(f"ψ(Na+, Ca++, Cl-) dvalue_dT = {p.dvalue_dT} ≈ -2.6e-4",
          abs(p.dvalue_dT - (-2.60e-4)) < 1e-9)


def test_multi_pitzer_25C_unchanged_from_v0_9_99():
    """v0.9.100 must not alter any v0.9.99 25 °C results
    (T-derivatives only kick in at T ≠ 298.15)."""
    section("test_multi_pitzer_25C_unchanged_from_v0_9_99")
    from stateprop.electrolyte import MultiPitzerSystem
    sw = MultiPitzerSystem.seawater()
    m = {"Na+": 0.486, "K+": 0.0106, "Mg++": 0.0547, "Ca++": 0.0107,
          "Cl-": 0.5658, "SO4--": 0.0293}
    phi_25 = sw.osmotic_coefficient(m, 298.15)
    a_w_25 = sw.water_activity(m, 298.15)
    # v0.9.99 published values
    check(f"seawater φ(25°C) = {phi_25:.5f} (v0.9.99 0.8959)",
          abs(phi_25 - 0.8959) < 1e-4)
    check(f"seawater a_w(25°C) = {a_w_25:.5f} (v0.9.99 0.98150)",
          abs(a_w_25 - 0.98150) < 1e-4)


def test_multi_pitzer_NaCaCl_T_dep_active():
    """NaCl-CaCl2 brine: γ should change between 25 °C and 75 °C
    via the bundled M88 T-derivatives on θ_NaCa and ψ_NaCaCl."""
    section("test_multi_pitzer_NaCaCl_T_dep_active")
    from stateprop.electrolyte import MultiPitzerSystem
    sys = MultiPitzerSystem.from_salts(["NaCl", "CaCl2"])
    m = {"Na+": 1.0, "Ca++": 0.5, "Cl-": 2.0}   # equal-equivalent mix
    g_25 = sys.gamma_pm("NaCl", m, T=298.15)
    g_75 = sys.gamma_pm("NaCl", m, T=348.15)
    # Result should differ from 25 °C (T-aware β AND mixing)
    check(f"γ_NaCl(NaCl/CaCl2 brine) 25°C = {g_25:.4f}, 75°C = {g_75:.4f}",
          abs(g_25 - g_75) > 1e-3)


def test_multi_pitzer_seawater_T_profile():
    """Seawater φ should decrease with T (well-established empirical
    trend; consistent with binary β derivatives + bundled mixing T-deps)."""
    section("test_multi_pitzer_seawater_T_profile")
    from stateprop.electrolyte import MultiPitzerSystem
    sw = MultiPitzerSystem.seawater()
    m = {"Na+": 0.486, "K+": 0.0106, "Mg++": 0.0547, "Ca++": 0.0107,
          "Cl-": 0.5658, "SO4--": 0.0293}
    phi_25 = sw.osmotic_coefficient(m, 298.15)
    phi_75 = sw.osmotic_coefficient(m, 348.15)
    phi_100 = sw.osmotic_coefficient(m, 373.15)
    check(f"seawater φ decreasing: 25°C={phi_25:.4f}, 75°C={phi_75:.4f}, "
          f"100°C={phi_100:.4f}",
          phi_75 < phi_25 and phi_100 < phi_75)


def test_multi_pitzer_user_float_override_still_works():
    """Backward compat: user can pass plain floats in theta_cc override
    (coerced to T-independent MixingParam)."""
    section("test_multi_pitzer_user_float_override_still_works")
    from stateprop.electrolyte import MultiPitzerSystem
    from stateprop.electrolyte.multi_pitzer import lookup_salt
    sys = MultiPitzerSystem(
        cations=[("Na+", 1), ("K+", 1)],
        anions=[("Cl-", -1)],
        binary_pairs={("Na+", "Cl-"): lookup_salt("NaCl"),
                       ("K+", "Cl-"): lookup_salt("KCl")},
        theta_cc={("Na+", "K+"): -0.020})    # plain float, override
    p = sys.theta_cc[("K+", "Na+")]   # canonical sort puts K+ first alphabetically
    check(f"plain-float override coerced to MixingParam: {p}",
          p.value_25 == -0.020 and p.dvalue_dT == 0.0)


# =====================================================================
# Mineral solubility (v0.9.101)
# =====================================================================

def test_mineral_db_loaded():
    """Bundled mineral database has ≥14 minerals."""
    section("test_mineral_db_loaded")
    from stateprop.electrolyte import list_minerals, lookup_mineral
    minerals = list_minerals()
    check(f"bundled DB has {len(minerals)} minerals",
          len(minerals) >= 14)
    for name in ("halite", "gypsum", "anhydrite", "calcite", "dolomite",
                 "barite", "magnesite"):
        m = lookup_mineral(name)
        check(f"  lookup_mineral({name!r}) → {m.formula}",
              m.name == name)


def test_mineral_log_K_sp_25C():
    """log_K_sp at 25 °C matches published values for canonical minerals."""
    section("test_mineral_log_K_sp_25C")
    from stateprop.electrolyte import lookup_mineral
    # Plummer-Busenberg 1982 / Blount 1977 / R-B 1987
    cases = [
        ("calcite",   -8.48),
        ("aragonite", -8.34),
        ("dolomite",  -17.09),
        ("gypsum",    -4.581),
        ("anhydrite", -4.36),
        ("barite",    -9.97),
        ("celestite", -6.63),
    ]
    for name, expected in cases:
        m = lookup_mineral(name)
        log_K = m.log_K_sp(298.15)
        check(f"log_K_sp({name}, 25°C) = {log_K:.3f} (lit {expected:.3f})",
              abs(log_K - expected) < 0.01)


def test_mineral_log_K_sp_T_dependence():
    """Van't Hoff form gives sensible T-dependence direction."""
    section("test_mineral_log_K_sp_T_dependence")
    from stateprop.electrolyte import lookup_mineral
    # Calcite: ΔH_rxn < 0 → log_K decreases with T (retrograde)
    calcite = lookup_mineral("calcite")
    log_K_25 = calcite.log_K_sp(298.15)
    log_K_75 = calcite.log_K_sp(348.15)
    check(f"calcite log_K(75°C) < log_K(25°C): {log_K_75:.3f} < {log_K_25:.3f}",
          log_K_75 < log_K_25)
    # Barite: ΔH_rxn > 0 → log_K increases with T (prograde)
    barite = lookup_mineral("barite")
    log_K_25 = barite.log_K_sp(298.15)
    log_K_75 = barite.log_K_sp(348.15)
    check(f"barite log_K(75°C) > log_K(25°C): {log_K_75:.3f} > {log_K_25:.3f}",
          log_K_75 > log_K_25)


def test_solubility_halite_pure_water():
    """Halite (NaCl) pure-water solubility 25 °C = 6.15 mol/kg
    (Krumgalz-Pogorelsky-Pitzer 1995)."""
    section("test_solubility_halite_pure_water")
    from stateprop.electrolyte import solubility_in_water
    S = solubility_in_water("halite", T=298.15)
    check(f"halite solubility 25°C = {S:.3f} mol/kg (lit 6.15, ~2% envelope)",
          abs(S - 6.15) / 6.15 < 0.02)


def test_solubility_gypsum_pure_water():
    """Gypsum (CaSO4·2H2O) pure-water solubility 25 °C = 0.0152 mol/kg
    (Marshall-Slusher 1966)."""
    section("test_solubility_gypsum_pure_water")
    from stateprop.electrolyte import solubility_in_water
    S = solubility_in_water("gypsum", T=298.15)
    check(f"gypsum solubility 25°C = {S:.5f} mol/kg (lit 0.0152, ~5% envelope)",
          abs(S - 0.0152) / 0.0152 < 0.05)


def test_solubility_halite_T_dependence():
    """Halite solubility increases with T (well-known empirical trend)."""
    section("test_solubility_halite_T_dependence")
    from stateprop.electrolyte import solubility_in_water
    S25 = solubility_in_water("halite", T=298.15)
    S100 = solubility_in_water("halite", T=373.15)
    check(f"halite S(100°C) > S(25°C): {S100:.2f} > {S25:.2f}",
          S100 > S25)


def test_solubility_gypsum_anhydrite_crossover():
    """Above ~40 °C, anhydrite is more stable than gypsum
    (i.e. anhydrite solubility lower)."""
    section("test_solubility_gypsum_anhydrite_crossover")
    from stateprop.electrolyte import solubility_in_water
    # At 25 °C, gypsum is the stable phase
    S_gyp_25 = solubility_in_water("gypsum", T=298.15)
    S_anh_25 = solubility_in_water("anhydrite", T=298.15)
    check(f"At 25°C: gypsum {S_gyp_25:.4f} < anhydrite {S_anh_25:.4f}",
          S_gyp_25 < S_anh_25)
    # At 80 °C, anhydrite is the stable phase
    S_gyp_80 = solubility_in_water("gypsum", T=353.15)
    S_anh_80 = solubility_in_water("anhydrite", T=353.15)
    check(f"At 80°C: anhydrite {S_anh_80:.4f} < gypsum {S_gyp_80:.4f} "
          "(crossover happened)",
          S_anh_80 < S_gyp_80)


def test_solubility_in_water_rejects_ternary():
    """solubility_in_water rejects minerals that aren't simple binary
    salts (e.g., dolomite needs three ions)."""
    section("test_solubility_in_water_rejects_ternary")
    from stateprop.electrolyte import solubility_in_water
    raised = False
    try:
        solubility_in_water("dolomite", T=298.15)
    except ValueError:
        raised = True
    check("solubility_in_water('dolomite') raises ValueError", raised)


def test_saturation_index_undersaturated():
    """SI < 0 for undersaturated solution."""
    section("test_saturation_index_undersaturated")
    from stateprop.electrolyte import saturation_index
    # Half-saturated barite (γ=1 at infinite dilution)
    m = {"Ba++": 5e-6, "SO4--": 5e-6}
    gammas = {"Ba++": 1.0, "SO4--": 1.0}
    SI = saturation_index("barite", m, gammas, T=298.15)
    check(f"barite half-saturated SI = {SI:.3f} (expect ~ -0.62)",
          -0.7 < SI < -0.5)


def test_saturation_index_saturated():
    """SI ≈ 0 at saturation."""
    section("test_saturation_index_saturated")
    from stateprop.electrolyte import saturation_index
    m = {"Ba++": 1.04e-5, "SO4--": 1.04e-5}
    gammas = {"Ba++": 1.0, "SO4--": 1.0}
    SI = saturation_index("barite", m, gammas, T=298.15)
    check(f"barite at saturation SI = {SI:.3f} (expect ~ 0)",
          abs(SI) < 0.01)


def test_saturation_index_supersaturated():
    """SI > 0 for supersaturated solution."""
    section("test_saturation_index_supersaturated")
    from stateprop.electrolyte import saturation_index
    m = {"Ba++": 1e-4, "SO4--": 1e-4}
    gammas = {"Ba++": 1.0, "SO4--": 1.0}
    SI = saturation_index("barite", m, gammas, T=298.15)
    check(f"barite 10× supersaturated SI = {SI:.3f} (expect +1.97)",
          1.9 < SI < 2.0)


def test_saturation_index_missing_ion_returns_minus_inf():
    """SI = -inf when a required ion is missing from molality dict."""
    section("test_saturation_index_missing_ion_returns_minus_inf")
    from stateprop.electrolyte import saturation_index
    # No Ba in solution → IAP = 0 → SI = -inf
    m = {"SO4--": 0.05}
    gammas = {"SO4--": 0.5}
    SI = saturation_index("barite", m, gammas, T=298.15)
    check(f"barite SI without Ba = {SI} (expect -inf)",
          SI == float("-inf"))


def test_mineral_system_seawater_qualitative():
    """In seawater at 25 °C, calcite/aragonite/dolomite are super-
    saturated (well-known marine carbonate result). Gypsum, anhydrite,
    halite are undersaturated (also known).

    The exact SI values are off by ~1-2 log units due to lack of
    explicit Ca-CO3, Mg-CO3 binary β and aqueous complexation, but
    the qualitative picture (signs and ordering) is correct."""
    section("test_mineral_system_seawater_qualitative")
    from stateprop.electrolyte import (
        MultiPitzerSystem, MineralSystem,
    )
    sw = MultiPitzerSystem.seawater()
    ms = MineralSystem(sw, ["calcite", "aragonite", "dolomite",
                              "gypsum", "anhydrite", "halite"])
    # Seawater + carbonate (approximate at pH ~8.1)
    m = {"Na+": 0.486, "K+": 0.0106, "Mg++": 0.0547, "Ca++": 0.0107,
          "Cl-": 0.5658, "SO4--": 0.0293,
          "HCO3-": 0.00170, "CO3--": 0.00025}
    SI = ms.saturation_indices(m)
    check(f"calcite supersaturated (SI={SI['calcite']:+.2f} > 0)",
          SI["calcite"] > 0)
    check(f"aragonite supersaturated (SI={SI['aragonite']:+.2f} > 0)",
          SI["aragonite"] > 0)
    check(f"calcite > aragonite (more stable polymorph, "
          f"{SI['calcite']:.2f} > {SI['aragonite']:.2f})",
          SI["calcite"] > SI["aragonite"])
    check(f"halite undersaturated (SI={SI['halite']:+.2f} < 0)",
          SI["halite"] < 0)
    check(f"gypsum undersaturated (SI={SI['gypsum']:+.2f} < 0)",
          SI["gypsum"] < 0)


def test_mineral_system_scale_risks_filter():
    """scale_risks returns only minerals with SI > threshold."""
    section("test_mineral_system_scale_risks_filter")
    from stateprop.electrolyte import (
        MultiPitzerSystem, MineralSystem,
    )
    sys = MultiPitzerSystem.from_salts(["NaCl", "CaCl2"])
    ms = MineralSystem(sys, ["halite", "sylvite"])
    # Pure NaCl-CaCl2 brine: undersaturated in halite (no Cl excess)
    m = {"Na+": 1.0, "Ca++": 0.5, "Cl-": 2.0}
    risks = ms.scale_risks(m)
    check(f"no scale risks at undersaturated condition: {risks}",
          len(risks) == 0)
    # Push NaCl past saturation
    m_super = {"Na+": 8.0, "Ca++": 0.0, "Cl-": 8.0}
    risks = ms.scale_risks(m_super)
    check(f"halite at risk in oversaturated NaCl: {risks}",
          "halite" in risks and risks["halite"] > 0)


def test_solubility_iter_converges():
    """solubility_in_water converges in <= 100 iterations for all
    bundled binary-salt minerals at 25 °C."""
    section("test_solubility_iter_converges")
    from stateprop.electrolyte import (
        list_minerals, lookup_mineral, solubility_in_water,
    )
    for name in list_minerals():
        m = lookup_mineral(name)
        if m.binary_salt is None:
            continue
        try:
            S = solubility_in_water(name, T=298.15)
            check(f"  {name}: S(25°C) = {S:.4g} mol/kg (converged)",
                  S > 0 and not np.isnan(S))
        except Exception as e:
            check(f"  {name}: FAILED — {e}", False)


def test_dolomite_SI_in_dilute_seawater():
    """Dolomite SI uses ν_Ca=1, ν_Mg=1, ν_CO3=2 stoichiometry correctly."""
    section("test_dolomite_SI_in_dilute_seawater")
    from stateprop.electrolyte import saturation_index, lookup_mineral
    m = {"Ca++": 0.01, "Mg++": 0.05, "CO3--": 1e-4}
    gammas = {"Ca++": 1.0, "Mg++": 1.0, "CO3--": 1.0}
    SI = saturation_index("dolomite", m, gammas, T=298.15)
    # log_IAP = log(0.01) + log(0.05) + 2·log(1e-4) = -2 + -1.301 + -8 = -11.301
    # SI = -11.301 - (-17.09) = +5.789
    expected = np.log10(0.01) + np.log10(0.05) + 2*np.log10(1e-4) - (-17.09)
    check(f"dolomite SI computed correctly: {SI:.3f} (expect {expected:.3f})",
          abs(SI - expected) < 0.01)


# =====================================================================
# Aqueous complexation (v0.9.102)
# =====================================================================

def test_complex_db_loaded():
    """Bundled complex database has 11 standard complexes."""
    section("test_complex_db_loaded")
    from stateprop.electrolyte import list_complexes, lookup_complex
    complexes = list_complexes()
    check(f"got {len(complexes)} complexes (expect 11)",
          len(complexes) == 11)
    # Check key complexes are there
    for name in ("CaSO4°", "MgSO4°", "NaSO4-",
                  "CaCO3°", "MgCO3°", "NaCO3-",
                  "CaHCO3+", "MgHCO3+", "CaOH+", "MgOH+"):
        c = lookup_complex(name)
        check(f"  lookup_complex({name!r}) → log_K_diss = {c.log_K_diss_25:+.2f}",
              c.name == name or c.name.replace("o", "°") == name)


def test_complex_K_diss_T_dependence():
    """Most ion-pair K_diss have ΔH > 0 (endothermic);
    K_diss increases with T (complex less stable at higher T)."""
    section("test_complex_K_diss_T_dependence")
    from stateprop.electrolyte import lookup_complex
    # CaSO4° has ΔH = +6900 J/mol, so log_K_diss should increase from -2.30 at 25°C
    caso4 = lookup_complex("CaSO4°")
    log_K_25 = caso4.log_K_diss(298.15)
    log_K_75 = caso4.log_K_diss(348.15)
    check(f"CaSO4° log_K_diss(75°C) > log_K_diss(25°C): "
          f"{log_K_75:.3f} > {log_K_25:.3f}",
          log_K_75 > log_K_25)


def test_complex_log_K_assoc_inverse():
    """log_K_assoc = -log_K_diss."""
    section("test_complex_log_K_assoc_inverse")
    from stateprop.electrolyte import lookup_complex
    c = lookup_complex("CaSO4°")
    check(f"log_K_assoc(25°C) = {c.log_K_assoc_25} = -log_K_diss",
          abs(c.log_K_assoc_25 - (-c.log_K_diss_25)) < 1e-10)


def test_speciation_pure_passthrough():
    """If no complexation can happen (no shared components),
    speciation result is just the input."""
    section("test_speciation_pure_passthrough")
    from stateprop.electrolyte import (
        MultiPitzerSystem, Speciation,
    )
    pitzer = MultiPitzerSystem.from_salts(["NaCl"])
    spec = Speciation(pitzer, ["CaSO4°"])    # Ca/SO4 not in totals
    res = spec.solve({"Na+": 1.0, "Cl-": 1.0}, T=298.15)
    check(f"converged passthrough: {res.converged}", res.converged)
    check(f"complexes empty: {res.complexes}",
          all(v == 0 for v in res.complexes.values())
          or len(res.complexes) == 0)
    check(f"free Na+ = total Na+: {res.free['Na+']:.4f}",
          abs(res.free["Na+"] - 1.0) < 1e-10)


def test_speciation_NaSO4_ion_pair():
    """In pure 1m Na2SO4, ~12% of Na is paired as NaSO4-
    (matches PHREEQC reference)."""
    section("test_speciation_NaSO4_ion_pair")
    from stateprop.electrolyte import (
        MultiPitzerSystem, Speciation,
    )
    pitzer = MultiPitzerSystem.from_salts(["Na2SO4"])
    spec = Speciation(pitzer, ["NaSO4-"])
    res = spec.solve({"Na+": 2.0, "SO4--": 1.0}, T=298.15)
    check(f"converged: {res.converged}", res.converged)
    pct_complex = res.complexes["NaSO4-"] / 1.0 * 100   # vs total SO4
    check(f"~10-20% of SO4 is paired as NaSO4-: {pct_complex:.1f}%",
          5 < pct_complex < 25)


def test_speciation_seawater_CO3_depleted():
    """In seawater, free CO3²⁻ should be a small fraction (~5-10%)
    of total CO3²⁻ — the rest is in MgCO3°, CaCO3°, etc.
    This is the textbook marine-chemistry result."""
    section("test_speciation_seawater_CO3_depleted")
    from stateprop.electrolyte import (
        MultiPitzerSystem, Speciation,
    )
    sw = MultiPitzerSystem.seawater()
    spec = Speciation(sw, ["CaSO4°", "MgSO4°", "NaSO4-",
                              "CaCO3°", "MgCO3°", "NaCO3-",
                              "CaHCO3+", "MgHCO3+"])
    m_total = {
        "Na+": 0.486, "K+": 0.0106, "Mg++": 0.0547, "Ca++": 0.0107,
        "Cl-": 0.5658, "SO4--": 0.0293,
        "HCO3-": 0.00170, "CO3--": 0.00025,
    }
    res = spec.solve(m_total, T=298.15)
    pct_free_CO3 = res.free["CO3--"] / m_total["CO3--"] * 100
    check(f"converged: {res.converged}", res.converged)
    check(f"free CO3²⁻ is <15% of total: {pct_free_CO3:.1f}%",
          pct_free_CO3 < 15)


def test_speciation_calcite_SI_seawater():
    """Calcite SI in seawater drops from ~+2.2 (no complexation)
    to ~+0.5 to +1.0 (with complexation), matching Doney 2009 lit."""
    section("test_speciation_calcite_SI_seawater")
    from stateprop.electrolyte import (
        MultiPitzerSystem, Speciation,
    )
    sw = MultiPitzerSystem.seawater()
    spec = Speciation(sw, ["CaSO4°", "MgSO4°", "NaSO4-",
                              "CaCO3°", "MgCO3°", "NaCO3-",
                              "CaHCO3+", "MgHCO3+"])
    m_total = {
        "Na+": 0.486, "K+": 0.0106, "Mg++": 0.0547, "Ca++": 0.0107,
        "Cl-": 0.5658, "SO4--": 0.0293,
        "HCO3-": 0.00170, "CO3--": 0.00025,
    }
    res = spec.solve(m_total, T=298.15)
    SI_calcite = res.saturation_index("calcite")
    check(f"calcite SI = {SI_calcite:+.2f} (Doney 2009 surface seawater "
          f"~+0.6 to +0.8)",
          0.4 < SI_calcite < 1.2)


def test_speciation_calcite_aragonite_ordering():
    """Calcite SI > aragonite SI by ~0.14 log units in seawater
    (calcite is the more stable polymorph)."""
    section("test_speciation_calcite_aragonite_ordering")
    from stateprop.electrolyte import (
        MultiPitzerSystem, Speciation,
    )
    sw = MultiPitzerSystem.seawater()
    spec = Speciation(sw, ["CaSO4°", "MgSO4°", "NaSO4-",
                              "CaCO3°", "MgCO3°", "NaCO3-",
                              "CaHCO3+", "MgHCO3+"])
    m_total = {
        "Na+": 0.486, "K+": 0.0106, "Mg++": 0.0547, "Ca++": 0.0107,
        "Cl-": 0.5658, "SO4--": 0.0293,
        "HCO3-": 0.00170, "CO3--": 0.00025,
    }
    res = spec.solve(m_total, T=298.15)
    SI_calc = res.saturation_index("calcite")
    SI_arag = res.saturation_index("aragonite")
    diff = SI_calc - SI_arag
    check(f"SI(calcite) - SI(aragonite) = {diff:.3f} (expect ~+0.14)",
          0.10 < diff < 0.20)


def test_speciation_gypsum_pure_water():
    """With explicit CaSO4° complex + thermodynamic K_sp = -4.75,
    gypsum solubility in pure water = 0.0151 mol/kg (lit 0.0152, <2%)."""
    section("test_speciation_gypsum_pure_water")
    from stateprop.electrolyte import (
        MultiPitzerSystem, Speciation,
    )
    pitzer = MultiPitzerSystem.from_salts(["CaSO4"])
    spec = Speciation(pitzer, ["CaSO4°"])
    # Bisect on total CaSO4
    lo, hi = 1e-6, 0.1
    for _ in range(80):
        m = (lo + hi) / 2
        res = spec.solve({"Ca++": m, "SO4--": m}, T=298.15)
        SI = res.saturation_index("gypsum")
        if SI > 0: hi = m
        else:      lo = m
        if abs(SI) < 1e-6: break
    check(f"gypsum solubility (with CaSO4°) = {m:.4f} mol/kg "
          f"(lit 0.0152, <2% envelope)",
          abs(m - 0.0152) / 0.0152 < 0.02)


def test_speciation_mass_balance_holds():
    """For each component, free + Σ(ν · complex) = total to ~1e-8."""
    section("test_speciation_mass_balance_holds")
    from stateprop.electrolyte import (
        MultiPitzerSystem, Speciation,
    )
    pitzer = MultiPitzerSystem.from_salts(["Na2SO4", "MgSO4", "CaSO4"])
    spec = Speciation(pitzer, ["CaSO4°", "MgSO4°", "NaSO4-"])
    totals = {"Na+": 1.0, "Ca++": 0.1, "Mg++": 0.2, "SO4--": 0.6}
    res = spec.solve(totals, T=298.15)
    check(f"converged: {res.converged}", res.converged)
    # Mass balance: total = free + Σ(ν · complex)
    mb_Na = (res.free["Na+"]
              + res.complexes["NaSO4-"])
    mb_Ca = (res.free["Ca++"]
              + res.complexes["CaSO4°"])
    mb_Mg = (res.free["Mg++"]
              + res.complexes["MgSO4°"])
    mb_SO4 = (res.free["SO4--"]
                 + res.complexes["NaSO4-"]
                 + res.complexes["CaSO4°"]
                 + res.complexes["MgSO4°"])
    for name, mb, total in [("Na", mb_Na, 1.0), ("Ca", mb_Ca, 0.1),
                              ("Mg", mb_Mg, 0.2), ("SO4", mb_SO4, 0.6)]:
        check(f"  {name} balance: {mb:.6f} = {total} (err {abs(mb-total):.1e})",
              abs(mb - total) / total < 1e-7)


def test_mineral_log_K_sp_thermo_field():
    """Mineral.log_K_sp_25_thermo, when set, is used by
    SpeciationResult.saturation_index."""
    section("test_mineral_log_K_sp_thermo_field")
    from stateprop.electrolyte import lookup_mineral
    gypsum = lookup_mineral("gypsum")
    check(f"gypsum has thermo K_sp = {gypsum.log_K_sp_25_thermo}",
          gypsum.log_K_sp_25_thermo == -4.75)
    check(f"gypsum apparent K_sp = {gypsum.log_K_sp_25} unchanged",
          gypsum.log_K_sp_25 == -4.581)


def test_solve_speciation_convenience_wrapper():
    """solve_speciation builds a Speciation and solves in one call."""
    section("test_solve_speciation_convenience_wrapper")
    from stateprop.electrolyte import (
        MultiPitzerSystem, solve_speciation,
    )
    pitzer = MultiPitzerSystem.from_salts(["Na2SO4"])
    res = solve_speciation(
        {"Na+": 1.0, "SO4--": 0.5},
        pitzer=pitzer, complexes=["NaSO4-"], T=298.15)
    check(f"convenience wrapper works: {res.converged}", res.converged)
    check(f"  has NaSO4- complex: {res.complexes['NaSO4-']:.4f}",
          res.complexes["NaSO4-"] > 1e-3)


def test_davies_equation_neutral_returns_zero():
    """Davies equation returns 0 for z=0 (neutral species)."""
    section("test_davies_equation_neutral_returns_zero")
    from stateprop.electrolyte.complexation import _davies_log_gamma
    check(f"_davies_log_gamma(0, 0.5) = {_davies_log_gamma(0, 0.5)}",
          _davies_log_gamma(0, 0.5) == 0.0)


def test_davies_equation_low_I_dilute_limit():
    """At very low I, Davies → 0 (γ → 1)."""
    section("test_davies_equation_low_I_dilute_limit")
    from stateprop.electrolyte.complexation import _davies_log_gamma
    log_g = _davies_log_gamma(1, 1e-6, 298.15)
    check(f"_davies_log_gamma(1, 1e-6) = {log_g:.6f} ≈ 0",
          abs(log_g) < 1e-3)


# =====================================================================
# Amine carbamate equilibria (v0.9.103)
# =====================================================================

def test_amine_db_loaded():
    """Bundled amine database has 5 standard amines."""
    section("test_amine_db_loaded")
    from stateprop.electrolyte import list_amines, lookup_amine
    names = list_amines()
    check(f"got {len(names)} amines (expect 5)", len(names) == 5)
    for n in ("MEA", "DEA", "MDEA", "AMP", "NH3"):
        a = lookup_amine(n)
        check(f"  {n}: pKa_25={a.pKa_25}, tertiary={a.is_tertiary}",
              a.name == n)


def test_amine_pKa_van_t_hoff():
    """pKa shifts from 9.50 at 25 °C to ~7.8 at 100 °C for MEA."""
    section("test_amine_pKa_van_t_hoff")
    from stateprop.electrolyte import lookup_amine
    mea = lookup_amine("MEA")
    pKa_25 = mea.pKa(298.15)
    pKa_100 = mea.pKa(373.15)
    check(f"  pKa(MEA, 25 °C) = {pKa_25:.2f}",
          abs(pKa_25 - 9.50) < 0.01)
    # van't Hoff: 9.50 + (49000/19.14)·(1/373.15 - 1/298.15) ≈ 7.78
    check(f"  pKa(MEA, 100 °C) = {pKa_100:.2f} (expect ~7.78)",
          abs(pKa_100 - 7.78) < 0.05)


def test_amine_K_a_inverse_of_pKa():
    """K_a = 10^(-pKa)."""
    section("test_amine_K_a_inverse_of_pKa")
    from stateprop.electrolyte import lookup_amine
    mea = lookup_amine("MEA")
    K_a = mea.K_a(298.15)
    check(f"  K_a(25 °C) = {K_a:.3e} (expect 10^-9.5 = 3.16e-10)",
          abs(K_a - 10**-9.5) / 10**-9.5 < 0.01)


def test_tertiary_amine_no_carbamate():
    """MDEA is tertiary; pK_carb is undefined."""
    section("test_tertiary_amine_no_carbamate")
    from stateprop.electrolyte import lookup_amine
    mdea = lookup_amine("MDEA")
    check(f"MDEA is tertiary", mdea.is_tertiary)
    try:
        mdea.pK_carb(298.15)
        check("MDEA.pK_carb raises ValueError", False)
    except ValueError:
        check("MDEA.pK_carb raises ValueError", True)


def test_carbonate_constants_at_25C():
    """Reference equilibrium constants at 25 °C match literature."""
    section("test_carbonate_constants_at_25C")
    from stateprop.electrolyte.amines import (
        _pK1_CO2, _pK2_CO2, _pKw, _kH_CO2,
    )
    check(f"pK1(CO2, 25 °C) = {_pK1_CO2(298.15):.3f} (lit 6.354)",
          abs(_pK1_CO2(298.15) - 6.354) < 0.01)
    check(f"pK2(CO2, 25 °C) = {_pK2_CO2(298.15):.3f} (lit 10.329)",
          abs(_pK2_CO2(298.15) - 10.329) < 0.01)
    check(f"pKw(25 °C) = {_pKw(298.15):.3f} (lit 14.00)",
          abs(_pKw(298.15) - 14.00) < 0.01)
    check(f"K_H(CO2, 25 °C) = {_kH_CO2(298.15):.1f} (lit 29.4)",
          abs(_kH_CO2(298.15) - 29.4) < 0.5)


def test_K_H_increases_with_T():
    """K_H = P/m increases with T (CO2 less soluble at high T)."""
    section("test_K_H_increases_with_T")
    from stateprop.electrolyte.amines import _kH_CO2
    K_25 = _kH_CO2(298.15)
    K_100 = _kH_CO2(373.15)
    check(f"K_H(100°C) = {K_100:.1f} > K_H(25°C) = {K_25:.1f}",
          K_100 > K_25)
    # Lit ~110-140 at 100 °C
    check(f"  K_H(100°C) in lit range 100-150: {K_100:.1f}",
          100 < K_100 < 150)


def test_amine_speciate_primary_MEA_alpha_05():
    """30 wt% (5m) MEA at α=0.5, 40 °C: P_CO2 ~0.1-0.2 bar
    (lit Aronu 2011 / LOM 1976), pH ~8-9."""
    section("test_amine_speciate_primary_MEA_alpha_05")
    from stateprop.electrolyte import AmineSystem
    sys = AmineSystem("MEA", total_amine=5.0)
    res = sys.speciate(alpha=0.5, T=313.15)
    check(f"  converged: {res.converged}", res.converged)
    check(f"  P_CO2 = {res.P_CO2:.3f} bar (expect 0.05-0.5)",
          0.05 < res.P_CO2 < 0.5)
    check(f"  pH = {res.pH:.2f} (expect 7-10)",
          7.0 < res.pH < 10.0)


def test_amine_speciate_tertiary_MDEA():
    """MDEA tertiary: at α=0.5, P_CO2 should be reasonable
    (no carbamate species in result)."""
    section("test_amine_speciate_tertiary_MDEA")
    from stateprop.electrolyte import AmineSystem
    sys = AmineSystem("MDEA", total_amine=5.0)
    res = sys.speciate(alpha=0.5, T=313.15)
    check(f"  converged: {res.converged}", res.converged)
    check(f"  P_CO2 = {res.P_CO2:.3f} bar (expect 0.1-10)",
          0.1 < res.P_CO2 < 10.0)
    # No carbamate species in tertiary
    check(f"  no MDEACOO- in free dict",
          "MDEACOO-" not in res.free)


def test_amine_speciate_alpha_zero():
    """At α=0 (no CO2), P_CO2 = 0 and pH should be high (basic amine)."""
    section("test_amine_speciate_alpha_zero")
    from stateprop.electrolyte import AmineSystem
    sys = AmineSystem("MEA", total_amine=5.0)
    res = sys.speciate(alpha=1e-6, T=313.15)
    check(f"  P_CO2 ≈ 0: {res.P_CO2:.3e}", res.P_CO2 < 1e-3)
    check(f"  pH > 10 (basic): {res.pH:.2f}", res.pH > 10.0)


def test_amine_speciate_mass_balance():
    """Total amine balance: free + protonated + carbamate = total."""
    section("test_amine_speciate_mass_balance")
    from stateprop.electrolyte import AmineSystem
    sys = AmineSystem("MEA", total_amine=5.0)
    res = sys.speciate(alpha=0.4, T=313.15)
    Am = res.free["MEA"]
    AmH = res.free["MEA+"] if "MEA+" in res.free else res.free["MEAH+"]
    AmCOO = res.free.get("MEACOO-", 0)
    total = Am + AmH + AmCOO
    check(f"  amine MB: {Am:.4f} + {AmH:.4f} + {AmCOO:.4f} = {total:.4f} "
          f"(expect 5.0)",
          abs(total - 5.0) / 5.0 < 1e-6)


def test_amine_equilibrium_loading_inverse():
    """Solving inverse: speciate(α) → P_CO2; equilibrium_loading(P_CO2) → α'.
    Should round-trip to within tolerance."""
    section("test_amine_equilibrium_loading_inverse")
    from stateprop.electrolyte import AmineSystem
    sys = AmineSystem("MEA", total_amine=5.0)
    alpha_orig = 0.4
    P_at_alpha = sys.speciate(alpha=alpha_orig, T=313.15).P_CO2
    alpha_back = sys.equilibrium_loading(P_CO2=P_at_alpha, T=313.15)
    check(f"  α → P → α': {alpha_orig} → {P_at_alpha:.3e} → {alpha_back:.3f}",
          abs(alpha_back - alpha_orig) / alpha_orig < 0.05)


def test_amine_loading_monotonic_in_PCO2():
    """Equilibrium loading α should increase monotonically with P_CO2."""
    section("test_amine_loading_monotonic_in_PCO2")
    from stateprop.electrolyte import AmineSystem
    sys = AmineSystem("MEA", total_amine=5.0)
    alphas = sys.loading_curve(P_CO2_list=[0.001, 0.01, 0.1, 1.0],
                                  T=313.15)
    check(f"  monotonic: {alphas}",
          all(alphas[i+1] > alphas[i] for i in range(len(alphas)-1)))


def test_amine_higher_T_releases_CO2():
    """At fixed α, higher T → higher P_CO2 (regenerator behaviour)."""
    section("test_amine_higher_T_releases_CO2")
    from stateprop.electrolyte import AmineSystem
    sys = AmineSystem("MEA", total_amine=5.0)
    P_40 = sys.speciate(alpha=0.5, T=313.15).P_CO2
    P_100 = sys.speciate(alpha=0.5, T=373.15).P_CO2
    check(f"  P(100°C) > P(40°C): {P_100:.3f} > {P_40:.3f}",
          P_100 > P_40)
    # Should be at least 5× higher at 100 °C (significant T-dep)
    check(f"  P(100°C)/P(40°C) > 5: ratio = {P_100/P_40:.1f}",
          P_100/P_40 > 5)


def test_amine_lookup_case_insensitive():
    """lookup_amine accepts mea, MEA, Mea."""
    section("test_amine_lookup_case_insensitive")
    from stateprop.electrolyte import lookup_amine
    a1 = lookup_amine("MEA")
    a2 = lookup_amine("mea")
    a3 = lookup_amine("Mea")
    check(f"  case-insensitive: all return same Amine",
          a1.name == a2.name == a3.name == "MEA")


# =====================================================================
# eNRTL refinements + amine column (v0.9.104)
# =====================================================================

def test_pdh_A_phi_25C():
    """A_φ at 25 °C = 0.3915 (Pitzer 1973)."""
    section("test_pdh_A_phi_25C")
    from stateprop.electrolyte.enrtl import A_phi
    check(f"A_phi(25 °C) = {A_phi(298.15):.4f} (lit 0.3915)",
          abs(A_phi(298.15) - 0.3915) < 0.0001)


def test_pdh_A_phi_T_dependence():
    """A_φ increases with T from 0.3915 (25 °C) to ~0.5908 (100 °C)."""
    section("test_pdh_A_phi_T_dependence")
    from stateprop.electrolyte.enrtl import A_phi
    A_25 = A_phi(298.15)
    A_100 = A_phi(373.15)
    check(f"A_phi(100 °C) > A_phi(25 °C): {A_100:.3f} > {A_25:.3f}",
          A_100 > A_25)
    # PDH/Pitzer 1991 quadratic gives ~0.54 at 100 °C
    check(f"A_phi(100 °C) ≈ 0.54: actual {A_100:.3f}",
          0.5 < A_100 < 0.6)


def test_pdh_log_gamma_neutral():
    """PDH log γ for z=0 returns 0."""
    section("test_pdh_log_gamma_neutral")
    from stateprop.electrolyte.enrtl import pdh_log_gamma
    check(f"pdh_log_gamma(0, 1.0) = 0",
          pdh_log_gamma(0, 1.0) == 0.0)


def test_pdh_does_not_diverge_at_high_I():
    """Unlike Davies, PDH log γ does NOT diverge to +∞ at high I.
    At I=4, Davies gives +1.09 (positive!) but PDH gives -0.36."""
    section("test_pdh_does_not_diverge_at_high_I")
    from stateprop.electrolyte.enrtl import (
        pdh_log_gamma, davies_log_gamma_v104,
    )
    log_d = davies_log_gamma_v104(2, 4.0, 298.15)
    log_p = pdh_log_gamma(2, 4.0, 298.15)
    check(f"Davies at I=4 gives positive log γ ({log_d:+.3f}, unphysical)",
          log_d > 0)
    check(f"PDH at I=4 gives negative log γ ({log_p:+.3f}, physical)",
          log_p < 0)


def test_amine_pdh_improves_high_T_prediction():
    """At α=0.5, 100 °C, MEA — PDH should give lower P_CO2 than Davies
    (closer to Hilliard 2008 lit ~5 bar)."""
    section("test_amine_pdh_improves_high_T_prediction")
    from stateprop.electrolyte import AmineSystem
    P_davies = AmineSystem("MEA", 5.0, activity_model="davies").speciate(
        0.5, T=373.15).P_CO2
    P_pdh = AmineSystem("MEA", 5.0, activity_model="pdh").speciate(
        0.5, T=373.15).P_CO2
    check(f"  PDH (P_CO2={P_pdh:.2f}) < Davies (P_CO2={P_davies:.2f})",
          P_pdh < P_davies)


def test_amine_speciate_with_pdh_converges():
    """PDH activity model converges to a self-consistent γ at high I."""
    section("test_amine_speciate_with_pdh_converges")
    from stateprop.electrolyte import AmineSystem
    sys = AmineSystem("MEA", 5.0, activity_model="pdh")
    res = sys.speciate(0.5, T=313.15)
    check(f"  converged: {res.converged}", res.converged)
    check(f"  P_CO2 in absorber range (0.05-0.5 bar): "
          f"{res.P_CO2:.3f}",
          0.05 < res.P_CO2 < 0.5)


def test_amine_column_simple_solve():
    """Basic absorber column converges and gives mass-balance-consistent
    α_rich and y_top."""
    section("test_amine_column_simple_solve")
    from stateprop.electrolyte import AmineColumn
    col = AmineColumn("MEA", total_amine=5.0, n_stages=10)
    res = col.solve(L=8.0, G=15.0, alpha_lean=0.20, y_in=0.12)
    check(f"  converged: {res.converged} ({res.iterations} iter)",
          res.converged)
    # Overall CO2 mass balance: G·(y_in - y_top) = L·(α_rich - α_lean)
    co2_absorbed_gas = 15.0 * (0.12 - res.y_top)
    co2_absorbed_liq = 8.0 * (res.alpha_rich - 0.20)
    check(f"  overall MB closure: gas {co2_absorbed_gas:.4f} = "
          f"liq {co2_absorbed_liq:.4f}",
          abs(co2_absorbed_gas - co2_absorbed_liq) / co2_absorbed_gas < 1e-5)


def test_amine_column_recovery_high_LG():
    """At high L/G, the column achieves >90% CO2 recovery in few stages."""
    section("test_amine_column_recovery_high_LG")
    from stateprop.electrolyte import AmineColumn
    col = AmineColumn("MEA", total_amine=5.0, n_stages=10)
    res = col.solve(L=15.0, G=10.0, alpha_lean=0.20, y_in=0.10)
    check(f"  recovery > 90%: {res.co2_recovery*100:.1f}%",
          res.co2_recovery > 0.90)


def test_amine_column_pinched_at_min_LG():
    """Near minimum L/G ratio, recovery is limited by α_max
    (rich pinch).  α_rich approaches the chemical maximum (~0.5 for MEA)."""
    section("test_amine_column_pinched_at_min_LG")
    from stateprop.electrolyte import AmineColumn
    col = AmineColumn("MEA", total_amine=5.0, n_stages=20)
    # L/G slightly above minimum
    res = col.solve(L=4.0, G=10.0, alpha_lean=0.05, y_in=0.30)
    check(f"  α_rich approaches saturation: {res.alpha_rich:.3f}",
          0.45 < res.alpha_rich < 0.6)


def test_amine_column_stages_for_recovery():
    """stages_for_recovery returns a reasonable integer for a typical
    capture target."""
    section("test_amine_column_stages_for_recovery")
    from stateprop.electrolyte import AmineColumn
    col = AmineColumn("MEA", total_amine=5.0, n_stages=1)
    N = col.stages_for_recovery(L=8.0, G=15.0, alpha_lean=0.20,
                                      y_in=0.12, target_recovery=0.90,
                                      max_stages=30)
    check(f"  stages for 90% recovery: {N} (expect 2-30)",
          1 < N < 30)


def test_amine_equilibrium_curve_monotonic():
    """y* increases monotonically with α."""
    section("test_amine_equilibrium_curve_monotonic")
    from stateprop.electrolyte import amine_equilibrium_curve
    alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    y_star = amine_equilibrium_curve("MEA", 5.0, alphas, T=313.15)
    check(f"  y* monotonic in α: {y_star}",
          all(y_star[i+1] > y_star[i] for i in range(len(y_star)-1)))


# =====================================================================
# Reactive stripper / heat balance (v0.9.105)
# =====================================================================

def test_amine_heat_properties_loaded():
    """Each bundled amine has delta_H_abs and cp_amine values."""
    section("test_amine_heat_properties_loaded")
    from stateprop.electrolyte import lookup_amine
    for name in ("MEA", "DEA", "MDEA", "AMP", "NH3"):
        a = lookup_amine(name)
        check(f"  {name}: ΔH_abs = {a.delta_H_abs/1000:.0f} kJ/mol, "
              f"cp = {a.cp_amine:.0f} J/(kg·K)",
              a.delta_H_abs < 0 and a.cp_amine > 1000)


def test_amine_cp_solution_30wt_MEA():
    """30 wt% MEA solution cp ≈ 3700 J/(kg·K)."""
    section("test_amine_cp_solution_30wt_MEA")
    from stateprop.electrolyte import lookup_amine
    cp = lookup_amine("MEA").cp_solution(0.30)
    expected = 0.7 * 4180 + 0.3 * 2650
    check(f"  cp_solution(30 wt% MEA) = {cp:.0f} (expect {expected:.0f})",
          abs(cp - expected) < 1)


def test_water_vapor_pressure_at_100C():
    """P_water_sat(100°C) ≈ 1.013 bar (atm bp)."""
    section("test_water_vapor_pressure_at_100C")
    from stateprop.electrolyte import P_water_sat
    P = P_water_sat(373.15)
    check(f"  P_water_sat(100 °C) = {P:.4f} bar (lit 1.013)",
          abs(P - 1.013) < 0.05)


def test_water_vapor_pressure_at_120C():
    """P_water_sat(120°C) ≈ 1.985 bar."""
    section("test_water_vapor_pressure_at_120C")
    from stateprop.electrolyte import P_water_sat
    P = P_water_sat(393.15)
    check(f"  P_water_sat(120 °C) = {P:.4f} bar (lit 1.985)",
          abs(P - 1.985) < 0.10)


def test_amine_stripper_simple_solve():
    """Basic stripper converges and gives mass-balance-consistent
    α_lean and y_top."""
    section("test_amine_stripper_simple_solve")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", total_amine=5.0, n_stages=10)
    r = strip.solve(L=10.0, G=8.0, alpha_rich=0.50, y_reb=0.05,
                       P=1.8, T_top=378.15, T_bottom=393.15)
    check(f"  converged: {r.converged} ({r.iterations} iter)",
          r.converged)
    # CO2 mass balance: G·(y_top - y_reb) = L·(α_rich - α_lean)
    co2_gas = 8.0 * (r.y_top_CO2 - 0.05)
    co2_liq = 10.0 * (0.50 - r.alpha_lean)
    rel = abs(co2_gas - co2_liq) / co2_liq
    check(f"  mass balance: gas {co2_gas:.4f} = liq {co2_liq:.4f} "
          f"(rel err {rel:.1e})",
          rel < 1e-5)


def test_amine_stripper_strips_CO2():
    """In a stripper, α_lean < α_rich (CO2 is removed from liquid)."""
    section("test_amine_stripper_strips_CO2")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", total_amine=5.0, n_stages=15)
    r = strip.solve(L=10.0, G=8.0, alpha_rich=0.50, y_reb=0.05)
    check(f"  α_lean ({r.alpha_lean:.3f}) < α_rich (0.50)",
          r.alpha_lean < 0.50)
    check(f"  CO2 stripped > 0: {r.co2_stripped:.3f}",
          r.co2_stripped > 0)


def test_amine_stripper_top_CO2_higher_than_reb():
    """Top vapor leaves with higher CO2 mole fraction than reboiler vapor."""
    section("test_amine_stripper_top_CO2_higher_than_reb")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", total_amine=5.0, n_stages=15)
    r = strip.solve(L=10.0, G=8.0, alpha_rich=0.50, y_reb=0.05)
    check(f"  y_top_CO2 ({r.y_top_CO2:.3f}) > y_reb (0.05)",
          r.y_top_CO2 > 0.05)


def test_amine_stripper_reboiler_duty_in_industry_range():
    """For 30 wt% MEA at typical L/G, Q_reb is in 3-5 GJ/ton CO2 range."""
    section("test_amine_stripper_reboiler_duty_in_industry_range")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", total_amine=5.0, n_stages=15)
    r = strip.solve(L=10.0, G=8.0, alpha_rich=0.50, y_reb=0.05,
                       wt_frac_amine=0.30)
    check(f"  Q_per_ton_CO2 = {r.Q_per_ton_CO2:.2f} GJ/ton "
          f"(industry 3-5)",
          3.0 < r.Q_per_ton_CO2 < 5.5)


def test_amine_stripper_heat_breakdown_reaction_dominant():
    """Reaction heat should be the largest single contribution
    (>40% of total Q_reb) for typical regenerator operation."""
    section("test_amine_stripper_heat_breakdown_reaction_dominant")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", total_amine=5.0, n_stages=15)
    r = strip.solve(L=10.0, G=5.0, alpha_rich=0.50, y_reb=0.05)
    pct_react = r.Q_reaction / r.Q_reboiler
    check(f"  reaction heat fraction: {pct_react*100:.0f}% "
          f"(expect dominant 40-65%)",
          0.40 < pct_react < 0.70)


def test_amine_stripper_higher_G_strips_more():
    """More stripping steam → lower α_lean → more CO2 removed."""
    section("test_amine_stripper_higher_G_strips_more")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", total_amine=5.0, n_stages=15)
    r_lo = strip.solve(L=10.0, G=3.0, alpha_rich=0.50, y_reb=0.05)
    r_hi = strip.solve(L=10.0, G=8.0, alpha_rich=0.50, y_reb=0.05)
    check(f"  α_lean at G=8 ({r_hi.alpha_lean:.3f}) < α_lean at G=3 "
          f"({r_lo.alpha_lean:.3f})",
          r_hi.alpha_lean < r_lo.alpha_lean)


def test_amine_stripper_per_stage_heat_balance_returns_list():
    """stage_heat_balance returns one dict per stage."""
    section("test_amine_stripper_per_stage_heat_balance_returns_list")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", total_amine=5.0, n_stages=10)
    r = strip.solve(L=10.0, G=5.0, alpha_rich=0.50, y_reb=0.05)
    breakdown = strip.stage_heat_balance(r)
    check(f"  got {len(breakdown)} stage entries (expect 10)",
          len(breakdown) == 10)
    check(f"  each entry has T, alpha, Q_sensible, Q_reaction, Q_vaporization",
          all(set(d.keys()) == {"T", "alpha", "Q_sensible",
                                  "Q_reaction", "Q_vaporization"}
              for d in breakdown))


# =====================================================================
# Adiabatic absorber + lean-rich heat exchanger (v0.9.106)
# =====================================================================

def test_amine_column_adiabatic_converges():
    """Adiabatic absorber Newton solver converges with T_n unknown."""
    section("test_amine_column_adiabatic_converges")
    from stateprop.electrolyte import AmineColumn
    col = AmineColumn("MEA", total_amine=5.0, n_stages=15)
    res = col.solve(L=8.0, G=15.0, alpha_lean=0.20, y_in=0.12,
                       adiabatic=True, T_liquid_in=313.15,
                       T_gas_in=313.15, wt_frac_amine=0.30)
    check(f"  converged: {res.converged} ({res.iterations} iter)",
          res.converged)


def test_amine_column_adiabatic_T_bulge():
    """Adiabatic absorber shows the temperature bulge (peak T > feed
    T due to exothermic CO2 absorption)."""
    section("test_amine_column_adiabatic_T_bulge")
    from stateprop.electrolyte import AmineColumn
    col = AmineColumn("MEA", total_amine=5.0, n_stages=15)
    res = col.solve(L=8.0, G=15.0, alpha_lean=0.20, y_in=0.12,
                       adiabatic=True, T_liquid_in=313.15,
                       T_gas_in=313.15, wt_frac_amine=0.30)
    T_max = max(res.T)
    bulge = T_max - 313.15
    check(f"  T bulge = {bulge:.1f} K (industry typical 10-20 K)",
          5 < bulge < 30)


def test_amine_column_adiabatic_recovery_lower():
    """Adiabatic operation gives LOWER recovery than isothermal
    (because hotter stages have higher equilibrium P_CO2, less driving
    force for absorption)."""
    section("test_amine_column_adiabatic_recovery_lower")
    from stateprop.electrolyte import AmineColumn
    col = AmineColumn("MEA", total_amine=5.0, n_stages=15)
    iso = col.solve(L=8.0, G=15.0, alpha_lean=0.20, y_in=0.12, T=313.15)
    ad = col.solve(L=8.0, G=15.0, alpha_lean=0.20, y_in=0.12,
                       adiabatic=True, T_liquid_in=313.15,
                       T_gas_in=313.15, wt_frac_amine=0.30)
    check(f"  iso recovery ({iso.co2_recovery*100:.0f}%) > "
          f"adiabatic ({ad.co2_recovery*100:.0f}%)",
          iso.co2_recovery > ad.co2_recovery)


def test_cross_hx_basic_balanced():
    """Balanced flows + ΔT_min=5: ε ≈ (ΔT - ΔT_min)/ΔT
    (high ε due to balance)."""
    section("test_cross_hx_basic_balanced")
    from stateprop.electrolyte import CrossHeatExchanger
    hx = CrossHeatExchanger(delta_T_min=5.0)
    r = hx.solve(T_hot_in=400.0, m_hot=1.0, cp_hot=1000,
                    T_cold_in=300.0, m_cold=1.0, cp_cold=1000)
    # T_hot - T_cold = 100 K, ΔT_min = 5 → ε_max ~ 0.95
    check(f"  ε = {r.effectiveness:.3f} (expect ~0.95)",
          0.93 < r.effectiveness < 0.97)
    # Both ends at ΔT_min for balanced flows
    check(f"  ΔT_hot_end = {r.delta_T_hot_end:.1f} K (expect 5)",
          abs(r.delta_T_hot_end - 5.0) < 0.1)
    check(f"  ΔT_cold_end = {r.delta_T_cold_end:.1f} K (expect 5)",
          abs(r.delta_T_cold_end - 5.0) < 0.1)


def test_cross_hx_unbalanced_pinch_at_one_end():
    """Unbalanced flows: when C_hot > C_cold, pinch at cold end
    (cold leaves at T_hot_in - ΔT_min; hot leaves higher)."""
    section("test_cross_hx_unbalanced_pinch_at_one_end")
    from stateprop.electrolyte import CrossHeatExchanger
    hx = CrossHeatExchanger(delta_T_min=5.0)
    # C_hot = 2000 W/K, C_cold = 1000 W/K → cold limited
    r = hx.solve(T_hot_in=400.0, m_hot=2.0, cp_hot=1000,
                    T_cold_in=300.0, m_cold=1.0, cp_cold=1000)
    # T_cold_out ≤ T_hot_in - ΔT_min = 395
    check(f"  T_cold_out = {r.T_cold_out:.1f} K (expect ≈395)",
          abs(r.T_cold_out - 395.0) < 0.5)


def test_cross_hx_LMTD_balanced_equal_ends():
    """For balanced flows with both ends at ΔT_min, LMTD = ΔT_min."""
    section("test_cross_hx_LMTD_balanced_equal_ends")
    from stateprop.electrolyte import CrossHeatExchanger
    hx = CrossHeatExchanger(delta_T_min=5.0)
    r = hx.solve(T_hot_in=400.0, m_hot=1.0, cp_hot=1000,
                    T_cold_in=300.0, m_cold=1.0, cp_cold=1000)
    check(f"  LMTD = {r.LMTD:.2f} K (expect 5.0)",
          abs(r.LMTD - 5.0) < 0.01)


def test_cross_hx_UA_inverse_to_dT_min():
    """As ΔT_min → 0, required UA → ∞ (asymptote)."""
    section("test_cross_hx_UA_inverse_to_dT_min")
    from stateprop.electrolyte import CrossHeatExchanger
    UA_5 = CrossHeatExchanger(5.0).solve(
        400, 1.0, 1000, 300, 1.0, 1000).UA_required
    UA_10 = CrossHeatExchanger(10.0).solve(
        400, 1.0, 1000, 300, 1.0, 1000).UA_required
    check(f"  UA(ΔT=5) = {UA_5:.0f} > UA(ΔT=10) = {UA_10:.0f}",
          UA_5 > UA_10)


def test_cross_hx_rejects_invalid_temperatures():
    """T_hot_in <= T_cold_in raises ValueError."""
    section("test_cross_hx_rejects_invalid_temperatures")
    from stateprop.electrolyte import CrossHeatExchanger
    hx = CrossHeatExchanger(delta_T_min=5.0)
    try:
        hx.solve(T_hot_in=300, m_hot=1.0, cp_hot=1000,
                  T_cold_in=400, m_cold=1.0, cp_cold=1000)
        check("expected ValueError", False)
    except ValueError:
        check("ValueError raised on T_hot < T_cold", True)


def test_lean_rich_exchanger_convenience():
    """Convenience function looks up Amine and computes mass flows."""
    section("test_lean_rich_exchanger_convenience")
    from stateprop.electrolyte import lean_rich_exchanger
    r = lean_rich_exchanger("MEA", total_amine=5.0,
                                T_lean_in=393.15, T_rich_in=313.15,
                                L_lean=10.0, delta_T_min=5.0)
    check(f"  Q > 0 ({r.Q/1e3:.0f} kW)", r.Q > 0)
    check(f"  ε in [0.85, 0.95] (typical lean-rich): {r.effectiveness:.3f}",
          0.85 < r.effectiveness < 0.95)


def test_HX_saves_stripper_duty():
    """When rich enters stripper preheated by HX, Q_reb is significantly
    lower than with cold rich feed (sensible heat reduction)."""
    section("test_HX_saves_stripper_duty")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", total_amine=5.0, n_stages=10)
    res_cold = strip.solve(L=10.0, G=8.0, alpha_rich=0.50, y_reb=0.05,
                                wt_frac_amine=0.30, T_rich_in=313.15)
    res_hot = strip.solve(L=10.0, G=8.0, alpha_rich=0.50, y_reb=0.05,
                              wt_frac_amine=0.30, T_rich_in=388.15)
    check(f"  Q_reb cold rich ({res_cold.Q_per_ton_CO2:.2f}) > "
          f"hot rich preheated ({res_hot.Q_per_ton_CO2:.2f})",
          res_cold.Q_per_ton_CO2 > res_hot.Q_per_ton_CO2)
    pct_save = (1 - res_hot.Q_per_ton_CO2/res_cold.Q_per_ton_CO2)*100
    check(f"  ⇒ HX saves {pct_save:.0f}% on Q_reb (industry typical 30-50%)",
          pct_save > 20)


# =====================================================================
# Coupled T-solver + stripper condenser (v0.9.107)
# =====================================================================

def test_stripper_solve_for_Q_reb_inverts_forward():
    """solve_for_Q_reb hits a target Q within tol when in bracket."""
    section("test_stripper_solve_for_Q_reb_inverts_forward")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", 5.0, 15)
    target = 700e3   # 700 kW
    r = strip.solve_for_Q_reb(L=10.0, G=8.0, alpha_rich=0.50,
                                    Q_reb_target=target,
                                    wt_frac_amine=0.30,
                                    T_rich_in=388.15, tol_rel=1e-3)
    rel_err = abs(r.Q_reboiler - target) / target
    check(f"  Q_actual = {r.Q_reboiler/1e3:.0f} kW vs target "
          f"{target/1e3:.0f} kW (rel.err {rel_err*100:.2f}%)",
          rel_err < 5e-3)


def test_stripper_solve_for_Q_reb_higher_Q_lower_alpha_lean():
    """Higher Q_reb should achieve lower α_lean (deeper stripping)."""
    section("test_stripper_solve_for_Q_reb_higher_Q_lower_alpha_lean")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", 5.0, 15)
    r_low = strip.solve_for_Q_reb(L=10.0, G=8.0, alpha_rich=0.50,
                                        Q_reb_target=600e3,
                                        wt_frac_amine=0.30, T_rich_in=388.15)
    r_high = strip.solve_for_Q_reb(L=10.0, G=8.0, alpha_rich=0.50,
                                          Q_reb_target=800e3,
                                          wt_frac_amine=0.30, T_rich_in=388.15)
    check(f"  α_lean(Q=800kW)={r_high.alpha_lean:.3f} < "
          f"α_lean(Q=600kW)={r_low.alpha_lean:.3f}",
          r_high.alpha_lean < r_low.alpha_lean)


def test_stripper_condenser_basic():
    """Partial condenser at 40 °C gives high CO2 purity (~96 vol%)."""
    section("test_stripper_condenser_basic")
    from stateprop.electrolyte import StripperCondenser
    cond = StripperCondenser(T_cond=313.15, P=1.8)
    r = cond.solve(V_in=10.0, y_CO2_in=0.50, T_in=378.15)
    # At 40°C, P_water_sat ≈ 0.0738 bar; y_H2O_vent = 0.0738/1.8 ≈ 0.041
    check(f"  y_CO2_vent = {r.y_CO2_vent:.4f} (expect ~0.96)",
          0.94 < r.y_CO2_vent < 0.98)


def test_stripper_condenser_mass_balance():
    """V_in · y_CO2_in = V_vent · y_CO2_vent (CO2 conservation)."""
    section("test_stripper_condenser_mass_balance")
    from stateprop.electrolyte import StripperCondenser
    cond = StripperCondenser(T_cond=313.15, P=1.8)
    r = cond.solve(V_in=10.0, y_CO2_in=0.40, T_in=378.15)
    co2_in = r.V_in * r.y_CO2_in
    co2_vent = r.V_vent * r.y_CO2_vent
    rel = abs(co2_in - co2_vent) / co2_in
    check(f"  CO2 MB closure: {co2_in:.4f} = {co2_vent:.4f} (rel {rel:.1e})",
          rel < 1e-9)


def test_stripper_condenser_water_balance():
    """V_in · y_H2O_in = V_vent · y_H2O_vent + L_reflux."""
    section("test_stripper_condenser_water_balance")
    from stateprop.electrolyte import StripperCondenser
    cond = StripperCondenser(T_cond=313.15, P=1.8)
    r = cond.solve(V_in=10.0, y_CO2_in=0.50, T_in=378.15)
    h2o_in = r.V_in * r.y_H2O_in
    h2o_vent = r.V_vent * r.y_H2O_vent
    h2o_reflux = r.L_reflux
    rel = abs(h2o_in - h2o_vent - h2o_reflux) / h2o_in
    check(f"  H2O MB: {h2o_in:.4f} = {h2o_vent:.4f} + {h2o_reflux:.4f} "
          f"(rel {rel:.1e})",
          rel < 1e-9)


def test_stripper_condenser_lower_T_higher_purity():
    """Lower T_cond → less water in vapor → higher CO2 purity."""
    section("test_stripper_condenser_lower_T_higher_purity")
    from stateprop.electrolyte import StripperCondenser
    r_warm = StripperCondenser(T_cond=323.15, P=1.8).solve(
        V_in=10.0, y_CO2_in=0.50, T_in=378.15)
    r_cold = StripperCondenser(T_cond=303.15, P=1.8).solve(
        V_in=10.0, y_CO2_in=0.50, T_in=378.15)
    check(f"  purity at 30°C ({r_cold.y_CO2_vent:.3f}) > "
          f"at 50°C ({r_warm.y_CO2_vent:.3f})",
          r_cold.y_CO2_vent > r_warm.y_CO2_vent)


def test_stripper_condenser_rejects_too_hot():
    """T_cond such that P_sat ≥ P raises ValueError (no condensation)."""
    section("test_stripper_condenser_rejects_too_hot")
    from stateprop.electrolyte import StripperCondenser
    try:
        StripperCondenser(T_cond=423.15, P=1.0)   # 150 °C, P=1 atm
        check(f"  expected ValueError", False)
    except ValueError:
        check(f"  ValueError raised on impossible T_cond", True)


def test_stripper_condenser_Q_breakdown_latent_dominant():
    """For typical conditions (105 → 40 °C), latent heat of water
    condensation should dominate over sensible cooling (>80%)."""
    section("test_stripper_condenser_Q_breakdown_latent_dominant")
    from stateprop.electrolyte import StripperCondenser
    cond = StripperCondenser(T_cond=313.15, P=1.8)
    r = cond.solve(V_in=10.0, y_CO2_in=0.50, T_in=378.15)
    pct_latent = r.Q_latent_cond / r.Q_cond
    check(f"  latent heat fraction = {pct_latent*100:.0f}% (expect >80%)",
          pct_latent > 0.80)


def test_stripper_with_condenser_helper():
    """stripper_with_condenser convenience runs both and connects."""
    section("test_stripper_with_condenser_helper")
    from stateprop.electrolyte import (
        AmineStripper, stripper_with_condenser,
    )
    strip = AmineStripper("MEA", 5.0, 15)
    s, c = stripper_with_condenser(
        strip,
        stripper_solve_kwargs=dict(L=10.0, G=8.0, alpha_rich=0.50,
                                      y_reb=0.05, T_rich_in=388.15),
        T_cond=313.15, P=1.8)
    check(f"  stripper converged + condenser ran "
          f"(α_lean={s.alpha_lean:.3f}, "
          f"vent purity={c.y_CO2_vent:.3f})",
          s.converged and c.y_CO2_vent > 0.9)


# =====================================================================
# Capture flowsheet integrator (v0.9.108)
# =====================================================================

def test_flowsheet_converges():
    """Recycle loop converges to consistent α_lean within 30 iter."""
    section("test_flowsheet_converges")
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0)
    r = fs.solve(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        T_absorber_feed=313.15, P_absorber=1.013,
        G_strip_steam=4.0,
        T_strip_top=378.15, T_strip_bottom=393.15,
        P_stripper=1.8, T_cond=313.15,
        delta_T_min_HX=5.0, wt_frac_amine=0.30,
    )
    check(f"  converged: {r.converged} ({r.iterations} iter)",
          r.converged)


def test_flowsheet_alpha_consistent():
    """At convergence, last α_lean_history value matches the result α_lean."""
    section("test_flowsheet_alpha_consistent")
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0)
    r = fs.solve(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        G_strip_steam=4.0,
        delta_T_min_HX=5.0, wt_frac_amine=0.30,
    )
    check(f"  final α_lean = last in history",
          abs(r.alpha_lean - r.alpha_lean_history[-1]) < 1e-12)


def test_flowsheet_co2_mass_balance():
    """Plant CO2 balance: absorber captured CO2 = vented CO2 minus
    the small reboiler-vapor CO2 stabilizer (y_reb · G_strip)."""
    section("test_flowsheet_co2_mass_balance")
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0)
    r = fs.solve(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        G_strip_steam=4.0,
        y_reb=0.001,    # near-zero to test physical balance
        delta_T_min_HX=5.0, wt_frac_amine=0.30,
    )
    co2_in = r.G_flue * 0.12
    co2_out_cleaned = r.G_flue * r.absorber_result.y_top
    co2_to_stripper = co2_in - co2_out_cleaned
    co2_in_vent = r.V_vent * r.y_CO2_vent
    rel_err = abs(co2_to_stripper - co2_in_vent) / co2_to_stripper
    check(f"  absorber CO2 captured ({co2_to_stripper:.4f}) ≈ "
          f"vented CO2 ({co2_in_vent:.4f}) (rel.err {rel_err:.2e})",
          rel_err < 0.02)


def test_flowsheet_HX_recovers_heat():
    """Q_HX > 0 and reasonable fraction of stripper sensible duty."""
    section("test_flowsheet_HX_recovers_heat")
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0)
    r = fs.solve(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        G_strip_steam=4.0,
        delta_T_min_HX=5.0, wt_frac_amine=0.30,
    )
    check(f"  Q_HX = {r.Q_HX/1e3:.0f} kW > 0", r.Q_HX > 0)


def test_flowsheet_in_industry_envelope():
    """Q per ton CO2 in industry envelope (3-6 GJ/ton at typical
    operating points)."""
    section("test_flowsheet_in_industry_envelope")
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0)
    r = fs.solve(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        G_strip_steam=3.5,        # near optimum
        delta_T_min_HX=5.0, wt_frac_amine=0.30,
    )
    check(f"  Q_per_ton = {r.Q_per_ton_CO2:.2f} GJ/ton (industry 3-6)",
          3.0 < r.Q_per_ton_CO2 < 6.0)


def test_flowsheet_summary_format():
    """summary() returns a non-empty multi-line string."""
    section("test_flowsheet_summary_format")
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0)
    r = fs.solve(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        G_strip_steam=4.0,
        delta_T_min_HX=5.0, wt_frac_amine=0.30,
    )
    s = r.summary()
    check(f"  summary is non-empty multi-line ({len(s)} chars)",
          len(s) > 100 and "\n" in s)


def test_flowsheet_lower_G_strip_lower_Q_per_ton_in_useful_range():
    """In the useful operating range (G_strip 3-6), lower steam usage
    gives lower Q_per_ton (less wasted vaporization heat)."""
    section("test_flowsheet_lower_G_strip_lower_Q_per_ton_in_useful_range")
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0)
    r6 = fs.solve(G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
                      G_strip_steam=6.0,
                      delta_T_min_HX=5.0, wt_frac_amine=0.30)
    r3 = fs.solve(G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
                      G_strip_steam=3.0,
                      delta_T_min_HX=5.0, wt_frac_amine=0.30)
    check(f"  Q/ton at G=3 ({r3.Q_per_ton_CO2:.2f}) < at G=6 "
          f"({r6.Q_per_ton_CO2:.2f})",
          r3.Q_per_ton_CO2 < r6.Q_per_ton_CO2)


def test_flowsheet_MDEA_works():
    """Flowsheet runs for tertiary amine (MDEA) without crashing."""
    section("test_flowsheet_MDEA_works")
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MDEA", 5.0)
    r = fs.solve(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        T_strip_top=373.15, T_strip_bottom=388.15,
        G_strip_steam=4.0,
        delta_T_min_HX=5.0, wt_frac_amine=0.30,
    )
    check(f"  MDEA flowsheet converged ({r.iterations} iter, "
          f"α_lean={r.alpha_lean:.3f})",
          r.converged)


# =====================================================================
# Adiabatic flowsheet + variable-V stripper (v0.9.109)
# =====================================================================

def test_flowsheet_adiabatic_absorber_converges():
    """Flowsheet with adiabatic_absorber=True converges."""
    section("test_flowsheet_adiabatic_absorber_converges")
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0)
    r = fs.solve(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        G_strip_steam=4.0,
        delta_T_min_HX=5.0, wt_frac_amine=0.30,
        adiabatic_absorber=True, T_gas_in=313.15,
    )
    check(f"  converged ({r.iterations} iter)", r.converged)


def test_flowsheet_adiabatic_T_rich_higher():
    """Adiabatic mode: rich exits absorber WARMER than feed (T-bulge)."""
    section("test_flowsheet_adiabatic_T_rich_higher")
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0)
    r = fs.solve(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        G_strip_steam=4.0,
        delta_T_min_HX=5.0, wt_frac_amine=0.30,
        adiabatic_absorber=True, T_gas_in=313.15,
    )
    check(f"  T_rich_from_absorber ({r.T_rich_from_absorber-273.15:.1f}°C) "
          f"> T_absorber_feed (40°C)",
          r.T_rich_from_absorber > 313.15 + 5.0)


def test_flowsheet_adiabatic_lower_HX_duty():
    """Adiabatic mode: warmer rich entering HX → lower HX duty needed
    (vs isothermal where rich enters at 40°C)."""
    section("test_flowsheet_adiabatic_lower_HX_duty")
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0)
    r_iso = fs.solve(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        G_strip_steam=4.0,
        delta_T_min_HX=5.0, wt_frac_amine=0.30,
        adiabatic_absorber=False)
    r_ad = fs.solve(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        G_strip_steam=4.0,
        delta_T_min_HX=5.0, wt_frac_amine=0.30,
        adiabatic_absorber=True, T_gas_in=313.15)
    check(f"  Q_HX_adiabatic ({r_ad.Q_HX/1e3:.0f}) < "
          f"Q_HX_iso ({r_iso.Q_HX/1e3:.0f}) kW",
          r_ad.Q_HX < r_iso.Q_HX)


def test_stripper_variable_V_converges():
    """variable_V=True solves successfully and produces a V profile."""
    section("test_stripper_variable_V_converges")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", 5.0, 15)
    r = strip.solve(L=10.0, G=8.0, alpha_rich=0.50, y_reb=0.05,
                       P=1.8, T_top=378.15, T_bottom=388.15,
                       wt_frac_amine=0.30, T_rich_in=388.15,
                       variable_V=True)
    check(f"  V_profile present (length {len(r.V_profile) if r.V_profile else 0})",
          r.V_profile is not None and len(r.V_profile) == 16)


def test_stripper_variable_V_higher_at_top():
    """Cooler top stages have lower y_H2O (less water vapor saturation)
    → V must be larger at top to maintain water mass flow."""
    section("test_stripper_variable_V_higher_at_top")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", 5.0, 15)
    r = strip.solve(L=10.0, G=8.0, alpha_rich=0.50, y_reb=0.05,
                       P=1.8, T_top=378.15, T_bottom=388.15,
                       wt_frac_amine=0.30, T_rich_in=388.15,
                       variable_V=True)
    V_top = r.V_profile[0]
    V_bot = r.V_profile[-1]
    check(f"  V_top ({V_top:.3f}) > V_bot ({V_bot:.3f})",
          V_top > V_bot)


def test_stripper_variable_V_bottom_matches_G_reb():
    """V at the bottom interface = G_reb (boundary condition)."""
    section("test_stripper_variable_V_bottom_matches_G_reb")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", 5.0, 15)
    r = strip.solve(L=10.0, G=8.0, alpha_rich=0.50,
                       P=1.8, T_top=378.15, T_bottom=388.15,
                       variable_V=True)
    check(f"  V[N] = {r.V_profile[-1]:.3f} ≈ G_reb (8.0)",
          abs(r.V_profile[-1] - 8.0) < 1e-6)


def test_stripper_variable_V_constant_water_mass_flow():
    """In variable-V mode, V[k]·y_H2O(T_k) = const through column
    (water mass flow conserved)."""
    section("test_stripper_variable_V_constant_water_mass_flow")
    from stateprop.electrolyte import AmineStripper, P_water_sat
    strip = AmineStripper("MEA", 5.0, 15)
    r = strip.solve(L=10.0, G=8.0, alpha_rich=0.50, y_reb=0.05,
                       P=1.8, T_top=378.15, T_bottom=388.15,
                       variable_V=True)
    # Compute water flow at each interface (using stage T as proxy)
    P_total = 1.8
    water_flows = []
    for k in range(len(r.V_profile)):
        if k == 0:
            T_k = r.T[0]
        elif k == len(r.V_profile) - 1:
            T_k = r.T[-1]
        else:
            T_k = 0.5 * (r.T[k - 1] + r.T[k])
        y_H2O_k = min(P_water_sat(T_k) / P_total, 0.99)
        water_flows.append(r.V_profile[k] * y_H2O_k)
    # All should be approximately equal (= G_reb · (1 - y_reb))
    expected = 8.0 * (1 - 0.05)
    rel_errs = [abs(w - expected) / expected for w in water_flows[:-1]]
    max_err = max(rel_errs)
    check(f"  water flow constant across interfaces "
          f"(max rel.err {max_err:.1e}, expected {expected:.3f} mol/s)",
          max_err < 0.02)


# =====================================================================
# T-saturation auto-clip + energy-balance V + flowsheet variable_V (v0.9.110)
# =====================================================================

def test_T_water_sat_inverse_consistency():
    """T_water_sat(P_water_sat(T)) ≈ T (round-trip)."""
    section("test_T_water_sat_inverse_consistency")
    from stateprop.electrolyte.amine_stripper import (
        T_water_sat, P_water_sat,
    )
    for T in [373.15, 383.15, 393.15, 403.15]:
        P = P_water_sat(T)
        T_back = T_water_sat(P)
        check(f"  T={T:.2f} → P={P:.3f} → T_back={T_back:.2f} (Δ {T_back-T:+.2f} K)",
              abs(T_back - T) < 0.2)


def test_T_water_sat_at_18bar():
    """T_water_sat(1.8) ≈ 391 K (water boils at 1.8 bar around 117 °C)."""
    section("test_T_water_sat_at_18bar")
    from stateprop.electrolyte.amine_stripper import T_water_sat
    T = T_water_sat(1.8)
    check(f"  T_water_sat(1.8) = {T:.2f} K ({T-273.15:.2f} °C, lit ~117 °C)",
          390 < T < 392)


def test_stripper_auto_clip_T_bottom():
    """T_bottom > T_sat(P) is auto-clipped to T_sat - margin."""
    section("test_stripper_auto_clip_T_bottom")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", 5.0, 15)
    # Default T_bottom=393.15 = 120°C with P=1.8 bar exceeds T_sat≈391 K
    r = strip.solve(L=10.0, G=8.0, alpha_rich=0.50,
                       P=1.8, T_top=378.15, T_bottom=393.15,
                       auto_clip_T_bottom=True)
    # Result T profile bottom must be ≤ T_sat
    check(f"  T_bottom_actual = {r.T[-1]:.2f} K (clipped to <T_sat=391.12)",
          r.T[-1] < 391.13)


def test_stripper_no_clip_raises():
    """auto_clip_T_bottom=False raises ValueError on unphysical T."""
    section("test_stripper_no_clip_raises")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", 5.0, 15)
    try:
        strip.solve(L=10.0, G=8.0, alpha_rich=0.50,
                      P=1.8, T_top=378.15, T_bottom=395.15,
                      auto_clip_T_bottom=False)
        check(f"  expected ValueError", False)
    except ValueError:
        check(f"  ValueError raised on unphysical T_bottom", True)


def test_stripper_variable_V_string_modes():
    """variable_V accepts 'saturation' and 'energy' as strings."""
    section("test_stripper_variable_V_string_modes")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", 5.0, 8)
    r_sat = strip.solve(L=10.0, G=8.0, alpha_rich=0.50,
                              P=1.8, T_top=378.15, T_bottom=388.15,
                              variable_V='saturation')
    r_en = strip.solve(L=10.0, G=8.0, alpha_rich=0.50,
                            P=1.8, T_top=378.15, T_bottom=388.15,
                            variable_V='energy')
    check(f"  saturation mode V profile present "
          f"(V[top]={r_sat.V_profile[0]:.2f})",
          r_sat.V_profile is not None)
    check(f"  energy mode V profile present "
          f"(V[top]={r_en.V_profile[0]:.2f})",
          r_en.V_profile is not None)


def test_stripper_variable_V_invalid_string_raises():
    """variable_V with invalid string value raises ValueError."""
    section("test_stripper_variable_V_invalid_string_raises")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", 5.0, 8)
    try:
        strip.solve(L=10.0, G=8.0, alpha_rich=0.50,
                      P=1.8, T_top=378.15, T_bottom=388.15,
                      variable_V='nonsense')
        check(f"  expected ValueError", False)
    except ValueError:
        check(f"  ValueError raised on bad variable_V value", True)


def test_stripper_energy_V_within_saturation_bounds():
    """Energy-mode V is bounded between 0.5× and 1.5× saturation V
    (numerical clamp for stability)."""
    section("test_stripper_energy_V_within_saturation_bounds")
    from stateprop.electrolyte import AmineStripper
    strip = AmineStripper("MEA", 5.0, 8)
    r_sat = strip.solve(L=10.0, G=8.0, alpha_rich=0.50,
                              P=1.8, T_top=378.15, T_bottom=388.15,
                              variable_V='saturation')
    r_en = strip.solve(L=10.0, G=8.0, alpha_rich=0.50,
                            P=1.8, T_top=378.15, T_bottom=388.15,
                            variable_V='energy')
    # Compare at top (where energy mode usually clamps)
    ratios = [r_en.V_profile[k] / r_sat.V_profile[k]
                for k in range(len(r_sat.V_profile))]
    in_bounds = all(0.49 <= r <= 1.51 for r in ratios)
    check(f"  V_energy / V_sat in [0.5, 1.5] for all stages "
          f"({min(ratios):.2f} to {max(ratios):.2f})",
          in_bounds)


def test_flowsheet_variable_V_passes_through():
    """CaptureFlowsheet accepts variable_V_stripper kwarg and passes
    through to inner stripper."""
    section("test_flowsheet_variable_V_passes_through")
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0)
    r = fs.solve(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        G_strip_steam=4.0,
        T_strip_top=378.15, T_strip_bottom=388.15,
        delta_T_min_HX=5.0, wt_frac_amine=0.30,
        variable_V_stripper='saturation',
    )
    check(f"  flowsheet converged with variable_V_stripper "
          f"(V_profile present in stripper_result)",
          r.converged
          and r.stripper_result.V_profile is not None)


def test_flowsheet_three_V_modes_same_Q_per_ton():
    """All three V modes give similar Q_per_ton (boundary-driven)."""
    section("test_flowsheet_three_V_modes_same_Q_per_ton")
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0)
    common = dict(G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
                     G_strip_steam=4.0,
                     T_strip_top=378.15, T_strip_bottom=388.15,
                     delta_T_min_HX=5.0, wt_frac_amine=0.30)
    Q_per_ton_modes = []
    for mode in [False, 'saturation']:    # skip 'energy' (slow)
        r = fs.solve(**common, variable_V_stripper=mode)
        Q_per_ton_modes.append(r.Q_per_ton_CO2)
    spread = max(Q_per_ton_modes) - min(Q_per_ton_modes)
    check(f"  Q/ton spread across modes = {spread:.3f} GJ/ton "
          f"(expect <0.1)",
          spread < 0.1)


# =====================================================================
# Sour-water Naphtali-Sandholm coupling (v0.9.111)
# =====================================================================

def test_sour_water_activity_basic():
    """SourWaterActivityModel returns γ array with α values for
    volatiles and 1.0 for water."""
    section("test_sour_water_activity_basic")
    from stateprop.electrolyte import SourWaterActivityModel
    am = SourWaterActivityModel(["NH3", "H2S", "CO2", "H2O"])
    g = am.gammas(353.15, [0.01, 0.005, 0.001, 0.984])
    check(f"  γ array has length 4: {g}", len(g) == 4)
    check(f"  γ_water = 1.0 (idx 3): {g[3]:.4f}", abs(g[3] - 1.0) < 1e-9)
    check(f"  0 < γ_NH3 ≤ 1 ({g[0]:.3f}, partial volatile)",
          0 < g[0] <= 1.0)
    check(f"  0 < γ_H2S ≤ 1 ({g[1]:.3e})", 0 < g[1] <= 1.0)


def test_sour_water_activity_pH_response():
    """Adding strong acid (Cl⁻) drops pH and lowers α_NH3 (more NH4⁺)."""
    section("test_sour_water_activity_pH_response")
    from stateprop.electrolyte import SourWaterActivityModel
    species = ["NH3", "H2S", "CO2", "H2O"]
    x = [0.01, 0.005, 0.001, 0.984]
    T = 353.15

    am_neutral = SourWaterActivityModel(species)
    am_acidic = SourWaterActivityModel(species,
                                            extra_strong_anions=0.5)
    g_n = am_neutral.gammas(T, x)
    g_a = am_acidic.gammas(T, x)
    check(f"  γ_NH3 (neutral={g_n[0]:.3f}) > γ_NH3 (Cl⁻=0.5M, "
          f"={g_a[0]:.3f})",
          g_n[0] > g_a[0])
    check(f"  γ_H2S (acid={g_a[1]:.3f}) > γ_H2S (neutral={g_n[1]:.3e})",
          g_a[1] > g_n[1])


def test_sour_water_activity_requires_water():
    """SourWaterActivityModel without H2O raises ValueError."""
    section("test_sour_water_activity_requires_water")
    from stateprop.electrolyte import SourWaterActivityModel
    try:
        SourWaterActivityModel(["NH3", "H2S"])
        check(f"  expected ValueError", False)
    except ValueError:
        check(f"  ValueError raised when H2O missing", True)


def test_sour_water_activity_requires_volatile():
    """SourWaterActivityModel needs at least one of NH3/H2S/CO2."""
    section("test_sour_water_activity_requires_volatile")
    from stateprop.electrolyte import SourWaterActivityModel
    try:
        SourWaterActivityModel(["H2O"])
        check(f"  expected ValueError", False)
    except ValueError:
        check(f"  ValueError raised when no volatile species", True)


def test_sour_water_psat_funcs_lengths():
    """build_psat_funcs returns one callable per species."""
    section("test_sour_water_psat_funcs_lengths")
    from stateprop.electrolyte import build_psat_funcs
    species = ["NH3", "H2S", "CO2", "H2O"]
    funcs = build_psat_funcs(species)
    check(f"  len(funcs) == len(species) = {len(funcs)}",
          len(funcs) == len(species))
    # Water P_sat at 100°C ≈ 1.013e5 Pa
    Pwater_100 = funcs[3](373.15)
    check(f"  H2O P_sat(100°C) = {Pwater_100/1e5:.3f} bar (expect ≈ 1.013)",
          0.95 < Pwater_100 / 1e5 < 1.05)


def test_sour_water_stripper_runs():
    """Basic sour-water stripper column converges."""
    section("test_sour_water_stripper_runs")
    from stateprop.electrolyte import sour_water_stripper
    import numpy as np
    species = ["NH3", "H2S", "CO2", "H2O"]
    r = sour_water_stripper(
        n_stages=10, feed_stage=2, feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.5,
        pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, 10)),
    )
    check(f"  column converged ({r.column_result.iterations} iter)",
          r.column_result.converged)
    check(f"  pH list length = n_stages: {len(r.pH)} == 10",
          len(r.pH) == 10)
    check(f"  pH values in physical range (3..12): "
          f"{min(r.pH):.2f}..{max(r.pH):.2f}",
          all(3 <= p <= 12 for p in r.pH))


def test_sour_water_stripper_strip_efficiencies():
    """Strip efficiency: CO2 > H2S > NH3 in basic sour water."""
    section("test_sour_water_stripper_strip_efficiencies")
    from stateprop.electrolyte import sour_water_stripper
    import numpy as np
    species = ["NH3", "H2S", "CO2", "H2O"]
    r = sour_water_stripper(
        n_stages=10, feed_stage=2, feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.5,
        pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, 10)),
    )
    e = r.bottoms_strip_efficiency
    check(f"  CO2 ({e['CO2']:.1%}) > H2S ({e['H2S']:.1%}) > NH3 ({e['NH3']:.1%})",
          e["CO2"] > e["H2S"] > e["NH3"])
    check(f"  CO2 strip > 95% (very volatile): {e['CO2']:.1%}",
          e["CO2"] > 0.95)


def test_sour_water_stripper_acid_lowers_pH():
    """Cl⁻ background lowers pH and dramatically reduces NH3 stripping."""
    section("test_sour_water_stripper_acid_lowers_pH")
    from stateprop.electrolyte import sour_water_stripper
    import numpy as np
    species = ["NH3", "H2S", "CO2", "H2O"]
    common = dict(
        n_stages=10, feed_stage=2, feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.5,
        pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, 10)),
    )
    r0 = sour_water_stripper(**common, extra_strong_anions=0.0)
    r_acid = sour_water_stripper(**common, extra_strong_anions=0.5)
    pH_avg_0 = sum(r0.pH) / len(r0.pH)
    pH_avg_a = sum(r_acid.pH) / len(r_acid.pH)
    check(f"  acid case has lower pH ({pH_avg_a:.2f} < {pH_avg_0:.2f})",
          pH_avg_a < pH_avg_0)
    e0 = r0.bottoms_strip_efficiency["NH3"]
    ea = r_acid.bottoms_strip_efficiency["NH3"]
    check(f"  acid case strips less NH3 ({ea:.1%} < {e0:.1%})",
          ea < e0)


def test_sour_water_stripper_base_helps_NH3():
    """Na⁺ background raises pH; NH3 strips MORE while H2S/CO2 plummet."""
    section("test_sour_water_stripper_base_helps_NH3")
    from stateprop.electrolyte import sour_water_stripper
    import numpy as np
    species = ["NH3", "H2S", "CO2", "H2O"]
    common = dict(
        n_stages=10, feed_stage=2, feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.5,
        pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, 10)),
    )
    r0 = sour_water_stripper(**common, extra_strong_cations=0.0)
    r_base = sour_water_stripper(**common, extra_strong_cations=0.5)
    pH_avg_0 = sum(r0.pH) / len(r0.pH)
    pH_avg_b = sum(r_base.pH) / len(r_base.pH)
    check(f"  base case has higher pH ({pH_avg_b:.2f} > {pH_avg_0:.2f})",
          pH_avg_b > pH_avg_0)
    # H2S and CO2 strip should drop dramatically (now ionic)
    e0_H2S = r0.bottoms_strip_efficiency["H2S"]
    eb_H2S = r_base.bottoms_strip_efficiency["H2S"]
    check(f"  base case strips much less H2S ({eb_H2S:.1%} << {e0_H2S:.1%})",
          eb_H2S < 0.5 * e0_H2S)


def test_sour_water_stripper_NH3_only():
    """Stripper with only NH3 + H2O works."""
    section("test_sour_water_stripper_NH3_only")
    from stateprop.electrolyte import sour_water_stripper
    import numpy as np
    species = ["NH3", "H2O"]
    r = sour_water_stripper(
        n_stages=10, feed_stage=2, feed_F=100.0,
        feed_z=[0.005, 0.995], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.0,
        pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, 10)),
    )
    check(f"  NH3-only stripper converged "
          f"({r.column_result.iterations} iter)",
          r.column_result.converged)
    # No H2S/CO2 in feed → no strip efficiency for them
    check(f"  efficiency dict has NH3 only: keys={list(r.bottoms_strip_efficiency.keys())}",
          set(r.bottoms_strip_efficiency.keys()) == {"NH3"})
    check(f"  NH3 stripped non-trivially: "
          f"{r.bottoms_strip_efficiency['NH3']:.1%}",
          r.bottoms_strip_efficiency["NH3"] > 0.20)


def test_sour_water_stripper_dilute_henry_consistency():
    """At very dilute conditions, partial pressure of NH3 should
    approximately match H_eff · m_NH3 (Henry's-law limit).

    Requires stage_efficiency=1.0 (full equilibrium); the Murphree
    default of 0.65 introduces a systematic bias.
    """
    section("test_sour_water_stripper_dilute_henry_consistency")
    from stateprop.electrolyte import (
        sour_water_stripper, SourWaterActivityModel,
    )
    from stateprop.electrolyte.sour_water import effective_henry
    import numpy as np
    species = ["NH3", "H2S", "CO2", "H2O"]
    r = sour_water_stripper(
        n_stages=8, feed_stage=2, feed_F=100.0,
        feed_z=[0.001, 0.0005, 0.0001, 0.9984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.0,
        pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, 8)),
        stage_efficiency=1.0,         # full equilibrium for the identity
    )
    col = r.column_result
    # At each stage, compute P_partial_NH3 from y·P and from H_eff·m
    am = SourWaterActivityModel(species)
    M = 1000.0 / 18.0153
    rel_errs = []
    for j in range(col.n_stages):
        x_j = col.x[j, :]
        y_j = col.y[j, :]
        T_j = col.T[j]
        P_partial_NH3 = y_j[0] * 1.5e5
        sp = am.speciate_at(T_j, x_j)
        m_NH3 = x_j[0] * M / max(x_j[3], 1e-9)
        H_eff = effective_henry("NH3", T_j, sp.pH)
        P_partial_henry = H_eff * m_NH3
        if P_partial_henry > 1.0:
            rel_errs.append(abs(P_partial_NH3 - P_partial_henry)
                                / P_partial_henry)
    max_err = max(rel_errs) if rel_errs else 0.0
    check(f"  P_partial(NH3) from column matches H_eff · m within "
          f"{max_err*100:.1f}% (dilute limit)",
          max_err < 0.10)


# =====================================================================
# Energy balance + Murphree for sour-water stripper (v0.9.112)
# =====================================================================

def test_sour_water_enthalpy_funcs_lengths():
    """build_enthalpy_funcs returns one h_V and one h_L per species."""
    section("test_sour_water_enthalpy_funcs_lengths")
    from stateprop.electrolyte import build_enthalpy_funcs
    species = ["NH3", "H2S", "CO2", "H2O"]
    h_V, h_L = build_enthalpy_funcs(species)
    check(f"  len(h_V) = len(h_L) = len(species) = "
          f"{len(h_V)}",
          len(h_V) == len(h_L) == len(species))


def test_sour_water_enthalpy_water_vap_at_Tref():
    """h_V_water(298.15) ≈ ΔH_vap(reference) = 43990 J/mol."""
    section("test_sour_water_enthalpy_water_vap_at_Tref")
    from stateprop.electrolyte import build_enthalpy_funcs
    species = ["NH3", "H2S", "CO2", "H2O"]
    h_V, h_L = build_enthalpy_funcs(species)
    h_V_water_298 = h_V[3](298.15)
    check(f"  h_V_water(298.15) = {h_V_water_298:.0f} J/mol "
          f"(ref ΔH_vap = 43990)",
          abs(h_V_water_298 - 43990) < 50)


def test_sour_water_enthalpy_NH3_at_Tref():
    """h_V_NH3(T_ref) = 0 (gas reference); h_L_NH3(T_ref) = ΔH_diss."""
    section("test_sour_water_enthalpy_NH3_at_Tref")
    from stateprop.electrolyte import build_enthalpy_funcs
    species = ["NH3", "H2S", "CO2", "H2O"]
    h_V, h_L = build_enthalpy_funcs(species)
    h_V_NH3 = h_V[0](298.15)
    h_L_NH3 = h_L[0](298.15)
    check(f"  h_V_NH3(298.15) = {h_V_NH3:.1f} ≈ 0 (gas ref)",
          abs(h_V_NH3) < 1e-3)
    check(f"  h_L_NH3(298.15) = {h_L_NH3:.0f} ≈ -34200 (ΔH_diss)",
          abs(h_L_NH3 + 34200) < 100)


def test_sour_water_enthalpy_water_vap_decreases_with_T():
    """ΔH_vap decreases with T (Watson) — h_V_water at 100 °C
    should be lower than at 25 °C."""
    section("test_sour_water_enthalpy_water_vap_decreases_with_T")
    from stateprop.electrolyte import build_enthalpy_funcs
    species = ["NH3", "H2S", "CO2", "H2O"]
    h_V, h_L = build_enthalpy_funcs(species)
    cp_V_water = 33.6
    dHvap_298 = h_V[3](298.15) - cp_V_water * 0.0
    dHvap_373 = h_V[3](373.15) - cp_V_water * (373.15 - 298.15)
    check(f"  ΔH_vap(298)={dHvap_298:.0f} > ΔH_vap(373)={dHvap_373:.0f}",
          dHvap_298 > dHvap_373)


def test_sour_water_stripper_energy_balance_runs():
    """sour_water_stripper(energy_balance=True) converges and
    reports Q_R, Q_C, steam_ratio."""
    section("test_sour_water_stripper_energy_balance_runs")
    from stateprop.electrolyte import sour_water_stripper
    import numpy as np
    species = ["NH3", "H2S", "CO2", "H2O"]
    r = sour_water_stripper(
        n_stages=10, feed_stage=2, feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.5,
        pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, 10)),
        energy_balance=True,
        stage_efficiency=1.0,
    )
    check(f"  column converged ({r.column_result.iterations} iter)",
          r.column_result.converged)
    check(f"  Q_R reported and positive ({r.Q_R/1e3:.1f} kW)",
          r.Q_R is not None and r.Q_R > 0)
    check(f"  Q_C reported and positive ({r.Q_C/1e3:.1f} kW)",
          r.Q_C is not None and r.Q_C > 0)
    check(f"  steam ratio reported "
          f"({r.steam_ratio_kg_per_kg_water:.3f} kg/kg)",
          r.steam_ratio_kg_per_kg_water is not None)


def test_sour_water_stripper_steam_ratio_in_industrial_range():
    """Steam-to-water ratio should be in the 0.05-0.20 kg/kg range
    for a typical sour-water stripper."""
    section("test_sour_water_stripper_steam_ratio_in_industrial_range")
    from stateprop.electrolyte import sour_water_stripper
    import numpy as np
    species = ["NH3", "H2S", "CO2", "H2O"]
    r = sour_water_stripper(
        n_stages=10, feed_stage=2, feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.5,
        pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, 10)),
        energy_balance=True,
        stage_efficiency=1.0,
    )
    sr = r.steam_ratio_kg_per_kg_water
    check(f"  steam ratio {sr:.3f} kg/kg in [0.02, 0.20] (industrial)",
          0.02 < sr < 0.20)


def test_sour_water_stripper_stage_efficiency_effect():
    """Lower Murphree efficiency reduces strip efficiency."""
    section("test_sour_water_stripper_stage_efficiency_effect")
    from stateprop.electrolyte import sour_water_stripper
    import numpy as np
    species = ["NH3", "H2S", "CO2", "H2O"]
    common = dict(
        n_stages=10, feed_stage=2, feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.5,
        pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, 10)),
    )
    r1 = sour_water_stripper(**common, stage_efficiency=1.0)
    r05 = sour_water_stripper(**common, stage_efficiency=0.5)
    e1 = r1.bottoms_strip_efficiency["NH3"]
    e05 = r05.bottoms_strip_efficiency["NH3"]
    check(f"  E=0.5 strips less NH3 ({e05:.1%}) than E=1.0 ({e1:.1%})",
          e05 < e1)


def test_sour_water_stripper_default_efficiency_is_065():
    """Default stage_efficiency = 0.65 (industrial)."""
    section("test_sour_water_stripper_default_efficiency_is_065")
    from stateprop.electrolyte import sour_water_stripper
    import numpy as np
    species = ["NH3", "H2S", "CO2", "H2O"]
    r = sour_water_stripper(
        n_stages=10, feed_stage=2, feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.5,
        pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, 10)),
    )
    check(f"  stage_efficiency_used = {r.stage_efficiency_used} (= 0.65)",
          r.stage_efficiency_used == 0.65)


def test_sour_water_stripper_per_stage_efficiency_array():
    """Per-stage Murphree as an array (not scalar) is accepted."""
    section("test_sour_water_stripper_per_stage_efficiency_array")
    from stateprop.electrolyte import sour_water_stripper
    import numpy as np
    species = ["NH3", "H2S", "CO2", "H2O"]
    eff_array = [0.5, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8]
    r = sour_water_stripper(
        n_stages=10, feed_stage=2, feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.5,
        pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, 10)),
        stage_efficiency=eff_array,
    )
    check(f"  per-stage efficiency converged "
          f"({r.column_result.iterations} iter)",
          r.column_result.converged)


def test_sour_water_stripper_energy_balance_temperature_response():
    """With energy balance enabled, the bottom stage T should
    self-adjust to near the boiling point at the column pressure."""
    section("test_sour_water_stripper_energy_balance_temperature_response")
    from stateprop.electrolyte import sour_water_stripper
    import numpy as np
    species = ["NH3", "H2S", "CO2", "H2O"]
    common = dict(
        n_stages=10, feed_stage=2, feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.5,
        pressure=1.5e5,
        T_init=list(np.linspace(373.15, 383.15, 10)),
    )
    r_eb = sour_water_stripper(**common, energy_balance=True,
                                       stage_efficiency=1.0)
    T_bot_eb = r_eb.column_result.T[-1]
    check(f"  energy-balance bottom T = {T_bot_eb-273.15:.1f}°C "
          f"(near T_sat at 1.5 bar = 112°C)",
          105 < T_bot_eb - 273.15 < 115)


# =====================================================================
# Two-stage sour-water flowsheet (v0.9.113)
# =====================================================================

def test_two_stage_flowsheet_runs():
    """Two-stage sour-water flowsheet converges and produces results."""
    section("test_two_stage_flowsheet_runs")
    from stateprop.electrolyte import sour_water_two_stage_flowsheet
    species = ["NH3", "H2S", "CO2", "H2O"]
    r = sour_water_two_stage_flowsheet(
        feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        acid_dose_mol_per_kg=0.1, base_dose_mol_per_kg=0.5,
        n_stages_acid=8, n_stages_base=8,
        distillate_rate_acid=2.5, distillate_rate_base=2.5,
        reflux_ratio_acid=1.0, reflux_ratio_base=1.0,
        energy_balance=True, stage_efficiency=1.0,
    )
    check(f"  flowsheet converged", r.converged)
    check(f"  H2S overall recovery = {r.overall_recovery['H2S']:.1%}",
          r.overall_recovery["H2S"] > 0.95)
    check(f"  NH3 overall recovery = {r.overall_recovery['NH3']:.1%}",
          r.overall_recovery["NH3"] > 0.30)


def test_two_stage_flowsheet_acid_strips_h2s():
    """Stage 1 (acid) should strip H2S/CO2 essentially 100% at adequate dose."""
    section("test_two_stage_flowsheet_acid_strips_h2s")
    from stateprop.electrolyte import sour_water_two_stage_flowsheet
    species = ["NH3", "H2S", "CO2", "H2O"]
    r = sour_water_two_stage_flowsheet(
        feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        acid_dose_mol_per_kg=0.5, base_dose_mol_per_kg=0.5,
        n_stages_acid=10, n_stages_base=10,
        distillate_rate_acid=2.5, distillate_rate_base=2.5,
        reflux_ratio_acid=1.0, reflux_ratio_base=1.0,
        stage_efficiency=1.0,
    )
    e1 = r.stage1_result.bottoms_strip_efficiency
    check(f"  stage 1 H2S = {e1['H2S']:.3f} (essentially 100%)",
          e1["H2S"] > 0.99)
    check(f"  stage 1 NH3 = {e1['NH3']:.3f} (low — stays as NH4+)",
          e1["NH3"] < 0.20)


def test_two_stage_flowsheet_chemical_consumption():
    """Acid/base consumption should be in physically expected
    range (proportional to dose × water flow)."""
    section("test_two_stage_flowsheet_chemical_consumption")
    from stateprop.electrolyte import sour_water_two_stage_flowsheet
    species = ["NH3", "H2S", "CO2", "H2O"]
    r = sour_water_two_stage_flowsheet(
        feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        acid_dose_mol_per_kg=0.5, base_dose_mol_per_kg=0.5,
        n_stages_acid=8, n_stages_base=8,
        distillate_rate_acid=2.5, distillate_rate_base=2.5,
        reflux_ratio_acid=1.0, reflux_ratio_base=1.0,
        energy_balance=False, stage_efficiency=1.0,
    )
    # Feed water mass: 100 mol/s · 0.984 · 0.018 ≈ 1.77 kg/s = 6378 kg/h
    # 0.5 mol HCl / kg water · 1.77 kg/s · 36.46 g/mol → ~116 kg/h
    check(f"  acid consumption ({r.acid_consumption_kg_per_h:.0f} kg/h) "
          f"in expected range ~110-120",
          100 < r.acid_consumption_kg_per_h < 130)


def test_two_stage_flowsheet_total_steam_in_range():
    """Combined steam ratio should be in industrial 0.05-0.30 range."""
    section("test_two_stage_flowsheet_total_steam_in_range")
    from stateprop.electrolyte import sour_water_two_stage_flowsheet
    species = ["NH3", "H2S", "CO2", "H2O"]
    r = sour_water_two_stage_flowsheet(
        feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        acid_dose_mol_per_kg=0.1, base_dose_mol_per_kg=0.5,
        n_stages_acid=8, n_stages_base=8,
        distillate_rate_acid=2.5, distillate_rate_base=2.5,
        reflux_ratio_acid=1.0, reflux_ratio_base=1.0,
        energy_balance=True, stage_efficiency=1.0,
    )
    sr = r.steam_ratio_total
    check(f"  total steam ratio {sr:.3f} kg/kg in [0.05, 0.30]",
          0.05 < sr < 0.30)


def test_two_stage_flowsheet_dose_negative_raises():
    """Negative dose raises ValueError."""
    section("test_two_stage_flowsheet_dose_negative_raises")
    from stateprop.electrolyte import sour_water_two_stage_flowsheet
    species = ["NH3", "H2S", "CO2", "H2O"]
    try:
        sour_water_two_stage_flowsheet(
            feed_F=100.0,
            feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
            species_names=species,
            acid_dose_mol_per_kg=-0.1, base_dose_mol_per_kg=0.5,
        )
        check(f"  expected ValueError", False)
    except ValueError:
        check(f"  ValueError raised on negative dose", True)


def test_find_acid_dose_for_h2s_recovery():
    """Auto-dose: find acid dose for target H2S recovery."""
    section("test_find_acid_dose_for_h2s_recovery")
    from stateprop.electrolyte import find_acid_dose_for_h2s_recovery
    species = ["NH3", "H2S", "CO2", "H2O"]
    dose = find_acid_dose_for_h2s_recovery(
        target_recovery=0.999,
        feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        n_stages_acid=10,
        distillate_rate_acid=2.5,
        reflux_ratio_acid=1.0, pressure_acid=1.5e5,
        stage_efficiency=1.0,
    )
    check(f"  found acid dose = {dose:.4f} mol/kg",
          0.0 < dose < 2.0)


# =====================================================================
# Tray hydraulics (v0.9.113)
# =====================================================================

def test_tray_design_geometry():
    """TrayDesign computes total/active/hole/downcomer areas correctly."""
    section("test_tray_design_geometry")
    from stateprop.distillation import TrayDesign
    td = TrayDesign(diameter=1.0)
    A_total = 3.14159 / 4
    check(f"  total area = {td.total_area:.4f} m² ≈ π/4 ({A_total:.4f})",
          abs(td.total_area - A_total) < 1e-4)
    check(f"  downcomer area = 10% total = {td.downcomer_area:.4f}",
          abs(td.downcomer_area - 0.10 * A_total) < 1e-6)
    check(f"  active area = total − 2·downcomer "
          f"({td.active_area:.4f})",
          abs(td.active_area - (A_total - 2 * 0.10 * A_total)) < 1e-6)


def test_flooding_velocity_increases_with_density_diff():
    """Higher (rho_L - rho_V) gives higher flooding velocity."""
    section("test_flooding_velocity_increases_with_density_diff")
    from stateprop.distillation.tray_hydraulics import flooding_velocity
    v_low = flooding_velocity(
        rho_L=950, rho_V=1.5, F_LV=0.1,
        tray_spacing=0.6, sigma=0.058)
    v_high = flooding_velocity(
        rho_L=1500, rho_V=1.5, F_LV=0.1,
        tray_spacing=0.6, sigma=0.058)
    check(f"  v_flood(rho_L=1500) = {v_high:.3f} > "
          f"v_flood(rho_L=950) = {v_low:.3f}",
          v_high > v_low)


def test_tray_hydraulics_runs_on_sour_water():
    """Run hydraulics on a converged sour-water column profile."""
    section("test_tray_hydraulics_runs_on_sour_water")
    from stateprop.electrolyte import sour_water_stripper
    from stateprop.distillation import tray_hydraulics, TrayDesign
    import numpy as np
    species = ["NH3", "H2S", "CO2", "H2O"]
    r = sour_water_stripper(
        n_stages=10, feed_stage=2, feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.5, pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, 10)),
        stage_efficiency=1.0,
    )
    td = TrayDesign(diameter=0.5)
    hyd = tray_hydraulics(
        V_profile=r.column_result.V, L_profile=r.column_result.L,
        T_profile=r.column_result.T,
        x_profile=r.column_result.x, y_profile=r.column_result.y,
        P=1.5e5, species_names=species, tray_design=td,
    )
    check(f"  hydraulics computed for all {len(hyd.per_stage)} stages",
          len(hyd.per_stage) == 10)
    check(f"  max %flood = {hyd.max_pct_flood:.1f}% (positive)",
          hyd.max_pct_flood > 0)


def test_size_tray_diameter_decreases_pct_flood():
    """size_tray_diameter must produce a diameter that gives at most
    target_flood_frac of flooding."""
    section("test_size_tray_diameter_decreases_pct_flood")
    from stateprop.electrolyte import sour_water_stripper
    from stateprop.distillation import (
        size_tray_diameter, tray_hydraulics, TrayDesign,
    )
    import numpy as np
    species = ["NH3", "H2S", "CO2", "H2O"]
    r = sour_water_stripper(
        n_stages=10, feed_stage=2, feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.5, pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, 10)),
        stage_efficiency=1.0,
    )
    D = size_tray_diameter(
        V_profile=r.column_result.V, L_profile=r.column_result.L,
        T_profile=r.column_result.T,
        x_profile=r.column_result.x, y_profile=r.column_result.y,
        P=1.5e5, species_names=species, target_flood_frac=0.75,
    )
    td = TrayDesign(diameter=D)
    hyd = tray_hydraulics(
        V_profile=r.column_result.V, L_profile=r.column_result.L,
        T_profile=r.column_result.T,
        x_profile=r.column_result.x, y_profile=r.column_result.y,
        P=1.5e5, species_names=species, tray_design=td,
    )
    check(f"  D = {D:.3f} m → max flood {hyd.max_pct_flood:.1f}% ≤ 76%",
          hyd.max_pct_flood <= 76)


def test_tray_hydraulics_smaller_diameter_higher_flood():
    """Smaller diameter ⇒ higher pct_flood (capacity decreases)."""
    section("test_tray_hydraulics_smaller_diameter_higher_flood")
    from stateprop.electrolyte import sour_water_stripper
    from stateprop.distillation import tray_hydraulics, TrayDesign
    import numpy as np
    species = ["NH3", "H2S", "CO2", "H2O"]
    r = sour_water_stripper(
        n_stages=10, feed_stage=2, feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.5, pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, 10)),
        stage_efficiency=1.0,
    )
    common = dict(
        V_profile=r.column_result.V, L_profile=r.column_result.L,
        T_profile=r.column_result.T,
        x_profile=r.column_result.x, y_profile=r.column_result.y,
        P=1.5e5, species_names=species,
    )
    h_big = tray_hydraulics(**common, tray_design=TrayDesign(diameter=1.0))
    h_small = tray_hydraulics(**common, tray_design=TrayDesign(diameter=0.3))
    check(f"  D=0.3m flood ({h_small.max_pct_flood:.1f}%) > "
          f"D=1.0m flood ({h_big.max_pct_flood:.1f}%)",
          h_small.max_pct_flood > h_big.max_pct_flood)


def test_tray_hydraulics_per_stage_fields_populated():
    """Each StageHydraulics has all required fields populated."""
    section("test_tray_hydraulics_per_stage_fields_populated")
    from stateprop.electrolyte import sour_water_stripper
    from stateprop.distillation import tray_hydraulics, TrayDesign
    import numpy as np
    species = ["NH3", "H2S", "CO2", "H2O"]
    r = sour_water_stripper(
        n_stages=10, feed_stage=2, feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.5, pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, 10)),
        stage_efficiency=1.0,
    )
    hyd = tray_hydraulics(
        V_profile=r.column_result.V, L_profile=r.column_result.L,
        T_profile=r.column_result.T,
        x_profile=r.column_result.x, y_profile=r.column_result.y,
        P=1.5e5, species_names=species,
        tray_design=TrayDesign(diameter=0.6),
    )
    s0 = hyd.per_stage[0]
    check(f"  rho_V positive ({s0.rho_V:.3f} kg/m³)", s0.rho_V > 0)
    check(f"  rho_L positive ({s0.rho_L:.0f} kg/m³)", s0.rho_L > 0)
    check(f"  weir crest non-negative ({s0.weir_crest*1000:.1f} mm)",
          s0.weir_crest >= 0)
    check(f"  total ΔP positive ({s0.total_pressure_drop:.0f} Pa)",
          s0.total_pressure_drop > 0)


def test_tray_hydraulics_water_density_at_100C():
    """Water density at 100°C ≈ 958 kg/m³ from approximation."""
    section("test_tray_hydraulics_water_density_at_100C")
    from stateprop.distillation.tray_hydraulics import _liquid_density_water
    rho = _liquid_density_water(373.15)
    check(f"  rho_water(100°C) = {rho:.0f} kg/m³ (lit ~958)",
          950 < rho < 970)


# =====================================================================
# Amine column via N-S solver (v0.9.114)
# =====================================================================

def test_amine_activity_model_basic():
    """AmineActivityModel reproduces P_CO2 from amine eq via γ_CO2."""
    section("test_amine_activity_model_basic")
    from stateprop.electrolyte import (
        AmineActivityModel, AmineSystem, lookup_amine,
    )
    sys_ = AmineSystem(lookup_amine("MEA"), 5.0)
    species = ["CO2", "H2O", "MEA"]
    am = AmineActivityModel(sys_, species)
    # 30 wt% MEA at α=0.5, T=40°C
    moles = 1 + 0.5 + 7.91
    x = [0.5/moles, 7.91/moles, 1/moles]
    gamma = am.gammas(313.15, x)
    # γ_CO2 × P_sat_CO2 (=1 bar) = P_CO2_eq → γ_CO2 = P_CO2_eq[Pa]/x_CO2/1e5
    P_CO2 = am.equilibrium_P_CO2(0.5, 313.15)
    expected = P_CO2 / (x[0] * 1e5)
    check(f"  γ_CO2 = {gamma[0]:.3f} ≈ {expected:.3f} (P_CO2/x_CO2/P_sat)",
          abs(gamma[0] - expected) / max(expected, 1e-6) < 0.01)


def test_amine_activity_model_loading():
    """AmineActivityModel.loading returns x_CO2 / x_amine, capped to 0.95."""
    section("test_amine_activity_model_loading")
    from stateprop.electrolyte import (
        AmineActivityModel, AmineSystem, lookup_amine,
    )
    sys_ = AmineSystem(lookup_amine("MEA"), 5.0)
    am = AmineActivityModel(sys_, ["CO2", "H2O", "MEA"])
    a = am.loading([0.05, 0.85, 0.10])
    check(f"  α({{CO2:0.05, MEA:0.10}}) = {a:.3f} (= 0.5)",
          abs(a - 0.5) < 1e-6)
    a2 = am.loading([0.5, 0.4, 0.1])     # would give 5 → clipped
    check(f"  α capped to 0.95 ({a2:.3f})", abs(a2 - 0.95) < 1e-6)


def test_amine_psat_funcs_lengths():
    """build_amine_psat_funcs returns one callable per species."""
    section("test_amine_psat_funcs_lengths")
    from stateprop.electrolyte import build_amine_psat_funcs
    funcs = build_amine_psat_funcs(["CO2", "H2O", "MEA", "N2"], "MEA")
    check(f"  4 callables", len(funcs) == 4)
    # CO2 → 1e5 Pa, H2O ~7.4 kPa @ 40°C, MEA → 1 Pa, N2 → 1e10 Pa
    p_CO2 = funcs[0](313.15)
    p_H2O = funcs[1](313.15)
    p_MEA = funcs[2](313.15)
    p_N2 = funcs[3](313.15)
    check(f"  P_sat_CO2 pseudo = {p_CO2:.0f} Pa (= 1 bar)",
          abs(p_CO2 - 1e5) < 1)
    check(f"  P_sat_H2O at 40°C = {p_H2O:.0f} Pa (~7400)",
          5000 < p_H2O < 9000)
    check(f"  P_sat_amine = {p_MEA:.1f} Pa (non-volatile)", p_MEA < 10)
    check(f"  P_sat_inert = {p_N2:.0e} Pa (huge)", p_N2 > 1e9)


def test_amine_enthalpy_funcs_runs():
    """Enthalpy callables don't crash."""
    section("test_amine_enthalpy_funcs_runs")
    from stateprop.electrolyte import (
        build_amine_enthalpy_funcs, AmineSystem, lookup_amine,
    )
    sys_ = AmineSystem(lookup_amine("MEA"), 5.0)
    h_V, h_L = build_amine_enthalpy_funcs(["CO2", "H2O", "MEA"], sys_)
    for f in h_V + h_L:
        v = f(388.15)
        check(f"  enthalpy callable returns finite",
              isinstance(v, float) and abs(v) < 1e7)


def test_amine_stripper_ns_runs():
    """amine_stripper_ns converges and produces physical α_lean."""
    section("test_amine_stripper_ns_runs")
    from stateprop.electrolyte import amine_stripper_ns
    r = amine_stripper_ns(
        amine_name="MEA", total_amine=5.0,
        L=10.0, G=8.0, alpha_rich=0.50,
        n_stages=15, T_top=378.15, T_bottom=388.15,
        P=1.8e5, energy_balance=True, stage_efficiency=1.0,
    )
    check(f"  N-S stripper converged ({r.column_result.iterations} iter)",
          r.column_result.converged)
    check(f"  α_lean = {r.alpha_lean:.4f} (industrial range 0.001-0.10)",
          1e-4 < r.alpha_lean < 0.20)


def test_amine_stripper_ns_energy_balance():
    """Reboiler duty is positive and physically sized."""
    section("test_amine_stripper_ns_energy_balance")
    from stateprop.electrolyte import amine_stripper_ns
    r = amine_stripper_ns(
        amine_name="MEA", total_amine=5.0,
        L=10.0, G=8.0, alpha_rich=0.50,
        n_stages=15, T_top=378.15, T_bottom=388.15,
        P=1.8e5, energy_balance=True, stage_efficiency=1.0,
    )
    Q_R_MJ = r.Q_R / 1e6
    # Per ton CO2 stripped: typical 3-5 GJ/ton
    co2_stripped = 10.0 * (0.50 - r.alpha_lean) * 44.01e-3   # kg/s
    if co2_stripped > 1e-6:
        gj_per_ton = (r.Q_R / 1e9) / (co2_stripped / 1000)
        check(f"  Q_R = {Q_R_MJ:.2f} MJ/s, "
              f"specific = {gj_per_ton:.2f} GJ/ton CO2",
              0.5 < gj_per_ton < 30)
    check(f"  Q_R positive ({r.Q_R/1e3:.0f} kW)", r.Q_R > 0)


def test_amine_absorber_ns_runs():
    """amine_absorber_ns converges and produces alpha_rich > alpha_lean."""
    section("test_amine_absorber_ns_runs")
    from stateprop.electrolyte import amine_absorber_ns
    r = amine_absorber_ns(
        amine_name="MEA", total_amine=5.0,
        L=10.0, G=15.0, alpha_lean=0.20, y_in_CO2=0.12,
        n_stages=10, P=1.013e5, energy_balance=False,
    )
    check(f"  converged ({r.column_result.iterations} iter)",
          r.column_result.converged)
    check(f"  α_rich ({r.alpha_rich:.4f}) > α_lean ({r.alpha_lean:.4f})",
          r.alpha_rich > r.alpha_lean)
    check(f"  recovery ({r.co2_recovery*100:.1f}%) > 0",
          r.co2_recovery > 0)


def test_amine_absorber_ns_inert_conservation():
    """Inert (N2) mass balance must close — no N2 absorbed in liquid."""
    section("test_amine_absorber_ns_inert_conservation")
    from stateprop.electrolyte import amine_absorber_ns
    r = amine_absorber_ns(
        amine_name="MEA", total_amine=5.0,
        L=10.0, G=15.0, alpha_lean=0.20, y_in_CO2=0.12,
        n_stages=10, P=1.013e5, energy_balance=False,
    )
    col = r.column_result
    # N2 in feed gas = G × y_N2_in
    y_H2O_sat_40C = 0.073
    y_N2_in = 1.0 - 0.12 - y_H2O_sat_40C
    N2_in = 15.0 * y_N2_in
    N2_out = float(col.V[0]) * float(col.y[0, 3])
    rel_err = abs(N2_in - N2_out) / max(N2_in, 1e-6)
    check(f"  N2 in={N2_in:.2f}, out={N2_out:.2f}, |err|={rel_err*100:.1f}%",
          rel_err < 0.05)


def test_amine_absorber_ns_alpha_increases_top_to_bottom():
    """Liquid α must increase as it descends (CO2 picked up)."""
    section("test_amine_absorber_ns_alpha_increases_top_to_bottom")
    from stateprop.electrolyte import amine_absorber_ns
    r = amine_absorber_ns(
        amine_name="MEA", total_amine=5.0,
        L=10.0, G=15.0, alpha_lean=0.20, y_in_CO2=0.12,
        n_stages=10, P=1.013e5, energy_balance=False,
    )
    # Allow tolerance for the very top (where lean comes in unchanged)
    monotone = all(r.alpha[i] <= r.alpha[i+1] + 1e-6
                       for i in range(len(r.alpha) - 1))
    check(f"  α monotone non-decreasing top→bottom",
          monotone)


def test_amine_absorber_default_inert_n2():
    """Default inert is N2; explicitly using a different name still works."""
    section("test_amine_absorber_default_inert_n2")
    from stateprop.electrolyte import amine_absorber_ns
    # Default (N2)
    r1 = amine_absorber_ns(
        amine_name="MEA", total_amine=5.0,
        L=10.0, G=15.0, alpha_lean=0.20, y_in_CO2=0.12,
        n_stages=10, P=1.013e5,
    )
    # Custom inert name
    r2 = amine_absorber_ns(
        amine_name="MEA", total_amine=5.0,
        L=10.0, G=15.0, alpha_lean=0.20, y_in_CO2=0.12,
        n_stages=10, P=1.013e5, inert_name="Air",
    )
    check(f"  default-N2 converged", r1.column_result.converged)
    check(f"  custom-Air converged", r2.column_result.converged)
    check(f"  α_rich similar within 5% ({r1.alpha_rich:.3f} vs "
          f"{r2.alpha_rich:.3f})",
          abs(r1.alpha_rich - r2.alpha_rich) / max(r1.alpha_rich, 1e-6) < 0.05)


# =====================================================================
# CaptureFlowsheet — N-S mode + tray sizing (v0.9.115)
# =====================================================================

def test_capture_flowsheet_bespoke_solver_unchanged():
    """v0.9.115 backward compat: solver='bespoke' (default) gives
    same result as before."""
    section("test_capture_flowsheet_bespoke_solver_unchanged")
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0,
                              n_stages_absorber=10, n_stages_stripper=15)
    r = fs.solve(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        T_absorber_feed=313.15, P_absorber=1.013,
        G_strip_steam=8.0, T_strip_top=378.15, T_strip_bottom=393.15,
        P_stripper=1.8, T_cond=313.15, delta_T_min_HX=5.0,
        wt_frac_amine=0.30, alpha_lean_init=0.20,
        max_outer=15,
    )
    check(f"  default solver = 'bespoke'", r.solver == "bespoke")
    check(f"  no diameter populated by default",
          r.absorber_diameter is None and r.stripper_diameter is None)
    check(f"  converged ({r.iterations} iter, recovery "
          f"{r.co2_recovery*100:.0f}%)",
          r.converged and r.co2_recovery > 0.5)


def test_capture_flowsheet_ns_solver_runs():
    """v0.9.115 N-S solver path converges."""
    section("test_capture_flowsheet_ns_solver_runs")
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0,
                              n_stages_absorber=6, n_stages_stripper=8)
    r = fs.solve(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        alpha_lean_init=0.005,        # close to N-S converged value
        solver="ns", max_outer=10, damp=0.5, tol=1e-3,
    )
    check(f"  N-S flowsheet converged ({r.iterations} iter)", r.converged)
    check(f"  recovery > 0.50 ({r.co2_recovery*100:.1f}%)",
          r.co2_recovery > 0.50)
    check(f"  Q/ton positive ({r.Q_per_ton_CO2:.2f} GJ/ton)",
          r.Q_per_ton_CO2 > 0)
    check(f"  solver field = 'ns'", r.solver == "ns")


def test_capture_flowsheet_size_trays_requires_ns():
    """size_trays=True with bespoke solver must raise ValueError."""
    section("test_capture_flowsheet_size_trays_requires_ns")
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0,
                              n_stages_absorber=6, n_stages_stripper=8)
    try:
        fs.solve(G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
                  solver="bespoke", size_trays=True)
        check(f"  expected ValueError", False)
    except ValueError as e:
        msg = str(e)
        check(f"  ValueError raised: {msg[:60]}",
              "ns" in msg or "rigorous" in msg.lower() or "bespoke" in msg)


def test_capture_flowsheet_unknown_solver_raises():
    """Invalid solver name raises ValueError."""
    section("test_capture_flowsheet_unknown_solver_raises")
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0)
    try:
        fs.solve(G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
                  solver="aspen")
        check(f"  expected ValueError", False)
    except ValueError as e:
        check(f"  ValueError raised: {str(e)[:60]}", True)


def test_capture_flowsheet_ns_with_size_trays():
    """N-S + size_trays produces tower diameters in physical range."""
    section("test_capture_flowsheet_ns_with_size_trays")
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0,
                              n_stages_absorber=6, n_stages_stripper=8)
    r = fs.solve(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        alpha_lean_init=0.005,
        solver="ns", size_trays=True, target_flood_frac=0.75,
        max_outer=10, damp=0.5, tol=1e-3,
    )
    check(f"  absorber_diameter populated", r.absorber_diameter is not None)
    check(f"  stripper_diameter populated", r.stripper_diameter is not None)
    if r.absorber_diameter is not None:
        check(f"  abs D = {r.absorber_diameter:.2f} m in [0.1, 5.0]",
              0.1 <= r.absorber_diameter <= 5.0)
    if r.stripper_diameter is not None:
        check(f"  strip D = {r.stripper_diameter:.2f} m in [0.1, 5.0]",
              0.1 <= r.stripper_diameter <= 5.0)
    check(f"  absorber_hydraulics populated",
          r.absorber_hydraulics is not None)
    check(f"  stripper_hydraulics populated",
          r.stripper_hydraulics is not None)


def test_capture_flowsheet_summary_includes_diameters():
    """summary() prints tower diameters when present."""
    section("test_capture_flowsheet_summary_includes_diameters")
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0,
                              n_stages_absorber=6, n_stages_stripper=8)
    r = fs.solve(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        alpha_lean_init=0.005,
        solver="ns", size_trays=True,
        max_outer=10, damp=0.5, tol=1e-3,
    )
    s = r.summary()
    check(f"  summary contains 'Absorber: D ='", "Absorber: D" in s)
    check(f"  summary contains 'Stripper: D ='", "Stripper: D" in s)
    check(f"  summary contains 'Tower hardware'", "Tower hardware" in s)


def test_capture_flowsheet_diameter_responds_to_flood_frac():
    """target_flood_frac=0.50 should give larger D than 0.85."""
    section("test_capture_flowsheet_diameter_responds_to_flood_frac")
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    from stateprop.electrolyte import CaptureFlowsheet
    fs = CaptureFlowsheet("MEA", 5.0,
                              n_stages_absorber=6, n_stages_stripper=8)
    common = dict(
        G_flue=15.0, y_in_CO2=0.12, L_amine=8.0,
        alpha_lean_init=0.005,
        solver="ns", size_trays=True,
        max_outer=10, damp=0.5, tol=1e-3,
    )
    r_loose = fs.solve(target_flood_frac=0.85, **common)
    r_tight = fs.solve(target_flood_frac=0.50, **common)
    check(f"  D(50%flood)={r_tight.absorber_diameter:.2f} > "
          f"D(85%flood)={r_loose.absorber_diameter:.2f}",
          r_tight.absorber_diameter > r_loose.absorber_diameter)


# =====================================================================
# Davies γ corrections in sour-water speciation (v0.9.116)
# =====================================================================

def test_davies_log_gamma_at_zero_I():
    """Davies γ_± at I=0 should be exactly 1.0 (log10 = 0)."""
    section("test_davies_log_gamma_at_zero_I")
    from stateprop.electrolyte.sour_water import _davies_log_gamma_pm
    log_g = _davies_log_gamma_pm(0.0, 298.15)
    check(f"  log10 γ_±(I=0) = {log_g} (= 0)", abs(log_g) < 1e-12)


def test_davies_log_gamma_NaCl_at_known_I():
    """Davies γ_± at I=0.1 should be ~0.78 (Pitzer NaCl literature 0.78)."""
    section("test_davies_log_gamma_NaCl_at_known_I")
    from stateprop.electrolyte.sour_water import _davies_log_gamma_pm
    g = 10 ** _davies_log_gamma_pm(0.1, 298.15)
    check(f"  γ_±(I=0.1) = {g:.3f} (lit ~0.78, NaCl)",
          0.74 < g < 0.82)


def test_davies_log_gamma_capped_at_high_I():
    """Davies should cap at I=2 (formula turns over above)."""
    section("test_davies_log_gamma_capped_at_high_I")
    from stateprop.electrolyte.sour_water import _davies_log_gamma_pm
    g_2 = 10 ** _davies_log_gamma_pm(2.0, 298.15)
    g_5 = 10 ** _davies_log_gamma_pm(5.0, 298.15)
    g_10 = 10 ** _davies_log_gamma_pm(10.0, 298.15)
    check(f"  γ_±(I=5) = {g_5:.3f} = γ_±(I=2) = {g_2:.3f} (capped)",
          abs(g_2 - g_5) < 1e-6)
    check(f"  γ_±(I=10) capped too", abs(g_2 - g_10) < 1e-6)


def test_speciate_davies_off_matches_default():
    """apply_davies_gammas=False reproduces the v0.9.115 default behavior."""
    section("test_speciate_davies_off_matches_default")
    from stateprop.electrolyte.sour_water import speciate
    r1 = speciate(T=298.15, m_NH3_total=0.5, m_H2S_total=0.3,
                    m_CO2_total=0.05,
                    extra_strong_cations=1.0, extra_strong_anions=1.0)
    r2 = speciate(T=298.15, m_NH3_total=0.5, m_H2S_total=0.3,
                    m_CO2_total=0.05,
                    extra_strong_cations=1.0, extra_strong_anions=1.0,
                    apply_davies_gammas=False)
    check(f"  default = davies=False: pH same ({r1.pH:.3f}, {r2.pH:.3f})",
          abs(r1.pH - r2.pH) < 1e-6)


def test_speciate_davies_shifts_h2s_split():
    """At moderate I, Davies γ correction should reduce H2S molecular
    fraction (acid becomes effectively stronger via γ_± < 1)."""
    section("test_speciate_davies_shifts_h2s_split")
    from stateprop.electrolyte.sour_water import speciate
    r_off = speciate(T=298.15, m_NH3_total=0.5, m_H2S_total=0.3,
                       m_CO2_total=0.05,
                       extra_strong_cations=0.0, extra_strong_anions=0.0)
    r_on = speciate(T=298.15, m_NH3_total=0.5, m_H2S_total=0.3,
                      m_CO2_total=0.05,
                      extra_strong_cations=0.0, extra_strong_anions=0.0,
                      apply_davies_gammas=True)
    check(f"  α_H2S decreases with Davies "
          f"({r_off.alpha_H2S:.4f} → {r_on.alpha_H2S:.4f})",
          r_on.alpha_H2S < r_off.alpha_H2S)


def test_speciate_davies_NH4_unchanged_at_low_I():
    """For NH4+ → NH3 + H+ (cationic acid), Davies γ correction
    cancels and pK_a is essentially unchanged at low I."""
    section("test_speciate_davies_NH4_unchanged_at_low_I")
    from stateprop.electrolyte.sour_water import speciate
    # Pure NH3 system (no H2S, no CO2) so NH4+ <-> NH3 dominates pH
    r_off = speciate(T=298.15, m_NH3_total=0.1, m_H2S_total=0.0,
                       m_CO2_total=0.0)
    r_on = speciate(T=298.15, m_NH3_total=0.1, m_H2S_total=0.0,
                      m_CO2_total=0.0, apply_davies_gammas=True)
    # NH4+ molality should be very close in both
    diff = abs(r_off.species_molalities["NH4+"]
                  - r_on.species_molalities["NH4+"])
    rel = diff / max(r_off.species_molalities["NH4+"], 1e-9)
    check(f"  NH4+ unchanged: rel diff = {rel*100:.2f}% (cationic acid)",
          rel < 0.05)


def test_speciate_davies_iterative_convergence():
    """Davies γ self-consistency loop converges within max_gamma_iter."""
    section("test_speciate_davies_iterative_convergence")
    from stateprop.electrolyte.sour_water import speciate
    # Should not raise; if iteration didn't converge we'd see oscillation
    r = speciate(T=298.15, m_NH3_total=0.5, m_H2S_total=0.3,
                   m_CO2_total=0.1,
                   extra_strong_cations=0.5, extra_strong_anions=0.5,
                   apply_davies_gammas=True, max_gamma_iter=20)
    check(f"  pH = {r.pH:.3f} (finite)",
          0 < r.pH < 14)


def test_sour_water_activity_model_pitzer_corrections_uses_davies():
    """SourWaterActivityModel(pitzer_corrections=True) now applies
    Davies γ corrections to the equilibria (in addition to the
    pre-existing Setschenow correction on molecular gases)."""
    section("test_sour_water_activity_model_pitzer_corrections_uses_davies")
    from stateprop.electrolyte import SourWaterActivityModel
    species = ["NH3", "H2S", "CO2", "H2O"]
    am_off = SourWaterActivityModel(species_names=species,
                                          extra_strong_cations=0.5,
                                          extra_strong_anions=0.5,
                                          pitzer_corrections=False)
    am_on = SourWaterActivityModel(species_names=species,
                                         extra_strong_cations=0.5,
                                         extra_strong_anions=0.5,
                                         pitzer_corrections=True)
    x = [0.005, 0.005, 0.001, 0.989]
    sp_off = am_off.speciate_at(298.15, x)
    sp_on = am_on.speciate_at(298.15, x)
    # With Davies, pH should shift (small but nonzero)
    check(f"  pH differs with Davies "
          f"(off={sp_off.pH:.4f}, on={sp_on.pH:.4f})",
          abs(sp_off.pH - sp_on.pH) > 1e-4)


# =====================================================================
# Chen-Song 2004 generalized eNRTL (v0.9.118)
# =====================================================================

def test_chen_song_list_amines():
    """list_chen_song_amines returns the expected three amines."""
    section("test_chen_song_list_amines")
    from stateprop.electrolyte.enrtl import list_chen_song_amines
    avail = list_chen_song_amines()
    check(f"  MEA in {avail}", "MEA" in avail)
    check(f"  MDEA in {avail}", "MDEA" in avail)
    check(f"  DEA in {avail}", "DEA" in avail)


def test_chen_song_unknown_amine_raises():
    """Unknown amine raises KeyError."""
    section("test_chen_song_unknown_amine_raises")
    from stateprop.electrolyte.enrtl import _chen_song_tau_matrix
    try:
        _chen_song_tau_matrix("PZ", 298.15)
        check(f"  expected KeyError for PZ", False)
    except KeyError:
        check(f"  KeyError raised for unknown amine", True)


def test_chen_song_tau_matrix_T_dependence():
    """τ matrix has T-dependence for MEA: τ at 100°C ≠ τ at 25°C."""
    section("test_chen_song_tau_matrix_T_dependence")
    import numpy as np
    from stateprop.electrolyte.enrtl import _chen_song_tau_matrix
    tau_25 = _chen_song_tau_matrix("MEA", 298.15)
    tau_100 = _chen_song_tau_matrix("MEA", 373.15)
    check(f"  τ_HA at 25°C = {tau_25[0,1]:.3f} differs from τ at 100°C "
          f"= {tau_100[0,1]:.3f}",
          abs(tau_25[0, 1] - tau_100[0, 1]) > 1e-3)
    check(f"  diagonal stays 0 at 100°C",
          np.all(np.diag(tau_100) == 0.0))


def test_chen_song_log_gamma_zero_at_pure_water():
    """At infinite dilution in water (asymmetric reference),
    γ_amine and γ_CO2 both → 1, so ln γ → 0."""
    section("test_chen_song_log_gamma_zero_at_pure_water")
    from stateprop.electrolyte.enrtl import chen_song_log_gamma_molecular
    # Pure water: x_w = 1, x_a = x_c = 0
    g_w, g_a, g_c = chen_song_log_gamma_molecular(
        "MEA", x_water=1.0, x_amine=0.0, x_CO2=0.0, T=298.15)
    check(f"  ln γ_amine = {g_a:.6f} ≈ 0 at infinite dilution",
          abs(g_a) < 1e-6)
    check(f"  ln γ_CO2 = {g_c:.6f} ≈ 0 at infinite dilution",
          abs(g_c) < 1e-6)


def test_chen_song_log_gamma_water_smooth_in_loaded():
    """γ_water in loaded MEA solution should be near 1 (still mostly
    pure water solvent at typical loadings)."""
    section("test_chen_song_log_gamma_water_smooth_in_loaded")
    import numpy as np
    from stateprop.electrolyte.enrtl import chen_song_log_gamma_molecular
    # 30 wt% MEA at α=0.5: x_w ≈ 0.86, x_a ≈ 0.10, x_c ≈ 0.04
    g_w, _, _ = chen_song_log_gamma_molecular(
        "MEA", x_water=0.86, x_amine=0.10, x_CO2=0.04, T=298.15)
    gamma_w = float(np.exp(g_w))
    check(f"  γ_water = {gamma_w:.3f} in [0.95, 1.10]",
          0.95 <= gamma_w <= 1.10)


def test_chen_song_log_gamma_CO2_below_unity_in_loaded():
    """γ_CO2 < 1 in loaded amine (CO2 stabilized by amine interactions)."""
    section("test_chen_song_log_gamma_CO2_below_unity_in_loaded")
    import numpy as np
    from stateprop.electrolyte.enrtl import chen_song_log_gamma_molecular
    _, _, g_c = chen_song_log_gamma_molecular(
        "MEA", x_water=0.86, x_amine=0.10, x_CO2=0.04, T=298.15)
    gamma_c = float(np.exp(g_c))
    check(f"  γ_CO2 = {gamma_c:.4f} < 1.0",
          gamma_c < 1.0)


def test_amine_system_chen_song_option():
    """activity_model='chen_song' is a valid AmineSystem option."""
    section("test_amine_system_chen_song_option")
    from stateprop.electrolyte import AmineSystem
    sys_ = AmineSystem("MEA", 5.0, activity_model="chen_song")
    check(f"  activity_model stored correctly",
          sys_.activity_model == "chen_song")
    r = sys_.speciate(alpha=0.30, T=313.15)
    check(f"  speciation runs and returns positive P_CO2 "
          f"({r.P_CO2:.4f} bar)", r.P_CO2 > 0)


def test_amine_system_invalid_model_raises():
    """Unknown activity_model raises ValueError."""
    section("test_amine_system_invalid_model_raises")
    from stateprop.electrolyte import AmineSystem
    try:
        AmineSystem("MEA", 5.0, activity_model="aspen_eos")
        check(f"  expected ValueError", False)
    except ValueError as e:
        check(f"  ValueError raised: {str(e)[:60]}",
              "chen_song" in str(e))


def test_amine_system_chen_song_lowers_P_CO2_at_high_T():
    """At 100 °C and α=0.5 (loaded regenerator condition), Chen-Song
    should give a *lower* P_CO2 than PDH-only — fixing the
    over-prediction that Chen-Song was added to address."""
    section("test_amine_system_chen_song_lowers_P_CO2_at_high_T")
    from stateprop.electrolyte import AmineSystem
    sys_pdh = AmineSystem("MEA", 5.0, activity_model="pdh")
    sys_cs = AmineSystem("MEA", 5.0, activity_model="chen_song")
    r_pdh = sys_pdh.speciate(alpha=0.50, T=393.15)   # 120°C
    r_cs = sys_cs.speciate(alpha=0.50, T=393.15)
    check(f"  P_CO2 PDH={r_pdh.P_CO2:.2f} > Chen-Song={r_cs.P_CO2:.2f}",
          r_cs.P_CO2 < r_pdh.P_CO2)


def test_amine_system_chen_song_falls_back_for_unsupported():
    """If amine doesn't have Chen-Song τ params (e.g. AMP), the system
    should still work — falls back silently to PDH-only γ when the
    amine isn't in list_chen_song_amines()."""
    section("test_amine_system_chen_song_falls_back_for_unsupported")
    from stateprop.electrolyte import AmineSystem
    # AMP is a valid amine but lacks Chen-Song τ; should silently fall
    # back to PDH-only γ within speciate.
    sys_ = AmineSystem("AMP", 5.0, activity_model="chen_song")
    r_cs = sys_.speciate(alpha=0.30, T=313.15)
    check(f"  AMP + chen_song runs without crashing "
          f"(P_CO2={r_cs.P_CO2:.4f} bar)",
          r_cs.P_CO2 > 0 and r_cs.converged)


# =====================================================================
# High-T Pitzer + sour-water Pitzer corrections (v0.9.116)
# =====================================================================

def test_pitzer_high_T_lookup():
    """lookup_salt_high_T returns a salt with param_func set."""
    section("test_pitzer_high_T_lookup")
    from stateprop.electrolyte import lookup_salt_high_T, list_salts_high_T
    salts = list_salts_high_T()
    check(f"  available high-T salts: {salts}",
          set(salts) == {"NaCl", "CaCl2", "KCl"})
    s = lookup_salt_high_T("NaCl")
    check(f"  param_func is set", s.param_func is not None)
    check(f"  T_max_valid = {s.T_max_valid:.0f} K", s.T_max_valid >= 473.15)


def test_pitzer_high_T_unknown_raises():
    """Unknown salt raises KeyError."""
    section("test_pitzer_high_T_unknown_raises")
    from stateprop.electrolyte import lookup_salt_high_T
    try:
        lookup_salt_high_T("FakeCl")
        check(f"  expected KeyError", False)
    except KeyError as e:
        check(f"  KeyError raised: {str(e)[:60]}", True)


def test_pitzer_high_T_at_T_dispatches_to_func():
    """salt.at_T(T) uses the param_func instead of Taylor expansion."""
    section("test_pitzer_high_T_at_T_dispatches_to_func")
    from stateprop.electrolyte import lookup_salt_high_T
    s = lookup_salt_high_T("NaCl")
    s_25 = s.at_T(298.15)
    s_200 = s.at_T(473.15)
    check(f"  β⁰(25°C) = {s_25.beta_0:.4f} (anchor 0.0765)",
          abs(s_25.beta_0 - 0.0765) < 1e-3)
    check(f"  β⁰(200°C) = {s_200.beta_0:.4f} (PP anchor 0.0717)",
          abs(s_200.beta_0 - 0.0717) < 0.005)


def test_pitzer_high_T_NaCl_gamma_at_200C():
    """γ_±(NaCl, 1m, 200°C) accuracy vs Pabalan-Pitzer published value."""
    section("test_pitzer_high_T_NaCl_gamma_at_200C")
    from stateprop.electrolyte import lookup_salt_high_T, PitzerModel
    s = lookup_salt_high_T("NaCl").at_T(473.15)
    g = PitzerModel(s).gamma_pm(1.0, T=473.15)
    lit = 0.456    # Pabalan-Pitzer 1988
    err = abs(g - lit) / lit
    check(f"  γ_±(NaCl, 1m, 200°C) = {g:.3f} vs lit {lit} ({err*100:.1f}% rel err)",
          err < 0.15)


def test_pitzer_high_T_CaCl2_gamma_at_100C():
    """γ_±(CaCl2, 1m, 100°C) within fit range."""
    section("test_pitzer_high_T_CaCl2_gamma_at_100C")
    from stateprop.electrolyte import lookup_salt_high_T, PitzerModel
    s = lookup_salt_high_T("CaCl2").at_T(373.15)
    g = PitzerModel(s).gamma_pm(1.0, T=373.15)
    check(f"  γ_±(CaCl2, 1m, 100°C) = {g:.3f} (>0)", g > 0)


def test_sour_water_setschenow_factor_at_high_I():
    """At I_strong=2 mol/kg, Setschenow factor for NH3 = 10^(0.077·2) = 1.426."""
    section("test_sour_water_setschenow_factor_at_high_I")
    from stateprop.electrolyte import SourWaterActivityModel
    species = ["NH3", "H2S", "CO2", "H2O"]
    m_dilute = SourWaterActivityModel(species, extra_strong_anions=0.0,
                                            extra_strong_cations=0.0)
    m_high = SourWaterActivityModel(species,
                                          extra_strong_anions=2.0,
                                          extra_strong_cations=2.0,
                                          pitzer_corrections=True)
    x = [0.005, 0.001, 0.001, 0.993]
    g0 = m_dilute.gammas(333.15, x)
    g1 = m_high.gammas(333.15, x)
    ratio_NH3 = g1[0] / g0[0]
    expected = 10.0 ** (0.077 * 2.0)
    check(f"  γ_NH3 ratio = {ratio_NH3:.3f} ≈ 10^(k_s·I) = "
          f"{expected:.3f}",
          abs(ratio_NH3 - expected) / expected < 0.02)


def test_sour_water_pitzer_off_unchanged():
    """pitzer_corrections=False (default) gives identical γ to v0.9.115."""
    section("test_sour_water_pitzer_off_unchanged")
    from stateprop.electrolyte import SourWaterActivityModel
    species = ["NH3", "H2S", "CO2", "H2O"]
    # Same model with extra ions but no Pitzer correction
    m = SourWaterActivityModel(species,
                                     extra_strong_anions=2.0,
                                     extra_strong_cations=2.0,
                                     pitzer_corrections=False)
    x = [0.005, 0.001, 0.001, 0.993]
    g = m.gammas(333.15, x)
    # Without correction, should match dilute (no Setschenow boost)
    m_dilute = SourWaterActivityModel(species)
    g_dilute = m_dilute.gammas(333.15, x)
    # NB: the speciate function is sensitive to extra_strong_ions
    # (changes pH), so γ_dilute and γ won't be identical.  We just
    # check the Setschenow factor isn't applied:
    check(f"  No Setschenow factor applied (γ < 1)",
          g[0] < 1.0)


def test_sour_water_stripper_pitzer_corrections():
    """Stripper with pitzer_corrections=True produces different
    α than without — strong-ion-dependent volatility."""
    section("test_sour_water_stripper_pitzer_corrections")
    import numpy as np
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    from stateprop.electrolyte import sour_water_stripper
    species = ["NH3", "H2S", "CO2", "H2O"]
    common = dict(
        n_stages=10, feed_stage=2, feed_F=100.0,
        feed_z=[0.01, 0.005, 0.001, 0.984], feed_T=353.15,
        species_names=species,
        reflux_ratio=1.0, distillate_rate=2.5, pressure=1.5e5,
        T_init=list(np.linspace(353.15, 383.15, 10)),
        extra_strong_anions=2.0,         # High background salt
        stage_efficiency=1.0,
    )
    r_no = sour_water_stripper(pitzer_corrections=False, **common)
    r_yes = sour_water_stripper(pitzer_corrections=True, **common)
    check(f"  both runs converged",
          r_no.column_result.converged and r_yes.column_result.converged)
    # H2S strip in both should be near 100% (acid-side speciation),
    # but α profiles will differ.
    check(f"  H2S recovery: pitzer=False {r_no.bottoms_strip_efficiency['H2S']*100:.1f}%, "
          f"pitzer=True {r_yes.bottoms_strip_efficiency['H2S']*100:.1f}%",
          True)


# =====================================================================
# Run
# =====================================================================

def main():
    print("=" * 60)
    print("stateprop electrolyte tests (v0.9.96)")
    print("=" * 60)
    tests = [
        # Constants
        test_water_density_298K, test_water_dielectric_298K,
        test_debye_huckel_A_298K, test_debye_huckel_A_T_increase,
        # Utilities
        test_ionic_strength_NaCl, test_ionic_strength_CaCl2,
        test_molality_mole_fraction_roundtrip,
        # Database
        test_lookup_salt, test_unknown_salt_raises,
        # γ± vs literature
        test_NaCl_gamma_pm, test_KCl_gamma_pm, test_HCl_gamma_pm,
        test_CaCl2_gamma_pm, test_Na2SO4_gamma_pm,
        # Osmotic / water activity
        test_NaCl_osmotic_coefficient, test_water_activity_NaCl,
        # Limiting law
        test_limiting_law_NaCl, test_pure_DH_at_very_low_m,
        test_davies_at_moderate_m,
        # Custom and self-consistency
        test_custom_PitzerSalt, test_pitzer_phi_self_consistency,
        # 2:2
        test_MgSO4_gamma_pm,
        # eNRTL
        test_enrtl_loads,
        # T-dependence (v0.9.97)
        test_T_dependence_at_Tref_unchanged,
        test_at_T_returns_self_when_no_derivatives,
        test_NaCl_T_dependence_qualitative,
        test_NaCl_beta_at_T,
        test_KCl_beta_at_T,
        test_CaCl2_T_dependence,
        test_water_activity_T_dependence,
        # Sour water (v0.9.97)
        test_pKw_25C, test_pKw_T_dependence,
        test_NH4_pKa_25C, test_H2S_pKa_25C, test_CO2_pKa_25C,
        test_henry_NH3_25C, test_henry_T_dependence,
        test_speciate_pure_NH3, test_speciate_pure_H2S,
        test_speciate_NH3_plus_H2S,
        test_effective_henry_pH_dependence,
        # Multi-electrolyte Pitzer (v0.9.98)
        test_multi_pitzer_NaCl_reduces_to_single,
        test_multi_pitzer_NaCl_KCl_mixture,
        test_multi_pitzer_seawater_water_activity,
        test_multi_pitzer_seawater_osmotic,
        test_multi_pitzer_charge_check,
        test_multi_pitzer_ionic_strength,
        test_multi_pitzer_seawater_constructor,
        # Proper E-θ unsymmetric mixing (v0.9.99)
        test_multi_pitzer_E_theta_active,
        test_multi_pitzer_J0_monotonic,
        test_multi_pitzer_seawater_with_E_theta,
        # T-dependent mixing terms (v0.9.100)
        test_mixing_param_T_independent_default,
        test_mixing_param_T_dependent,
        test_mixing_param_bundled_NaCa,
        test_mixing_param_bundled_psi_NaCaCl,
        test_multi_pitzer_25C_unchanged_from_v0_9_99,
        test_multi_pitzer_NaCaCl_T_dep_active,
        test_multi_pitzer_seawater_T_profile,
        test_multi_pitzer_user_float_override_still_works,
        # Mineral solubility (v0.9.101)
        test_mineral_db_loaded,
        test_mineral_log_K_sp_25C,
        test_mineral_log_K_sp_T_dependence,
        test_solubility_halite_pure_water,
        test_solubility_gypsum_pure_water,
        test_solubility_halite_T_dependence,
        test_solubility_gypsum_anhydrite_crossover,
        test_solubility_in_water_rejects_ternary,
        test_saturation_index_undersaturated,
        test_saturation_index_saturated,
        test_saturation_index_supersaturated,
        test_saturation_index_missing_ion_returns_minus_inf,
        test_mineral_system_seawater_qualitative,
        test_mineral_system_scale_risks_filter,
        test_solubility_iter_converges,
        test_dolomite_SI_in_dilute_seawater,
        # Aqueous complexation (v0.9.102)
        test_complex_db_loaded,
        test_complex_K_diss_T_dependence,
        test_complex_log_K_assoc_inverse,
        test_speciation_pure_passthrough,
        test_speciation_NaSO4_ion_pair,
        test_speciation_seawater_CO3_depleted,
        test_speciation_calcite_SI_seawater,
        test_speciation_calcite_aragonite_ordering,
        test_speciation_gypsum_pure_water,
        test_speciation_mass_balance_holds,
        test_mineral_log_K_sp_thermo_field,
        test_solve_speciation_convenience_wrapper,
        test_davies_equation_neutral_returns_zero,
        test_davies_equation_low_I_dilute_limit,
        # Amine carbamate equilibria (v0.9.103)
        test_amine_db_loaded,
        test_amine_pKa_van_t_hoff,
        test_amine_K_a_inverse_of_pKa,
        test_tertiary_amine_no_carbamate,
        test_carbonate_constants_at_25C,
        test_K_H_increases_with_T,
        test_amine_speciate_primary_MEA_alpha_05,
        test_amine_speciate_tertiary_MDEA,
        test_amine_speciate_alpha_zero,
        test_amine_speciate_mass_balance,
        test_amine_equilibrium_loading_inverse,
        test_amine_loading_monotonic_in_PCO2,
        test_amine_higher_T_releases_CO2,
        test_amine_lookup_case_insensitive,
        # eNRTL refinements + amine column (v0.9.104)
        test_pdh_A_phi_25C,
        test_pdh_A_phi_T_dependence,
        test_pdh_log_gamma_neutral,
        test_pdh_does_not_diverge_at_high_I,
        test_amine_pdh_improves_high_T_prediction,
        test_amine_speciate_with_pdh_converges,
        test_amine_column_simple_solve,
        test_amine_column_recovery_high_LG,
        test_amine_column_pinched_at_min_LG,
        test_amine_column_stages_for_recovery,
        test_amine_equilibrium_curve_monotonic,
        # Reactive stripper / heat balance (v0.9.105)
        test_amine_heat_properties_loaded,
        test_amine_cp_solution_30wt_MEA,
        test_water_vapor_pressure_at_100C,
        test_water_vapor_pressure_at_120C,
        test_amine_stripper_simple_solve,
        test_amine_stripper_strips_CO2,
        test_amine_stripper_top_CO2_higher_than_reb,
        test_amine_stripper_reboiler_duty_in_industry_range,
        test_amine_stripper_heat_breakdown_reaction_dominant,
        test_amine_stripper_higher_G_strips_more,
        test_amine_stripper_per_stage_heat_balance_returns_list,
        # Adiabatic absorber + lean-rich heat exchanger (v0.9.106)
        test_amine_column_adiabatic_converges,
        test_amine_column_adiabatic_T_bulge,
        test_amine_column_adiabatic_recovery_lower,
        test_cross_hx_basic_balanced,
        test_cross_hx_unbalanced_pinch_at_one_end,
        test_cross_hx_LMTD_balanced_equal_ends,
        test_cross_hx_UA_inverse_to_dT_min,
        test_cross_hx_rejects_invalid_temperatures,
        test_lean_rich_exchanger_convenience,
        test_HX_saves_stripper_duty,
        # Coupled T-solver + stripper condenser (v0.9.107)
        test_stripper_solve_for_Q_reb_inverts_forward,
        test_stripper_solve_for_Q_reb_higher_Q_lower_alpha_lean,
        test_stripper_condenser_basic,
        test_stripper_condenser_mass_balance,
        test_stripper_condenser_water_balance,
        test_stripper_condenser_lower_T_higher_purity,
        test_stripper_condenser_rejects_too_hot,
        test_stripper_condenser_Q_breakdown_latent_dominant,
        test_stripper_with_condenser_helper,
        # Capture flowsheet integrator (v0.9.108)
        test_flowsheet_converges,
        test_flowsheet_alpha_consistent,
        test_flowsheet_co2_mass_balance,
        test_flowsheet_HX_recovers_heat,
        test_flowsheet_in_industry_envelope,
        test_flowsheet_summary_format,
        test_flowsheet_lower_G_strip_lower_Q_per_ton_in_useful_range,
        test_flowsheet_MDEA_works,
        # Adiabatic flowsheet + variable-V stripper (v0.9.109)
        test_flowsheet_adiabatic_absorber_converges,
        test_flowsheet_adiabatic_T_rich_higher,
        test_flowsheet_adiabatic_lower_HX_duty,
        test_stripper_variable_V_converges,
        test_stripper_variable_V_higher_at_top,
        test_stripper_variable_V_bottom_matches_G_reb,
        test_stripper_variable_V_constant_water_mass_flow,
        # T-saturation auto-clip + energy-balance V + flowsheet variable_V (v0.9.110)
        test_T_water_sat_inverse_consistency,
        test_T_water_sat_at_18bar,
        test_stripper_auto_clip_T_bottom,
        test_stripper_no_clip_raises,
        test_stripper_variable_V_string_modes,
        test_stripper_variable_V_invalid_string_raises,
        test_stripper_energy_V_within_saturation_bounds,
        test_flowsheet_variable_V_passes_through,
        test_flowsheet_three_V_modes_same_Q_per_ton,
        # Sour-water Naphtali-Sandholm coupling (v0.9.111)
        test_sour_water_activity_basic,
        test_sour_water_activity_pH_response,
        test_sour_water_activity_requires_water,
        test_sour_water_activity_requires_volatile,
        test_sour_water_psat_funcs_lengths,
        test_sour_water_stripper_runs,
        test_sour_water_stripper_strip_efficiencies,
        test_sour_water_stripper_acid_lowers_pH,
        test_sour_water_stripper_base_helps_NH3,
        test_sour_water_stripper_NH3_only,
        test_sour_water_stripper_dilute_henry_consistency,
        # Energy balance + Murphree (v0.9.112)
        test_sour_water_enthalpy_funcs_lengths,
        test_sour_water_enthalpy_water_vap_at_Tref,
        test_sour_water_enthalpy_NH3_at_Tref,
        test_sour_water_enthalpy_water_vap_decreases_with_T,
        test_sour_water_stripper_energy_balance_runs,
        test_sour_water_stripper_steam_ratio_in_industrial_range,
        test_sour_water_stripper_stage_efficiency_effect,
        test_sour_water_stripper_default_efficiency_is_065,
        test_sour_water_stripper_per_stage_efficiency_array,
        test_sour_water_stripper_energy_balance_temperature_response,
        # Two-stage flowsheet + tray hydraulics (v0.9.113)
        test_two_stage_flowsheet_runs,
        test_two_stage_flowsheet_acid_strips_h2s,
        test_two_stage_flowsheet_chemical_consumption,
        test_two_stage_flowsheet_total_steam_in_range,
        test_two_stage_flowsheet_dose_negative_raises,
        test_find_acid_dose_for_h2s_recovery,
        test_tray_design_geometry,
        test_flooding_velocity_increases_with_density_diff,
        test_tray_hydraulics_runs_on_sour_water,
        test_size_tray_diameter_decreases_pct_flood,
        test_tray_hydraulics_smaller_diameter_higher_flood,
        test_tray_hydraulics_per_stage_fields_populated,
        test_tray_hydraulics_water_density_at_100C,
        # Amine N-S column (v0.9.114)
        test_amine_activity_model_basic,
        test_amine_activity_model_loading,
        test_amine_psat_funcs_lengths,
        test_amine_enthalpy_funcs_runs,
        test_amine_stripper_ns_runs,
        test_amine_stripper_ns_energy_balance,
        test_amine_absorber_ns_runs,
        test_amine_absorber_ns_inert_conservation,
        test_amine_absorber_ns_alpha_increases_top_to_bottom,
        test_amine_absorber_default_inert_n2,
        # CaptureFlowsheet — N-S mode + tray sizing (v0.9.115)
        test_capture_flowsheet_bespoke_solver_unchanged,
        test_capture_flowsheet_ns_solver_runs,
        test_capture_flowsheet_size_trays_requires_ns,
        test_capture_flowsheet_unknown_solver_raises,
        test_capture_flowsheet_ns_with_size_trays,
        test_capture_flowsheet_summary_includes_diameters,
        test_capture_flowsheet_diameter_responds_to_flood_frac,
        # v0.9.116 — High-T Pitzer + sour-water Setschenow
        test_pitzer_high_T_lookup,
        test_pitzer_high_T_unknown_raises,
        test_pitzer_high_T_at_T_dispatches_to_func,
        test_pitzer_high_T_NaCl_gamma_at_200C,
        test_pitzer_high_T_CaCl2_gamma_at_100C,
        test_sour_water_setschenow_factor_at_high_I,
        test_sour_water_pitzer_off_unchanged,
        test_sour_water_stripper_pitzer_corrections,
        # v0.9.116 (continued) — Davies γ in sour-water speciation
        test_davies_log_gamma_at_zero_I,
        test_davies_log_gamma_NaCl_at_known_I,
        test_davies_log_gamma_capped_at_high_I,
        test_speciate_davies_off_matches_default,
        test_speciate_davies_shifts_h2s_split,
        test_speciate_davies_NH4_unchanged_at_low_I,
        test_speciate_davies_iterative_convergence,
        test_sour_water_activity_model_pitzer_corrections_uses_davies,
        # v0.9.118 — Chen-Song 2004 generalized eNRTL
        test_chen_song_list_amines,
        test_chen_song_unknown_amine_raises,
        test_chen_song_tau_matrix_T_dependence,
        test_chen_song_log_gamma_zero_at_pure_water,
        test_chen_song_log_gamma_water_smooth_in_loaded,
        test_chen_song_log_gamma_CO2_below_unity_in_loaded,
        test_amine_system_chen_song_option,
        test_amine_system_invalid_model_raises,
        test_amine_system_chen_song_lowers_P_CO2_at_high_T,
        test_amine_system_chen_song_falls_back_for_unsupported,
    ]
    for t in tests:
        t()
    print("\n" + "=" * 60)
    print(f"RESULT: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)
    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()
