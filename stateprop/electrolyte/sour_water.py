"""Sour-water thermodynamics: weak-electrolyte speciation in water.

Sour water is the wastewater from refineries, gas-treating plants, and
chemical complexes containing dissolved acidic and basic gases — most
commonly H₂S, NH₃, and CO₂.  These species partially dissociate in
water, with the pH set by self-consistent solution of:

    Henry's law:    P_i = H_i · m_i^molecular   (for dissolved gas i)
    Dissociation:   K_a (or K_b) for each weak acid/base
    Charge balance: Σ z_i · m_i = 0
    Mass balance:   m_i^total = m_i^molecular + m_i^ionic forms

This module provides:

    * Henry's-law constants H(T) for NH₃, H₂S, CO₂
    * Dissociation constants K(T) for NH₃ (basic), H₂S (acid),
      CO₂/H₂CO₃ (acid), H₂O (autoionization)
    * `SourWaterSpeciation` — solves all the above for a given
      total composition and T to give pH, ionic strength, and
      effective volatility of each species
    * `effective_henry(species, T, pH)` — apparent Henry's constant
      after accounting for dissociation; the form a distillation
      column needs as its "Psat" function

The big simplification used here is the standard engineering one:
activity coefficients of the molecular species are taken as 1
(reasonable at <1 mol/kg total dissolved species), and the
dissociation constants K_a are stoichiometric (concentration-based).
For high ionic strength, multiply each K by the appropriate γ ratio
using ``stateprop.electrolyte.PitzerModel``; this refinement is on
the roadmap.

Henry's-law constants are from Edwards-Maurer-Newman-Prausnitz 1978
AIChE J. 24, 966, with T-dependence in their original Eq. 5 form
(typically valid 0-150 °C).

References
----------
* Edwards, T. J., Maurer, G., Newman, J., Prausnitz, J. M. (1978).
  Vapor-liquid equilibria in multicomponent aqueous solutions of
  volatile weak electrolytes. AIChE J. 24, 966.
* Wilhelm, E., Battino, R., Wilcock, R. J. (1977).
  Low-pressure solubility of gases in liquid water. Chem. Rev. 77, 219.
* Harned, H. S., Owen, B. B. (1958). *The Physical Chemistry of
  Electrolytic Solutions* (3rd ed.).  K_w(T) and CO₂ K_a(T).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


# Universal gas constant [J/(mol·K)]
_R = 8.314462618


# =====================================================================
# Henry's-law constants H(T): partial pressure / molality basis [Pa·kg/mol]
# =====================================================================
#
# We use a clean van't Hoff form anchored to well-tabulated 25 °C
# values from Wilhelm-Battino-Wilcock (1977) and Carroll (1991):
#
#   H(T) = H_25 · exp[-ΔH_sol/R · (1/T - 1/298.15)]
#
# where H = P_partial / m_dissolved [Pa·kg/mol], so larger H means
# less soluble (more volatile).  ΔH_sol is the enthalpy of solution
# (negative for exothermic dissolution).

@dataclass
class HenryConstants:
    """Van't Hoff form Henry's-law T-dependence anchored at 25 °C.

    H(T) [Pa · kg / mol] = H_25 · exp[-ΔH_sol/R · (1/T - 1/298.15)]

    Valid range: typically 273-423 K (0-150 °C).
    """
    name: str
    H_25: float       # Henry's at 298.15 K [Pa·kg/mol]
    dH_sol: float      # Enthalpy of solution [J/mol], NEGATIVE if exothermic

    def H_Pa(self, T: float) -> float:
        """Henry's coefficient H [Pa·kg/mol] = P_partial / m_dissolved.

        Uses van't Hoff for the solubility constant K = 1/H:
            K(T) = K_25 · exp[-ΔH_sol/R · (1/T - 1/Tr)]
        Therefore:
            H(T) = H_25 · exp[+ΔH_sol/R · (1/T - 1/Tr)]
        For exothermic dissolution (ΔH_sol < 0), H rises with T:
        gases become less soluble at high T (familiar from boiling water).
        """
        return self.H_25 * np.exp(self.dH_sol / _R
                                     * (1.0 / T - 1.0 / 298.15))


# Well-tabulated reference values at 25 °C from
# Wilhelm-Battino-Wilcock 1977 (Chem. Rev. 77, 219) and standard
# refinery sour-water modeling references.  H_Pa = (atm·kg/mol)·101325.

_HENRY = {
    # NH3: highly soluble in water (H ≈ 1800 Pa·kg/mol at 25 °C)
    # ΔH_sol ≈ -34 kJ/mol (Wilhelm 1977)
    "NH3":  HenryConstants("NH3", H_25=1791.0, dH_sol=-34000.0),
    # H2S: moderately soluble (H ≈ 1.0e6 Pa·kg/mol at 25 °C)
    # ΔH_sol ≈ -19 kJ/mol
    "H2S":  HenryConstants("H2S", H_25=9.93e5, dH_sol=-19000.0),
    # CO2: moderately soluble (H ≈ 3.0e6 Pa·kg/mol at 25 °C)
    # ΔH_sol ≈ -19 kJ/mol
    "CO2":  HenryConstants("CO2", H_25=2.99e6, dH_sol=-19000.0),
}


# =====================================================================
# Dissociation constants K(T)
# =====================================================================
#
# Standard form: van't Hoff anchored to well-tabulated pK values at
# 25 °C from Harned-Owen 1958 (Kw) and Bates-Pinching 1949 / Hershey
# et al. 1988 (NH4+, H2S, CO2).
#
#   K(T) = K_25 · exp[-ΔH_rxn/R · (1/T - 1/298.15)]

@dataclass
class DissociationConstants:
    """T-dependent equilibrium constant for a weak acid/base dissociation.

    K(T) = K_25 · exp[-ΔH_rxn/R · (1/T - 1/298.15)]
    """
    name: str
    reaction: str
    pK_25: float        # pK at 25 °C; K_25 = 10^(-pK_25)
    dH_rxn: float       # ΔH for the reaction [J/mol]

    def K(self, T: float) -> float:
        K_25 = 10 ** (-self.pK_25)
        return K_25 * np.exp(-self.dH_rxn / _R
                                * (1.0 / T - 1.0 / 298.15))


_DISSOCIATIONS = {
    # NH4+ ⇌ NH3 + H+   (acid dissociation of NH4+)
    # pKa(25°C) = 9.245, ΔH = +52.0 kJ/mol (Bates-Pinching 1949)
    "NH4+":   DissociationConstants(
        "NH4+", "NH4+ ⇌ NH3 + H+",
        pK_25=9.245, dH_rxn=52000.0),
    # H2S ⇌ HS- + H+    (first dissociation)
    # pKa1(25°C) = 6.99, ΔH = +21.5 kJ/mol (Hershey et al. 1988)
    "H2S":    DissociationConstants(
        "H2S", "H2S ⇌ HS- + H+",
        pK_25=6.99, dH_rxn=21500.0),
    # CO2 + H2O ⇌ HCO3- + H+   (first apparent dissociation)
    # pKa1(25°C) = 6.35, ΔH = +9.2 kJ/mol (Harned-Davis 1943)
    "CO2":    DissociationConstants(
        "CO2", "CO2 + H2O ⇌ HCO3- + H+",
        pK_25=6.35, dH_rxn=9200.0),
    # H2O ⇌ H+ + OH-   (water autoionization)
    # pKw(25°C) = 14.00, ΔH = +55.84 kJ/mol (Harned-Owen 1958)
    "H2O":    DissociationConstants(
        "H2O", "H2O ⇌ H+ + OH-",
        pK_25=14.00, dH_rxn=55840.0),
}


def henry_constant(species: str, T: float = 298.15) -> float:
    """Henry's-law coefficient H [Pa·kg/mol] for NH3, H2S, or CO2 at T.

    H is defined by P_partial = H · m, where m is the **molecular**
    species molality (not the total dissolved including ionic forms).

    To get the apparent volatility of a partially-dissociating species
    in solution, use ``effective_henry(species, T, pH, ...)``.
    """
    if species not in _HENRY:
        raise KeyError(
            f"No Henry's data for {species!r}. "
            f"Available: {sorted(_HENRY.keys())}")
    return _HENRY[species].H_Pa(T)


def dissociation_K(species: str, T: float = 298.15) -> float:
    """T-dependent dissociation constant K_a (or K_w) for a weak
    electrolyte.  Returns the equilibrium constant in concentration
    (molality) units."""
    if species not in _DISSOCIATIONS:
        raise KeyError(
            f"No dissociation data for {species!r}. "
            f"Available: {sorted(_DISSOCIATIONS.keys())}")
    return _DISSOCIATIONS[species].K(T)


def pK_water(T: float = 298.15) -> float:
    """pKw = -log10(Kw) for water self-ionization at T.

    At 25 °C: pKw ≈ 14.00.  At 100 °C: pKw ≈ 12.27.
    """
    return -np.log10(dissociation_K("H2O", T))


# =====================================================================
# Speciation: solve for pH given total compositions
# =====================================================================

@dataclass
class SourWaterSpeciation:
    """Speciation result for an aqueous H₂S/NH₃/CO₂ system.

    Attributes
    ----------
    T : float
        Temperature [K].
    pH : float
        Equilibrium pH (negative log of H+ molality).
    I : float
        Ionic strength [mol/kg].
    species_molalities : dict
        Molality of each species: 'H+', 'OH-', 'NH3', 'NH4+',
        'H2S', 'HS-', 'CO2', 'HCO3-', 'H2CO3'.
    alpha_NH3 : float
        Fraction of total nitrogen present as molecular NH3 (volatile).
    alpha_H2S : float
        Fraction of total sulfide present as molecular H2S (volatile).
    alpha_CO2 : float
        Fraction of total carbonate species as molecular CO2 (volatile).
    """
    T: float
    pH: float
    I: float
    species_molalities: Dict[str, float]
    alpha_NH3: float
    alpha_H2S: float
    alpha_CO2: float


def _davies_log_gamma_pm(I: float, T: float = 298.15) -> float:
    """Davies activity coefficient for a 1:1 monovalent ion.

    log10 γ_± = −A_DH(T) · z² · [√I / (1 + √I) − 0.3·I]

    Accurate to ~5 % at I ≤ 0.5 mol/kg, ~15 % at I = 2 mol/kg.
    For I > 2 mol/kg use a Pitzer model (specific-ion interaction) —
    the Davies expression turns over and ceases to be physical.

    Parameters
    ----------
    I : float
        Ionic strength [mol/kg].
    T : float
        Temperature [K].  T-dependence of A_DH is weak (5 % over 273-373 K).

    Returns
    -------
    float
        log10 γ_± for a |z|=1 ion.  Multiply by z² for higher-charge
        species.
    """
    if I <= 0:
        return 0.0
    # Cap at 2 mol/kg — Davies turns over above this and goes positive,
    # which is unphysical.  Beyond 2 mol/kg the value at 2 is held flat
    # as a reasonable engineering bound; users wanting accuracy here
    # should use a Pitzer model.
    I_eff = min(I, 2.0)
    # A_DH(T) — Robinson-Stokes 1959 polynomial fit, water solvent
    T_C = T - 273.15
    A_DH = 0.4918 + 6.6098e-4 * T_C + 5.0231e-6 * T_C ** 2
    sI = np.sqrt(I_eff)
    return -A_DH * (sI / (1.0 + sI) - 0.3 * I_eff)


def speciate(T: float, m_NH3_total: float, m_H2S_total: float,
              m_CO2_total: float = 0.0,
              extra_strong_cations: float = 0.0,
              extra_strong_anions: float = 0.0,
              apply_davies_gammas: bool = False,
              max_gamma_iter: int = 20,
              gamma_tol: float = 1e-4) -> SourWaterSpeciation:
    """Compute equilibrium speciation and pH of a sour-water sample.

    Solves the system:
        K_a(NH4+) = [NH3][H+] / [NH4+]
        K_a(H2S)  = [HS-][H+] / [H2S]
        K_a1(CO2) = [HCO3-][H+] / [CO2]
        K_w       = [H+][OH-]
        balances:
            [NH3] + [NH4+] = m_NH3_total
            [H2S] + [HS-]  = m_H2S_total
            [CO2] + [HCO3-] = m_CO2_total
            [Na+] (or extra cation) + [NH4+] + [H+]
                = [Cl-] (or extra anion) + [HS-] + [HCO3-] + [OH-]

    Parameters
    ----------
    T : float
        Temperature [K].
    m_NH3_total : float
        Total ammonia in solution [mol/kg], including NH₄⁺.
    m_H2S_total : float
        Total sulfide in solution [mol/kg], including HS⁻.
    m_CO2_total : float
        Total dissolved CO₂ [mol/kg], including HCO₃⁻.
    extra_strong_cations : float
        Sum of strong-cation molalities (Na+, K+, etc.) NOT generated
        from the weak-electrolyte equilibria.  Default 0.
    extra_strong_anions : float
        Sum of strong-anion molalities (Cl-, SO4²-, etc.) NOT generated
        from the weak-electrolyte equilibria.  Default 0.
    apply_davies_gammas : bool, default False (v0.9.116)
        If True, apply Davies activity coefficient corrections to all
        K_a values via the relation
            K_a^effective = K_a^thermo · γ_HA / (γ_A · γ_H)
        Each acid-base pair (NH4+/NH3, H2S/HS-, CO2/HCO3-, H2O/OH-)
        gets divided by γ_±² where γ_± is computed from the converged
        ionic strength.  The molecular-species γ is taken as 1 (a good
        approximation at I < ~5 mol/kg; full treatment uses Setschenow
        coefficients in :class:`SourWaterActivityModel`).  The total
        ionic strength is iterated to self-consistency in ≤ 5 outer
        passes for typical inputs.
    max_gamma_iter : int, default 20
        Outer-loop iteration limit for γ self-consistency.
    gamma_tol : float, default 1e-4
        Convergence tolerance on |Δlog10 γ_±|.

    Returns
    -------
    SourWaterSpeciation
        Solved speciation.

    Notes
    -----
    For low total concentrations (< 1 mol/kg) and pH near neutral,
    γ_i = 1 is a good approximation and ``apply_davies_gammas=False``
    is fine.  At high I (≥ 2 mol/kg, e.g. brine background or
    refinery sour water with ammonium chloride loading), the Davies
    correction shifts pK_a values by 0.3-0.7 units which materially
    changes both pH and the molecular/ionic split.  For I > 5 mol/kg
    use a Pitzer model with specific-ion parameters.
    """
    # Get K's at the operating T (thermodynamic, infinite-dilution)
    K_NH4_thermo = dissociation_K("NH4+", T)
    K_H2S_thermo = dissociation_K("H2S", T)
    K_CO2_thermo = dissociation_K("CO2", T)
    K_w_thermo = dissociation_K("H2O", T)

    log_gamma_pm_prev = 0.0   # γ_± = 1 initially
    for outer in range(max_gamma_iter if apply_davies_gammas else 1):
        # Effective stoichiometric K's after γ correction.
        # For a generic acid HA → A^q + H+:
        #   K_thermo = γ_A · γ_H · m_A · m_H / (γ_HA · m_HA)
        #   K_bare   = m_A · m_H / m_HA = K_thermo · γ_HA / (γ_A · γ_H)
        #
        # NH4+ → NH3 + H+ (cationic acid, z_HA=+1, z_A=0, z_H=+1):
        #   γ_NH4 ≈ γ_+, γ_NH3 ≈ 1, γ_H ≈ γ_+
        #   K_NH4_bare = K_NH4_thermo · γ_+ / (1 · γ_+) = K_NH4_thermo
        #   → cationic-acid γ correction CANCELS (well-known result).
        # H2S → HS- + H+, CO2 → HCO3- + H+, H2O → OH- + H+ (neutral acids):
        #   γ_HA ≈ 1, γ_A ≈ γ_-, γ_H ≈ γ_+
        #   K_bare = K_thermo / (γ_- · γ_+) = K_thermo / γ_±²
        log_gamma_pm = log_gamma_pm_prev
        gamma_pm_sq = (10.0 ** log_gamma_pm) ** 2
        K_NH4 = K_NH4_thermo                           # no γ correction
        K_H2S = K_H2S_thermo / gamma_pm_sq             # neutral acid
        K_CO2 = K_CO2_thermo / gamma_pm_sq
        K_w = K_w_thermo / gamma_pm_sq

        # Charge balance as a function of [H+]:
        #   [Na+] + [NH4+] + [H+] - [Cl-] - [HS-] - [HCO3-] - [OH-] = 0
        def charge_balance(h):
            nh4 = m_NH3_total / (1.0 + K_NH4 / h)
            hs = m_H2S_total / (1.0 + h / K_H2S)
            hco3 = m_CO2_total / (1.0 + h / K_CO2)
            oh = K_w / h
            return (extra_strong_cations + nh4 + h
                     - extra_strong_anions - hs - hco3 - oh)

        # Bracket the root in [H+]
        h_lo, h_hi = 1e-13, 1.0
        f_lo = charge_balance(h_lo)
        f_hi = charge_balance(h_hi)
        if f_lo * f_hi > 0:
            for h_try in [1e-14, 10.0, 100.0]:
                if charge_balance(h_try) * f_lo < 0:
                    h_hi = h_try if h_try > h_lo else h_lo
                    h_lo = h_lo if h_try > h_lo else h_try
                    break
        for _ in range(80):
            h_mid = np.sqrt(h_lo * h_hi)   # log-bisection
            f_mid = charge_balance(h_mid)
            if f_mid * charge_balance(h_lo) < 0:
                h_hi = h_mid
            else:
                h_lo = h_mid
            if abs(np.log10(h_hi / h_lo)) < 1e-10:
                break
        h_plus = np.sqrt(h_lo * h_hi)

        # Compute species at this h+
        nh4_x = m_NH3_total / (1.0 + K_NH4 / h_plus)
        hs_x = m_H2S_total / (1.0 + h_plus / K_H2S)
        hco3_x = m_CO2_total / (1.0 + h_plus / K_CO2)
        oh_x = K_w / h_plus

        if not apply_davies_gammas:
            break

        # Update γ_± from ionic strength
        I_curr = 0.5 * (extra_strong_cations + extra_strong_anions
                            + nh4_x + h_plus + hs_x + hco3_x + oh_x)
        log_gamma_pm_new = _davies_log_gamma_pm(I_curr, T)
        if abs(log_gamma_pm_new - log_gamma_pm_prev) < gamma_tol:
            log_gamma_pm_prev = log_gamma_pm_new
            break
        # Damped update for stability at very high I
        log_gamma_pm_prev = (0.5 * log_gamma_pm_new
                                  + 0.5 * log_gamma_pm_prev)

    # Compute all species (use final K values which already include γ)
    nh4 = m_NH3_total / (1.0 + K_NH4 / h_plus)
    nh3 = m_NH3_total - nh4
    hs = m_H2S_total / (1.0 + h_plus / K_H2S)
    h2s = m_H2S_total - hs
    hco3 = m_CO2_total / (1.0 + h_plus / K_CO2)
    co2 = m_CO2_total - hco3
    oh = K_w / h_plus

    species = {
        "H+": h_plus, "OH-": oh,
        "NH3": nh3, "NH4+": nh4,
        "H2S": h2s, "HS-": hs,
        "CO2": co2, "HCO3-": hco3,
    }

    # Ionic strength (using charge magnitudes)
    I = 0.5 * (h_plus + oh + nh4 + hs + hco3
                + extra_strong_cations + extra_strong_anions)

    # Volatile fractions (molecular / total)
    alpha_NH3 = nh3 / m_NH3_total if m_NH3_total > 1e-30 else 1.0
    alpha_H2S = h2s / m_H2S_total if m_H2S_total > 1e-30 else 1.0
    alpha_CO2 = (co2 / m_CO2_total if m_CO2_total > 1e-30 else 1.0)

    return SourWaterSpeciation(
        T=T, pH=-np.log10(h_plus), I=I,
        species_molalities=species,
        alpha_NH3=alpha_NH3, alpha_H2S=alpha_H2S, alpha_CO2=alpha_CO2)


def effective_henry(species: str, T: float, pH: float) -> float:
    """Apparent Henry's-law coefficient at given pH [Pa·kg/mol].

    Accounts for partial dissociation: the ionic forms (NH₄⁺, HS⁻,
    HCO₃⁻) are non-volatile, so the **effective** P/m relationship
    uses only the molecular fraction.

    H_eff = H_molecular(T) · α_molecular(T, pH)

    where α = 1/(1 + 10^(pK_a - pH)) for an acid (H₂S, CO₂)
       or α = 1/(1 + 10^(pH - pK_a))   for a base (NH₃)

    This is the form a distillation column wants as its "P_sat"
    function for sour-water stripper modeling.
    """
    H = henry_constant(species, T)

    if species == "NH3":
        # NH₄⁺ ⇌ NH₃ + H⁺ has K_a; α_NH3 = 1/(1 + [H+]/K_a)
        K_a = dissociation_K("NH4+", T)
        h_plus = 10 ** (-pH)
        alpha = 1.0 / (1.0 + h_plus / K_a)
    elif species == "H2S":
        # H₂S ⇌ HS⁻ + H⁺; α_H2S = 1/(1 + K_a/[H+])
        K_a = dissociation_K("H2S", T)
        h_plus = 10 ** (-pH)
        alpha = 1.0 / (1.0 + K_a / h_plus)
    elif species == "CO2":
        K_a = dissociation_K("CO2", T)
        h_plus = 10 ** (-pH)
        alpha = 1.0 / (1.0 + K_a / h_plus)
    else:
        raise ValueError(
            f"effective_henry only defined for NH3/H2S/CO2, got {species!r}")
    return H * alpha
