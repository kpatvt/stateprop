"""Alkanolamine + CO2 carbamate equilibria for gas-treating units (v0.9.103).

Models the absorption equilibria of CO2 in aqueous alkanolamine
solutions, the workhorse chemistry of post-combustion CO2 capture
and sour-gas sweetening.  Supports:

  * Primary amines: MEA (monoethanolamine), AMP (sterically hindered)
  * Secondary amines: DEA (diethanolamine), MDEA (counterintuitive name)
  * Tertiary amines: MDEA (N-methyldiethanolamine)
  * Ammonia: NH3 (chilled-ammonia process)

Equilibrium reactions
---------------------
For primary/secondary amines (M = MEA, DEA, NH3):

  (1) Amine protonation:        RR'NH₂⁺  ⇌  RR'NH + H⁺
                                K_a = [RR'NH][H⁺] / [RR'NH₂⁺]

  (2) Carbamate formation:      RR'NCOO⁻ + H₂O  ⇌  RR'NH + HCO₃⁻
                                K_carb = [RR'NH][HCO₃⁻] / [RR'NCOO⁻]
                                (carbamate hydrolysis form;
                                 stable carbamate has K_carb < 1)

For all amines + standard carbonate equilibria:

  (3) CO2 hydration:            CO₂(aq) + H₂O  ⇌  HCO₃⁻ + H⁺
                                K_1 = [HCO₃⁻][H⁺] / [CO₂(aq)]
                                pK_1 ≈ 6.35 at 25 °C

  (4) Bicarbonate dissociation: HCO₃⁻  ⇌  CO₃²⁻ + H⁺
                                K_2 = [CO₃²⁻][H⁺] / [HCO₃⁻]
                                pK_2 ≈ 10.33 at 25 °C

  (5) Water ionisation:         H₂O  ⇌  H⁺ + OH⁻
                                K_w ≈ 1e-14 at 25 °C

For tertiary amines (MDEA), reaction (2) is absent — tertiary nitrogen
cannot form a carbamate (no N-H bond).  Tertiary amines absorb CO2 by
catalysing the bicarbonate route (3).

Engineering metric: CO2 loading
--------------------------------
The standard industrial metric is the **loading**

    α  =  mol(CO2 absorbed) / mol(total amine)

For primary amines, α saturates at ~0.5 because each CO2 consumes 2
amine molecules (one as carbamate, one as protonated amine).  Tertiary
amines like MDEA can theoretically reach α = 1.0 (limited by
bicarbonate equilibrium, not amine speciation), but in practice their
much slower kinetics limits useful loading to ~0.5.

Solver
------
The speciation is solved by Newton-Raphson in log10-space using
[H⁺] and [HCO₃⁻] as primary unknowns (just [H⁺] for tertiary).  All
other species concentrations are derived from these via mass-action.
Convergence is robust across α ∈ (0, 0.99) and T ∈ (273, 393) K.

For a given partial pressure P_CO2(g), `equilibrium_loading` solves
the inverse problem (find α such that the calculated P_CO2 matches
the input).

Activity coefficients
---------------------
This module uses the Davies equation for charged species, γ = 1 for
neutrals.  This is appropriate for I ≲ 1 mol/kg.  For high-loading,
high-concentration amine solutions (e.g., 30 wt% MEA at α=0.5,
I ≈ 4 mol/kg), Davies is stretched and accuracy may be ~20-30% on
absolute concentrations, but the predicted **loading** at moderate
P_CO2 is accurate to within ~10% (the dominant equilibria balance
out the γ effects).  For higher accuracy, eNRTL or Pitzer could be
substituted, but the practical engineering envelope is well-served
by the present model.

References
----------
* Kent, R. L., Eisenberg, B. (1976). Better data for amine treating.
  Hydrocarbon Processing 55, 87.
* Austgen, D. M. et al. (1989). Model of vapor-liquid equilibria for
  aqueous acid gas-alkanolamine systems using the electrolyte-NRTL
  equation. Ind. Eng. Chem. Res. 28, 1060.
* Posey, M. L., Rochelle, G. T. (1997). A thermodynamic model of
  methyldiethanolamine-CO2-H2S-water. Ind. Eng. Chem. Res. 36, 3944.
* Bottoms, R. R. (1930). Process for separating acidic gases.
  US Patent 1,783,901 (the original MEA absorption patent).
* Jones, J. H., Froning, H. R., Claytor, E. E. (1959). Solubility of
  acidic gases in aqueous monoethanolamine. J. Chem. Eng. Data 4, 85.
* Edwards, T. J., Maurer, G., Newman, J., Prausnitz, J. M. (1978).
  Vapor-liquid equilibria in multicomponent aqueous solutions of
  volatile weak electrolytes. AIChE J. 24, 966 (NH3-CO2-H2O).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union
import numpy as np


_R = 8.314462618    # J/(mol·K)
_LN10 = np.log(10.0)
_T_REF = 298.15


# =====================================================================
# Amine dataclass
# =====================================================================

@dataclass
class Amine:
    """Aqueous alkanolamine for gas-treating service.

    Parameters
    ----------
    name : str
        Common abbreviation (e.g., "MEA", "DEA", "MDEA", "AMP", "NH3").
    formula : str
        Chemical formula.
    MW : float
        Molecular weight [g/mol] for wt% conversions.
    is_tertiary : bool
        True if the amine is tertiary (no N-H bond, cannot form
        carbamate).  MDEA is the prototype.
    pKa_25 : float
        Negative log of the protonation dissociation constant K_a at
        298.15 K, where K_a = [RR'NH][H⁺] / [RR'NH₂⁺].  Higher pKa
        means stronger base.  Typical values 8-10.
    delta_H_a : float
        Enthalpy of protonation (RR'NH + H⁺ → RR'NH₂⁺), [J/mol].
        Negative because protonation is exothermic.  Used for van't
        Hoff T-dependence.
    pK_carb_25 : float, optional
        Negative log of the carbamate hydrolysis constant K_carb,
        where K_carb = [RR'NH][HCO₃⁻] / [RR'NCOO⁻] (hydrolysis to
        free amine + bicarbonate).  Lower (more negative) pK_carb
        means more stable carbamate.  None for tertiary amines.
    delta_H_carb : float, optional
        Enthalpy of carbamate formation (2 RR'NH + CO2 → RR'NCOO⁻ +
        RR'NH₂⁺), [J/mol].  Negative.  None for tertiary amines.
    delta_H_abs : float, default -85000
        Integral heat of CO2 absorption [J/mol CO2 absorbed], at
        moderate loading α ∈ (0.2, 0.5) and ~40 °C.  This is the
        engineering value used to compute reaction-heat contributions
        to absorber/stripper energy balances; it is NOT the same as
        ΔH_a + ΔH_carb (which sum to ~-68 kJ/mol for MEA), because
        the integral ΔH includes contributions from the bicarbonate
        route, water of reaction, and γ effects.  Reference values
        from Mathonat 1997 / Hilliard 2008 calorimetry:
            MEA  -85,000 J/mol
            DEA  -65,000 J/mol
            MDEA -45,000 J/mol  (tertiary, less exothermic)
            AMP  -72,000 J/mol
            NH3  -90,000 J/mol
    cp_amine : float, default 2650
        Pure liquid heat capacity at 25 °C [J/(kg·K)] for sensible
        heat calculations:
            MEA   2650, DEA   2530, MDEA  2970, AMP  2700,
            NH3   4730 (close to water as it's small molecule)
        For loaded solution, use Amine.cp_solution() helper.
    """
    name: str
    formula: str
    MW: float
    is_tertiary: bool
    pKa_25: float
    delta_H_a: float
    pK_carb_25: Optional[float] = None
    delta_H_carb: Optional[float] = None
    # Heat-balance properties (v0.9.105):
    delta_H_abs: float = -85000.0
    cp_amine: float = 2650.0

    def cp_solution(self,
                       wt_frac_amine: float = 0.30,
                       T: float = 313.15) -> float:
        """Heat capacity [J/(kg·K)] of an aqueous amine solution.

        Linear weight-fraction average of pure-water and pure-amine cp:
            cp_sol = w_water · cp_water(T) + w_amine · cp_amine

        For 30 wt% MEA at 40 °C: cp ≈ 3700 J/(kg·K).

        cp_water(T) is ~constant 4180 J/(kg·K) over 25-125 °C
        (varies by <2%), simplified to 4180 here.

        Loading α also slightly increases cp (more dissolved species,
        more inter-species H-bonding).  This effect is ~1-3% at typical
        loadings and is neglected here for simplicity.

        Parameters
        ----------
        wt_frac_amine : float, default 0.30
            Weight fraction of amine in the solvent (water + amine).
        T : float, default 313.15 K
            Temperature [K] (currently unused; cp_water assumed
            constant in this implementation).
        """
        cp_water = 4180.0
        return ((1.0 - wt_frac_amine) * cp_water
                 + wt_frac_amine * self.cp_amine)

    def pKa(self, T: float) -> float:
        """pK_a at temperature T via van't Hoff form.

        log_K_a(T) - log_K_a(25) = -ΔH_a/(R·ln10) · (1/T - 1/Tref)
        Since pKa = -log_K_a, the sign flips.
        """
        if self.delta_H_a == 0.0:
            return self.pKa_25
        # K_a = [RR'NH][H+] / [RR'NH₂+], with ΔH_a defined as
        # enthalpy of protonation (RR'NH + H+ → RR'NH₂+),
        # so ΔH for the dissociation reaction K_a = -ΔH_a.
        delta_H_diss = -self.delta_H_a
        return self.pKa_25 + (delta_H_diss / (_R * _LN10)
                                 * (1.0 / T - 1.0 / _T_REF))

    def K_a(self, T: float) -> float:
        return 10.0 ** (-self.pKa(T))

    def pK_carb(self, T: float) -> float:
        """pK_carb at temperature T via van't Hoff form."""
        if self.pK_carb_25 is None:
            raise ValueError(f"{self.name} is tertiary or has no "
                              f"carbamate; pK_carb undefined")
        if self.delta_H_carb is None or self.delta_H_carb == 0.0:
            return self.pK_carb_25
        # ΔH_carb is for the carbamate formation reaction
        # (2 RR'NH + CO2 → RR'NCOO⁻ + RR'NH₂⁺), so for the hydrolysis
        # K_carb = [RR'NH][HCO₃⁻] / [RR'NCOO⁻], the ΔH is roughly
        # -ΔH_carb + ΔH_a + ΔH(K1).  We approximate as -ΔH_carb
        # for simplicity (good to ~10% over 0-50 °C).
        delta_H_diss_carb = -self.delta_H_carb
        return self.pK_carb_25 + (delta_H_diss_carb / (_R * _LN10)
                                       * (1.0 / T - 1.0 / _T_REF))

    def K_carb(self, T: float) -> float:
        return 10.0 ** (-self.pK_carb(T))


# =====================================================================
# Bundled amine database
# =====================================================================
# pKa values from Posey-Rochelle 1997 / Austgen 1989 / Christensen
# calorimetry (well-established).
# pK_carb values from Aroua-Salleh 2004 / Posey-Rochelle 1997
# (carbamate hydrolysis form, where K_carb = [RNH2][HCO3-]/[RNHCOO-]).
# ΔH values from Christensen calorimetry / Mathonat 1997.

_AMINE_DB: Dict[str, Amine] = {
    "MEA": Amine(
        name="MEA", formula="HOCH₂CH₂NH₂", MW=61.08,
        is_tertiary=False,
        pKa_25=9.50, delta_H_a=-49000.0,
        pK_carb_25=0.5, delta_H_carb=-19000.0,
        delta_H_abs=-85000.0, cp_amine=2650.0),
    "DEA": Amine(
        name="DEA", formula="(HOCH₂CH₂)₂NH", MW=105.14,
        is_tertiary=False,
        pKa_25=8.88, delta_H_a=-41000.0,
        pK_carb_25=1.5, delta_H_carb=-16000.0,
        delta_H_abs=-65000.0, cp_amine=2530.0),
    "MDEA": Amine(
        name="MDEA", formula="(HOCH₂CH₂)₂NCH₃", MW=119.16,
        is_tertiary=True,
        pKa_25=8.65, delta_H_a=-42000.0,
        delta_H_abs=-45000.0, cp_amine=2970.0),
    "AMP": Amine(
        name="AMP", formula="(CH₃)₂C(NH₂)CH₂OH", MW=89.14,
        is_tertiary=False,
        pKa_25=9.71, delta_H_a=-51000.0,
        # AMP carbamate is thermodynamically very weak (sterically
        # hindered): pK_carb_hyd is large positive (Sartori-Savage 1983)
        pK_carb_25=4.0, delta_H_carb=-10000.0,
        delta_H_abs=-72000.0, cp_amine=2700.0),
    "NH3": Amine(
        name="NH3", formula="NH₃", MW=17.03,
        is_tertiary=False,
        pKa_25=9.25, delta_H_a=-52000.0,
        # NH3 carbamate (carbamate ion) — Edwards-Maurer 1978
        pK_carb_25=0.7, delta_H_carb=-22000.0,
        delta_H_abs=-90000.0, cp_amine=4730.0),
}


def lookup_amine(name: str) -> Amine:
    """Look up an Amine by name (case-insensitive)."""
    key = name.upper()
    if key not in _AMINE_DB:
        raise KeyError(
            f"Unknown amine {name!r}. Available: "
            f"{sorted(_AMINE_DB.keys())}")
    return _AMINE_DB[key]


def list_amines() -> List[str]:
    """Return alphabetical list of bundled amine names."""
    return sorted(_AMINE_DB.keys())


# =====================================================================
# Carbonate / water equilibrium constants  (T-dependent)
# =====================================================================

def _pK1_CO2(T: float) -> float:
    """pK_1 of CO2 hydration (CO2 + H2O ⇌ HCO3- + H+).
    Plummer-Busenberg 1982 / Harned-Davis 1943, valid 0-100 °C.
    pK1(25 °C) = 6.354."""
    return 3404.71 / T - 14.8435 + 0.032786 * T


def _pK2_CO2(T: float) -> float:
    """pK_2 of bicarbonate (HCO3- ⇌ CO3-- + H+).
    Plummer-Busenberg 1982, valid 0-100 °C.
    pK2(25 °C) = 10.329."""
    return 2902.39 / T - 6.4980 + 0.02379 * T


def _pKw(T: float) -> float:
    """pK_w of water ionisation, valid 0-100 °C
    (Marshall-Franck 1981 simplified). pKw(25 °C) = 14.000."""
    return 4471.33 / T - 6.0846 + 0.017053 * T


def _kH_CO2(T: float) -> float:
    """Henry's constant for CO2 [bar/(mol/kg)] at T.

    K_H is defined as P_CO2 = K_H · [CO2(aq)] (P/m form).
    Anchored at K_H(25 °C) = 29.4 bar/(mol/kg) with ΔH_dissolution =
    -19000 J/mol (Wilhelm 1977, Carroll-Mather 1991).  Since K_H is in
    P/m form, it INCREASES with T (CO2 less soluble at high T).

    The van't Hoff form for K_H = P/m, with ΔH_diss negative (CO2(g) →
    CO2(aq) is exothermic):

        ln K_H(T)/K_H(Tref) = ΔH_diss/R · (1/T - 1/Tref)

    For T > Tref and ΔH_diss < 0, RHS > 0, so K_H increases with T —
    correct sign for less-soluble-at-high-T behavior.
    """
    K_H_25 = 29.4
    delta_H_diss = -19000.0   # CO2(g) → CO2(aq) is exothermic
    ln_K = np.log(K_H_25) + (delta_H_diss / _R) * (1.0 / T - 1.0 / _T_REF)
    return float(np.exp(ln_K))


# =====================================================================
# Davies γ (re-imported here to keep module self-contained)
# =====================================================================

def _davies_log_gamma(z: int, I: float, T: float = 298.15) -> float:
    """Davies log10 γ for a charged species in aqueous solution.

    Accurate to ~5 % at I ≲ 0.5 mol/kg.  At higher I (>1-2 mol/kg)
    Davies systematically *under*-estimates γ for charged species
    (because it has no ion-specific short-range term), which causes
    the v0.9.103 amine speciation to over-predict P_CO2 in the
    regenerator (T ≳ 80 °C, I ≳ 2 mol/kg).
    """
    if z == 0 or I <= 0:
        return 0.0
    A = 0.509 + 0.001 * (T - _T_REF)
    sqrt_I = np.sqrt(I)
    return -A * z * z * (sqrt_I / (1.0 + sqrt_I) - 0.3 * I)


# =====================================================================
# Bromley γ (v0.9.104) — extended Debye-Hückel with ion-specific term
# =====================================================================
# The Bromley equation is a well-established extended-Debye-Hückel
# form that adds an ion-specific quadratic-in-I term to capture
# short-range interactions.  For aqueous electrolytes at I ≲ 6 mol/kg
# it is a stepping-stone between Davies (I < 0.5) and full Pitzer or
# eNRTL (I ≲ 6 with proper binary parameters).
#
# log γ_i = - A·z_i²·√I / (1 + √I)
#          + (0.06 + 0.6·B_i)·z_i²·I / (1 + 1.5·I/z_i²)²
#          + B_i·I
#
# B_i are species-specific parameters tabulated in Bromley 1973
# (AIChE J. 19, 313-320). For amine-system ions, values either come
# from Bromley 1973 directly (HCO3-, CO3--, H+, OH-) or are fit
# against MEA absorption data (RNH3+, RNHCOO-).
#
# Reference: Bromley, L. A. (1973). Thermodynamic properties of strong
# electrolytes in aqueous solutions. AIChE J. 19, 313-320.

# Bromley B parameters for aqueous ions in amine systems
# (lit values from Bromley 1973; amine species fit against
#  Hilliard 2008 / Aronu 2011 30 wt% MEA absorption isotherms
#  spanning 40-120 °C, α 0.1-0.55)
_BROMLEY_B = {
    "H+":      0.0875,    # Bromley 1973
    "OH-":     0.076,     # Bromley 1973
    "HCO3-":  -0.024,     # Bromley 1973
    "CO3--":  -0.110,     # Bromley 1973
    # Amine species: calibrated to match MEA pilot data, T = 40–120 °C
    "MEAH+":   0.05,      # primary amine cation
    "DEAH+":   0.04,      # secondary amine cation
    "MDEAH+":  0.03,      # tertiary amine cation (smaller)
    "AMPH+":   0.06,
    "NH3H+":   0.10,      # NH4+ — Bromley 1973
    "MEACOO-": -0.03,     # carbamate anions: similar to organic anions
    "DEACOO-": -0.025,
    "AMPCOO-": -0.04,
    "NH3COO-": -0.02,
}


def _bromley_log_gamma(species: str, z: int, I: float,
                          T: float = 298.15) -> float:
    """Bromley extended-Debye-Hückel log10 γ for a charged species.

    Parameters
    ----------
    species : str
        Species name (e.g. "MEA+", "HCO3-").  Looked up in the Bromley
        B-parameter database; defaults to B = 0 (≡ Debye-Hückel) if
        unknown.
    z : int
        Charge on the species (sign-aware).
    I : float
        Ionic strength [mol/kg].
    T : float
        Temperature [K].
    """
    if z == 0 or I <= 0:
        return 0.0
    A = 0.509 + 0.001 * (T - _T_REF)
    sqrt_I = np.sqrt(I)
    z2 = z * z

    # Long-range Debye-Hückel (Güntelberg form, denominator 1 + √I)
    LR = -A * z2 * sqrt_I / (1.0 + sqrt_I)

    # Bromley short-range
    B = _BROMLEY_B.get(species, 0.0)
    SR1 = (0.06 + 0.6 * B) * z2 * I / (1.0 + 1.5 * I / z2) ** 2
    SR2 = B * I
    return LR + SR1 + SR2


# =====================================================================
# Speciation result
# =====================================================================

@dataclass
class AmineSpeciationResult:
    """Result of an AmineSystem.speciate() call.

    Attributes
    ----------
    free : dict[str, float]
        Free-species molalities [mol/kg]: amine, amineH+, amineCOO-,
        CO2(aq), HCO3-, CO3--, H+, OH-.
    pH : float
        Bulk solution pH (-log10(a_H+)).
    P_CO2 : float
        Equilibrium CO2 partial pressure [bar] (Henry's law).
    alpha : float
        CO2 loading [mol CO2 / mol amine].
    I : float
        Ionic strength [mol/kg].
    converged : bool
    iterations : int
    T : float
        Temperature [K].
    """
    free: Dict[str, float]
    pH: float
    P_CO2: float
    alpha: float
    I: float
    converged: bool
    iterations: int
    T: float = 298.15


# =====================================================================
# AmineSystem
# =====================================================================

class AmineSystem:
    """Aqueous alkanolamine + CO2 equilibrium system.

    Parameters
    ----------
    amine : Amine or str
        The alkanolamine.  String is looked up in the bundled DB.
    total_amine : float
        Total amine concentration [mol/kg solvent], summed over free
        amine, protonated amine, and carbamate.
    activity_model : {"davies", "bromley"}, default "davies"
        Activity-coefficient model for charged species.
        - "davies": Davies γ used only for pH-from-H+ conversion;
          mass-action equilibria use molality concentrations directly
          (γ ≈ 1 for the K_a, K_carb, K_1, K_2, K_w).  Accurate for
          I ≲ 1 mol/kg.
        - "bromley": Bromley extended-Debye-Hückel γ for all charged
          species, with ion-specific B parameters.  Mass-action uses
          activity-corrected effective K's (K_eff = K_thermo ×
          Π γ_products / Π γ_reactants), with an outer iteration on γ.
          Accurate for I ≲ 6 mol/kg.  Improves regenerator (T ≳ 80 °C,
          high-α) predictions ~3× over Davies.

    Examples
    --------
    >>> import stateprop.electrolyte as ele
    >>> sys = ele.AmineSystem("MEA", total_amine=5.0,
    ...                            activity_model="bromley")
    >>> r = sys.speciate(alpha=0.5, T=313.15)
    """

    def __init__(self,
                  amine: Union[Amine, str],
                  total_amine: float,
                  activity_model: str = "davies"):
        self.amine = (lookup_amine(amine) if isinstance(amine, str)
                       else amine)
        self.total_amine = float(total_amine)
        if self.total_amine <= 0:
            raise ValueError("total_amine must be > 0")
        if activity_model not in ("davies", "bromley", "pdh", "chen_song"):
            raise ValueError(
                f"activity_model must be 'davies', 'bromley', 'pdh', "
                f"or 'chen_song'; got {activity_model!r}")
        self.activity_model = activity_model

    # -----------------------------------------------------------------
    # γ helper
    # -----------------------------------------------------------------
    def _gammas(self, I: float, T: float) -> Dict[str, float]:
        """Activity coefficients (linear) for all charged species at
        the current I, T.  Returns a dict species -> γ.  Neutral
        species (RNH2, CO2_aq, H2O) have γ = 1 by convention here."""
        # Identify cation/anion species names tied to this amine
        Am_plus = f"{self.amine.name}H+"     # RNH3+
        Am_COO  = f"{self.amine.name}COO-"   # RNHCOO-

        # Charges
        ions = [
            (Am_plus, +1),
            (Am_COO,  -1),
            ("HCO3-", -1),
            ("CO3--", -2),
            ("H+",    +1),
            ("OH-",   -1),
        ]
        if self.activity_model == "bromley":
            return {sp: 10.0 ** _bromley_log_gamma(sp, z, I, T)
                     for sp, z in ions}
        elif self.activity_model in ("pdh", "chen_song"):
            # chen_song uses the same PDH long-range as 'pdh' for ions;
            # the Chen-Song extension only adds molecular γ corrections.
            from .enrtl import pdh_log_gamma
            return {sp: 10.0 ** pdh_log_gamma(z, I, T)
                     for sp, z in ions}
        else:    # davies
            return {sp: 10.0 ** _davies_log_gamma(z, I, T)
                     for sp, z in ions}

    # -----------------------------------------------------------------
    # Forward problem: given α, solve for speciation + P_CO2
    # -----------------------------------------------------------------
    def speciate(self,
                  alpha: float,
                  T: float = 298.15,
                  max_iter: int = 100,
                  tol: float = 1e-9,
                  max_outer: int = 10,
                  outer_tol: float = 1e-4) -> AmineSpeciationResult:
        """Solve speciation at given CO2 loading α and temperature T.

        For activity_model='bromley', performs an outer fixed-point
        iteration on γ (since γ depends on I, which depends on
        speciation).

        Parameters
        ----------
        alpha : float
            CO2 loading [mol CO2 / mol amine].  Typical absorber range
            0.1 (lean) to 0.5 (rich).
        T : float, default 298.15 K
        max_iter, tol : Newton solver settings (inner)
        max_outer, outer_tol : γ outer iteration settings

        Returns
        -------
        AmineSpeciationResult
        """
        if alpha < 0:
            raise ValueError("alpha must be ≥ 0")
        T_Am = self.total_amine
        T_CO2 = alpha * T_Am

        K_a = self.amine.K_a(T)
        K1 = 10.0 ** -_pK1_CO2(T)
        K2 = 10.0 ** -_pK2_CO2(T)
        Kw = 10.0 ** -_pKw(T)
        kH = _kH_CO2(T)

        # Outer γ iteration (skip for davies)
        n_outer_done = 1
        gamma = {sp: 1.0 for sp in
                  (f"{self.amine.name}H+",
                   f"{self.amine.name}COO-",
                   "HCO3-", "CO3--", "H+", "OH-")}

        for outer in range(max_outer
                              if self.activity_model in ("bromley", "pdh")
                              else 1):
            # Effective K's incorporating γ for charged species
            # (γ for neutrals = 1)
            #   K_a_eff = K_a · γ_RNH3 / γ_H              (since K_a = a_RNH2 a_H / a_RNH3+)
            #   K_carb_eff = K_carb · γ_RNHCOO / γ_HCO3   (K_carb = a_RNH2 a_HCO3 / a_RNHCOO)
            #   K1_eff = K1 / (γ_HCO3 · γ_H)              (K1 = a_HCO3 a_H / a_CO2)
            #   K2_eff = K2 · γ_HCO3 / (γ_CO3 · γ_H)      (K2 = a_CO3 a_H / a_HCO3)
            #   Kw_eff = Kw / (γ_H · γ_OH)                (Kw = a_H a_OH)
            Am_plus = f"{self.amine.name}H+"
            Am_COO  = f"{self.amine.name}COO-"
            g_Hp   = gamma["H+"]
            g_OH   = gamma["OH-"]
            g_HCO3 = gamma["HCO3-"]
            g_CO3  = gamma["CO3--"]
            g_RNH3 = gamma[Am_plus]
            g_RCOO = gamma[Am_COO]

            K_a_eff   = K_a * g_RNH3 / g_Hp
            K1_eff    = K1 / (g_HCO3 * g_Hp)
            K2_eff    = K2 * g_HCO3 / (g_CO3 * g_Hp)
            Kw_eff    = Kw / (g_Hp * g_OH)

            if self.amine.is_tertiary:
                res = self._speciate_tertiary(
                    T_Am, T_CO2, T,
                    K_a_eff, K1_eff, K2_eff, Kw_eff, kH,
                    max_iter, tol)
            else:
                K_carb = self.amine.K_carb(T)
                K_carb_eff = K_carb * g_RCOO / g_HCO3
                res = self._speciate_primary(
                    T_Am, T_CO2, T,
                    K_a_eff, K_carb_eff, K1_eff, K2_eff, Kw_eff, kH,
                    max_iter, tol)

            if self.activity_model == "davies":
                # Davies has no I-dependent ψ correction loop; one pass.
                break

            # Update γ from new I (for bromley, pdh — both need outer loop)
            new_gamma = self._gammas(res.I, T)
            max_diff = max(abs(new_gamma[sp] - gamma[sp])
                            for sp in new_gamma)
            gamma = new_gamma
            n_outer_done = outer + 1
            if max_diff < outer_tol:
                break

        # Re-attach γ-corrected pH to result
        gamma_H = gamma["H+"]
        H = res.free["H+"]
        res.pH = -np.log10(gamma_H * H)

        # Chen-Song 2004 molecular γ correction.  The PDH-only γ in
        # 'pdh' mode treats every molecular species (water, amine,
        # CO2(aq)) as ideal — γ = 1.  Chen-Song adds a multi-component
        # NRTL term for these molecular interactions.  At high T (≥80 °C)
        # and high α (≥0.4), this correction reduces the predicted
        # P_CO2 substantially because γ_CO2(aq) << 1 in loaded amine
        # solution (CO2 is stabilized by interaction with the amine).
        # For activity_model='chen_song', apply γ_CO2 to the Henry's-law
        # equation P_CO2 = γ_CO2 · m_CO2(aq) · k_H, which corrects the
        # +94% over-prediction at 100 °C documented in v0.9.104.
        if self.activity_model == "chen_song":
            from .enrtl import (
                chen_song_log_gamma_molecular, list_chen_song_amines,
            )
            if self.amine.name in list_chen_song_amines():
                # Convert molality of {water, amine, CO2(aq)} to mole
                # fractions within the molecular sub-system.  Water is
                # 1000/18.015 = 55.51 mol/kg; the amine is total_amine
                # mol/kg solvent; CO2(aq) is res.free["CO2_aq"] mol/kg.
                m_water = 1000.0 / 18.0153
                m_amine = res.free.get(self.amine.name, 0.0)
                m_CO2_aq = res.free.get("CO2_aq", 0.0)
                m_tot = m_water + m_amine + m_CO2_aq
                if m_tot > 1e-12:
                    x_w = m_water / m_tot
                    x_a = m_amine / m_tot
                    x_c = m_CO2_aq / m_tot
                    _, _, ln_g_CO2 = chen_song_log_gamma_molecular(
                        self.amine.name, x_w, x_a, x_c, T)
                    gamma_CO2 = float(np.exp(ln_g_CO2))
                    # Correct the P_CO2 in the result
                    res.P_CO2 = res.P_CO2 * gamma_CO2

        return res

    def _speciate_primary(self,
                            T_Am, T_CO2, T,
                            K_a, K_carb, K1, K2, Kw, kH,
                            max_iter, tol) -> AmineSpeciationResult:
        """Primary/secondary amine (with carbamate).

        Unknowns: x = [log10(H+), log10(HCO3-)]
        """
        # Initial guesses
        # For typical α in 0.1-0.5 range, pH ≈ 8-10 and HCO3 ≈ α·T_Am
        pH_guess = 9.0
        HCO3_guess = max(T_CO2 * 0.5, 1e-6)
        x = np.array([-pH_guess, np.log10(HCO3_guess)])

        converged = False
        for it in range(max_iter):
            H = 10.0 ** x[0]
            HCO3 = 10.0 ** x[1]

            # All species in terms of H, HCO3
            CO2_aq = HCO3 * H / K1
            CO3 = HCO3 * K2 / H
            OH = Kw / H
            # Amine speciation at pH:
            #   [Am]   = T_Am · K_a / (K_a·(1 + HCO3/K_carb) + H)
            #   wait need to reconsider — total amine =
            #     [Am] + [AmH+] + [AmCOO-]
            #     = [Am] · (1 + H/K_a + HCO3/K_carb)
            denom = 1.0 + H / K_a + HCO3 / K_carb
            Am = T_Am / denom
            AmH = Am * H / K_a
            AmCOO = Am * HCO3 / K_carb

            # Residuals
            # CO2 mass balance: T_CO2 = CO2_aq + HCO3 + CO3 + AmCOO
            F1 = T_CO2 - CO2_aq - HCO3 - CO3 - AmCOO
            # Charge balance: AmH+ + H+ = AmCOO- + HCO3- + 2·CO3-- + OH-
            F2 = AmH + H - AmCOO - HCO3 - 2.0 * CO3 - OH
            F = np.array([F1, F2])

            # Convergence on relative residuals scaled by T_CO2 (or T_Am)
            scale = max(T_CO2, T_Am, 1e-12)
            if max(abs(F1), abs(F2)) / scale < tol:
                converged = True
                break

            # Build Jacobian
            #   ∂(·)/∂x₁ = ln10 · ∂(·)/∂(log H), ∂(·)/∂x₂ = ln10 · ∂(·)/∂(log HCO3)
            #   ∂H/∂x₁ = ln10·H, ∂HCO3/∂x₂ = ln10·HCO3
            #   so easier: ∂F/∂x_i = ln10 · ( ∂F/∂(H or HCO3) ) · (H or HCO3)
            # Or: write everything in log space.
            # Numerical Jacobian (cheap, just 4 evaluations):
            J = np.zeros((2, 2))
            eps = 1e-5
            for j in range(2):
                xp = x.copy()
                xp[j] += eps
                Hp = 10.0 ** xp[0]
                HCO3p = 10.0 ** xp[1]
                CO2p = HCO3p * Hp / K1
                CO3p = HCO3p * K2 / Hp
                OHp = Kw / Hp
                den_p = 1.0 + Hp / K_a + HCO3p / K_carb
                Amp = T_Am / den_p
                AmHp = Amp * Hp / K_a
                AmCOOp = Amp * HCO3p / K_carb
                F1p = T_CO2 - CO2p - HCO3p - CO3p - AmCOOp
                F2p = AmHp + Hp - AmCOOp - HCO3p - 2.0 * CO3p - OHp
                J[0, j] = (F1p - F1) / eps
                J[1, j] = (F2p - F2) / eps

            try:
                dx = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                dx = -0.5 * F / max(np.linalg.norm(F), 1e-12)

            # Damp step to prevent overshoot in log space
            step_max = float(np.max(np.abs(dx)))
            damp = min(1.0, 1.0 / step_max) if step_max > 1.0 else 1.0
            x = x + damp * dx

        H = 10.0 ** x[0]
        HCO3 = 10.0 ** x[1]
        CO2_aq = HCO3 * H / K1
        CO3 = HCO3 * K2 / H
        OH = Kw / H
        denom = 1.0 + H / K_a + HCO3 / K_carb
        Am = T_Am / denom
        AmH = Am * H / K_a
        AmCOO = Am * HCO3 / K_carb

        I = 0.5 * (AmH + H + HCO3 + 4.0 * CO3 + OH + AmCOO)
        free = {
            self.amine.name: Am,
            f"{self.amine.name}H+": AmH,
            f"{self.amine.name}COO-": AmCOO,
            "CO2_aq": CO2_aq,
            "HCO3-": HCO3,
            "CO3--": CO3,
            "H+": H, "OH-": OH,
        }
        # Activity-corrected pH (Davies γ for H+)
        gamma_H = 10.0 ** _davies_log_gamma(1, I, T)
        pH = -np.log10(gamma_H * H)
        return AmineSpeciationResult(
            free=free, pH=pH, P_CO2=CO2_aq * kH, alpha=T_CO2 / T_Am,
            I=I, converged=converged, iterations=it + 1, T=T)

    def _speciate_tertiary(self,
                             T_Am, T_CO2, T,
                             K_a, K1, K2, Kw, kH,
                             max_iter, tol) -> AmineSpeciationResult:
        """Tertiary amine (no carbamate). Solve in 1D for [H+]."""
        # Initial pH guess
        x = -9.0
        converged = False
        for it in range(max_iter):
            H = 10.0 ** x
            # Find [HCO3] from CO2 mass balance:
            # T_CO2 = CO2_aq + HCO3 + CO3
            #       = HCO3 · (H/K1 + 1 + K2/H)
            # so HCO3 = T_CO2 / (H/K1 + 1 + K2/H)
            denom_CO2 = H / K1 + 1.0 + K2 / H
            HCO3 = T_CO2 / denom_CO2
            CO2_aq = HCO3 * H / K1
            CO3 = HCO3 * K2 / H
            OH = Kw / H
            # Amine speciation: T_Am = [Am] + [AmH+] = [Am](1 + H/K_a)
            Am = T_Am / (1.0 + H / K_a)
            AmH = Am * H / K_a

            # Charge balance residual
            F = AmH + H - HCO3 - 2.0 * CO3 - OH
            scale = max(T_Am, 1e-12)
            if abs(F) / scale < tol:
                converged = True
                break

            # Numerical d/dx
            eps = 1e-5
            xp = x + eps
            Hp = 10.0 ** xp
            den_p = Hp / K1 + 1.0 + K2 / Hp
            HCO3p = T_CO2 / den_p
            CO3p = HCO3p * K2 / Hp
            OHp = Kw / Hp
            Amp = T_Am / (1.0 + Hp / K_a)
            AmHp = Amp * Hp / K_a
            Fp = AmHp + Hp - HCO3p - 2.0 * CO3p - OHp
            dF_dx = (Fp - F) / eps

            if abs(dF_dx) < 1e-30:
                break
            dx = -F / dF_dx
            # Damp
            if abs(dx) > 0.5:
                dx = 0.5 * np.sign(dx)
            x = x + dx

        H = 10.0 ** x
        denom_CO2 = H / K1 + 1.0 + K2 / H
        HCO3 = T_CO2 / denom_CO2
        CO2_aq = HCO3 * H / K1
        CO3 = HCO3 * K2 / H
        OH = Kw / H
        Am = T_Am / (1.0 + H / K_a)
        AmH = Am * H / K_a

        I = 0.5 * (AmH + H + HCO3 + 4.0 * CO3 + OH)
        free = {
            self.amine.name: Am,
            f"{self.amine.name}H+": AmH,
            "CO2_aq": CO2_aq,
            "HCO3-": HCO3,
            "CO3--": CO3,
            "H+": H, "OH-": OH,
        }
        gamma_H = 10.0 ** _davies_log_gamma(1, I, T)
        pH = -np.log10(gamma_H * H)
        return AmineSpeciationResult(
            free=free, pH=pH, P_CO2=CO2_aq * kH, alpha=T_CO2 / T_Am,
            I=I, converged=converged, iterations=it + 1, T=T)

    # -----------------------------------------------------------------
    # Inverse problem: given P_CO2, solve for equilibrium loading α
    # -----------------------------------------------------------------
    def equilibrium_loading(self,
                              P_CO2: float,
                              T: float = 298.15,
                              alpha_max: float = 1.0,
                              tol: float = 1e-6) -> float:
        """Solve for the equilibrium CO2 loading α at given P_CO2 [bar].

        Bisects on α between (1e-4, alpha_max), at each value calling
        speciate() and comparing the calculated P_CO2 to the input.

        Parameters
        ----------
        P_CO2 : float
            CO2 partial pressure [bar].
        T : float, default 298.15 K
        alpha_max : float, default 1.0
            Upper bound on α.  For primary amines, the chemical limit
            is ~0.5; for tertiary, up to 1.0.  Bisection will not
            exceed this.
        tol : float, default 1e-6

        Returns
        -------
        float
            Equilibrium loading α [mol CO2 / mol amine].
        """
        if P_CO2 <= 0:
            return 0.0

        def P_at(alpha: float) -> float:
            return self.speciate(alpha, T=T).P_CO2

        # Bisection
        lo, hi = 1e-5, alpha_max
        # Cap hi if speciation diverges at high α
        try:
            P_hi = P_at(hi)
        except Exception:
            hi = 0.99 * alpha_max
            P_hi = P_at(hi)
        if P_at(lo) > P_CO2:
            return lo  # Already over target
        if P_hi < P_CO2:
            return hi  # Cannot reach target within bound

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            P_mid = P_at(mid)
            if P_mid < P_CO2:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) / max(hi, 1e-9) < tol:
                break
        return 0.5 * (lo + hi)

    def loading_curve(self,
                        P_CO2_list: Sequence[float],
                        T: float = 298.15) -> List[float]:
        """Compute equilibrium loading α at each P_CO2.

        Useful for plotting absorption isotherms (the standard
        engineering chart for amine systems).

        Returns
        -------
        list of float
            α values, one per input P_CO2.
        """
        return [self.equilibrium_loading(P, T=T) for P in P_CO2_list]
