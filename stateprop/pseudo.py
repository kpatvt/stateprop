"""Hydrocarbon pseudo-components for refinery and gas-processing problems.

Heavy hydrocarbon mixtures (crudes, diesel, atmospheric residue, naphtha
cuts, gas condensates) cannot be described by named molecular species
because they contain thousands of isomers above C7.  Instead, the
chemical engineering practice since the 1960s has been to characterize
each cut by:

    * Normal boiling point (NBP, K)
    * Specific gravity (SG, dimensionless, water = 1 at 60°F)
    * Molecular weight (MW, g/mol; can be estimated from NBP and SG)

These two-or-three numbers determine — via a network of empirical
correlations — the critical properties (Tc, Pc, Vc), acentric factor,
ideal-gas heat capacity, latent heat of vaporization, and liquid density
needed to drop the pseudo-component into a cubic equation of state, an
activity model, or a distillation column.

This module implements the industry-standard correlations:

    * Riazi-Daubert (1980, 1987) for Tc, Pc, MW from NBP and SG
    * Edmister (1958) for the acentric factor
    * Lee-Kesler (1976) for ideal-gas Cp
    * Watson-Nelson-Murphy (1935) for Watson K factor and latent heat
    * Riazi (2005, "Characterization and Properties of Petroleum
      Fractions") for the cohesive correlations

References
----------
* Riazi, M. R. & Daubert, T. E. (1980).  Simplify property predictions.
  *Hydrocarbon Processing*, 59(3), 115-116.
* Riazi, M. R. (2005).  *Characterization and Properties of Petroleum
  Fractions* (1st ed.).  ASTM International.  ISBN 978-0-8031-3361-4.
* Edmister, W. C. (1958).  Applied hydrocarbon thermodynamics, Part 4.
  *Petroleum Refiner*, 37(4), 173-179.
* Whitson, C. H. & Brule, M. R. (2000).  *Phase Behavior*.  SPE Monograph 20.
* Lee, B. I. & Kesler, M. G. (1975).  A generalized thermodynamic
  correlation based on three-parameter corresponding states.
  *AIChE Journal*, 21(3), 510-527.
* Watson, K. M., Nelson, E. F. & Murphy, G. B. (1935).  Characterization
  of petroleum fractions.  *Industrial & Engineering Chemistry*, 27, 1460.
* Kesler, M. G. & Lee, B. I. (1976).  Improve prediction of enthalpy of
  fractions.  *Hydrocarbon Processing*, 55(3), 153.

Conventions
-----------
* Specific gravity is defined at 60°F / 60°F (water reference at 60°F),
  the standard refinery convention.  This is essentially equivalent to
  density at 288.7 K / density of water at 288.7 K.
* Watson K factor (also called UOP K): K = (1.8 * NBP)^(1/3) / SG.
  K ≈ 13 for paraffinic stocks, 10-11 for aromatic, 12 for naphthenic.
* All temperatures in K, pressures in Pa, MW in kg/mol unless explicitly
  noted otherwise.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, List
import numpy as np


# Reference values (60°F = 288.706 K, water at 60°F = 999.012 kg/m³)
T_60F = 288.7056      # 60°F in K
RHO_WATER_60F = 999.012   # kg/m³


# =====================================================================
# Riazi-Daubert correlations (1980, 1987)
# =====================================================================

def riazi_daubert_Tc(NBP: float, SG: float) -> float:
    """Critical temperature [K] from NBP and SG via Riazi-Daubert (1980).

    Tc = 19.06232 * NBP^0.58848 * SG^0.3596

    Valid range: NBP in 273-867 K, SG in 0.63-0.97 (light naphtha through
    heavy gas oil).  Errors typically 1-2% within this range; up to 5-8%
    for extra-heavy cuts (NBP > 850 K).
    """
    return 19.06232 * (NBP ** 0.58848) * (SG ** 0.3596)


def riazi_daubert_Pc(NBP: float, SG: float) -> float:
    """Critical pressure [Pa] from NBP and SG via Riazi-Daubert (1980).

    Original form (Eq. 2.65 of Riazi 2005, in psia and °R):
        Pc[psia] = 3.12281e9 * NBP_R^(-2.3125) * SG^(2.3201)

    Then converted to Pa.  Errors typically 4-6% for naphthenic and
    paraffinic cuts; up to 10% for highly aromatic cuts in the
    middle-distillate range.
    """
    NBP_R = 1.8 * NBP    # Rankine
    Pc_psia = 3.12281e9 * (NBP_R ** -2.3125) * (SG ** 2.3201)
    return Pc_psia * 6894.757


def riazi_daubert_MW(NBP: float, SG: float) -> float:
    """Molecular weight [g/mol] from NBP and SG via Riazi-Daubert (1980).

    MW = 1.6607e-4 * NBP^2.1962 * SG^(-1.0164)        (NBP in K)

    Valid for hydrocarbons in 70-700 g/mol (NBP roughly 300-700 K).
    Errors typically 3-7%.  Riazi 2005 also provides a variant for
    higher-MW (Eq. 2.51) but the simple form above is adequate for
    typical refinery cuts.
    """
    return 1.6607e-4 * (NBP ** 2.1962) * (SG ** -1.0164)


def riazi_daubert_Vc(NBP: float, SG: float) -> float:
    """Critical molar volume [m³/mol] from NBP and SG.

    Computed from Tc, Pc, and a Riedel-style critical compressibility:
        Vc = Zc * R * Tc / Pc
        Zc = 0.290 - 0.080 * ω    (Riazi 2005 Eq. 2.69)

    where ω is computed via Lee-Kesler from the Tc/Pc/NBP triple.
    Accuracy 3-5% across the middle-distillate range, better than
    direct correlations of Vc on NBP and SG which have 5-10% scatter.
    """
    Tc = riazi_daubert_Tc(NBP, SG)
    Pc = riazi_daubert_Pc(NBP, SG)
    omega = lee_kesler_acentric(NBP, Tc, Pc)
    Zc = 0.290 - 0.080 * omega
    R_gas = 8.314462618
    return Zc * R_gas * Tc / Pc


# =====================================================================
# Edmister acentric factor
# =====================================================================

def edmister_acentric(NBP: float, Tc: float, Pc: float) -> float:
    """Acentric factor by the Edmister (1958) correlation.

    omega = (3/7) * (theta) / (1 - theta) * log10(Pc/Patm) - 1
    where theta = NBP/Tc.

    Edmister is the textbook default for petroleum cuts when ω is not
    measured.  Accuracy ~5% for paraffinic and naphthenic cuts; less
    accurate for highly aromatic stocks where Lee-Kesler-improved
    correlations (1976) should be used.
    """
    theta = NBP / Tc
    if theta >= 1.0:
        raise ValueError(f"Edmister: NBP ({NBP}) >= Tc ({Tc}), undefined")
    p_atm = 101325.0
    omega = (3.0 / 7.0) * (theta / (1.0 - theta)) * np.log10(Pc / p_atm) - 1.0
    return float(omega)


def lee_kesler_acentric(NBP: float, Tc: float, Pc: float) -> float:
    """Lee-Kesler (1976) acentric factor, more accurate than Edmister.

    ω = -ln(Pc/1.01325) - 5.92714 + 6.09648/Tbr + 1.28862·ln(Tbr) - 0.169347·Tbr^6
        ─────────────────────────────────────────────────────────────────────
                    15.2518 - 15.6875/Tbr - 13.4721·ln(Tbr) + 0.43577·Tbr^6

    Where Tbr = NBP/Tc and Pc must be in bar in the formula above.

    Accuracy ~2-3% for petroleum cuts including aromatics.
    """
    Tbr = NBP / Tc
    if Tbr >= 1.0:
        raise ValueError(f"Lee-Kesler: NBP/Tc = {Tbr} >= 1, undefined")
    Pc_bar = Pc / 1e5
    num = (-np.log(Pc_bar / 1.01325) - 5.92714
           + 6.09648 / Tbr + 1.28862 * np.log(Tbr) - 0.169347 * Tbr**6)
    den = (15.2518 - 15.6875 / Tbr - 13.4721 * np.log(Tbr) + 0.43577 * Tbr**6)
    return float(num / den)


# =====================================================================
# Lee-Kesler ideal-gas Cp
# =====================================================================

def lee_kesler_cp_ig_coeffs(NBP: float, SG: float, MW: float
                              ) -> Tuple[float, float, float, float]:
    """Ideal-gas Cp polynomial coefficients (a0, a1, a2, a3) such that

        Cp^ig(T) = a0 + a1*T + a2*T^2 + a3*T^3   [J/(mol K)]

    Uses a Watson-K-corrected scaling of a per-mass quadratic fit to
    n-paraffin NIST data (C5-C20, T = 298-900 K):

        Cp/MW [J/(g·K)] = b0 + b1*T + b2*T^2

    with (b0, b1, b2) = (-0.0703, 7.008e-3, -4.03e-6) for paraffinic
    cuts (K_W = 12.7).  A small linear correction in (K_W - 12.0) is
    applied to lighten the polynomial for aromatic stocks (K_W < 12)
    and to heavy it for highly paraffinic stocks.

    Output is in molar SI form ready to pass to ``CubicEOS`` as
    ``ideal_gas_cp_poly``.  Accuracy: 5-10% across typical refinery
    cuts (NBP 350-700 K).  For more accuracy, override Cp_ig directly
    on the EOS object using a measured polynomial.
    """
    K_W = watson_K(NBP, SG)
    # Per-mass Cp [J/(g·K)] quadratic from n-paraffin NIST fit
    b0_par, b1_par, b2_par = -0.0703, 7.008e-3, -4.03e-6
    # Watson K correction: aromatics have ~10-15% lower Cp_ig than
    # paraffins of same MW (more rigid molecules).  Linear scaling:
    #   correction = 1 + 0.04 * (K_W - 12.7)
    #   K_W=12.7 (paraffin): factor = 1.0
    #   K_W=10.0 (aromatic): factor = 0.892  -> ~11% reduction
    #   K_W=11.5 (naphthenic): factor = 0.952
    factor = 1.0 + 0.04 * (K_W - 12.7)
    b0 = b0_par * factor
    b1 = b1_par * factor
    b2 = b2_par * factor
    # Convert per-mass to molar: Cp_mol [J/(mol K)] = Cp_mass [J/(g·K)] * MW [g/mol]
    a0 = b0 * MW
    a1 = b1 * MW
    a2 = b2 * MW
    a3 = 0.0
    return a0, a1, a2, a3


# =====================================================================
# Watson K factor and latent heat
# =====================================================================

def watson_K(NBP: float, SG: float) -> float:
    """Watson K (UOP K) characterization factor.

    K = (1.8 * NBP_K)^(1/3) / SG

    Empirical interpretation:
      K ≈ 13.0   highly paraffinic
      K ≈ 12.0   naphthenic
      K ≈ 11.0   intermediate
      K ≈ 10.0   highly aromatic
    """
    return ((1.8 * NBP) ** (1.0 / 3.0)) / SG


def watson_latent_heat(T: float, NBP: float, MW: float) -> float:
    """Latent heat of vaporization [J/mol] at T via Watson (1943) scaling.

    H_vap(T) = H_vap(NBP) * ((Tc - T)/(Tc - NBP))^0.38

    Uses Riedel (1954) for H_vap at NBP:
      H_vap(NBP) [J/mol] = R * NBP * 1.092 * (ln(Pc[bar]) - 1.013)
                            / (0.930 - NBP/Tc)

    This requires Tc and Pc, which are not arguments here; this helper
    is therefore mainly a placeholder.  The PseudoComponent class
    computes latent heat properly using its own Tc and Pc.  Use
    ``PseudoComponent.latent_heat(T)`` instead in user code.
    """
    raise NotImplementedError(
        "Use PseudoComponent.latent_heat(T) which has Tc and Pc.")


# =====================================================================
# Vapor pressure (Lee-Kesler / Reid-Prausnitz form)
# =====================================================================

def lee_kesler_psat(T: float, Tc: float, Pc: float, omega: float) -> float:
    """Vapor pressure [Pa] from Lee-Kesler corresponding-states form.

    ln(Pr) = f0(Tr) + omega * f1(Tr)
    f0(Tr) = 5.92714 - 6.09648/Tr - 1.28862*ln(Tr) + 0.169347*Tr^6
    f1(Tr) = 15.2518 - 15.6875/Tr - 13.4721*ln(Tr) + 0.43577*Tr^6

    Valid for Tr in 0.5-1.0 (i.e., from 50% of Tc up to Tc).  This is
    the Lee-Kesler form universally used in PR/SRK petroleum work, and
    matches Antoine-style vapor pressures within 1% for heavy
    hydrocarbon cuts.
    """
    Tr = T / Tc
    if Tr <= 0:
        return 0.0
    if Tr >= 1.0:
        return Pc
    f0 = 5.92714 - 6.09648 / Tr - 1.28862 * np.log(Tr) + 0.169347 * Tr**6
    f1 = (15.2518 - 15.6875 / Tr - 13.4721 * np.log(Tr) + 0.43577 * Tr**6)
    ln_Pr = f0 + omega * f1
    return Pc * np.exp(ln_Pr)


# =====================================================================
# Liquid density (Yen-Woods / Rackett-Spencer)
# =====================================================================

def rackett_density(T: float, Tc: float, Pc: float, omega: float,
                     Vc: float) -> float:
    """Saturated liquid molar density [mol/m³] via Rackett-Spencer (1972).

    V_sat = Vc * Z_RA^((1 - Tr)^(2/7))
    Z_RA = 0.29056 - 0.08775 * omega
    rho = 1 / V_sat
    """
    Tr = T / Tc
    Z_RA = 0.29056 - 0.08775 * omega
    V_sat = Vc * Z_RA ** ((1.0 - Tr) ** (2.0 / 7.0))
    return 1.0 / V_sat


# =====================================================================
# PseudoComponent dataclass
# =====================================================================

@dataclass
class PseudoComponent:
    """A petroleum pseudo-component characterized by NBP and SG.

    Once instantiated, exposes Tc, Pc, omega, MW, Vc, Watson K,
    ideal-gas Cp coefficients, and methods for vapor pressure, latent
    heat, and liquid density.  Drops directly into stateprop's
    ``CubicEOS(T_c, p_c, acentric_factor, ideal_gas_cp_poly,
    molar_mass)`` constructor.

    Parameters
    ----------
    NBP : float
        Normal boiling point [K].  Required.
    SG : float
        Specific gravity at 60°F / 60°F.  Required.
    name : str
        Display name (e.g., ``"C12+"``, ``"Diesel cut 460-540K"``).
    MW : float, optional
        Molecular weight [g/mol].  If None, estimated from
        Riazi-Daubert.  Provide explicitly when known to improve
        accuracy on light cuts (NBP < 380 K).
    Tc : float, optional
        Critical temperature [K].  Default: Riazi-Daubert.
    Pc : float, optional
        Critical pressure [Pa].  Default: Riazi-Daubert.
    Vc : float, optional
        Critical molar volume [m³/mol].  Default: Riazi-Daubert.
    omega : float, optional
        Acentric factor.  Default: Lee-Kesler (more accurate than
        Edmister for the typical NBP range of petroleum cuts).
    omega_method : str
        ``"lee_kesler"`` (default) or ``"edmister"``.

    Attributes
    ----------
    All inputs are echoed plus the computed ``T_c``, ``p_c``,
    ``acentric_factor``, ``molar_mass`` (kg/mol), ``Vc`` (m³/mol),
    ``Watson_K``, and ``ideal_gas_cp_poly`` (4-tuple).
    """
    NBP: float
    SG: float
    name: str = "pseudo"
    MW: Optional[float] = None
    Tc: Optional[float] = None
    Pc: Optional[float] = None
    Vc: Optional[float] = None
    omega: Optional[float] = None
    omega_method: str = "lee_kesler"

    # Computed (post-init)
    T_c: float = field(init=False)
    p_c: float = field(init=False)
    acentric_factor: float = field(init=False)
    molar_mass: float = field(init=False)             # kg/mol
    Watson_K: float = field(init=False)
    ideal_gas_cp_poly: Tuple[float, float, float, float] = field(init=False)

    def __post_init__(self):
        if self.NBP <= 0:
            raise ValueError(f"NBP must be positive, got {self.NBP}")
        if not (0.4 < self.SG < 1.5):
            raise ValueError(
                f"SG must be in (0.4, 1.5) for hydrocarbon cuts, "
                f"got {self.SG}")
        # Fill in defaults from Riazi-Daubert
        if self.MW is None:
            self.MW = riazi_daubert_MW(self.NBP, self.SG)
        if self.Tc is None:
            self.Tc = riazi_daubert_Tc(self.NBP, self.SG)
        if self.Pc is None:
            self.Pc = riazi_daubert_Pc(self.NBP, self.SG)
        if self.Vc is None:
            self.Vc = riazi_daubert_Vc(self.NBP, self.SG)
        if self.omega is None:
            if self.omega_method == "lee_kesler":
                self.omega = lee_kesler_acentric(self.NBP, self.Tc, self.Pc)
            elif self.omega_method == "edmister":
                self.omega = edmister_acentric(self.NBP, self.Tc, self.Pc)
            else:
                raise ValueError(
                    f"omega_method must be 'lee_kesler' or 'edmister', "
                    f"got {self.omega_method!r}")
        # Aliased fields for direct EOS interop
        self.T_c = self.Tc
        self.p_c = self.Pc
        self.acentric_factor = self.omega
        self.molar_mass = self.MW * 1e-3
        self.Watson_K = watson_K(self.NBP, self.SG)
        self.ideal_gas_cp_poly = lee_kesler_cp_ig_coeffs(
            self.NBP, self.SG, self.MW)

    # --- Methods -----------------------------------------------------
    def psat(self, T: float) -> float:
        """Vapor pressure [Pa] at T via Lee-Kesler corresponding states."""
        return lee_kesler_psat(T, self.Tc, self.Pc, self.omega)

    def latent_heat(self, T: float) -> float:
        """Latent heat of vaporization [J/mol] at T via Watson scaling
        from Riedel's NBP estimate.

        H_vap(NBP) = R * NBP * 1.092 * (ln(Pc_bar) - 1.013)
                     / (0.930 - NBP/Tc)              [Riedel 1954]
        H_vap(T)   = H_vap(NBP) * ((Tc - T)/(Tc - NBP))^0.38   [Watson 1943]
        """
        if T >= self.Tc:
            return 0.0
        R = 8.314462618
        Pc_bar = self.Pc / 1e5
        H_NBP = (R * self.NBP * 1.092 * (np.log(Pc_bar) - 1.013)
                  / (0.930 - self.NBP / self.Tc))
        scale = ((self.Tc - T) / (self.Tc - self.NBP)) ** 0.38
        return float(H_NBP * scale)

    def liquid_density(self, T: float) -> float:
        """Saturated liquid molar density [mol/m³] via Rackett-Spencer."""
        return rackett_density(T, self.Tc, self.Pc, self.omega, self.Vc)

    def liquid_density_kg(self, T: float) -> float:
        """Saturated liquid mass density [kg/m³]."""
        return self.liquid_density(T) * self.molar_mass

    def cp_ig(self, T: float) -> float:
        """Ideal-gas heat capacity [J/(mol K)] at T."""
        a0, a1, a2, a3 = self.ideal_gas_cp_poly
        return a0 + a1 * T + a2 * T**2 + a3 * T**3

    def h_ig(self, T: float, T_ref: float = 298.15) -> float:
        """Ideal-gas enthalpy [J/mol] relative to T_ref."""
        a0, a1, a2, a3 = self.ideal_gas_cp_poly
        H = lambda Tx: (a0 * Tx + a1 * Tx**2 / 2 + a2 * Tx**3 / 3
                         + a3 * Tx**4 / 4)
        return H(T) - H(T_ref)

    def __repr__(self):
        return (f"PseudoComponent(name={self.name!r}, NBP={self.NBP:.1f} K, "
                f"SG={self.SG:.4f}, MW={self.MW:.1f} g/mol, "
                f"Tc={self.Tc:.1f} K, Pc={self.Pc/1e5:.2f} bar, "
                f"omega={self.omega:.4f}, K_W={self.Watson_K:.2f})")


# =====================================================================
# Convenience constructors and cut distributions
# =====================================================================

def make_pseudo_from_NBP_SG(NBP: float, SG: float, name: Optional[str] = None,
                              **kwargs) -> PseudoComponent:
    """Convenience wrapper: build a PseudoComponent from NBP and SG.

    Equivalent to ``PseudoComponent(NBP=NBP, SG=SG, name=name, **kwargs)``;
    provided for symmetry with ``make_pseudo_cut_distribution``.
    """
    if name is None:
        name = f"pseudo({NBP:.0f}K)"
    return PseudoComponent(NBP=NBP, SG=SG, name=name, **kwargs)


def make_pseudo_cut_distribution(
    NBP_cuts: Sequence[float],
    SG_cuts: Optional[Sequence[float]] = None,
    SG_avg: Optional[float] = None,
    Watson_K: Optional[float] = None,
    name_prefix: str = "cut",
) -> List[PseudoComponent]:
    """Generate a list of PseudoComponents for a refinery cut distribution.

    A common scenario in petroleum engineering is to have measured TBP
    (true boiling point) data — a curve of NBP vs cumulative volume %
    distilled — but only an average SG or Watson K for the whole cut.
    This helper expands the TBP into individual pseudo-components.

    Three input modes:

    1. ``NBP_cuts`` and ``SG_cuts`` both given: each cut has its own SG.
    2. ``NBP_cuts`` and ``SG_avg`` given: each cut gets ``SG_avg``;
       use this for back-of-envelope work.
    3. ``NBP_cuts`` and ``Watson_K`` given: SG of each cut is computed
       from K and its NBP via SG = (1.8*NBP)^(1/3) / K.  This is the
       standard refinery practice when the K factor is constant across
       a narrow cut.

    Parameters
    ----------
    NBP_cuts : sequence of float
        NBPs of the cuts [K], one per pseudo-component.
    SG_cuts : sequence of float, optional
        Per-cut SGs.
    SG_avg : float, optional
        Single SG applied to all cuts.
    Watson_K : float, optional
        Single Watson K applied to all cuts (each cut's SG is computed).
    name_prefix : str
        Prefix for the auto-generated names (e.g., ``"diesel"`` →
        ``"diesel_1"``, ``"diesel_2"``...).

    Returns
    -------
    list of PseudoComponent
    """
    n = len(NBP_cuts)
    if SG_cuts is not None:
        if len(SG_cuts) != n:
            raise ValueError(
                f"SG_cuts length ({len(SG_cuts)}) != NBP_cuts length ({n})")
        SGs = list(SG_cuts)
    elif SG_avg is not None:
        SGs = [float(SG_avg)] * n
    elif Watson_K is not None:
        SGs = [(1.8 * NBP) ** (1.0 / 3.0) / Watson_K for NBP in NBP_cuts]
    else:
        raise ValueError(
            "Must provide one of: SG_cuts, SG_avg, or Watson_K")
    return [PseudoComponent(NBP=float(NBP), SG=float(SG),
                              name=f"{name_prefix}_{k+1}")
            for k, (NBP, SG) in enumerate(zip(NBP_cuts, SGs))]


# =====================================================================
# EOS interop
# =====================================================================

def make_PR_from_pseudo(pseudo: PseudoComponent, **kwargs):
    """Construct a PR-EOS object from a PseudoComponent.

    Forwards Tc, Pc, acentric_factor, molar_mass, ideal_gas_cp_poly to
    ``stateprop.cubic.eos.PR``.  Extra kwargs override the defaults.
    """
    from stateprop.cubic.eos import CubicEOS
    return CubicEOS(
        T_c=pseudo.T_c, p_c=pseudo.p_c,
        acentric_factor=pseudo.acentric_factor,
        family="pr",
        molar_mass=pseudo.molar_mass,
        ideal_gas_cp_poly=pseudo.ideal_gas_cp_poly,
        name=pseudo.name,
        **kwargs)


def make_SRK_from_pseudo(pseudo: PseudoComponent, **kwargs):
    """Construct an SRK-EOS object from a PseudoComponent."""
    from stateprop.cubic.eos import CubicEOS
    return CubicEOS(
        T_c=pseudo.T_c, p_c=pseudo.p_c,
        acentric_factor=pseudo.acentric_factor,
        family="srk",
        molar_mass=pseudo.molar_mass,
        ideal_gas_cp_poly=pseudo.ideal_gas_cp_poly,
        name=pseudo.name,
        **kwargs)
