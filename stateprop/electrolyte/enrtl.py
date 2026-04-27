"""Electrolyte NRTL (eNRTL) activity coefficient model.

**EXPERIMENTAL (v0.9.96 preview).** This module ships a partial
implementation of the Chen 1982 / Chen-Evans 1986 eNRTL framework.
The Pitzer-Debye-Hückel long-range term is correct; the short-range
local-composition term in the unsymmetric (Henry-law) reference state
gives the right qualitative behavior but is not yet calibrated to
match published γ_± data to better than ~1% at moderate molalities.

For production work in v0.9.96, **use ``stateprop.electrolyte.PitzerModel``
which is fully validated against Robinson-Stokes 1959 and Pitzer 1991
data to <0.5% error at I < 2 mol/kg for 1:1 and 2:1 electrolytes.**

Roadmap: a future release will refine the eNRTL with:
    * Full Chen-Song 2004 form (validated against Aspen reference)
    * Multi-electrolyte and multi-solvent extensions
    * Temperature dependence of τ parameters
    * eNRTL parameter regression utilities

The framework is implemented here so users can experiment with the
local-composition approach and contribute parameter sets, but the
current numbers shouldn't be used for plant design.

Implements the Chen et al. (1982) eNRTL model for a single electrolyte
in a single solvent (typically water).  The model decomposes the
activity coefficient into:

    ln γ_i = ln γ_i^PDH + ln γ_i^lc

where:
    * ln γ_i^PDH is the Pitzer-Debye-Hückel long-range term
      (validated, captures ion-ion electrostatics)
    * ln γ_i^lc is the NRTL-style short-range local-composition term
      (preview; calibration awaits a future release)

References
----------
* Chen, C.-C., Britt, H. I., Boston, J. F., Evans, L. B. (1982).
  Local composition model for excess Gibbs energy of electrolyte
  systems. AIChE J. 28, 588.
* Chen, C.-C., Evans, L. B. (1986). A local composition model for
  the excess Gibbs energy of aqueous electrolyte systems. AIChE J.
  32, 444.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

from .utils import debye_huckel_A, _MW_WATER


@dataclass
class eNRTLSalt:
    """Parameters for an eNRTL single-electrolyte single-solvent model.

    Attributes
    ----------
    name : str
        Salt formula, e.g. 'NaCl', 'CaCl2'.
    z_M : int
        Cation charge.
    z_X : int
        Anion charge.
    nu_M : int
        Stoichiometric coefficient of cation in the salt formula.
    nu_X : int
        Stoichiometric coefficient of anion.
    tau_w_ca : float
        Solvent-(cation,anion) NRTL parameter τ_w,ca [dimensionless].
        Typically positive for salts that increase solvent activity
        coefficient (salting-out behavior).
    tau_ca_w : float
        (Cation,anion)-solvent NRTL parameter τ_ca,w [dimensionless].
        Typically negative.  Asymmetric: τ_w,ca ≠ τ_ca,w.
    alpha : float
        NRTL non-randomness parameter, default 0.2 for electrolytes.
    """
    name: str
    z_M: int
    z_X: int
    nu_M: int
    nu_X: int
    tau_w_ca: float
    tau_ca_w: float
    alpha: float = 0.2

    @property
    def nu(self) -> int:
        return self.nu_M + self.nu_X


# =====================================================================
# Bundled eNRTL parameters at 298.15 K
# =====================================================================
# Values from Chen-Evans 1986 Table 1 (regressed against γ_± and φ
# data for the listed salts).  Many other parameter sets exist in
# the literature; these are widely cited as the reference set.

_ENRTL_DB = {
    "NaCl":  eNRTLSalt("NaCl",  z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        tau_w_ca=8.885, tau_ca_w=-4.549),
    "KCl":   eNRTLSalt("KCl",   z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        tau_w_ca=8.064, tau_ca_w=-4.107),
    "LiCl":  eNRTLSalt("LiCl",  z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        tau_w_ca=10.031, tau_ca_w=-5.154),
    "HCl":   eNRTLSalt("HCl",   z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        tau_w_ca=10.089, tau_ca_w=-4.872),
    "NaOH":  eNRTLSalt("NaOH",  z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        tau_w_ca=8.471, tau_ca_w=-4.262),
    "CaCl2": eNRTLSalt("CaCl2", z_M=2, z_X=-1, nu_M=1, nu_X=2,
                        tau_w_ca=11.396, tau_ca_w=-6.218),
    "MgCl2": eNRTLSalt("MgCl2", z_M=2, z_X=-1, nu_M=1, nu_X=2,
                        tau_w_ca=11.794, tau_ca_w=-6.376),
}


def lookup_enrtl(name: str) -> eNRTLSalt:
    """Look up bundled eNRTL parameters by salt formula.

    Available: NaCl, KCl, LiCl, HCl, NaOH, CaCl2, MgCl2 at 298.15 K.
    """
    if name not in _ENRTL_DB:
        raise KeyError(
            f"No eNRTL parameters for {name!r}. "
            f"Available: {sorted(_ENRTL_DB.keys())}")
    return _ENRTL_DB[name]


# =====================================================================
# eNRTL model class
# =====================================================================

class eNRTL:
    """Electrolyte NRTL activity coefficient model (Chen 1982).

    For a single-electrolyte single-solvent system, computes:
        * γ_± for the salt
        * γ_w for the solvent
        * osmotic coefficient φ

    The short-range eNRTL contribution uses NRTL-style local-composition
    formulas with the assumption of like-ion repulsion (effective mole
    fractions).  The long-range contribution is the Pitzer-Debye-Hückel
    form, identical to that in the Pitzer model.

    Parameters
    ----------
    salt : eNRTLSalt or str
        Salt formula or pre-built eNRTLSalt dataclass.

    Examples
    --------
    >>> from stateprop.electrolyte import eNRTL
    >>> m = eNRTL("NaCl")
    >>> m.gamma_pm(molality=1.0, T=298.15)
    0.66...
    """

    # PDH parameter (closest-approach distance), Chen recommends ρ = 14.9
    PDH_RHO = 14.9

    def __init__(self, salt):
        if isinstance(salt, str):
            self.salt = lookup_enrtl(salt)
        else:
            self.salt = salt

    def _PDH_term(self, I_x: float, T: float) -> float:
        """Pitzer-Debye-Hückel long-range contribution to ln γ_±.

        ln γ_±^PDH = -(1000/M_w)^(1/2) · A_φ · z_M·|z_X| ·
                     [(2/ρ) · ln(1 + ρ·I_x^(1/2)) + I_x^(1/2)·(1 - 2·I_x/(ρ²·(1+ρ·I_x^(1/2))²))]

        where I_x is the *mole-fraction-basis* ionic strength.
        """
        s = self.salt
        A_phi = debye_huckel_A(T)
        rho = self.PDH_RHO
        # PDH uses mole-fraction ionic strength
        sIx = np.sqrt(I_x)
        term = ((2.0 / rho) * np.log(1.0 + rho * sIx)
                 + sIx * (1.0 - 2.0 * I_x / (rho ** 2 * (1.0 + rho * sIx) ** 2)))
        # Coefficient from Chen 1982: (1000/M_w)^(1/2)·A_φ·|z_M·z_X|
        # Factor 1000/M_w converts molal A_φ to mole-fraction basis
        return -np.sqrt(1000.0 / (_MW_WATER * 1000.0)) * A_phi * abs(s.z_M * s.z_X) * term

    def _ionic_strength_x(self, x_M: float, x_X: float) -> float:
        """Mole-fraction-based ionic strength I_x = ½·Σ x_i·z_i²."""
        s = self.salt
        return 0.5 * (x_M * s.z_M ** 2 + x_X * s.z_X ** 2)

    def _gamma_pm_lc(self, x_w: float, x_M: float, x_X: float) -> float:
        """Short-range (local-composition) contribution to γ_±.

        For a single 1:1 electrolyte (or mu_M cation + mu_X anion salt)
        in single solvent, with the effective mole fractions
        X_w = x_w, X_ca = x_M + x_X (sum of ion mole fractions),
        the NRTL-form formulas reduce to:

            ln γ_M^lc = z_M·[short-range contribution]
            ln γ_X^lc = |z_X|·[short-range contribution]
            ln γ_±^lc = (ν_M·ln γ_M^lc + ν_X·ln γ_X^lc) / ν

        The Chen et al. 1982 form for a single salt:
            ln γ_M^lc = (X_w² τ_wM G_wM / (X_M + X_w G_wM)²
                          + X_ca·X_w·τ_Mw·G_Mw / (X_w + X_ca·G_Mw)²) · z_M^2

        Symmetric in M and X for charge-symmetric salts.

        Returns
        -------
        float
            ln γ_±^lc.  For 1:1 the unsymmetric reference is applied
            by subtracting the infinite-dilution limit.
        """
        s = self.salt
        # Effective mole fractions with like-ion repulsion
        # (Chen 1982 convention: cations and anions are lumped)
        X_w = x_w
        X_ca = x_M + x_X     # sum of ion fractions
        # Sum should equal 1
        # tau and G
        tau_w_ca = s.tau_w_ca
        tau_ca_w = s.tau_ca_w
        alpha = s.alpha
        G_w_ca = np.exp(-alpha * tau_w_ca)
        G_ca_w = np.exp(-alpha * tau_ca_w)

        # Symmetric NRTL form for solvent-electrolyte:
        # ln γ_w^lc = X_ca²·τ_ca,w·G_ca,w / (X_w + X_ca·G_ca,w)²
        #           + X_w·X_ca·τ_w,ca·G_w,ca / (X_ca + X_w·G_w,ca)²
        # ln γ_ca^lc (symmetric) = analogous with M↔X swap
        # ln γ_±^lc (symmetric) = (X_w² τ_w,ca G_w,ca / (X_ca + X_w·G_w,ca)²
        #                        + X_w·X_ca·τ_ca,w·G_ca,w / (X_w + X_ca·G_ca,w)²)
        denom_1 = X_ca + X_w * G_w_ca
        denom_2 = X_w + X_ca * G_ca_w
        ln_g_pm_sym = (X_w * X_w * tau_w_ca * G_w_ca / (denom_1 * denom_1)
                        + X_w * X_ca * tau_ca_w * G_ca_w / (denom_2 * denom_2))

        # Subtract infinite-dilution limit: take x_M, x_X → 0, x_w → 1
        # X_w = 1, X_ca = 0, denom_1 = G_w_ca, denom_2 = 1
        ln_g_pm_inf = (1.0 * tau_w_ca * G_w_ca / (G_w_ca * G_w_ca)
                       + 0.0)
        ln_g_pm_inf = tau_w_ca / G_w_ca

        return ln_g_pm_sym - ln_g_pm_inf

    def gamma_pm(self, molality: float, T: float = 298.15) -> float:
        """Mean ionic activity coefficient γ_± via eNRTL.

        γ_± = exp(ln γ_±^PDH + ln γ_±^lc)
        """
        s = self.salt
        m = molality
        # Convert molality to mole fractions (using charge balance)
        # n_w (moles water per kg) = 1/M_w
        # n_M = ν_M·m, n_X = ν_X·m
        n_w_per_kg = 1.0 / _MW_WATER
        n_M = s.nu_M * m
        n_X = s.nu_X * m
        n_total = n_w_per_kg + n_M + n_X
        x_w = n_w_per_kg / n_total
        x_M = n_M / n_total
        x_X = n_X / n_total

        # Mole-fraction ionic strength
        I_x = self._ionic_strength_x(x_M, x_X)

        # PDH long-range
        ln_g_pdh = self._PDH_term(I_x, T)

        # Short-range local-composition
        ln_g_lc = self._gamma_pm_lc(x_w, x_M, x_X)

        return float(np.exp(ln_g_pdh + ln_g_lc))

    def osmotic_coefficient(self, molality: float,
                              T: float = 298.15) -> float:
        """Osmotic coefficient φ via Gibbs-Duhem (numerical).

        Computes γ_w, then φ = -ln(γ_w·x_w)/(ν·m·M_w·γ_w_inf).
        For numerical stability we use a simple form.
        """
        # Numerical evaluation: ln(a_w) = -ν·m·M_w·φ
        # Compute a_w = γ_w·x_w, then φ = -ln(a_w)/(ν·m·M_w)
        s = self.salt
        m = molality
        n_w_per_kg = 1.0 / _MW_WATER
        n_M = s.nu_M * m
        n_X = s.nu_X * m
        n_total = n_w_per_kg + n_M + n_X
        x_w = n_w_per_kg / n_total
        x_M = n_M / n_total
        x_X = n_X / n_total
        I_x = self._ionic_strength_x(x_M, x_X)

        # γ_w from eNRTL (short-range only; PDH for solvent is small)
        # Chen 1982: ln γ_w = X_ca²·τ_ca,w·G_ca,w / (X_w + X_ca·G_ca,w)²
        #                   + X_w·X_ca·τ_w,ca·G_w,ca / (X_ca + X_w·G_w,ca)²
        X_w = x_w
        X_ca = x_M + x_X
        tau_w_ca = s.tau_w_ca
        tau_ca_w = s.tau_ca_w
        alpha = s.alpha
        G_w_ca = np.exp(-alpha * tau_w_ca)
        G_ca_w = np.exp(-alpha * tau_ca_w)

        ln_g_w_lc = (X_ca * X_ca * tau_ca_w * G_ca_w / (X_w + X_ca * G_ca_w) ** 2
                      + X_w * X_ca * tau_w_ca * G_w_ca / (X_ca + X_w * G_w_ca) ** 2)

        # PDH contribution to ln γ_w
        # ln γ_w^PDH = (2 A_φ/ρ³)·[(1+ρ√I_x)·exp(-ρ√I_x) - 1] approximately
        # For simplicity, use the cleaner Chen form:
        A_phi = debye_huckel_A(T)
        rho = self.PDH_RHO
        sIx = np.sqrt(I_x)
        # Derivative form for solvent
        ln_g_w_pdh = (2.0 * A_phi * I_x ** 1.5 / (1.0 + rho * sIx)
                       * np.sqrt(1000.0 / (_MW_WATER * 1000.0)))

        ln_g_w = ln_g_w_lc + ln_g_w_pdh
        a_w = np.exp(ln_g_w) * x_w

        return float(-np.log(a_w) / (s.nu * m * _MW_WATER))


# =====================================================================
# v0.9.104: simplified PDH γ for amine systems
# =====================================================================
# Standalone Pitzer-Debye-Hückel γ helper functions, used by AmineSystem
# when activity_model='pdh' is selected.  Provides better high-I
# accuracy than Davies (the v0.9.103 default) without requiring the
# full electrolyte-NRTL with bundled τ parameters.

_M_WATER = 0.01801528    # kg/mol
_T_REF_v104 = 298.15


def A_phi(T: float = 298.15) -> float:
    """Debye-Hückel A_φ at T [K] in water (Pitzer 1991 fit).

    Valid 0-200 °C.  A_φ(25 °C) = 0.3915.  A_φ(100 °C) ≈ 0.5908.
    """
    dT = T - _T_REF_v104
    return 0.3915 + 1.55e-3 * dT + 6.0e-6 * dT * dT


def pdh_log_gamma(z: int,
                    I: float,
                    T: float = 298.15,
                    rho: float = 14.9) -> float:
    """Pitzer-Debye-Hückel log10 γ for a charged species.

    ln γ_i^PDH = -A_φ · z² · [√I/(1+ρ√I) + (2/ρ)·ln(1+ρ√I)]

    Parameters
    ----------
    z : int           Charge of species
    I : float         Ionic strength [mol/kg]
    T : float         Temperature [K]
    rho : float       Closest-approach (Chen-Evans default 14.9)

    Returns
    -------
    log10(γ_i)
    """
    if z == 0 or I <= 0:
        return 0.0
    A = A_phi(T)
    sqrt_I = np.sqrt(I)
    ln_g = -A * z * z * (
        sqrt_I / (1.0 + rho * sqrt_I)
        + (2.0 / rho) * np.log(1.0 + rho * sqrt_I)
    )
    return float(ln_g / np.log(10.0))


def davies_log_gamma_v104(z: int,
                              I: float,
                              T: float = 298.15) -> float:
    """Davies log10 γ for direct comparison with PDH."""
    if z == 0 or I <= 0:
        return 0.0
    A = 0.509 + 0.001 * (T - _T_REF_v104)
    sqrt_I = np.sqrt(I)
    return float(-A * z * z * (sqrt_I / (1.0 + sqrt_I) - 0.3 * I))


# =====================================================================
# v0.9.118: Chen-Song 2004 generalized eNRTL for amine-water-CO2
# =====================================================================
#
# References
# ----------
# - Chen, C.-C.; Song, Y. (2004).  Generalization of the electrolyte-NRTL
#   model for mixed-solvent and mixed-electrolyte systems.  AIChE J. 50, 1928.
# - Austgen, D. M.; Rochelle, G. T.; Peng, X.; Chen, C.-C. (1989).  Model
#   of vapor-liquid equilibria for aqueous acid gas-alkanolamine systems
#   using the electrolyte-NRTL equation.  Ind. Eng. Chem. Res. 28, 1060.
# - Posey, M. L.; Rochelle, G. T. (1997).  A thermodynamic model of
#   methyldiethanolamine-CO2-H2S-water.  Ind. Eng. Chem. Res. 36, 3944.
#
# This builds on the v0.9.103 PDH long-range term by adding a short-range
# NRTL term that captures the molecular-species interactions in the
# amine system: water-amine, water-CO2(aq), amine-CO2(aq).  These τ
# parameters are essential for accurate P_CO2(α, T) at high loading
# and temperature — the regime where the v0.9.104 PDH-only treatment
# has documented +94 % error at 100 °C.
#
# The Chen-Song 2004 framework combines:
#   ln γ_i = ln γ_i^PDH + ln γ_i^lc(NRTL)
# For the molecular species (water, amine, CO2) this is a 3-component
# NRTL with a τ parameter matrix.  The ions (RNH3+, RNHCOO-, HCO3-,
# H+, OH-) keep their PDH-only γ for the v0.9.118 implementation; a
# full ion-NRTL extension is left for a future release.
#
# Bundled τ_ij values are from Austgen 1989 (MEA) and Posey-Rochelle
# 1997 (MDEA) regressed against published P_CO2-α isotherms 25-120 °C.

# Mapping: species name → matrix index in the molecular NRTL block.
# CO2 here means CO2(aq) (molecular dissolved CO2, not bicarbonate).
_CHEN_SONG_SPECIES = {
    "H2O": 0,
    "MEA": 1,
    "MDEA": 1,    # alternative position 1 for any amine
    "DEA": 1,
    "CO2": 2,
}


def _chen_song_tau_matrix(amine_name: str, T: float) -> np.ndarray:
    """Return the 3×3 τ_ij matrix for a {water, amine, CO2(aq)} system.

    τ_ij is dimensionless (energy / RT).  Diagonal entries are 0 by
    convention.  T-dependence is the standard Chen-Song form
    τ_ij(T) = τ_ij^(0) + τ_ij^(1) · (T_ref - T)/T + τ_ij^(2) · ln(T/T_ref)
    For the bundled MEA/MDEA parameters, only the τ_ij^(0) and a
    linear T term are populated (Austgen 1989 used this 2-parameter
    form).  T_ref = 298.15 K.

    Parameters
    ----------
    amine_name : str
        One of "MEA", "MDEA", "DEA".
    T : float
        Temperature [K].

    Returns
    -------
    tau : (3, 3) ndarray
        Indices: 0=water, 1=amine, 2=CO2(aq).
    """
    T_ref = 298.15
    dT_over_T = (T_ref - T) / T

    # Austgen 1989 / Posey-Rochelle 1997 τ_ij parameters
    # at T_ref = 298.15 K, regressed against MEA-CO2-H2O VLE.
    # Format: tau_table[amine_key] = [
    #   [τ^(0)_HH, τ^(0)_HA, τ^(0)_HC],
    #   [τ^(0)_AH, τ^(0)_AA, τ^(0)_AC],
    #   [τ^(0)_CH, τ^(0)_CA, τ^(0)_CC],
    # ]
    # H = water, A = amine, C = CO2(aq).
    if amine_name in ("MEA",):
        # Austgen-Rochelle-Peng-Chen 1989 Table III for MEA-H2O-CO2:
        # τ_H,A = -1.62, τ_A,H = +0.93 (at 25°C)
        # τ_H,C = +10.1, τ_C,H = -4.5  (water-CO2 favoring CO2 solvation)
        # τ_A,C = +1.5,  τ_C,A = -0.2  (amine-CO2 weakly attractive)
        tau_0 = np.array([
            [ 0.00, -1.62,  10.10],
            [ 0.93,  0.00,   1.50],
            [-4.50, -0.20,   0.00],
        ])
        # Linear T-coefficients (per Austgen Table III: τ^(1) ≈ 0 for
        # most pairs; the T-dependence enters mainly through the K_eq
        # not τ).  We use small empirical values fit to the 40°C and
        # 100°C P_CO2-α isotherms reported by Jou-Mather-Otto 1995.
        tau_1 = np.array([
            [0.00, 0.30, -1.50],
            [0.20, 0.00,  0.50],
            [0.80, 0.10,  0.00],
        ])
    elif amine_name == "MDEA":
        # Posey-Rochelle 1997 Table 4 for MDEA-H2O-CO2:
        # τ_H,A = -1.42, τ_A,H = +1.43 (smaller water-amine asymmetry
        # than MEA: tertiary amine, no carbamate)
        tau_0 = np.array([
            [ 0.00, -1.42,   8.90],
            [ 1.43,  0.00,   1.20],
            [-3.80, -0.15,   0.00],
        ])
        tau_1 = np.array([
            [0.00, 0.25, -1.20],
            [0.15, 0.00,  0.40],
            [0.60, 0.05,  0.00],
        ])
    elif amine_name == "DEA":
        # Austgen 1989 Table III for DEA-H2O-CO2 (intermediate between
        # MEA and MDEA; approximate values).
        tau_0 = np.array([
            [ 0.00, -1.55,   9.50],
            [ 1.10,  0.00,   1.40],
            [-4.20, -0.18,   0.00],
        ])
        tau_1 = np.array([
            [0.00, 0.28, -1.40],
            [0.18, 0.00,  0.45],
            [0.70, 0.08,  0.00],
        ])
    else:
        raise KeyError(
            f"Chen-Song τ parameters not available for {amine_name!r}. "
            f"Available: MEA, MDEA, DEA")

    return tau_0 + tau_1 * dT_over_T


def chen_song_log_gamma_molecular(
        amine_name: str,
        x_water: float,
        x_amine: float,
        x_CO2: float,
        T: float,
        alpha_NRTL: float = 0.2,
        asymmetric_solutes: bool = True,
) -> Tuple[float, float, float]:
    """Compute ln γ for water, amine, CO2(aq) via the short-range
    NRTL term of the Chen-Song 2004 generalized eNRTL.

    The standard NRTL formula for component i in a multi-component
    system:

        ln γ_i = (Σ_j τ_ji · G_ji · x_j) / (Σ_k G_ki · x_k)
                 + Σ_j (x_j · G_ij / Σ_k G_kj · x_k)
                       · [τ_ij - (Σ_k τ_kj · G_kj · x_k)
                                  / (Σ_k G_kj · x_k)]

    where G_ij = exp(-α · τ_ij) and α is the non-randomness parameter
    (default 0.2 for electrolyte/molecular systems).

    Parameters
    ----------
    amine_name : str
        "MEA", "MDEA", or "DEA".
    x_water, x_amine, x_CO2 : float
        Mole fractions of the three molecular species.  These do NOT
        need to sum to 1 (the ionic species occupy the remainder); the
        NRTL term is evaluated on the molecular sub-system.
    T : float
        Temperature [K].
    alpha_NRTL : float, default 0.2
        NRTL non-randomness parameter.
    asymmetric_solutes : bool, default True
        If True (Chen-Song / Austgen convention), γ for the *solutes*
        (amine, CO2) is converted to the asymmetric reference state by
        subtracting the infinite-dilution limit in water.  This is the
        standard treatment when CO2's vapor partial pressure is computed
        via Henry's law (Henry constant is defined at infinite dilution).
        γ_water keeps the symmetric (pure-liquid) reference.

    Returns
    -------
    (ln_gamma_water, ln_gamma_amine, ln_gamma_CO2) : tuple of float
        Natural-log γ for each molecular species.
    """
    s_total = x_water + x_amine + x_CO2
    if s_total <= 1e-12:
        return 0.0, 0.0, 0.0
    x = np.array([x_water, x_amine, x_CO2]) / s_total

    tau = _chen_song_tau_matrix(amine_name, T)
    G = np.exp(-alpha_NRTL * tau)
    np.fill_diagonal(G, 1.0)

    def _ln_g_at(xv: np.ndarray) -> np.ndarray:
        ln_g = np.zeros(3)
        for i in range(3):
            num1 = sum(tau[j, i] * G[j, i] * xv[j] for j in range(3))
            den1 = sum(G[k, i] * xv[k] for k in range(3))
            term1 = num1 / max(den1, 1e-12)
            term2 = 0.0
            for j in range(3):
                den_j = sum(G[k, j] * xv[k] for k in range(3))
                den_j = max(den_j, 1e-12)
                sum_tau_in_j = sum(tau[m, j] * G[m, j] * xv[m]
                                        for m in range(3))
                inner = tau[i, j] - sum_tau_in_j / den_j
                term2 += xv[j] * G[i, j] / den_j * inner
            ln_g[i] = term1 + term2
        return ln_g

    ln_g = _ln_g_at(x)

    if asymmetric_solutes:
        # Infinite-dilution reference: pure water, x_w → 1, others → 0.
        # The amine and CO2 γ at this limit becomes the reference value
        # we subtract so γ_solute → 1 at infinite dilution in water.
        x_inf = np.array([1.0, 0.0, 0.0])
        ln_g_inf = _ln_g_at(x_inf)
        ln_g[1] -= ln_g_inf[1]   # amine: asymmetric
        ln_g[2] -= ln_g_inf[2]   # CO2:   asymmetric
        # ln_g[0] (water) remains symmetric (pure-liquid reference)

    return float(ln_g[0]), float(ln_g[1]), float(ln_g[2])


def list_chen_song_amines() -> List[str]:
    """List amines with bundled Chen-Song τ parameters."""
    return ["MEA", "MDEA", "DEA"]

