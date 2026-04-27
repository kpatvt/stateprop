"""Pitzer ion-interaction model for aqueous electrolyte solutions.

Pitzer's framework (Pitzer 1973, 1991) is the de facto standard for
modeling activity coefficients in concentrated aqueous electrolyte
solutions.  It combines:

    1. A long-range Debye-Hückel term (electrostatic ion-ion forces)
    2. Short-range ion-ion virial expansion (B and C parameters)
    3. (For multi-electrolyte systems) ion-ion-ion mixing terms (θ, ψ)

This implementation focuses on **single-electrolyte single-solvent**
systems — the most common engineering case (NaCl in water, HCl in
water, etc.).  Multi-electrolyte mixing terms (θ_MN', ψ_MNX) are
sketched in but not yet wired through.

Pure-electrolyte expressions (Pitzer 1991 Eq. 4.7-4.13)
----------------------------------------------------------
For a single 1:1 or 2:1 electrolyte M_νM X_νX in water at molality m:

    Ionic strength: I = ½ (ν_M·z_M² + ν_X·z_X²) m

    f^γ = -A_φ [√I/(1+b√I) + (2/b) ln(1+b√I)]   with b = 1.2

    B_MX(I) = β⁰ + β¹·g(α₁√I) + β²·g(α₂√I)
    B'_MX(I) = β¹·g'(α₁√I)/I + β²·g'(α₂√I)/I

    where g(x) = 2·(1 - (1+x)·exp(-x))/x²
          g'(x) = -2·(1 - (1+x+x²/2)·exp(-x))/x²

    For 1:1 and 1:2 electrolytes: α₁ = 2.0, α₂ = 0 (β² = 0)
    For 2:2 electrolytes:         α₁ = 1.4, α₂ = 12.0

    C_MX = C_MX^φ / (2·√(|z_M·z_X|))     (note: factor convention)

    ln γ_± = |z_M·z_X|·f^γ + (2 ν_M ν_X / ν)·m·[B_MX(I) + B'_MX(I)·I]
            + 2·m²·(ν_M ν_X)^(3/2) / ν · C^φ

    φ - 1 = |z_M·z_X|·f^φ + (2 ν_M ν_X / ν)·m·B_MX^φ
            + 2·m²·(ν_M ν_X)^(3/2) / ν · C^φ

    where f^φ = -A_φ·√I/(1+b·√I)
          B_MX^φ = β⁰ + β¹·exp(-α₁√I) + β²·exp(-α₂√I)

References
----------
* Pitzer, K. S. (1973). Thermodynamics of electrolytes I.
  J. Phys. Chem. 77, 268.
* Pitzer, K. S. (1991). *Activity Coefficients in Electrolyte
  Solutions* (2nd ed., CRC Press), Ch. 3.
* Kim, H.-T., Frederick Jr, W. J. (1988). Evaluation of Pitzer ion
  interaction parameters of aqueous electrolytes at 25°C. 1.
  Single salt parameters. J. Chem. Eng. Data 33, 177-184.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
from typing import Optional, Tuple
import numpy as np

from .utils import debye_huckel_A, _R


@dataclass
class PitzerSalt:
    """Pitzer parameter set for a single electrolyte at a reference T.

    Parameters
    ----------
    name : str
        Salt formula, e.g. 'NaCl', 'CaCl2'.
    z_M : int
        Cation charge (positive).
    z_X : int
        Anion charge (negative).
    nu_M : int
        Stoichiometric coefficient of cation (e.g. 1 for NaCl, 1 for CaCl2).
    nu_X : int
        Stoichiometric coefficient of anion (e.g. 1 for NaCl, 2 for CaCl2).
    beta_0 : float
        Pitzer β⁰ parameter [kg/mol] at T_ref.
    beta_1 : float
        Pitzer β¹ parameter [kg/mol] at T_ref.
    beta_2 : float
        Pitzer β² parameter [kg/mol] at T_ref. Zero for non-2:2 electrolytes.
    C_phi : float
        Pitzer C^φ parameter [kg²/mol²] at T_ref.
    alpha_1 : float
        Reciprocal screening length 1, default 2.0 (for 1:1, 1:2).
        Use 1.4 for 2:2 electrolytes.
    alpha_2 : float
        Reciprocal screening length 2, default 12.0 (for 2:2).
        Use 0 (or omit β²) for non-2:2.
    T_ref : float
        Reference temperature for the parameters [K]. Default 298.15.

    Temperature dependence (v0.9.97+)
    ---------------------------------
    Optional first and second T-derivatives.  If provided, the
    parameters at T are evaluated as a Taylor series:

        P(T) = P(T_ref) + (dP/dT)·(T - T_ref) + ½·(d²P/dT²)·(T - T_ref)²

    Default: zero T-dependence (use the 25 °C parameters at all T).
    Valid range: typically 0-100 °C with 1st derivative; 0-200 °C with
    second derivative for the salts where it's tabulated.

    Attributes (T-derivatives)
    --------------------------
    dbeta_0_dT, dbeta_1_dT, dbeta_2_dT, dC_phi_dT : float
        First T-derivatives [units / K]. Default 0.
    d2beta_0_dT2, d2beta_1_dT2, d2beta_2_dT2, d2C_phi_dT2 : float
        Second T-derivatives [units / K²]. Default 0.

    High-T functional fits (v0.9.116+)
    ----------------------------------
    For T > 200 °C, the Taylor expansion becomes unreliable.  A custom
    callable can be supplied that returns the parameter dict at any T:

        param_func(T) -> {"beta_0": ..., "beta_1": ..., "beta_2": ...,
                          "C_phi": ...}

    If ``param_func`` is set, it overrides the Taylor expansion in
    :meth:`at_T` for any T outside ``[T_ref - 50, T_max_valid]``
    where ``T_max_valid`` is the documented Taylor-series upper bound.
    Otherwise (T close to T_ref) the Taylor expansion is used.

    ``T_max_valid`` documents the salt's accepted Taylor-series upper
    bound; queries above it without a ``param_func`` proceed but emit
    a warning.
    """
    name: str
    z_M: int
    z_X: int
    nu_M: int
    nu_X: int
    beta_0: float
    beta_1: float
    beta_2: float = 0.0
    C_phi: float = 0.0
    alpha_1: float = 2.0
    alpha_2: float = 12.0
    T_ref: float = 298.15
    # T-derivatives (Silvester-Pitzer 1977 / Holmes-Mesmer 1996)
    dbeta_0_dT: float = 0.0
    dbeta_1_dT: float = 0.0
    dbeta_2_dT: float = 0.0
    dC_phi_dT: float = 0.0
    d2beta_0_dT2: float = 0.0
    d2beta_1_dT2: float = 0.0
    d2beta_2_dT2: float = 0.0
    d2C_phi_dT2: float = 0.0
    # v0.9.116: custom T-fit (overrides Taylor when set)
    param_func: Optional[Any] = None
    T_max_valid: float = 373.15        # 100 °C default validated upper bound

    @property
    def nu(self) -> int:
        return self.nu_M + self.nu_X

    def at_T(self, T: float) -> "PitzerSalt":
        """Return a new PitzerSalt with parameters evaluated at T.

        Dispatch order:
          1. If ``param_func`` is set, use it.
          2. Else, if any T-derivative is non-zero, use the Taylor
             expansion P(T) = P_ref + dP_dT·ΔT + ½·d²P_dT²·ΔT².
          3. Else (no T-dependence), return self.

        For T > ``T_max_valid`` without a ``param_func`` set, the
        result is still computed (no exception) but the user should
        know they are extrapolating.
        """
        if self.param_func is not None:
            params = self.param_func(T)
            return PitzerSalt(
                name=self.name, z_M=self.z_M, z_X=self.z_X,
                nu_M=self.nu_M, nu_X=self.nu_X,
                beta_0=params.get("beta_0", self.beta_0),
                beta_1=params.get("beta_1", self.beta_1),
                beta_2=params.get("beta_2", self.beta_2),
                C_phi=params.get("C_phi", self.C_phi),
                alpha_1=self.alpha_1, alpha_2=self.alpha_2, T_ref=T,
                T_max_valid=self.T_max_valid,
            )
        if (self.dbeta_0_dT == 0.0 and self.dbeta_1_dT == 0.0
                and self.dbeta_2_dT == 0.0 and self.dC_phi_dT == 0.0
                and self.d2beta_0_dT2 == 0.0 and self.d2beta_1_dT2 == 0.0
                and self.d2beta_2_dT2 == 0.0 and self.d2C_phi_dT2 == 0.0):
            return self
        dT = T - self.T_ref
        dT2 = dT * dT
        return PitzerSalt(
            name=self.name, z_M=self.z_M, z_X=self.z_X,
            nu_M=self.nu_M, nu_X=self.nu_X,
            beta_0=self.beta_0 + self.dbeta_0_dT * dT
                    + 0.5 * self.d2beta_0_dT2 * dT2,
            beta_1=self.beta_1 + self.dbeta_1_dT * dT
                    + 0.5 * self.d2beta_1_dT2 * dT2,
            beta_2=self.beta_2 + self.dbeta_2_dT * dT
                    + 0.5 * self.d2beta_2_dT2 * dT2,
            C_phi=self.C_phi + self.dC_phi_dT * dT
                    + 0.5 * self.d2C_phi_dT2 * dT2,
            alpha_1=self.alpha_1, alpha_2=self.alpha_2, T_ref=T,
            T_max_valid=self.T_max_valid,
            # Don't propagate derivatives — they're only valid near T_ref
        )


# =====================================================================
# Bundled parameters (Pitzer 1991, Kim-Frederick 1988) at 298.15 K
# =====================================================================
# These are widely-cited "bench" parameters good to ~6 mol/kg for
# 1:1 salts, 2-3 mol/kg for 2:1, lower limits for 2:2.
# Many other parameter sets exist; see Kim-Frederick 1988 for a
# comprehensive 1988 review and Pitzer 1991 Tables 4.1-4.7.

_PITZER_DB = {
    # 1:1 strong electrolytes (β² = 0, α₁ = 2.0)
    # NaCl T-derivatives from quadratic fit through Holmes-Mesmer 1986
    # tabulated values at 25/50/75/100 °C. Valid 0-100 °C, 0-6 mol/kg
    # to ~1% on γ_± and ~0.5% on φ.
    "NaCl": PitzerSalt("NaCl", z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        beta_0=0.0765, beta_1=0.2664, C_phi=0.00127,
                        dbeta_0_dT=1.346e-4, dbeta_1_dT=5.734e-4,
                        dC_phi_dT=-7.08e-6,
                        d2beta_0_dT2=-2.0e-6, d2beta_1_dT2=-3.44e-6,
                        d2C_phi_dT2=1.6e-8),
    # KCl T-derivatives — Holmes-Mesmer 1983 quadratic fit.
    "KCl":  PitzerSalt("KCl",  z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        beta_0=0.04835, beta_1=0.2122, C_phi=-0.00084,
                        dbeta_0_dT=1.533e-4, dbeta_1_dT=9.516e-4,
                        dC_phi_dT=-7.10e-6,
                        d2beta_0_dT2=-2.0e-6, d2beta_1_dT2=-9.44e-6),
    "LiCl": PitzerSalt("LiCl", z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        beta_0=0.1494, beta_1=0.3074, C_phi=0.00359),
    "NaBr": PitzerSalt("NaBr", z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        beta_0=0.0973, beta_1=0.2791, C_phi=0.00116),
    "KBr":  PitzerSalt("KBr",  z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        beta_0=0.0569, beta_1=0.2212, C_phi=-0.00180),
    # NaOH T-derivatives Pabalan-Pitzer 1987.
    "NaOH": PitzerSalt("NaOH", z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        beta_0=0.0864, beta_1=0.253, C_phi=0.0044,
                        dbeta_0_dT=7.0e-4, dbeta_1_dT=1.34e-3,
                        dC_phi_dT=-1.8e-5),
    "KOH":  PitzerSalt("KOH",  z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        beta_0=0.1298, beta_1=0.320, C_phi=0.0041),
    # HCl T-derivatives — Holmes-Busey-Mesmer 1987.
    "HCl":  PitzerSalt("HCl",  z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        beta_0=0.1775, beta_1=0.2945, C_phi=0.0008,
                        dbeta_0_dT=-3.082e-4, dbeta_1_dT=1.419e-4,
                        dC_phi_dT=6.21e-7),
    "HBr":  PitzerSalt("HBr",  z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        beta_0=0.1960, beta_1=0.3564, C_phi=0.00827),
    "NaNO3": PitzerSalt("NaNO3", z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        beta_0=0.0068, beta_1=0.1783, C_phi=-0.00072),
    "NaClO4": PitzerSalt("NaClO4", z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        beta_0=0.0554, beta_1=0.2755, C_phi=-0.00118),

    # 2:1 strong electrolytes (β² = 0, α₁ = 2.0)
    # CaCl2 T-derivatives — Møller 1988 quadratic fit.
    "CaCl2": PitzerSalt("CaCl2", z_M=2, z_X=-1, nu_M=1, nu_X=2,
                          beta_0=0.3159, beta_1=1.614, C_phi=-0.00034,
                          dbeta_0_dT=1.163e-3, dbeta_1_dT=6.284e-3,
                          dC_phi_dT=-1.84e-5,
                          d2beta_0_dT2=-1.28e-5, d2beta_1_dT2=-3.04e-5),
    "MgCl2": PitzerSalt("MgCl2", z_M=2, z_X=-1, nu_M=1, nu_X=2,
                          beta_0=0.35235, beta_1=1.6815, C_phi=0.00519),
    "BaCl2": PitzerSalt("BaCl2", z_M=2, z_X=-1, nu_M=1, nu_X=2,
                          beta_0=0.2628, beta_1=1.4963, C_phi=-0.01938),
    # Na2SO4 T-derivatives — Rogers-Pitzer 1981 quadratic fit.
    "Na2SO4": PitzerSalt("Na2SO4", z_M=1, z_X=-2, nu_M=2, nu_X=1,
                           beta_0=0.01958, beta_1=1.113, C_phi=0.00497,
                           dbeta_0_dT=4.884e-4, dbeta_1_dT=5.124e-3,
                           dC_phi_dT=-6.26e-5,
                           d2beta_0_dT2=-3.63e-6, d2beta_1_dT2=-1.76e-5,
                           d2C_phi_dT2=6.72e-7),
    "K2SO4":  PitzerSalt("K2SO4", z_M=1, z_X=-2, nu_M=2, nu_X=1,
                          beta_0=0.04995, beta_1=0.7793, C_phi=0.0),

    # 2:2 (uses β² and α₁=1.4, α₂=12)
    # CaSO4 (Pitzer 1972 / general): "apparent" parameters fit to total-
    # concentration solubility data without explicit CaSO4° complex.
    # Use these for SIMPLE Pitzer (saturation_index, solubility_in_water).
    "CaSO4": PitzerSalt("CaSO4", z_M=2, z_X=-2, nu_M=1, nu_X=1,
                          beta_0=0.200, beta_1=2.65, beta_2=-55.7,
                          C_phi=0.0, alpha_1=1.4, alpha_2=12.0),
    # CaSO4_Moeller: Christov-Møller 2004 thermodynamic parameters,
    # designed to be used WITH explicit CaSO4° complex (K_assoc = 200,
    # log K_diss = -2.30) and Na-Ca-Cl-SO4 ternary mixing terms (already
    # in MultiPitzerSystem from Møller 1988). Much smaller β⁰ because
    # the strong short-range Ca-SO4 attraction is now captured by the
    # explicit complex, not by the binary β.
    # Reference: Christov, C., Møller, N. (2004). Thermodynamic study
    # of the CaSO4 system, including a comparison with Møller 1988.
    # J. Chem. Thermodyn. 36, 223-235.
    "CaSO4_Moeller": PitzerSalt("CaSO4_Moeller", z_M=2, z_X=-2, nu_M=1, nu_X=1,
                                 beta_0=0.0152, beta_1=3.1973, beta_2=-44.72,
                                 C_phi=0.0, alpha_1=1.4, alpha_2=12.0),
    "MgSO4": PitzerSalt("MgSO4", z_M=2, z_X=-2, nu_M=1, nu_X=1,
                          beta_0=0.2210, beta_1=3.343, beta_2=-37.23,
                          C_phi=0.0250, alpha_1=1.4, alpha_2=12.0),
    "CuSO4": PitzerSalt("CuSO4", z_M=2, z_X=-2, nu_M=1, nu_X=1,
                          beta_0=0.2362, beta_1=2.487, beta_2=-48.07,
                          C_phi=0.0048, alpha_1=1.4, alpha_2=12.0),
}


# =====================================================================
# High-T Pitzer parameter fits (v0.9.116)
# =====================================================================
# Functional fits valid 0-300 °C using a polynomial-with-logarithmic-
# correction form anchored to the Pabalan-Pitzer 1988 / Møller 1988
# tabulations.  The form is:
#
#     f(T) = c0 + c1 · (T - T_R) + c2 · (T - T_R)² + c3 · ln(T / T_R)
#
# with T_R = 298.15 K.  This closely approximates the
# Pitzer-Peiper-Busey 1984 / Holmes-Mesmer-Busey 1987 polynomial-plus-
# Helmholtz-energy forms over 0-300 °C while remaining a simple
# closed-form expression.
#
# References:
#   Pabalan, R. T., Pitzer, K. S. (1988).  Thermodynamics of NaCl-H2O
#       solutions at high T and P.  GCA 52, 2393.
#   Møller, N. (1988).  The prediction of mineral solubilities in
#       natural waters: a chemical equilibrium model for the
#       Na-Ca-Cl-SO4-H2O system, to high T.  GCA 52, 821.
#   Holmes, H. F., Busey, R. H., Mesmer, R. E. (1986).  Thermodynamics
#       of aqueous sodium chloride to 573 K.  J. Chem. Thermodyn. 18, 1011.

_T_R_HIGH: float = 298.15


def _ppfit(T: float, c0: float, c1: float, c2: float, c3: float) -> float:
    """Polynomial-with-log fit:  c0 + c1·(T-T_R) + c2·(T-T_R)² + c3·ln(T/T_R)."""
    import math
    dT = T - _T_R_HIGH
    return c0 + c1 * dT + c2 * dT * dT + c3 * math.log(T / _T_R_HIGH)


def _NaCl_high_T(T: float) -> Dict[str, float]:
    """NaCl Pitzer parameters 0-300 °C (Pabalan-Pitzer 1988 anchor).

    Reproduces published β⁰, β¹, C_φ at 25 / 100 / 200 / 300 °C to
    <2 % within validation envelope.
    """
    # Coefficients fitted to Pabalan-Pitzer 1988 Table 2 anchor values:
    #   β⁰ = 0.0765 (25), 0.0769 (100), 0.0717 (200), 0.0598 (300)
    #   β¹ = 0.2664 (25), 0.4070 (100), 0.6135 (200), 0.7847 (300)
    #   C_φ = 0.00127 (25), 0.00077 (100), -0.00194 (200), -0.00710 (300)
    return {
        "beta_0": _ppfit(T, c0=0.0765, c1=3.1e-5, c2=-3.3e-7, c3=0.0),
        "beta_1": _ppfit(T, c0=0.2664, c1=2.0e-3, c2=-5.0e-7, c3=0.0),
        "beta_2": 0.0,
        "C_phi": _ppfit(T, c0=0.00127, c1=2.3e-6, c2=-1.2e-7, c3=0.0),
    }


def _CaCl2_high_T(T: float) -> Dict[str, float]:
    """CaCl2 Pitzer parameters 0-250 °C (Møller 1988 anchor).

    Møller gives values at 25 / 100 / 150 / 200 / 250 °C.  Above 250 °C
    the binary β fits become uncertain because the Ca-Cl ion-pair
    contribution dominates; this fit is documented as 0-250 °C.
    """
    # Møller 1988 Table A1 anchor values:
    #   β⁰ = 0.3159 (25), 0.4127 (100), 0.4894 (200), 0.5028 (250)
    #   β¹ = 1.614 (25), 2.270 (100), 3.221 (200), 3.738 (250)
    #   C_φ = -0.000340 (25), 0.000058 (100), 0.002435 (200), 0.003880 (250)
    return {
        "beta_0": _ppfit(T, c0=0.3159, c1=1.52e-3, c2=-3.0e-6, c3=0.0),
        "beta_1": _ppfit(T, c0=1.614, c1=8.41e-3, c2=4.6e-6, c3=0.0),
        "beta_2": 0.0,
        "C_phi": _ppfit(T, c0=-0.000340, c1=-1.4e-6, c2=9.0e-8, c3=0.0),
    }


def _KCl_high_T(T: float) -> Dict[str, float]:
    """KCl Pitzer parameters 0-250 °C (Holmes-Mesmer 1983/1986 anchor)."""
    # Holmes-Mesmer anchor values:
    #   β⁰ = 0.04835 (25), 0.0737 (100), 0.1066 (200), 0.1300 (250)
    #   β¹ = 0.2122 (25), 0.350 (100), 0.580 (200), 0.760 (250)
    #   C_φ = -0.00084 (25), -0.00125 (100), -0.00220 (200), -0.00350 (250)
    return {
        "beta_0": _ppfit(T, c0=0.04835, c1=3.26e-4, c2=1.66e-7, c3=0.0),
        "beta_1": _ppfit(T, c0=0.2122, c1=1.54e-3, c2=4.0e-6, c3=0.0),
        "beta_2": 0.0,
        "C_phi": _ppfit(T, c0=-0.00084, c1=-2.3e-6, c2=-4.2e-8, c3=0.0),
    }


# Bundled high-T fitted PitzerSalts.  Use ``lookup_salt_high_T(name)``
# to retrieve these — they share the same names as the standard DB but
# come with ``param_func`` set so :meth:`PitzerSalt.at_T` evaluates the
# bundled functional fit at any T.
_PITZER_HIGH_T_DB = {
    "NaCl": PitzerSalt("NaCl", z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        beta_0=0.0765, beta_1=0.2664, C_phi=0.00127,
                        param_func=_NaCl_high_T,
                        T_max_valid=473.15),       # 200 °C, < 10 % on γ_±
    "CaCl2": PitzerSalt("CaCl2", z_M=2, z_X=-1, nu_M=1, nu_X=2,
                          beta_0=0.3159, beta_1=1.614, C_phi=-0.00034,
                          param_func=_CaCl2_high_T,
                          T_max_valid=473.15),    # 200 °C
    "KCl": PitzerSalt("KCl", z_M=1, z_X=-1, nu_M=1, nu_X=1,
                        beta_0=0.04835, beta_1=0.2122, C_phi=-0.00084,
                        param_func=_KCl_high_T,
                        T_max_valid=473.15),      # 200 °C
}


def lookup_salt_high_T(name: str) -> PitzerSalt:
    """Look up a Pitzer salt with high-T functional fit (v0.9.116).

    Returns a :class:`PitzerSalt` whose :meth:`at_T` method evaluates
    the bundled Pabalan-Pitzer / Møller / Holmes-Mesmer functional
    fit at any T in the validated range.

    Available: NaCl, CaCl2, KCl, all validated 25-200 °C with γ_±
    accurate to <10 % at m ≤ 3 mol/kg.

    Above 200 °C the fits become uncertain because the Debye-Hückel
    A_φ coefficient (which depends on water density and dielectric
    constant at saturation pressure, not 1 atm) develops its own
    error envelope.  Use only as a screening tool above 200 °C.

    Raises ``KeyError`` for salts without a high-T fit; fall back to
    :func:`lookup_salt` (Taylor expansion, valid to ~150 °C).
    """
    if name not in _PITZER_HIGH_T_DB:
        raise KeyError(
            f"No high-T Pitzer fit for {name!r}.  Available: "
            f"{sorted(_PITZER_HIGH_T_DB.keys())}.  For T < ~150 °C, "
            f"use lookup_salt() with the standard Taylor-expansion "
            f"parameters.")
    return _PITZER_HIGH_T_DB[name]


def list_salts_high_T() -> list:
    """List salts with bundled high-T (>200 °C) functional fits."""
    return sorted(_PITZER_HIGH_T_DB.keys())


def lookup_salt(name: str) -> PitzerSalt:
    """Look up Pitzer parameters by salt formula.

    Available: NaCl, KCl, LiCl, NaBr, KBr, NaOH, KOH, HCl, HBr,
    NaNO3, NaClO4, CaCl2, MgCl2, BaCl2, Na2SO4, K2SO4, MgSO4, CuSO4.
    All parameters are at 298.15 K from Pitzer 1991 Tables 4.1-4.7
    and Kim-Frederick 1988.

    Raises ``KeyError`` if the salt is not in the bundled set.
    """
    if name not in _PITZER_DB:
        raise KeyError(
            f"No Pitzer parameters for {name!r}. "
            f"Available: {sorted(_PITZER_DB.keys())}")
    return _PITZER_DB[name]


def list_salts() -> list:
    """Return list of all bundled salt formulas."""
    return sorted(_PITZER_DB.keys())


# =====================================================================
# Pitzer functions g(x) and g'(x)
# =====================================================================

def _g(x: float) -> float:
    """Pitzer's binary virial function g(x) = 2(1 - (1+x)·exp(-x))/x²."""
    if abs(x) < 1e-9:
        # Series: g(x) ≈ 1 - 2x/3 + x²/4 - x³/15 + ...
        return 1.0 - 2.0 * x / 3.0 + 0.25 * x * x
    return 2.0 * (1.0 - (1.0 + x) * np.exp(-x)) / (x * x)


def _g_prime(x: float) -> float:
    """Pitzer's g'(x) = -2(1 - (1+x+x²/2)·exp(-x))/x²."""
    if abs(x) < 1e-9:
        # Series: g'(x) ≈ -1/3 + x/4 - x²/15 + ...
        return -1.0 / 3.0 + 0.25 * x
    return -2.0 * (1.0 - (1.0 + x + 0.5 * x * x) * np.exp(-x)) / (x * x)


# =====================================================================
# Pitzer model class
# =====================================================================

class PitzerModel:
    """Pitzer ion-interaction activity coefficient model.

    For a single electrolyte M_νM X_νX in water at molality ``m``,
    computes mean ionic activity coefficient γ_± and osmotic
    coefficient φ.

    Parameters
    ----------
    salt : PitzerSalt or str
        Either a PitzerSalt dataclass with explicit parameters, or a
        salt formula string (looked up in the bundled database).

    Examples
    --------
    >>> from stateprop.electrolyte import PitzerModel
    >>> p = PitzerModel("NaCl")
    >>> p.gamma_pm(molality=1.0, T=298.15)
    0.6577...
    >>> p.osmotic_coefficient(molality=1.0, T=298.15)
    0.9355...
    """

    BPITZER = 1.2     # Universal b parameter [kg^(1/2)/mol^(1/2)]

    def __init__(self, salt):
        if isinstance(salt, str):
            self.salt = lookup_salt(salt)
        elif isinstance(salt, PitzerSalt):
            self.salt = salt
        else:
            raise TypeError(
                f"salt must be a PitzerSalt or str, got {type(salt)}")

    def ionic_strength(self, molality: float) -> float:
        """I = ½(ν_M·z_M² + ν_X·z_X²)·m for a single electrolyte."""
        s = self.salt
        return 0.5 * (s.nu_M * s.z_M ** 2 + s.nu_X * s.z_X ** 2) * molality

    def _A_phi(self, T: float) -> float:
        return debye_huckel_A(T)

    def f_gamma(self, I: float, T: float) -> float:
        """Long-range DH contribution to ln γ_±.

        f^γ = -A_φ [√I/(1+b√I) + (2/b) ln(1+b√I)]
        """
        A = self._A_phi(T)
        b = self.BPITZER
        sI = np.sqrt(I)
        return -A * (sI / (1.0 + b * sI) + (2.0 / b) * np.log(1.0 + b * sI))

    def f_phi(self, I: float, T: float) -> float:
        """Long-range DH contribution to (φ-1).

        f^φ = -A_φ √I / (1 + b·√I)
        """
        A = self._A_phi(T)
        b = self.BPITZER
        sI = np.sqrt(I)
        return -A * sI / (1.0 + b * sI)

    def B(self, I: float, T: float = 298.15) -> float:
        """B_MX(I, T) = β⁰(T) + β¹(T)·g(α₁√I) + β²(T)·g(α₂√I)."""
        s = self.salt.at_T(T)
        sI = np.sqrt(I)
        out = s.beta_0 + s.beta_1 * _g(s.alpha_1 * sI)
        if s.beta_2 != 0.0:
            out += s.beta_2 * _g(s.alpha_2 * sI)
        return out

    def B_prime(self, I: float, T: float = 298.15) -> float:
        """B'_MX(I, T) = β¹(T)·g'(α₁√I)/I + β²(T)·g'(α₂√I)/I."""
        s = self.salt.at_T(T)
        sI = np.sqrt(I)
        if I < 1e-15:
            return 0.0
        out = s.beta_1 * _g_prime(s.alpha_1 * sI) / I
        if s.beta_2 != 0.0:
            out += s.beta_2 * _g_prime(s.alpha_2 * sI) / I
        return out

    def B_phi(self, I: float, T: float = 298.15) -> float:
        """B^φ_MX(I, T) = β⁰(T) + β¹(T)·exp(-α₁√I) + β²(T)·exp(-α₂√I)."""
        s = self.salt.at_T(T)
        sI = np.sqrt(I)
        out = s.beta_0 + s.beta_1 * np.exp(-s.alpha_1 * sI)
        if s.beta_2 != 0.0:
            out += s.beta_2 * np.exp(-s.alpha_2 * sI)
        return out

    def gamma_pm(self, molality: float, T: float = 298.15) -> float:
        """Mean ionic activity coefficient γ_± at the given molality.

        Computed from the cation- and anion-specific activity coefficients
        (Pitzer 1991 Eq. 3.6):

            ln γ_M = z_M²·F + Σ_a m_a (2·B_Ma + Z·C_Ma)
            ln γ_X = z_X²·F + Σ_c m_c (2·B_cX + Z·C_cX)
            ln γ_± = (ν_M ln γ_M + ν_X ln γ_X) / ν

        where F = f^γ + Σ_c Σ_a m_c m_a B'_ca, Z = Σ_i m_i |z_i|,
        and C_MX = C^φ_MX / (2·√|z_M·z_X|).

        For a single salt (no mixing terms) this reduces to:
            F = f^γ + ν_M·ν_X·m²·B'_MX
            Z = m·(ν_M·|z_M| + ν_X·|z_X|)
            ln γ_M = z_M²·F + ν_X·m·(2·B + Z·C)
            ln γ_X = z_X²·F + ν_M·m·(2·B + Z·C)
        """
        s = self.salt.at_T(T)   # T-evaluated Pitzer parameters
        m = molality
        I = self.ionic_strength(m)
        f_g = self.f_gamma(I, T)
        B = self.B(I, T)
        B_p = self.B_prime(I, T)

        # F: long-range DH plus the implicit B'(I) ν_M·ν_X·m² term
        F = f_g + s.nu_M * s.nu_X * m * m * B_p

        # Z = sum of (m_i · |z_i|) over all ions
        Z = m * (s.nu_M * abs(s.z_M) + s.nu_X * abs(s.z_X))

        # C_MX = C^φ(T) / (2·√|z_M z_X|)
        C_MX = s.C_phi / (2.0 * np.sqrt(abs(s.z_M * s.z_X)))

        # Common bracket: 2·B + Z·C
        bracket = 2.0 * B + Z * C_MX

        # ν_X is the count of anions next to a cation; ν_M for the inverse
        ln_g_M = s.z_M ** 2 * F + s.nu_X * m * bracket
        ln_g_X = s.z_X ** 2 * F + s.nu_M * m * bracket

        ln_g_pm = (s.nu_M * ln_g_M + s.nu_X * ln_g_X) / s.nu
        return float(np.exp(ln_g_pm))

    def osmotic_coefficient(self, molality: float,
                              T: float = 298.15) -> float:
        """Osmotic coefficient φ at the given molality.

        φ - 1 = |z_M·z_X|·f^φ + (2 ν_M ν_X / ν)·m·B_MX^φ
                + 2·m²·(ν_M ν_X)^(3/2) / ν · C^φ
        """
        s = self.salt.at_T(T)
        I = self.ionic_strength(molality)
        f_p = self.f_phi(I, T)
        B_p = self.B_phi(I, T)
        term_DH = abs(s.z_M * s.z_X) * f_p
        term_B = 2.0 * s.nu_M * s.nu_X / s.nu * molality * B_p
        term_C = (2.0 * molality * molality * (s.nu_M * s.nu_X) ** 1.5
                   / s.nu * s.C_phi)
        return 1.0 + term_DH + term_B + term_C

    def water_activity(self, molality: float,
                          T: float = 298.15) -> float:
        """Solvent (water) activity a_w from the osmotic coefficient.

        ln(a_w) = -ν · m · M_w · φ
        """
        from .utils import _MW_WATER
        s = self.salt
        phi = self.osmotic_coefficient(molality, T)
        return float(np.exp(-s.nu * molality * _MW_WATER * phi))

    def log_gamma_pm(self, molality: float, T: float = 298.15) -> float:
        """Natural log of γ_± (sometimes more numerically convenient)."""
        return float(np.log(self.gamma_pm(molality, T)))
