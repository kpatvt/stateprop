"""Electrolyte thermodynamics utilities (v0.9.96).

Implements the foundational machinery for aqueous electrolyte
solution modeling:

    * Ionic strength I = ½·Σ m_i·z_i²
    * Debye-Hückel A coefficient A_φ(T, ρ_w, ε_w)
    * Davies equation (extended Debye-Hückel)
    * Molality ↔ mole-fraction conversions

The Pitzer and eNRTL models build on top of these; see ``pitzer.py``
and ``enrtl.py`` for the full activity-coefficient implementations.

References
----------
* Pitzer, K. S. (1991). *Activity Coefficients in Electrolyte
  Solutions* (2nd ed., CRC Press). Comprehensive reference for the
  Pitzer formulation.
* Helgeson, H. C., Kirkham, D. H. (1974). Theoretical prediction of
  the thermodynamic behavior of aqueous electrolytes at high
  pressures and temperatures. Am. J. Sci. 274, 1199-1261.
* Bradley, D. J., Pitzer, K. S. (1979). Thermodynamics of
  electrolytes. 12. Dielectric properties of water and Debye-Hückel
  parameters to 350°C and 1 kbar. J. Phys. Chem. 83, 1599-1603.
"""
from __future__ import annotations
from typing import Dict, Sequence
import numpy as np


# Physical constants (CODATA 2018)
_NA = 6.02214076e23           # Avogadro [1/mol]
_KB = 1.380649e-23            # Boltzmann [J/K]
_E = 1.602176634e-19          # elementary charge [C]
_EPS0 = 8.8541878128e-12      # vacuum permittivity [F/m]
_MW_WATER = 0.0180153          # kg/mol
_R = 8.314462618              # universal gas constant [J/(mol·K)]


# =====================================================================
# Ionic strength and basic conversions
# =====================================================================

def ionic_strength(molalities: Dict[str, float],
                    charges: Dict[str, int]) -> float:
    """Stoichiometric ionic strength [mol/kg].

    I = (1/2) * Σ m_i * z_i²

    Parameters
    ----------
    molalities : dict
        Map species name → molality [mol/kg solvent].
    charges : dict
        Map species name → integer charge.

    Returns
    -------
    float
        Ionic strength [mol/kg].
    """
    I = 0.0
    for species, m in molalities.items():
        z = charges.get(species, 0)
        if z != 0:
            I += m * z * z
    return 0.5 * I


def molality_to_mole_fraction(molalities: Dict[str, float],
                                MW_solvent: float = _MW_WATER) -> Dict[str, float]:
    """Convert molality basis (mol/kg solvent) to mole-fraction basis.

    Mole fraction of solvent x_w = 1 / (1 + MW_w * Σ m_i)
    Mole fraction of solute i:    x_i = MW_w * m_i / (1 + MW_w * Σ m_i)

    Parameters
    ----------
    molalities : dict
        Solute molalities (solvent not included).
    MW_solvent : float
        Solvent molar mass [kg/mol]. Default = water.

    Returns
    -------
    dict
        {'water': x_w, **solute_x}, summing to 1.
    """
    sum_m = sum(molalities.values())
    denom = 1.0 + MW_solvent * sum_m
    out = {"solvent": 1.0 / denom}
    for species, m in molalities.items():
        out[species] = MW_solvent * m / denom
    return out


def mole_fraction_to_molality(mole_fractions: Dict[str, float],
                                solvent_key: str = "solvent",
                                MW_solvent: float = _MW_WATER) -> Dict[str, float]:
    """Convert mole-fraction basis to molality (mol/kg solvent).

    m_i = x_i / (x_w * MW_w)

    Parameters
    ----------
    mole_fractions : dict
        Map name → mole fraction.  Sum should be 1.
    solvent_key : str
        Name of the solvent component. Default ``"solvent"``.
    MW_solvent : float
        Solvent molar mass [kg/mol].

    Returns
    -------
    dict
        Solute molalities (solvent not included since m_solvent isn't
        a meaningful concept).
    """
    x_w = mole_fractions.get(solvent_key)
    if x_w is None or x_w <= 0:
        raise ValueError(f"Solvent fraction at key {solvent_key!r} "
                          f"missing or non-positive")
    out = {}
    for k, x in mole_fractions.items():
        if k == solvent_key:
            continue
        out[k] = x / (x_w * MW_solvent)
    return out


# =====================================================================
# Debye-Hückel A coefficient
# =====================================================================

def water_density(T: float, p: float = 101325.0) -> float:
    """Water density [kg/m³] vs T [K] from a simple polynomial fit.

    Accurate to ~0.1% over 273-373 K at 1 atm.  Used internally for
    A_φ at temperatures other than 298.15 K.  For a more accurate
    value at high T, use stateprop.fluids.load_fluid("water") and
    its IAPWS-95 reference EOS.
    """
    # Polynomial fit to IAPWS data, T in K, returns kg/m³
    Tc = T - 273.15
    return (999.83952 + 16.945176*Tc - 7.9870401e-3*Tc**2
            - 46.170461e-6*Tc**3 + 105.56302e-9*Tc**4
            - 280.54253e-12*Tc**5) / (1.0 + 16.879850e-3*Tc)


def water_dielectric(T: float) -> float:
    """Static relative permittivity of water (Bradley-Pitzer 1979 fit).

    Fits IAPWS data within ~0.5% over 273-650 K.

    ε_r(T) = U1 * exp(U2 + U3·T + U4·T²) for low-pressure water.
    Simpler fit valid up to ~100 °C:
       ε_r = 87.740 - 0.40008(T-273.15) + 9.398e-4(T-273.15)²
              - 1.410e-6(T-273.15)³
    """
    Tc = T - 273.15
    return (87.740 - 0.40008*Tc + 9.398e-4*Tc**2 - 1.410e-6*Tc**3)


def debye_huckel_A(T: float, p: float = 101325.0) -> float:
    """Debye-Hückel A_φ coefficient for the osmotic-coefficient form.

    A_φ = (1/3) * (2π·N_A·ρ_w / 1000)^(1/2) * (e² / (4π·ε₀·ε_r·k·T))^(3/2)

    Returns a dimensionless coefficient.

    Parameters
    ----------
    T : float
        Temperature [K].
    p : float, optional
        Pressure [Pa]. Default = 1 atm. Affects ρ_w slightly; for
        most aqueous-solution work pressure dependence is negligible
        below 100 bar.

    Returns
    -------
    float
        A_φ. At 25°C, 1 atm, water: A_φ ≈ 0.392.

    Notes
    -----
    The activity-coefficient form A_γ = 3·A_φ (some references swap
    a factor of 3 and confuse the two).  Pitzer's formulation uses
    A_φ throughout.
    """
    rho_w = water_density(T, p)            # kg/m³
    eps_r = water_dielectric(T)
    rho_water_per_kg = rho_w / 1000.0      # kg/m³ × 1 m³/1000 L = kg/L
    # NB: rho_w in kg/m³, but Pitzer's formula uses moles per kg.  The
    # standard form is:
    # A_φ = (1/3) sqrt(2 π N_A ρ_w_kg_per_m3 / 1) * [e² / (4πε₀εRkT)]^(3/2)
    # Many texts express ρ_w in kg/m³ directly.
    # We follow Pitzer 1991 Eq. (1.14):
    pre = (1.0/3.0) * np.sqrt(2.0 * np.pi * _NA * rho_w)
    bracket = (_E ** 2 / (4.0 * np.pi * _EPS0 * eps_r * _KB * T)) ** 1.5
    return pre * bracket


# =====================================================================
# Davies equation (extended Debye-Hückel)
# =====================================================================

def davies_log_gamma_pm(z_plus: int, z_minus: int,
                          molality: float,
                          T: float = 298.15) -> float:
    """Davies equation: log10(γ_±) for a 1:1 electrolyte.

    log γ_± = -|z_+ z_-| · A · [√I/(1+√I) - 0.3·I]

    where A = 3·A_φ / ln(10) is the Davies/DH activity-coefficient
    coefficient (≈ 0.509 for water at 25 °C). Valid to roughly 0.5
    mol/kg.  Above this, use the full Pitzer model.
    """
    A_phi = debye_huckel_A(T)
    A_DH = 3.0 * A_phi / np.log(10.0)
    nu_plus = abs(z_minus)    # for 1:1, simple
    nu_minus = abs(z_plus)
    nu = nu_plus + nu_minus
    # For a 1:1 electrolyte at molality m, ionic strength = m
    # For general ν+:ν-, I = (1/2)(ν+·z+² + ν-·z-²) m
    I = 0.5 * (nu_plus * z_plus**2 + nu_minus * z_minus**2) * molality
    return -abs(z_plus * z_minus) * A_DH * (
        np.sqrt(I) / (1.0 + np.sqrt(I)) - 0.3 * I)


def debye_huckel_log_gamma_pm(z_plus: int, z_minus: int,
                                  molality: float, T: float = 298.15) -> float:
    """Pure Debye-Hückel limiting law: log γ_± = -A·|z+z-|·√I.

    Valid only at very low ionic strength (I < 0.01).  Use the
    Davies or Pitzer equations for engineering work.
    """
    A_phi = debye_huckel_A(T)
    A_DH = 3.0 * A_phi / np.log(10.0)
    nu_plus = abs(z_minus)
    nu_minus = abs(z_plus)
    I = 0.5 * (nu_plus * z_plus**2 + nu_minus * z_minus**2) * molality
    return -abs(z_plus * z_minus) * A_DH * np.sqrt(I)
