"""Jossi-Stiel-Thodos high-pressure viscosity correlation.

Jossi, J.A., L.I. Stiel, G. Thodos, "The Viscosity of Pure Substances
in the Dense Gaseous and Liquid Phases", AIChE J. 8, 59 (1962).

Stiel, L.I., G. Thodos, "The Viscosity of Polar Substances in the
Dense Gaseous and Liquid Regions", AIChE J. 10, 26 (1964).

This is a residual-viscosity correlation: given a dilute-gas viscosity
mu_0 (e.g., from Chapman-Enskog or Chung dilute limit), it adds a
density-dependent correction valid up to liquid-like densities. The
correlation is an alternative to extending Chung's full polynomial to
high densities; it tends to be more accurate for the dense liquid
region near and above the critical density.

Form:
    [(mu - mu_0) xi * 10^4 + 10^-4]^(1/4) = poly(rho_r)
    poly(rho_r) = 0.1023 + 0.023364 rho_r + 0.058533 rho_r^2
                  - 0.040758 rho_r^3 + 0.0093324 rho_r^4
    xi = T_c^(1/6) / (sqrt(M_g) p_c_atm^(2/3))   [units: (cP)^-1]
    rho_r = rho_mol * V_c                          [reduced density]

Validity: rho_r in [0.1, 3.0]; tested mostly for nonpolar fluids and
weakly polar fluids. For strongly-polar/associating fluids the polar
extension (Stiel-Thodos 1964) gives modest improvements but is not
included here -- use Chung dense-fluid expression for those.

For mixtures, mole-fraction-weighted critical properties are used
(Lee-Kesler-style mixing). Cross-interactions are ignored (k_ij=0
implicit).

Units throughout (SI):
    rho_mol [mol/m^3], T [K], M [kg/mol], T_c [K], p_c [Pa]
    V_c [m^3/mol], mu [Pa.s]
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


# Jossi-Stiel-Thodos polynomial coefficients
_JST = np.array([0.1023, 0.023364, 0.058533, -0.040758, 0.0093324])


def _xi_factor(T_c: float, p_c: float, M: float) -> float:
    """Stiel-Thodos kinematic-viscosity scale parameter xi in (cP)^-1.

    Inputs in SI. Internally converts to (T_c [K], p_c [atm], M [g/mol]).
    """
    M_g = M * 1000.0
    p_c_atm = p_c / 101325.0
    return T_c ** (1.0 / 6.0) / (np.sqrt(M_g) * p_c_atm ** (2.0 / 3.0))


def viscosity_stiel_thodos(
    rho_mol: float,
    T_c: float,
    p_c: float,
    V_c: float,
    M: float,
    mu_dilute: float,
) -> float:
    """Total viscosity = dilute-gas + Jossi-Stiel-Thodos dense correction.

    Parameters
    ----------
    rho_mol : float
        Molar density of the fluid [mol/m^3]. Use the EOS solution.
    T_c, p_c : float
        Critical temperature [K] and pressure [Pa].
    V_c : float
        Critical molar volume [m^3/mol]. (rho_c = 1/V_c.)
    M : float
        Molar mass [kg/mol].
    mu_dilute : float
        Dilute-gas viscosity at the given T [Pa.s]. Compute from
        Chung's mu_0 expression or any other dilute-gas correlation
        (Chapman-Enskog, etc.) before calling this.

    Returns
    -------
    mu : float
        Total viscosity [Pa.s].

    Notes
    -----
    For rho_r outside the validated range [0.1, 3.0], the formula is
    extrapolated and accuracy degrades. Returns mu_dilute (no
    correction) for very low densities where the residual correlation
    would amplify numerical noise.
    """
    rho_r = rho_mol * V_c
    if rho_r < 0.05:
        # Below the validated range; dilute-gas value is sufficient
        return mu_dilute
    xi = _xi_factor(T_c, p_c, M)
    # Polynomial evaluation: 0.1023 + 0.023364 rho_r + ...
    poly = _JST[0] + rho_r * (_JST[1] + rho_r * (_JST[2]
            + rho_r * (_JST[3] + rho_r * _JST[4])))
    # (mu - mu_0) xi + 10^-4 = poly^4
    # mu - mu_0 in cP = (poly^4 - 1e-4) / xi
    rhs = poly ** 4 - 1e-4
    delta_mu_cP = rhs / xi
    delta_mu = delta_mu_cP * 1e-3   # cP -> Pa.s
    return mu_dilute + delta_mu


def _mixture_critical_properties(
    x: np.ndarray, T_c: np.ndarray, p_c: np.ndarray,
    V_c: np.ndarray, M: np.ndarray
) -> tuple[float, float, float, float]:
    """Lee-Kesler-style mole-fraction mixing rules for pseudo-component
    critical properties used in mixture Stiel-Thodos.

    Returns (T_cm, p_cm, V_cm, M_m).
    """
    M_m = float(np.sum(x * M))
    # Volumetric mixing for V_c (Lee-Kesler):
    # V_cm = sum_ij x_i x_j V_cij,  V_cij = ((V_ci^(1/3) + V_cj^(1/3))/2)^3
    Vc13 = V_c ** (1.0 / 3.0)
    Vcij = (0.5 * (Vc13[:, None] + Vc13[None, :])) ** 3
    V_cm = float(x @ Vcij @ x)
    # T_cm: combining rule, T_cij = sqrt(T_ci T_cj)
    Tcij = np.sqrt(np.outer(T_c, T_c))
    T_cm = float(x @ (Tcij * Vcij) @ x) / V_cm
    # p_cm via Z_c estimate: use mole-fraction-weighted Z_c
    Z_c = p_c * V_c / (8.31446 * T_c)
    Z_cm = float(np.sum(x * Z_c))
    p_cm = Z_cm * 8.31446 * T_cm / V_cm
    return T_cm, p_cm, V_cm, M_m


def viscosity_mixture_stiel_thodos(
    rho_mol: float,
    x: Sequence[float],
    T_c: Sequence[float],
    p_c: Sequence[float],
    V_c: Sequence[float],
    M: Sequence[float],
    mu_dilute_mix: float,
) -> float:
    """Mixture viscosity via Stiel-Thodos with Lee-Kesler critical-property
    mixing.

    Parameters
    ----------
    rho_mol : float
        Mixture molar density [mol/m^3].
    x : sequence
        Mole fractions, length N, sum to 1.
    T_c, p_c : sequence
        Pure-component critical T [K] and p [Pa], length N each.
    V_c : sequence
        Pure-component critical molar volume [m^3/mol], length N.
    M : sequence
        Pure-component molar masses [kg/mol], length N.
    mu_dilute_mix : float
        Mixture dilute-gas viscosity [Pa.s]. Compute via Wilke's rule
        on the pure-component dilute viscosities, or via Chung's
        mixture mu_0 expression.

    Returns
    -------
    mu : float
        Mixture viscosity [Pa.s].
    """
    x = np.asarray(x, dtype=float)
    T_c = np.asarray(T_c, dtype=float)
    p_c = np.asarray(p_c, dtype=float)
    V_c = np.asarray(V_c, dtype=float)
    M = np.asarray(M, dtype=float)
    T_cm, p_cm, V_cm, M_m = _mixture_critical_properties(x, T_c, p_c, V_c, M)
    return viscosity_stiel_thodos(rho_mol, T_cm, p_cm, V_cm, M_m, mu_dilute_mix)
