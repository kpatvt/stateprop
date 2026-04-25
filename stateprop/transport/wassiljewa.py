"""Wassiljewa equation with Mason-Saxena (Wilke) coefficients for
mixture thermal conductivity.

Wassiljewa, A., Phys. Z. 5, 737 (1904).
Mason, E.A., S.C. Saxena, "Approximate Formula for the Thermal
Conductivity of Gas Mixtures", Phys. Fluids 1, 361 (1958).
Wilke, C.R., "A Viscosity Equation for Gas Mixtures", J. Chem. Phys.
18, 517 (1950).

This is the classical alternative to Chung's mixture rule. Given pure-
component thermal conductivities lambda_i and pure-component
viscosities mu_i (used to compute the inter-species coefficients),
mixture thermal conductivity is:

    lambda_m = sum_i [ x_i lambda_i / sum_j x_j Phi_ij ]

with Wilke-style cross coefficients

    Phi_ij = [1 + (mu_i/mu_j)^(1/2) (M_j/M_i)^(1/4)]^2
              / sqrt( 8 (1 + M_i/M_j) )

The simplified form (epsilon=1) gives results within ~5% of more
elaborate methods (e.g. Lindsay-Bromley) for typical hydrocarbon
mixtures and is widely used in process simulators.

Note on naming: the user's "Predictive-Soave for mixture lambda" does
not appear to match a standard published method (Predictive-Soave-
Redlich-Kwong is a group-contribution EOS for vapor-liquid equilibria,
not a transport-property method). Wassiljewa-Mason-Saxena is the most
commonly-implemented classical alternative to Chung's mixture rule and
serves the same purpose.

Units:
    M [kg/mol] (mass per mole; relative to other species only)
    mu [Pa.s], lambda [W/(m.K)]
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


def thermal_conductivity_mixture_wassiljewa(
    x: Sequence[float],
    M: Sequence[float],
    lambda_pure: Sequence[float],
    mu_pure: Sequence[float],
) -> float:
    """Mixture thermal conductivity via Wassiljewa-Mason-Saxena.

    Parameters
    ----------
    x : sequence
        Mole fractions, length N (sum to 1).
    M : sequence
        Pure-component molar masses, length N. SI [kg/mol]; only
        relative ratios M_i/M_j matter so any consistent unit works.
    lambda_pure : sequence
        Pure-component thermal conductivities [W/(m.K)], length N.
        Compute these at the mixture T using a per-species correlation
        (Chung pure, Stiel-Thodos pure, etc.) before calling.
    mu_pure : sequence
        Pure-component viscosities [Pa.s], length N. Same conditions
        as lambda_pure. Used only for the cross-coefficient ratios.

    Returns
    -------
    lambda_m : float
        Mixture thermal conductivity [W/(m.K)].
    """
    x = np.asarray(x, dtype=float)
    M = np.asarray(M, dtype=float)
    lam = np.asarray(lambda_pure, dtype=float)
    mu = np.asarray(mu_pure, dtype=float)
    N = x.size

    # Phi_ij = (1 + (mu_i/mu_j)^0.5 (M_j/M_i)^0.25)^2 / sqrt(8 (1 + M_i/M_j))
    mu_ratio = mu[:, None] / mu[None, :]              # (i, j) -> mu_i/mu_j
    M_ratio_ji = M[None, :] / M[:, None]              # (i, j) -> M_j/M_i
    M_ratio_ij = M[:, None] / M[None, :]              # (i, j) -> M_i/M_j
    numer = (1.0 + np.sqrt(mu_ratio) * M_ratio_ji ** 0.25) ** 2
    denom = np.sqrt(8.0 * (1.0 + M_ratio_ij))
    Phi = numer / denom                                # (N, N)

    # lambda_m = sum_i x_i lambda_i / (sum_j x_j Phi_ij)
    denominators = Phi @ x                             # (N,) -> sum_j x_j Phi_ij
    return float(np.sum(x * lam / denominators))


def viscosity_mixture_wilke(
    x: Sequence[float],
    M: Sequence[float],
    mu_pure: Sequence[float],
) -> float:
    """Mixture viscosity via Wilke's equation (a companion to
    Wassiljewa-Mason-Saxena, using the same Phi_ij coefficients).

    Often more accurate than simple mole-fraction averaging for
    species of disparate molar masses.

    Parameters
    ----------
    x : sequence
        Mole fractions, length N.
    M : sequence
        Pure-component molar masses, length N [kg/mol] (or any
        consistent unit).
    mu_pure : sequence
        Pure-component viscosities [Pa.s], length N.

    Returns
    -------
    mu_m : float
        Mixture viscosity [Pa.s].
    """
    x = np.asarray(x, dtype=float)
    M = np.asarray(M, dtype=float)
    mu = np.asarray(mu_pure, dtype=float)

    mu_ratio = mu[:, None] / mu[None, :]
    M_ratio_ji = M[None, :] / M[:, None]
    M_ratio_ij = M[:, None] / M[None, :]
    numer = (1.0 + np.sqrt(mu_ratio) * M_ratio_ji ** 0.25) ** 2
    denom = np.sqrt(8.0 * (1.0 + M_ratio_ij))
    Phi = numer / denom
    denominators = Phi @ x
    return float(np.sum(x * mu / denominators))
