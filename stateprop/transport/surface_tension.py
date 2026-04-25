"""Surface tension correlations.

- Brock-Bird 1955: pure-fluid corresponding-states (Reid-Prausnitz-Poling
  Eq. 12-3.5).
- Macleod-Sugden 1923/Sugden 1924: parachor-based pure-fluid and mixture
  correlation using phase densities from an EOS.

Units:
    T [K], rho_mol [mol/m^3], sigma [N/m]
"""

from __future__ import annotations
from typing import Sequence
import numpy as np


def surface_tension_brock_bird(comp, T: float) -> float:
    """Pure-fluid surface tension via Brock-Bird [N/m]."""
    T_c = float(comp.T_c)
    p_c = float(comp.p_c)
    omega = float(comp.acentric_factor)
    T_r = T / T_c
    if T_r >= 1.0:
        return 0.0

    T_b = getattr(comp, 'T_b', None)
    if T_b is None or T_b <= 0.0:
        T_br = 0.567 + 0.1 * omega - 0.05 * omega ** 2
    else:
        T_br = float(T_b) / T_c

    p_c_bar = p_c / 1e5
    Q = 0.1196 * (1.0 + T_br * np.log(p_c_bar / 1.01325) / (1.0 - T_br)) - 0.279
    sigma_dyn_cm = p_c_bar ** (2.0/3.0) * T_c ** (1.0/3.0) * Q * (1.0 - T_r) ** (11.0/9.0)
    return float(sigma_dyn_cm * 1e-3)


# -------------------------------------------------------------------------
# Macleod-Sugden parachor method (v0.9.33)
# -------------------------------------------------------------------------

def surface_tension_macleod_sugden(comp, rho_L_mol: float, rho_V_mol: float) -> float:
    """Pure-fluid surface tension via Macleod-Sugden [N/m].

    sigma^(1/4) = P (rho_L - rho_V)    with rho in mol/cm^3, sigma in dyn/cm

    Parameters
    ----------
    comp : component-like
        Must have `parachor` [cm^3/mol * (dyn/cm)^(1/4)].
    rho_L_mol : float
        Saturated-liquid molar density [mol/m^3].
    rho_V_mol : float
        Saturated-vapor molar density [mol/m^3].
    """
    P = float(getattr(comp, 'parachor', 0.0) or 0.0)
    if P <= 0.0:
        raise ValueError(
            f"component {getattr(comp, 'name', '?')!r} has parachor={P}; "
            "Macleod-Sugden requires a parachor value.")
    drho = (rho_L_mol - rho_V_mol) * 1e-6   # mol/cm^3
    if drho <= 0.0:
        return 0.0
    sigma_dyn_cm = (P * drho) ** 4
    return float(sigma_dyn_cm * 1e-3)


def surface_tension_mixture_macleod_sugden(
    comps: Sequence,
    x: Sequence[float],
    y: Sequence[float],
    rho_L_mol: float,
    rho_V_mol: float,
) -> float:
    """Mixture surface tension via Macleod-Sugden [N/m].

    sigma^(1/4) = Sum_i P_i (rho_L * x_i - rho_V * y_i)
                   with rho in mol/cm^3.

    Parameters
    ----------
    comps : list of components (length N)
    x : array-like (N,)  liquid-phase mole fractions
    y : array-like (N,)  vapor-phase mole fractions
    rho_L_mol, rho_V_mol : float
        Saturated liquid and vapor molar densities [mol/m^3].

    Notes
    -----
    Macleod-Sugden is the standard engineering default for mixture
    surface tension (used in Aspen, HYSYS). Typical accuracy 5-20%
    when densities come from a reliable EOS.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    P = np.array([float(getattr(c, 'parachor', 0.0) or 0.0) for c in comps])
    if np.any(P <= 0.0):
        missing = [getattr(c, 'name', '?') for c, p in zip(comps, P) if p <= 0.0]
        raise ValueError(f"components missing parachor: {missing}")
    rho_L_cm3 = rho_L_mol * 1e-6
    rho_V_cm3 = rho_V_mol * 1e-6
    s_quarter = float(np.sum(P * (rho_L_cm3 * x - rho_V_cm3 * y)))
    if s_quarter <= 0.0:
        return 0.0
    return float(s_quarter ** 4 * 1e-3)

