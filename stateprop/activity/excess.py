"""Excess thermodynamic properties from activity-coefficient models.

T-derivatives of gE give access to excess enthalpy hE, entropy sE,
heat capacity cpE, and per-component dln(gamma_i)/dT (e.g., for
non-isothermal flash, energy balances on distillation columns,
heat-of-mixing predictions, and verifying Gibbs-Helmholtz
consistency of fitted models).

The mixin provides:

    model.gE(T, x)              -> excess Gibbs energy [J/mol]
    model.hE(T, x)              -> excess enthalpy     [J/mol]
    model.sE(T, x)              -> excess entropy      [J/(mol K)]
    model.cpE(T, x)             -> excess heat cap.    [J/(mol K)]
    model.dgE_RT_dT(T, x)       -> d(gE/RT)/dT
    model.dlngammas_dT(T, x)    -> d(ln gamma_i)/dT  (per component)

All implemented via central finite difference on `gE_over_RT(T, x)`
or `lngammas(T, x)`, which the host class must already provide.

Relations:
    gE  = RT * (gE/RT)
    hE  = -RT^2 * d(gE/RT)/dT          (Gibbs-Helmholtz)
    sE  = (hE - gE) / T
    cpE = dhE/dT                       (numerical, second-order in gE)

For all activity models in `stateprop.activity` (NRTL, UNIQUAC,
UNIFAC, UNIFAC-Dortmund, UNIFAC-Lyngby) gE/RT is smooth in T and
the central FD with h ~ T * 1e-4 gives 6-8 digits of accuracy on
hE / dlngammas_dT, and 4-6 digits on cpE (which is a second
derivative).

For applications requiring more accuracy or symbolic differentiation,
each model could be extended with analytical T-derivatives of
its T-dependent functions (tau for NRTL/UNIQUAC, Psi_mn for UNIFAC).
This is on the roadmap; the FD implementation gives correct values
to engineering accuracy without code duplication across five
different model families.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


# CODATA 2018 / SI molar gas constant
R_GAS = 8.31446261815324   # J / (mol K)


class GibbsExcessTDerivatives:
    """Mixin adding T-derivatives of the excess Gibbs energy.

    The host class must provide:

        gE_over_RT(T: float, x: ndarray) -> float
        lngammas(T: float, x: ndarray) -> ndarray

    The host need not store anything; this mixin only adds methods.
    Step size defaults are conservative; can be overridden by setting
    `T_DERIV_H_REL` (relative) and `T_DERIV_H_MIN` (absolute) on the
    class or instance.
    """

    # Central-FD step: h = max(H_MIN, H_REL * T). H_REL of 1e-4 gives
    # roundoff at ~1e-12 and truncation at ~1e-8 relative for typical
    # gE_over_RT functions, so total is ~1e-7 rel for the first
    # derivative (excellent for hE).
    T_DERIV_H_REL: float = 1.0e-4
    T_DERIV_H_MIN: float = 1.0e-3   # K

    # Larger step for cpE since it's a numerical second derivative of
    # gE/RT (or first derivative of hE which is itself FD-based).
    T_DERIV_H_REL_2: float = 1.0e-3
    T_DERIV_H_MIN_2: float = 5.0e-3

    def _h_step(self, T: float, second: bool = False) -> float:
        if second:
            return max(self.T_DERIV_H_MIN_2, T * self.T_DERIV_H_REL_2)
        return max(self.T_DERIV_H_MIN, T * self.T_DERIV_H_REL)

    # -- First-order T-derivatives ------------------------------------------

    def dlngammas_dT(self, T: float, x: Sequence[float]) -> np.ndarray:
        """d(ln gamma_i)/dT for each component i.

        Default implementation: central finite difference on lngammas.
        Subclasses may override with analytical formulas for higher
        precision (typically ~1e-12 relative agreement with FD)."""
        T = float(T)
        h = self._h_step(T)
        plus = np.asarray(self.lngammas(T + h, x))
        minus = np.asarray(self.lngammas(T - h, x))
        return (plus - minus) / (2.0 * h)

    def dgE_RT_dT(self, T: float, x: Sequence[float]) -> float:
        """d(gE/RT)/dT at fixed x. Uses the identity
        gE/RT = sum_i x_i ln gamma_i, so this routes through
        dlngammas_dT and automatically benefits from any analytical
        override the model provides."""
        x_arr = np.asarray(x, dtype=float)
        return float(np.sum(x_arr * self.dlngammas_dT(T, x_arr)))

    def dlngammas_dT_FD(self, T: float, x: Sequence[float]) -> np.ndarray:
        """Force the finite-difference computation regardless of any
        analytical override. Useful for testing that an analytical
        implementation agrees with FD."""
        return GibbsExcessTDerivatives.dlngammas_dT(self, T, x)

    # -- Excess properties (J/mol or J/(mol K)) -----------------------------

    def gE(self, T: float, x: Sequence[float]) -> float:
        """Excess Gibbs energy gE = RT * (gE/RT) [J/mol]."""
        T = float(T)
        return R_GAS * T * self.gE_over_RT(T, x)

    def hE(self, T: float, x: Sequence[float]) -> float:
        """Excess enthalpy hE = -RT^2 * d(gE/RT)/dT [J/mol].

        Gibbs-Helmholtz relation. Equivalent forms:
            hE = -R T^2 sum_i x_i d(ln gamma_i)/dT
            hE = gE + T sE
        """
        T = float(T)
        return -R_GAS * T * T * self.dgE_RT_dT(T, x)

    def sE(self, T: float, x: Sequence[float]) -> float:
        """Excess entropy sE = (hE - gE)/T [J/(mol K)]."""
        T = float(T)
        return (self.hE(T, x) - self.gE(T, x)) / T

    def cpE(self, T: float, x: Sequence[float]) -> float:
        """Excess heat capacity cpE = dhE/dT at fixed x [J/(mol K)].

        Numerical FD on hE (which is itself FD-based on gE/RT). The
        underlying second derivative of gE limits this to ~4-6 digit
        accuracy, which is adequate for engineering use.
        """
        T = float(T)
        h = self._h_step(T, second=True)
        return (self.hE(T + h, x) - self.hE(T - h, x)) / (2.0 * h)
