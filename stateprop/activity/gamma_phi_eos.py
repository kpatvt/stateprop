"""Gamma-phi flash with EOS-based vapor fugacities (high-pressure).

The v0.9.40 `GammaPhiFlash` treats the vapor as an ideal gas, which
breaks down above ~5 bar or near the critical region. This module
extends the gamma-phi formulation by sourcing vapor fugacity
coefficients from an equation of state (cubic or SAFT) for the
vapor phase while keeping the activity coefficient model for the
liquid:

    f_i^L = x_i * gamma_i(T, x) * f_pure_i^L(T, p)
    f_i^V = y_i * phi_i^V(T, p, y) * p

At equilibrium f_i^L = f_i^V, giving:

    K_i = y_i / x_i
        = gamma_i * f_pure_i^L(T, p) / (p * phi_i^V(T, p, y))
        = gamma_i * p_i^sat * phi_i^sat * exp(V_i^L (p - p_i^sat) / RT)
                  / (p * phi_i^V)

where:
- gamma_i(T, x) from activity model (NRTL/UNIQUAC/UNIFAC/...)
- p_i^sat(T) from Antoine or other psat correlation
- phi_i^sat(T) is the pure-component vapor fugacity coefficient
  at saturation, optionally computed from the same EOS
- exp(V_i^L (p - p_i^sat) / RT) is the Poynting correction,
  optionally included if pure-component liquid molar volumes are
  supplied
- phi_i^V(T, p, y) computed from the vapor EOS at the current
  vapor composition, requires SS iteration as y changes during flash

This is the engineering-standard formulation for non-ideal mixtures
above ambient pressure (Aspen, ProSim, ProMax, HYSYS, DWSIM all
implement variants of this scheme).

Reference: Walas, "Phase Equilibria in Chemical Engineering" Ch. 4;
Reid-Prausnitz-Poling 5th ed. Ch. 8 + 11.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import warnings
import numpy as np

from .gamma_phi import (BubbleResult, DewResult, FlashResult,
                          _solve_rachford_rice)


_R_GAS = 8.31446261815324   # J/(mol K)


class GammaPhiEOSFlash:
    """High-pressure gamma-phi flash. Activity model for the liquid,
    cubic / SAFT EOS for the vapor.

    Parameters
    ----------
    activity_model : NRTL / UNIQUAC / UNIFAC / UNIFAC_Dortmund / UNIFAC_Lyngby
        Anything implementing gammas(T, x).
    psat_funcs : sequence of callables
        psat_i(T) -> Pa for each component.
    vapor_eos : EOS mixture object
        Must implement:
        - density_from_pressure(p, T, x, phase_hint='vapor') -> rho [mol/m^3]
        - ln_phi(rho, T, x) -> ndarray of ln(phi_i)
        Both `CubicMixture` and `SAFTMixture` from this package qualify.
    pure_liquid_volumes : sequence of float, optional
        V_i^L in m^3/mol per component. If supplied, the Poynting
        correction is included in K-values. If None (default), the
        Poynting factor defaults to 1, accurate to ~5% at 100 bar
        for typical mixtures.
    phi_sat_funcs : sequence of callables, optional
        phi_i^sat(T) -> dimensionless fugacity coefficient of pure i
        at saturation. If None, defaults to 1 (low-pressure
        approximation). For higher-precision work, pass
        phi_sat = pure_eos_phi_sat helpers.

    Notes
    -----
    For a binary at moderate pressure (10-30 bar), the dominant
    correction is `phi_i^V` for the vapor; phi_sat and Poynting
    contribute ~1-5% each. At p > 100 bar all three matter.
    """

    def __init__(self,
                 activity_model,
                 psat_funcs: Sequence[Callable[[float], float]],
                 vapor_eos,
                 pure_liquid_volumes: Optional[Sequence[float]] = None,
                 phi_sat_funcs: Optional[Sequence[Callable[[float], float]]] = None):
        self.model = activity_model
        self.psat = list(psat_funcs)
        self.eos = vapor_eos
        self.N = len(self.psat)
        self._VL = (np.asarray(pure_liquid_volumes, dtype=float)
                    if pure_liquid_volumes is not None else None)
        self._phi_sat = list(phi_sat_funcs) if phi_sat_funcs is not None else None
        if hasattr(activity_model, 'N') and activity_model.N != self.N:
            raise ValueError(f"activity_model N={activity_model.N} != "
                             f"{self.N} psat_funcs")

    # -----------------------------------------------------------------
    # Helper: K-value at (T, p, x, y) given current vapor composition
    # -----------------------------------------------------------------

    def _K_values(self, T: float, p: float,
                   x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute K_i = γ_i p_i^sat φ_i^sat exp(Poynting) / (p φ_i^V).

        Requires y (vapor composition) to evaluate φ_i^V at the
        current iterate. SS iteration on y as composition changes."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        psat = np.array([f(T) for f in self.psat])
        gammas = np.asarray(self.model.gammas(T, x))
        # Vapor fugacity coefficients
        rho_v = self.eos.density_from_pressure(p, T, y, phase_hint='vapor')
        ln_phi_v = np.asarray(self.eos.ln_phi(rho_v, T, y))
        phi_v = np.exp(ln_phi_v)
        # Pure component saturation fugacity coefficients
        if self._phi_sat is not None:
            phi_sat = np.array([f(T) for f in self._phi_sat])
        else:
            phi_sat = np.ones(self.N)
        # Poynting correction
        if self._VL is not None:
            poynting = np.exp(self._VL * (p - psat) / (_R_GAS * T))
        else:
            poynting = np.ones(self.N)
        return gammas * psat * phi_sat * poynting / (p * phi_v)

    # -----------------------------------------------------------------
    # Bubble pressure (high-p)
    # -----------------------------------------------------------------

    def bubble_p(self, T: float, x: Sequence[float],
                 p_guess: float = None,
                 tol: float = 1e-7, maxiter: int = 100) -> BubbleResult:
        """Bubble pressure with EOS vapor fugacity.

        Outer SS iteration: at fixed (T, x), iterate on (p, y) until
        K-values are self-consistent and sum y = 1. Initial guess
        from Raoult's law.
        """
        x = np.asarray(x, dtype=float)
        psat = np.array([f(T) for f in self.psat])
        gammas = np.asarray(self.model.gammas(T, x))
        # Raoult initial guess
        if p_guess is None:
            p = float(np.sum(x * gammas * psat))
        else:
            p = float(p_guess)
        y = (x * gammas * psat / p)
        y = y / y.sum()
        for it in range(maxiter):
            K = self._K_values(T, p, x, y)
            y_new = K * x
            sum_y = float(y_new.sum())
            # Pressure correction: at true bubble, sum_y should be 1.
            # Rescale p to drive sum_y toward 1.
            p_new = p * sum_y
            y_new = y_new / sum_y
            err_p = abs(p_new - p) / max(p, 1.0)
            err_y = float(np.max(np.abs(y_new - y)))
            p = p_new
            y = y_new
            if err_p < tol and err_y < tol:
                K_final = self._K_values(T, p, x, y)
                return BubbleResult(T=T, p=p, x=x, y=y, K=K_final,
                                     iterations=it + 1)
        raise RuntimeError(f"bubble_p (EOS) did not converge in {maxiter} iter "
                            f"(p={p:.2e} Pa, sum_y={sum_y:.4f})")

    # -----------------------------------------------------------------
    # Bubble temperature
    # -----------------------------------------------------------------

    def bubble_t(self, p: float, x: Sequence[float],
                 T_guess: float = None,
                 tol: float = 1e-7, maxiter: int = 50) -> BubbleResult:
        """Bubble T at fixed p, x. Outer Newton on T using bubble_p
        as the inner solver."""
        x = np.asarray(x, dtype=float)
        if T_guess is None:
            # Crude Raoult guess: invert sum x_i p_i^sat = p (no EOS)
            T = self._raoult_bubble_t_guess(p, x)
        else:
            T = float(T_guess)
        for it in range(maxiter):
            try:
                r = self.bubble_p(T, x, tol=1e-8, maxiter=200)
            except RuntimeError:
                T = T * 0.99
                continue
            f = r.p - p
            if abs(f / p) < tol:
                return BubbleResult(T=T, p=p, x=x, y=r.y, K=r.K,
                                     iterations=it + 1)
            h = max(0.01, T * 1e-4)
            try:
                r_h = self.bubble_p(T + h, x, tol=1e-8, maxiter=200)
            except RuntimeError:
                r_h = self.bubble_p(T - h, x, tol=1e-8, maxiter=200)
                df_dT = -(r_h.p - r.p) / h
            else:
                df_dT = (r_h.p - r.p) / h
            if abs(df_dT) < 1e-30:
                raise RuntimeError("bubble_t (EOS): zero Jacobian")
            T = T - f / df_dT
            T = max(50.0, min(2000.0, T))
        raise RuntimeError(f"bubble_t (EOS) did not converge in {maxiter} iter")

    # -----------------------------------------------------------------
    # Dew point (high-p)
    # -----------------------------------------------------------------

    def dew_p(self, T: float, y: Sequence[float],
              p_guess: float = None,
              tol: float = 1e-7, maxiter: int = 200) -> DewResult:
        """Dew pressure at fixed T, y. SS iteration on x (and p)
        because gamma_i depends on liquid composition and phi_i^V
        depends on vapor composition (here fixed)."""
        y = np.asarray(y, dtype=float)
        psat = np.array([f(T) for f in self.psat])
        # Initial guess: gamma=1 (Raoult), 1/p = sum y/psat
        if p_guess is None:
            p = 1.0 / float(np.sum(y / psat))
        else:
            p = float(p_guess)
        x = y * p / psat
        x = x / x.sum()
        for it in range(maxiter):
            K = self._K_values(T, p, x, y)
            x_new = y / K
            sum_x = float(x_new.sum())
            # p correction: at true dew, sum_x should be 1
            p_new = p / sum_x
            x_new = x_new / sum_x
            err_p = abs(p_new - p) / max(p, 1.0)
            err_x = float(np.max(np.abs(x_new - x)))
            p = p_new
            x = x_new
            if err_p < tol and err_x < tol:
                K_final = self._K_values(T, p, x, y)
                return DewResult(T=T, p=p, x=x, y=y, K=K_final,
                                  iterations=it + 1)
        raise RuntimeError(f"dew_p (EOS) did not converge in {maxiter} iter "
                            f"(p={p:.2e}, sum_x={sum_x:.4f})")

    def dew_t(self, p: float, y: Sequence[float],
              T_guess: float = None,
              tol: float = 1e-7, maxiter: int = 50) -> DewResult:
        """Dew T at fixed p, y."""
        y = np.asarray(y, dtype=float)
        if T_guess is None:
            T = self._raoult_dew_t_guess(p, y)
        else:
            T = float(T_guess)
        for it in range(maxiter):
            try:
                r = self.dew_p(T, y, tol=1e-8, maxiter=300)
            except RuntimeError:
                T = T * 1.01
                continue
            f = r.p - p
            if abs(f / p) < tol:
                return DewResult(T=T, p=p, x=r.x, y=y, K=r.K,
                                  iterations=it + 1)
            h = max(0.01, T * 1e-4)
            try:
                r_h = self.dew_p(T + h, y, tol=1e-8, maxiter=300)
                df_dT = (r_h.p - r.p) / h
            except RuntimeError:
                r_h = self.dew_p(T - h, y, tol=1e-8, maxiter=300)
                df_dT = -(r_h.p - r.p) / h
            if abs(df_dT) < 1e-30:
                raise RuntimeError("dew_t (EOS): zero Jacobian")
            T = T - f / df_dT
            T = max(50.0, min(2000.0, T))
        raise RuntimeError(f"dew_t (EOS) did not converge in {maxiter} iter")

    # -----------------------------------------------------------------
    # Isothermal flash
    # -----------------------------------------------------------------

    def isothermal(self, T: float, p: float, z: Sequence[float],
                   K_guess: Optional[Sequence[float]] = None,
                   tol: float = 1e-6, maxiter: int = 200) -> FlashResult:
        """PT-flash at given T, p, feed z. Returns vapor fraction V
        and equilibrium x, y with EOS vapor fugacity coupling.

        Parameters
        ----------
        K_guess : sequence of float, optional
            Initial K-values for SS warm-start. Useful in batch grid
            generation where neighboring (T, p, z) points are similar.
            Often gives 30-50% iteration reduction for smooth grids.
            If None, uses Raoult's law initial guess.
        """
        z = np.asarray(z, dtype=float)
        psat = np.array([f(T) for f in self.psat])
        if K_guess is not None:
            K = np.asarray(K_guess, dtype=float)
            if K.size != self.N:
                raise ValueError(f"K_guess must have length {self.N}")
        else:
            # Initial K from Raoult (gamma=1, phi_v=1)
            K = psat / p
        x = z.copy()
        y = z.copy()
        for it in range(maxiter):
            V = _solve_rachford_rice(z, K)
            x_new = z / (1.0 + V * (K - 1.0))
            x_new = x_new / x_new.sum()
            y_new = K * x_new
            y_new = y_new / y_new.sum()
            # Update K with new compositions
            K_new = self._K_values(T, p, x_new, y_new)
            err = float(np.max(np.abs(K_new - K) / np.maximum(K, 1e-12)))
            x, y, K = x_new, y_new, K_new
            if err < tol:
                return FlashResult(T=T, p=p, z=z, V=V, x=x, y=y, K=K,
                                    iterations=it + 1)
        raise RuntimeError(f"isothermal (EOS) did not converge in {maxiter} iter "
                            f"(max K err {err:.2e})")

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _raoult_bubble_t_guess(self, p: float, x: np.ndarray) -> float:
        """Crude initial T guess for bubble: weighted pure-component Tsat."""
        T_b = np.array([self._pure_Tsat(i, p) for i in range(self.N)])
        return float(np.sum(x * T_b))

    def _raoult_dew_t_guess(self, p: float, y: np.ndarray) -> float:
        T_b = np.array([self._pure_Tsat(i, p) for i in range(self.N)])
        return float(np.sum(y * T_b))

    def _pure_Tsat(self, i: int, p: float, T_low: float = 100.0,
                    T_high: float = 1500.0) -> float:
        """Pure-component saturation T at p via bisection on psat(T) = p."""
        psat_fn = self.psat[i]
        def f(T):
            return psat_fn(T) - p
        f_low = f(T_low); f_high = f(T_high)
        for _ in range(10):
            if f_low * f_high < 0:
                break
            if f_low > 0:
                T_low = max(50.0, T_low * 0.5)
                f_low = f(T_low)
            else:
                T_high = T_high * 1.5
                f_high = f(T_high)
        if f_low * f_high > 0:
            raise RuntimeError(f"pure_Tsat: no bracket for component {i}")
        for _ in range(200):
            T_mid = 0.5 * (T_low + T_high)
            f_mid = f(T_mid)
            if abs(f_mid / p) < 1e-10:
                return T_mid
            if f_low * f_mid < 0:
                T_high = T_mid; f_high = f_mid
            else:
                T_low = T_mid; f_low = f_mid
        return 0.5 * (T_low + T_high)


def make_phi_sat_funcs(
    mixture,
    psat_funcs: Sequence[Callable[[float], float]],
) -> list:
    """Build per-component saturation fugacity coefficient functions
    ``Phi_sat,i(T) = phi_V^pure(T, p_sat,i(T))`` from a mixture EOS.

    For each component i, the function evaluates the EOS at the pure-i
    composition (a unit vector) at temperature T and pressure
    p_sat,i(T), takes the vapor root, and returns the i-th component
    of ``exp(ln_phi)``.  This is the textbook saturation-fugacity
    coefficient that enters the gamma-phi K-value formula:

        K_i = gamma_i * p_sat,i * Phi_sat,i * Poynting_i / (p * phi_V,i)

    Parameters
    ----------
    mixture : EOS mixture (CubicMixture or SAFTMixture)
        Must implement ``density_from_pressure(p, T, x, phase_hint)``
        and ``ln_phi(rho, T, x)``.
    psat_funcs : sequence of callables T -> p_sat,i(T) [Pa]
        Pure-component saturation pressures; one per mixture component
        in the same order as the EOS.

    Returns
    -------
    list of callables T -> Phi_sat,i(T)

    Notes
    -----
    For an ideal gas (or a vapor at low p where phi_V -> 1), each
    Phi_sat -> 1 and these helpers add no information.  Their
    contribution becomes meaningful above ~10 bar; near or above the
    critical pressure of any pure component, p_sat is extrapolated
    and Phi_sat is unreliable.
    """
    N = len(psat_funcs)
    if hasattr(mixture, "N") and mixture.N != N:
        raise ValueError(
            f"mixture has N={mixture.N} but psat_funcs has length {N}")

    def _make(i: int):
        x_pure = np.zeros(N)
        x_pure[i] = 1.0
        psat_i = psat_funcs[i]
        def phi_sat_i(T: float) -> float:
            p = float(psat_i(T))
            if p <= 0:
                return 1.0
            try:
                rho = mixture.density_from_pressure(p, T, x_pure,
                                                     phase_hint="vapor")
                ln_phi = mixture.ln_phi(rho, T, x_pure)
                return float(np.exp(ln_phi[i]))
            except Exception:
                # Fall back to ideal-gas if EOS fails (e.g. supercritical)
                return 1.0
        phi_sat_i.__name__ = f"phi_sat_{i}"
        return phi_sat_i

    return [_make(i) for i in range(N)]
