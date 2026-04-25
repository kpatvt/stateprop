"""Gamma-phi flash (modified Raoult's law) for non-ideal liquid mixtures.

Wires the v0.9.39 activity-coefficient models (NRTL, UNIQUAC, UNIFAC)
into VLE calculations:

    K_i = gamma_i(T, x) * p_i^sat(T) / p     (modified Raoult)

For low to moderate pressures with mostly-ideal vapor this is the
classical chemical-engineering approach for highly non-ideal
mixtures (azeotropes, distillation columns, LLE). At higher
pressures the user should switch to a phi-phi formulation with a
cubic / SAFT EOS for both phases (existing facilities in
`stateprop.cubic.flash`, `stateprop.saft`, etc.).

Three calculation types provided:

    flash.bubble_p(T, x)       -> (p, y, K)         bubble pressure
    flash.bubble_t(p, x)       -> (T, y, K)         bubble temperature
    flash.dew_p(T, y)          -> (p, x, K)         dew pressure
    flash.dew_t(p, y)          -> (T, x, K)         dew temperature
    flash.isothermal(T, p, z)  -> (V, x, y, K)      P-T flash

Pure-component vapor pressures are supplied via callables psat(T) -> Pa
or via the convenience class `AntoinePsat`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import warnings
import numpy as np


# ---------------------------------------------------------------------------
# Vapor-pressure helper
# ---------------------------------------------------------------------------


class AntoinePsat:
    """NIST WebBook Antoine form:
        log10(p_sat[bar]) = A - B / (T[K] + C)

    Returns p_sat in Pa when called.

    Parameters
    ----------
    A, B, C : float
        Antoine coefficients in the NIST convention (T in Kelvin,
        p in bar). For other unit conventions, convert before passing.
    T_min, T_max : float or None
        Optional validity range. Warning issued if T outside.

    Examples
    --------
    >>> # Ethanol from NIST WebBook (273-352 K)
    >>> ethanol_psat = AntoinePsat(A=5.37229, B=1670.409, C=-40.191)
    >>> ethanol_psat(298.15)   # ~7864 Pa
    """

    def __init__(self, A: float, B: float, C: float,
                 T_min: float = None, T_max: float = None):
        self.A = float(A)
        self.B = float(B)
        self.C = float(C)
        self.T_min = T_min
        self.T_max = T_max

    def __call__(self, T: float) -> float:
        T = float(T)
        if self.T_min is not None and T < self.T_min - 5.0:
            warnings.warn(f"T={T:.2f} below Antoine range [{self.T_min:.2f}, "
                          f"{self.T_max:.2f}]")
        if self.T_max is not None and T > self.T_max + 5.0:
            warnings.warn(f"T={T:.2f} above Antoine range [{self.T_min:.2f}, "
                          f"{self.T_max:.2f}]")
        # 1 bar = 100000 Pa
        return 1.0e5 * 10.0 ** (self.A - self.B / (T + self.C))


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class BubbleResult:
    T: float
    p: float
    x: np.ndarray
    y: np.ndarray
    K: np.ndarray
    iterations: int = 0


@dataclass
class DewResult:
    T: float
    p: float
    x: np.ndarray
    y: np.ndarray
    K: np.ndarray
    iterations: int = 0


@dataclass
class FlashResult:
    T: float
    p: float
    z: np.ndarray
    V: float            # vapor mole fraction
    x: np.ndarray
    y: np.ndarray
    K: np.ndarray
    iterations: int = 0


# ---------------------------------------------------------------------------
# Gamma-phi flash
# ---------------------------------------------------------------------------


class GammaPhiFlash:
    """Modified Raoult's law VLE flash with activity coefficient model
    for the liquid phase and ideal-gas vapor.

    Parameters
    ----------
    activity_model : NRTL, UNIQUAC, or UNIFAC instance
        Must implement gammas(T, x) -> ndarray.
    psat_funcs : sequence of callables
        Length-N sequence of callables psat(T) -> Pa for each component.
        Use AntoinePsat for tabulated NIST coefficients, or a custom
        callable for DIPPR / Wagner / etc.

    Notes
    -----
    Vapor phase is treated as ideal gas (phi_i^V = 1, no Poynting
    correction, no fugacity correction at saturation). For pressures
    above ~5 bar or temperatures near critical, use phi-phi flash
    with a cubic/SAFT EOS instead.
    """

    def __init__(self, activity_model, psat_funcs: Sequence[Callable[[float], float]]):
        self.model = activity_model
        self.psat = list(psat_funcs)
        self.N = len(self.psat)
        if hasattr(activity_model, 'N') and activity_model.N != self.N:
            raise ValueError(f"activity_model has N={activity_model.N} "
                             f"but {self.N} psat_funcs provided")

    def _psat_array(self, T: float) -> np.ndarray:
        return np.array([f(T) for f in self.psat])

    def _gammas(self, T: float, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.gammas(T, x))

    # -----------------------------------------------------------------
    # Bubble point
    # -----------------------------------------------------------------

    def bubble_p(self, T: float, x: Sequence[float]) -> BubbleResult:
        """Bubble pressure at fixed T, liquid composition x."""
        x = np.asarray(x, dtype=float)
        psat = self._psat_array(T)
        gammas = self._gammas(T, x)
        # Modified Raoult: y_i p = x_i gamma_i p_i^sat; sum y_i = 1 -> p
        p = float(np.sum(x * gammas * psat))
        K = gammas * psat / p
        y = K * x
        # Renormalize to handle round-off
        y = y / y.sum()
        return BubbleResult(T=T, p=p, x=x, y=y, K=K, iterations=1)

    def bubble_t(self, p: float, x: Sequence[float],
                 T_guess: float = None,
                 tol: float = 1e-8, maxiter: int = 50) -> BubbleResult:
        """Bubble temperature at fixed p, liquid composition x.
        Newton iteration on T using current bubble_p estimate."""
        x = np.asarray(x, dtype=float)
        if T_guess is None:
            # Crude guess: weighted average of pure-component boiling points
            # at p, found by solving p_i^sat(T_b) = p for each i (bisection)
            T_b = np.array([self._pure_Tsat(i, p) for i in range(self.N)])
            T = float(np.sum(x * T_b))
        else:
            T = float(T_guess)
        for it in range(maxiter):
            psat = self._psat_array(T)
            gammas = self._gammas(T, x)
            p_calc = float(np.sum(x * gammas * psat))
            f = p_calc - p
            if abs(f / p) < tol:
                K = gammas * psat / p_calc
                y = (K * x); y = y / y.sum()
                return BubbleResult(T=T, p=p, x=x, y=y, K=K,
                                     iterations=it + 1)
            # Numerical dT
            h = max(0.001, T * 1e-4)
            psat_h = self._psat_array(T + h)
            gammas_h = self._gammas(T + h, x)
            f_h = float(np.sum(x * gammas_h * psat_h)) - p
            df_dT = (f_h - f) / h
            if abs(df_dT) < 1e-30:
                raise RuntimeError("bubble_t: zero Jacobian")
            T_new = T - f / df_dT
            # Damped Newton
            T = max(50.0, min(2000.0, T_new))
        raise RuntimeError(f"bubble_t did not converge in {maxiter} iter")

    # -----------------------------------------------------------------
    # Dew point
    # -----------------------------------------------------------------

    def dew_p(self, T: float, y: Sequence[float],
              tol: float = 1e-8, maxiter: int = 100) -> DewResult:
        """Dew pressure at fixed T, vapor composition y. Iterates over
        liquid composition x to handle gamma(x) coupling."""
        y = np.asarray(y, dtype=float)
        psat = self._psat_array(T)
        # Initial guess: assume gamma=1 (Raoult), x_i = y_i p / p_i^sat normalized
        # 1/p = sum y_i / p_i^sat
        p = 1.0 / float(np.sum(y / psat))
        x = y * p / psat
        x = x / x.sum()
        for it in range(maxiter):
            gammas = self._gammas(T, x)
            # 1/p = sum y_i / (gamma_i p_i^sat)
            p_new = 1.0 / float(np.sum(y / (gammas * psat)))
            x_new = y * p_new / (gammas * psat)
            x_new = x_new / x_new.sum()
            err = float(np.max(np.abs(x_new - x)))
            x = x_new
            p = p_new
            if err < tol:
                K = gammas * psat / p
                return DewResult(T=T, p=p, x=x, y=y, K=K, iterations=it + 1)
        raise RuntimeError(f"dew_p did not converge in {maxiter} iter")

    def dew_t(self, p: float, y: Sequence[float],
              T_guess: float = None,
              tol: float = 1e-8, maxiter: int = 50) -> DewResult:
        """Dew temperature at fixed p, vapor composition y."""
        y = np.asarray(y, dtype=float)
        if T_guess is None:
            T_b = np.array([self._pure_Tsat(i, p) for i in range(self.N)])
            T = float(np.sum(y * T_b))
        else:
            T = float(T_guess)
        # Outer Newton on T to drive sum x_i = 1 (already enforced inside dew_p);
        # equivalent: drive 1/p - sum y/(gamma * psat) = 0.
        for it in range(maxiter):
            try:
                r = self.dew_p(T, y, tol=tol, maxiter=200)
            except RuntimeError:
                # Fall back: shrink step
                T = T * 0.99
                continue
            p_calc = r.p
            f = p_calc - p
            if abs(f / p) < tol:
                return DewResult(T=T, p=p, x=r.x, y=y, K=r.K, iterations=it + 1)
            h = max(0.001, T * 1e-4)
            r_h = self.dew_p(T + h, y, tol=tol, maxiter=200)
            df_dT = (r_h.p - p_calc) / h
            if abs(df_dT) < 1e-30:
                raise RuntimeError("dew_t: zero Jacobian")
            T_new = T - f / df_dT
            T = max(50.0, min(2000.0, T_new))
        raise RuntimeError(f"dew_t did not converge in {maxiter} iter")

    # -----------------------------------------------------------------
    # Isothermal flash (PT flash)
    # -----------------------------------------------------------------

    def isothermal(self, T: float, p: float, z: Sequence[float],
                   K_guess: Optional[Sequence[float]] = None,
                   tol: float = 1e-8, maxiter: int = 100) -> FlashResult:
        """PT-flash at given T, p, feed z. Returns vapor fraction V and
        equilibrium compositions x, y.

        Outer iteration: SS update of x for the activity coefficients;
        inner: Rachford-Rice for V given current K.

        Parameters
        ----------
        K_guess : sequence of float, optional
            Initial K-values for SS warm-start. Useful in batch grid
            generation where neighboring (T, p, z) points are similar.
            If None, uses Raoult's law initial guess (gamma = 1).
        """
        z = np.asarray(z, dtype=float)
        psat = self._psat_array(T)
        if K_guess is not None:
            K = np.asarray(K_guess, dtype=float)
            if K.size != self.N:
                raise ValueError(f"K_guess must have length {self.N}")
        else:
            # Initial K: Raoult (gamma = 1)
            gammas = np.ones(self.N)
            K = gammas * psat / p
        x = z.copy()
        for it in range(maxiter):
            # Rachford-Rice for V
            V = _solve_rachford_rice(z, K)
            x_new = z / (1.0 + V * (K - 1.0))
            x_new = x_new / x_new.sum()
            y_new = K * x_new
            y_new = y_new / y_new.sum()
            # Update gammas with new x
            gammas = self._gammas(T, x_new)
            K_new = gammas * psat / p
            err = float(np.max(np.abs(K_new - K) / np.maximum(K, 1e-12)))
            x = x_new
            K = K_new
            if err < tol:
                return FlashResult(T=T, p=p, z=z, V=V, x=x, y=y_new, K=K,
                                    iterations=it + 1)
        raise RuntimeError(f"isothermal flash did not converge in {maxiter} iter")

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _pure_Tsat(self, i: int, p: float, T_low: float = 100.0,
                    T_high: float = 1500.0) -> float:
        """Pure-component saturation temperature at p via bisection on
        p_i^sat(T) = p."""
        psat_fn = self.psat[i]
        def f(T):
            return psat_fn(T) - p
        f_low = f(T_low); f_high = f(T_high)
        # Expand if needed
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
            raise RuntimeError(f"pure_Tsat: no bracket found for component {i} at p={p}")
        # Bisection
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


# ---------------------------------------------------------------------------
# Rachford-Rice solver
# ---------------------------------------------------------------------------


def _solve_rachford_rice(z: np.ndarray, K: np.ndarray,
                          tol: float = 1e-12, maxiter: int = 100) -> float:
    """Solve sum_i z_i (K_i - 1) / (1 + V (K_i - 1)) = 0 for V in [0, 1].

    Uses bisection-bounded Newton (always converges)."""
    if np.all(K >= 1.0):
        return 1.0   # All K > 1 -> all vapor
    if np.all(K <= 1.0):
        return 0.0   # All K < 1 -> all liquid
    # Brackets: V_min and V_max where the function is bracketed
    V_min = 1.0 / (1.0 - K.max()) + 1e-9
    V_max = 1.0 / (1.0 - K.min()) - 1e-9
    V_lo = max(0.0, V_min)
    V_hi = min(1.0, V_max)
    def f(V):
        return float(np.sum(z * (K - 1.0) / (1.0 + V * (K - 1.0))))
    f_lo = f(V_lo); f_hi = f(V_hi)
    if f_lo <= 0:
        return V_lo
    if f_hi >= 0:
        return V_hi
    # Bisection
    for _ in range(maxiter):
        V_mid = 0.5 * (V_lo + V_hi)
        f_mid = f(V_mid)
        if abs(f_mid) < tol or (V_hi - V_lo) < tol:
            return V_mid
        if f_lo * f_mid < 0:
            V_hi = V_mid; f_hi = f_mid
        else:
            V_lo = V_mid; f_lo = f_mid
    return 0.5 * (V_lo + V_hi)
