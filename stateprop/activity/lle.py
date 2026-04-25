"""Two-phase liquid-liquid equilibrium flash.

Solves binary or multicomponent LLE by SS iteration on the equal-fugacity
condition for two liquid phases:

    x1_i gamma_i^L1(T, x1) = x2_i gamma_i^L2(T, x2)

i.e., K_i = x2_i / x1_i = gamma_i^L1 / gamma_i^L2.

This is a prerequisite for LLE-fitted parameter regression
(stateprop.activity.regression). The user supplies initial liquid
compositions x1_guess, x2_guess that are sufficiently different to seed
the LL split; without good guesses the iteration converges to the trivial
solution x1 = x2 = z (single liquid phase).

Reference: Sandler, "Chemical, Biochemical, and Engineering
Thermodynamics", 4th ed. Ch. 11. Or Reid-Prausnitz-Poling Ch. 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

from .gamma_phi import _solve_rachford_rice


@dataclass
class LLEResult:
    """Result of a 2-phase LLE flash."""
    T: float
    z: np.ndarray
    beta: float           # mole fraction of feed in phase 2
    x1: np.ndarray        # phase-1 (reference) composition
    x2: np.ndarray        # phase-2 composition
    K: np.ndarray         # K_i = x2_i / x1_i = gamma_L1/gamma_L2
    iterations: int = 0
    converged: bool = True


class LLEFlash:
    """Two-phase liquid-liquid flash for a fixed activity model.

    Parameters
    ----------
    activity_model : object with .gammas(T, x) and .N attribute
        Any model from stateprop.activity (NRTL, UNIQUAC, UNIFAC, ...).
    """

    def __init__(self, activity_model):
        self.model = activity_model
        self.N = activity_model.N

    def solve(self, T: float, z: Sequence[float],
              x1_guess: Sequence[float], x2_guess: Sequence[float],
              tol: float = 1e-8, maxiter: int = 200,
              collapse_tol: float = 1e-5) -> LLEResult:
        """Solve LLE at T for feed z.

        Parameters
        ----------
        T : float
            Temperature [K].
        z : sequence
            Feed composition (length N).
        x1_guess, x2_guess : sequence
            Initial guesses for the two liquid compositions. Must be
            sufficiently different (max |x1 - x2| > collapse_tol) to
            seed the LL split.
        tol : float
            Convergence tolerance on max |Δx| between iterations.
        maxiter : int
            Outer SS iteration limit.
        collapse_tol : float
            If max |x1 - x2| drops below this, declare collapse to
            single phase and raise.

        Returns
        -------
        LLEResult

        Raises
        ------
        RuntimeError if SS fails to converge or phases collapse.
        """
        z = np.asarray(z, dtype=float)
        z = z / z.sum()
        x1 = np.asarray(x1_guess, dtype=float).copy()
        x1 = x1 / x1.sum()
        x2 = np.asarray(x2_guess, dtype=float).copy()
        x2 = x2 / x2.sum()

        if float(np.max(np.abs(x1 - x2))) < collapse_tol:
            raise ValueError("x1_guess and x2_guess too similar; "
                             "LLE iteration would collapse to z")

        for it in range(maxiter):
            gamma_1 = np.asarray(self.model.gammas(T, x1))
            gamma_2 = np.asarray(self.model.gammas(T, x2))
            K = gamma_1 / gamma_2

            # 2-phase Rachford-Rice for beta = fraction in phase 2
            try:
                beta = _solve_rachford_rice(z, K)
            except (ValueError, RuntimeError):
                raise RuntimeError(
                    f"LLE: K-values do not admit 2-phase split "
                    f"at iter {it} (K = {K})"
                )

            # Compositions from RR solution
            denom = 1.0 + beta * (K - 1.0)
            x1_new = z / denom
            x1_new = x1_new / x1_new.sum()
            x2_new = K * x1_new
            x2_new = x2_new / x2_new.sum()

            # Detect collapse
            if float(np.max(np.abs(x1_new - x2_new))) < collapse_tol:
                raise RuntimeError(
                    f"LLE collapsed to single phase at iter {it}: "
                    f"max |x1 - x2| = "
                    f"{float(np.max(np.abs(x1_new - x2_new))):.2e}"
                )

            err = max(float(np.max(np.abs(x1_new - x1))),
                      float(np.max(np.abs(x2_new - x2))))
            x1, x2 = x1_new, x2_new
            if err < tol:
                return LLEResult(T=T, z=z, beta=beta, x1=x1, x2=x2,
                                  K=K, iterations=it + 1, converged=True)

        raise RuntimeError(f"LLE flash did not converge in {maxiter} iter")
