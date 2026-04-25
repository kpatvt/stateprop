"""Michelsen stability analysis (tangent plane distance test).

Determines whether a candidate single-phase composition z is stable
or wants to split into multiple phases at given (T, p). The test is
the foundation for automatic phase-count detection in two- and three-
phase flash, removing the need for users to provide initial guesses.

Michelsen, M.L. "The isothermal flash problem. Part I. Stability",
Fluid Phase Equilib. 9, 1 (1982).

Algorithm (liquid-phase, activity-model formulation)
====================================================

For a single liquid-phase candidate z, the tangent plane distance
function for a trial composition Y is:

    TPD(Y) = sum_i Y_i (ln Y_i + ln gamma_i(Y) - ln z_i - ln gamma_i(z))

If TPD(Y) >= 0 for all valid Y on the simplex, z is **stable** (a
single phase is the equilibrium state). If TPD(Y) < 0 for some Y,
z is **unstable** and will split.

Michelsen's iteration uses unnormalized variables Y* with constraint
sum Y* free, and the fixed-point map:

    ln Y_i*_(k+1) = h_i - ln gamma_i(Y*_k / sum Y*_k)
    where h_i = ln z_i + ln gamma_i(z)

At a stationary point Y_inf, the (normalized) TPD reduces to
TPD = -ln(sum Y*_inf). So:

- sum Y*_inf > 1  ->  TPD < 0  ->  z is unstable, Y_inf is a candidate
                                    second-phase composition
- sum Y*_inf <= 1 ->  TPD >= 0 ->  this stationary point doesn't
                                    indicate instability

The algorithm runs the iteration from multiple initial trial guesses
to find ALL stationary points (a single composition can have several;
the test must check the most negative TPD).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import numpy as np


@dataclass
class StabilityResult:
    """Result of a Michelsen stability test."""
    stable: bool
    tpd_min: float          # most negative TPD found (>= 0 if stable)
    Y_min: np.ndarray       # composition at the minimum
    n_stationary: int       # number of stationary points found
    iterations_total: int   # total iterations across all trials
    trials_evaluated: int


def stability_test(activity_model, T: float, z: Sequence[float],
                   trial_initials: Optional[List[Sequence[float]]] = None,
                   tol: float = 1e-10,
                   tpd_tol: float = 1e-7,
                   maxiter: int = 200,
                   include_default_trials: bool = True
                   ) -> StabilityResult:
    """Michelsen stability test for a liquid-phase candidate.

    Parameters
    ----------
    activity_model : object with .gammas(T, x) and .N
        Activity coefficient model (NRTL, UNIQUAC, UNIFAC, ...).
    T : float
        Temperature [K].
    z : sequence
        Candidate single-phase composition (length N).
    trial_initials : list of sequences, optional
        Initial guesses for the trial composition Y. If None and
        `include_default_trials` is True, automatic guesses are
        generated (each pure-i-rich corner plus equimolar perturbed).
    tol : float
        Convergence tolerance for the SS iteration on Y.
    tpd_tol : float
        TPD value below which the candidate is declared unstable.
        Default 1e-7 (any negative TPD indicates instability, but
        tiny negative values can be numerical noise).
    maxiter : int
        Max SS iterations per trial.
    include_default_trials : bool
        If True (default), add pure-i-rich and perturbed trials to
        any user-supplied list.

    Returns
    -------
    StabilityResult
        - stable: True if no trial yielded TPD < -tpd_tol.
        - tpd_min: most negative TPD over all stationary points found.
        - Y_min: composition at the minimum (useful as initial guess
                 for a second phase if z is unstable).
    """
    z = np.asarray(z, dtype=float)
    z = z / z.sum()
    N = z.size

    if N < 2:
        raise ValueError("Stability test requires at least 2 components")

    # Reference quantities
    gammas_z = np.asarray(activity_model.gammas(T, z), dtype=float)
    ln_gz = np.log(gammas_z)
    # h_i = ln z_i + ln gamma_i(z)
    # Floor z to avoid -inf for components present in trace amounts
    h = np.log(np.maximum(z, 1e-300)) + ln_gz

    # Build trial list
    trials: List[np.ndarray] = []
    if trial_initials is not None:
        for Y0 in trial_initials:
            Y0_arr = np.asarray(Y0, dtype=float)
            Y0_arr = Y0_arr / Y0_arr.sum()
            trials.append(Y0_arr)
    if include_default_trials:
        eps = 1.0 / max(100.0, 10.0 * N)   # small perturbation
        # Pure-i-rich trials: Y close to pure-component i
        for i in range(N):
            Y = np.full(N, eps / max(N - 1, 1))
            Y[i] = 1.0 - eps
            Y = Y / Y.sum()
            trials.append(Y)
        # Equimolar perturbed (away from z)
        # The straight equimolar would converge to trivial if z is symmetric.
        # Use a perturbation that's distinct from z.
        if N == 2:
            for delta in [0.3, -0.3]:
                Y = np.array([0.5 + delta, 0.5 - delta])
                if abs(Y[0] - z[0]) > 0.05:
                    trials.append(Y)
        else:
            # For N >= 3, use random perturbations of equimolar
            rng = np.random.default_rng(42)
            for _ in range(N):
                Y = np.ones(N) / N + 0.2 * rng.standard_normal(N)
                Y = np.maximum(Y, 0.01)
                Y = Y / Y.sum()
                if np.max(np.abs(Y - z)) > 0.05:
                    trials.append(Y)

    tpd_min = float('inf')
    Y_min = z.copy()
    n_stationary = 0
    iter_total = 0
    trials_evaluated = 0

    for Y0 in trials:
        # Skip if too close to z (would converge to trivial)
        if float(np.max(np.abs(Y0 - z))) < 1e-3:
            continue
        trials_evaluated += 1
        # Initialize unnormalized Y* with sum = 1 (Y* = Y_norm initially)
        Y_unn = Y0.copy()
        Y_unn = Y_unn / Y_unn.sum()

        converged = False
        for it in range(maxiter):
            iter_total += 1
            s = float(Y_unn.sum())
            if s <= 0:
                break
            Y_norm = Y_unn / s
            # Bail out if iterate collapsed onto z (trivial stationary point)
            if float(np.max(np.abs(Y_norm - z))) < 1e-7:
                break
            # gamma at normalized Y
            try:
                gammas_Y = np.asarray(activity_model.gammas(T, Y_norm),
                                        dtype=float)
            except Exception:
                break
            ln_gY = np.log(np.maximum(gammas_Y, 1e-300))
            ln_Y_new = h - ln_gY
            # Clip extreme values to avoid overflow
            ln_Y_new = np.clip(ln_Y_new, -700.0, 700.0)
            Y_new = np.exp(ln_Y_new)
            err = float(np.max(np.abs(Y_new - Y_unn)
                                / np.maximum(np.abs(Y_unn), 1e-15)))
            Y_unn = Y_new
            if err < tol:
                converged = True
                break

        if not converged:
            continue

        s = float(Y_unn.sum())
        Y_norm = Y_unn / s
        # Reject trivial stationary point (Y_norm ~= z)
        if float(np.max(np.abs(Y_norm - z))) < 1e-4:
            continue

        # TPD at this stationary point
        tpd = -np.log(s)
        n_stationary += 1
        if tpd < tpd_min:
            tpd_min = tpd
            Y_min = Y_norm.copy()

    # If no non-trivial stationary points were found, declare stable
    if n_stationary == 0:
        return StabilityResult(stable=True, tpd_min=0.0, Y_min=z.copy(),
                                n_stationary=0,
                                iterations_total=iter_total,
                                trials_evaluated=trials_evaluated)

    stable = bool(tpd_min > -tpd_tol)
    return StabilityResult(stable=stable, tpd_min=float(tpd_min),
                            Y_min=Y_min,
                            n_stationary=n_stationary,
                            iterations_total=iter_total,
                            trials_evaluated=trials_evaluated)
