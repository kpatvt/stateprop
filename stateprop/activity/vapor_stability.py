"""Vapor-phase stability test using EOS fugacity coefficients.

Complement to v0.9.48's liquid-phase Michelsen TPD test
(stateprop.activity.stability). Where the v0.9.48 test treats z as
a candidate single liquid and uses gamma values from an activity
coefficient model, this module treats z as a candidate single vapor
and uses phi values from an equation of state (cubic, GERG, SAFT).

Algorithm
=========

For trial composition Y near the candidate vapor z, define:

    TPD(Y) = sum_i Y_i (ln Y_i + ln phi_i(Y) - ln z_i - ln phi_i(z))

If TPD >= 0 for all Y on the simplex, z is stable as a single
vapor. If TPD(Y) < 0 for some Y, z is unstable and Y is a candidate
second-vapor-phase composition.

The fixed-point iteration (Michelsen 1982) on unnormalized Y* with
sum free is:

    ln Y*_i_(k+1) = h_i - ln phi_i(Y*_k / sum Y*_k)
    where h_i = ln z_i + ln phi_i(z)

At a stationary point: TPD = -ln(sum Y_inf*). Sum > 1 means
TPD < 0 -- vapor instability.

Use cases
=========

Vapor-vapor splitting is rare in typical chemical engineering
mixtures but can occur for:

- Highly polar + nonpolar systems at supercritical conditions
- Some quantum gas mixtures at very low T
- Polymer + small-molecule systems

For typical 1-30 bar VLE/VLLE workflows, vapor splitting is
extremely uncommon. This module is provided for completeness and
for systems where it matters (e.g., supercritical CO2 separations
of polar compounds).

References
==========
Michelsen, M.L. "The isothermal flash problem. Part I. Stability",
Fluid Phase Equilib. 9, 1 (1982).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np


@dataclass
class VaporStabilityResult:
    """Result of an EOS-based vapor stability test."""
    stable: bool
    tpd_min: float          # most negative TPD found (>= 0 if stable)
    Y_min: np.ndarray       # composition at the minimum
    n_stationary: int
    iterations_total: int
    trials_evaluated: int


def vapor_phase_stability_test(eos, T: float, p: float,
                                  z: Sequence[float],
                                  trial_initials: Optional[List[Sequence[float]]] = None,
                                  tol: float = 1e-10,
                                  tpd_tol: float = 1e-7,
                                  maxiter: int = 200,
                                  include_default_trials: bool = True
                                  ) -> VaporStabilityResult:
    """Vapor-phase Michelsen stability test using EOS fugacity coefficients.

    Parameters
    ----------
    eos : object with .density_from_pressure(p, T, x, phase_hint='vapor')
          and .ln_phi(rho, T, x)
        Equation of state from `stateprop.cubic`, `stateprop.gerg`,
        or `stateprop.saft`.
    T : float
        Temperature [K].
    p : float
        Pressure [Pa].
    z : sequence
        Candidate vapor composition (length N).
    trial_initials : list of sequences, optional
        User-supplied trial compositions for the iteration. If None
        and `include_default_trials` is True, automatic guesses are
        generated (each pure-i-rich corner plus perturbations).
    tol : float
        Convergence tolerance on the SS step.
    tpd_tol : float
        TPD value below which z is declared unstable. Default 1e-7.
    maxiter : int
        Max SS iterations per trial.
    include_default_trials : bool
        Whether to add automatic trial guesses to any user-supplied list.

    Returns
    -------
    VaporStabilityResult
        - stable: True if no trial yielded TPD < -tpd_tol.
        - tpd_min: most negative TPD found.
        - Y_min: composition at the minimum (a candidate second vapor
                 phase if z is unstable).
    """
    z = np.asarray(z, dtype=float)
    z = z / z.sum()
    N = z.size

    if N < 2:
        raise ValueError("Vapor stability test requires N >= 2 components")

    # Reference: ln phi at z (vapor root)
    try:
        rho_z = eos.density_from_pressure(p, T, z, phase_hint='vapor')
        ln_phi_z = np.asarray(eos.ln_phi(rho_z, T, z), dtype=float)
    except Exception as e:
        raise RuntimeError(
            f"EOS failed to evaluate vapor at z={z}, T={T}, p={p}: {e}"
        )

    # h_i = ln z_i + ln phi_i(z); the +ln p term cancels in TPD differences
    h = np.log(np.maximum(z, 1e-300)) + ln_phi_z

    # Build trial list
    trials: List[np.ndarray] = []
    if trial_initials is not None:
        for Y0 in trial_initials:
            Y0_arr = np.asarray(Y0, dtype=float)
            Y0_arr = Y0_arr / Y0_arr.sum()
            trials.append(Y0_arr)
    if include_default_trials:
        eps = 1.0 / max(100.0, 10.0 * N)
        for i in range(N):
            Y = np.full(N, eps / max(N - 1, 1))
            Y[i] = 1.0 - eps
            Y = Y / Y.sum()
            trials.append(Y)
        if N == 2:
            for delta in [0.3, -0.3]:
                Y = np.array([0.5 + delta, 0.5 - delta])
                if abs(Y[0] - z[0]) > 0.05:
                    trials.append(Y)
        else:
            rng = np.random.default_rng(42)
            for _ in range(N):
                Y = np.ones(N) / N + 0.2 * rng.standard_normal(N)
                Y = np.maximum(Y, 0.01)
                Y = Y / Y.sum()
                if np.max(np.abs(Y - z)) > 0.05:
                    trials.append(Y)

    tpd_min = float('inf')
    Y_at_min = z.copy()
    n_stationary = 0
    iter_total = 0
    trials_evaluated = 0

    for Y0 in trials:
        if float(np.max(np.abs(Y0 - z))) < 1e-3:
            continue
        trials_evaluated += 1
        Y_unn = Y0.copy()
        Y_unn = Y_unn / Y_unn.sum()

        converged = False
        for it in range(maxiter):
            iter_total += 1
            s = float(Y_unn.sum())
            if s <= 0:
                break
            Y_norm = Y_unn / s
            if float(np.max(np.abs(Y_norm - z))) < 1e-7:
                break
            try:
                rho_Y = eos.density_from_pressure(p, T, Y_norm,
                                                    phase_hint='vapor')
                ln_phi_Y = np.asarray(eos.ln_phi(rho_Y, T, Y_norm),
                                        dtype=float)
            except Exception:
                break
            ln_Y_new = h - ln_phi_Y
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
        if float(np.max(np.abs(Y_norm - z))) < 1e-4:
            continue

        tpd = -np.log(s)
        n_stationary += 1
        if tpd < tpd_min:
            tpd_min = tpd
            Y_at_min = Y_norm.copy()

    if n_stationary == 0:
        return VaporStabilityResult(stable=True, tpd_min=0.0,
                                       Y_min=z.copy(),
                                       n_stationary=0,
                                       iterations_total=iter_total,
                                       trials_evaluated=trials_evaluated)
    stable = bool(tpd_min > -tpd_tol)
    return VaporStabilityResult(stable=stable, tpd_min=float(tpd_min),
                                  Y_min=Y_at_min,
                                  n_stationary=n_stationary,
                                  iterations_total=iter_total,
                                  trials_evaluated=trials_evaluated)
