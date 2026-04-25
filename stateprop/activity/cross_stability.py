"""Cross-phase Michelsen TPD test: vapor candidate against liquid trial,
and liquid candidate against vapor trial, using the unified
gamma-phi-EOS framework.

Where v0.9.48 (`stability_test`) is liquid-against-liquid (gamma only)
and v0.9.51 (`vapor_phase_stability_test`) is vapor-against-vapor
(phi only), this module handles the two remaining cases:

- **Liquid candidate, vapor trial**: detects whether a candidate
  single liquid composition z would actually want to vaporize
  (single-phase liquid is unstable against vapor formation).

- **Vapor candidate, liquid trial**: detects whether a candidate
  single vapor composition z would actually want to condense
  (single-phase vapor is unstable against liquid formation).

Together with the same-phase tests, this completes the Michelsen
TPD framework for the gamma-phi-EOS framework.

Algorithm
=========

In the gamma-phi-EOS framework, the fugacities are:

    f_i^L = x_i gamma_i^L(x) p_sat_i phi_sat_i exp(V_L_i (p - p_sat_i) / RT)
    f_i^V = y_i phi_i^V(T, p, y) p

For TPD analysis we measure the Gibbs energy of forming a trial
phase from the candidate composition:

    D(Y) / RT = sum_i Y_i (ln f_i^trial(Y) - ln f_i^candidate(z))

For *liquid candidate z, vapor trial Y*:

    h_i = ln(z_i gamma_i^L(z) p_sat_i phi_sat_i e^P / p)
    Stationarity:  ln Y*_i = h_i - ln phi_i^V(Y*/||Y*||)

For *vapor candidate z, liquid trial Y*:

    h_i = ln(z_i phi_i^V(z) p / (p_sat_i phi_sat_i e^P))
    Stationarity:  ln Y*_i = h_i - ln gamma_i^L(Y*/||Y*||)

In both cases, at the stationary point:

    TPD = -ln(sum Y_inf*)

so sum > 1 means TPD < 0 (instability) and Y_norm = Y*/||Y*|| is a
candidate second-phase composition in the *trial* phase.

Use cases
=========

- **Auto VLE detection**: given (T, p, z), test the liquid candidate;
  if unstable against vapor, vapor exists and a 2-phase VLE flash is
  appropriate. Symmetrically, test vapor candidate.
- **Heteroazeotrope detection**: liquid candidate may be unstable
  both against another liquid (v0.9.48) AND against vapor (this
  module), implying 3-phase VLLE.
- **Robust auto_isothermal seeding**: cross-phase TPD informs which
  phase combinations to try in the auto-flash branch logic.

References
==========
Michelsen (1982) Fluid Phase Equilib. 9, 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np

_R_GAS = 8.314462618


@dataclass
class CrossPhaseStabilityResult:
    """Result of a cross-phase Michelsen TPD test."""
    stable: bool
    tpd_min: float                 # most negative TPD found
    Y_min: np.ndarray              # composition of trial-phase candidate
    candidate_phase: str           # 'liquid' or 'vapor'
    trial_phase: str               # the OTHER phase
    n_stationary: int
    iterations_total: int
    trials_evaluated: int


def cross_phase_stability_test(
    activity_model,
    eos,
    psat_funcs: Sequence,
    T: float,
    p: float,
    z: Sequence[float],
    candidate_phase: str = 'liquid',
    phi_sat_funcs: Optional[Sequence] = None,
    liquid_molar_volumes: Optional[Sequence[float]] = None,
    trial_initials: Optional[List[Sequence[float]]] = None,
    tol: float = 1e-10,
    tpd_tol: float = 1e-7,
    maxiter: int = 200,
    include_default_trials: bool = True,
) -> CrossPhaseStabilityResult:
    """Cross-phase TPD test: candidate-phase z vs trial in opposite phase.

    Parameters
    ----------
    activity_model : object with .gammas(T, x) -> array
    eos : EOS with .density_from_pressure(p, T, x, phase_hint=...) and
          .ln_phi(rho, T, x)
    psat_funcs : sequence of callables psat(T) -> Pa (length N)
    T, p : float
        Temperature [K] and pressure [Pa]
    z : sequence of length N
        Candidate-phase composition.
    candidate_phase : 'liquid' or 'vapor'
        Which phase z is being tested as.
    phi_sat_funcs : optional sequence of callables phi_sat(T) -> float
        Saturation fugacity coefficients (default: 1.0 for each component).
    liquid_molar_volumes : optional sequence of floats, m^3/mol
        For Poynting correction. If None, Poynting = 1.
    trial_initials, tol, tpd_tol, maxiter, include_default_trials :
        See `stability_test` and `vapor_phase_stability_test`.

    Returns
    -------
    CrossPhaseStabilityResult
        Stable iff TPD >= -tpd_tol over all stationary points found.
    """
    if candidate_phase not in ('liquid', 'vapor'):
        raise ValueError("candidate_phase must be 'liquid' or 'vapor'")
    z = np.asarray(z, dtype=float)
    z = z / z.sum()
    N = z.size
    if N < 2:
        raise ValueError("Cross-phase stability test requires N >= 2")

    psat = np.array([f(T) for f in psat_funcs], dtype=float)
    if phi_sat_funcs is not None:
        phi_sat = np.array([f(T) for f in phi_sat_funcs], dtype=float)
    else:
        phi_sat = np.ones(N)
    if liquid_molar_volumes is not None:
        VL = np.asarray(liquid_molar_volumes, dtype=float)
        poynting = np.exp(VL * (p - psat) / (_R_GAS * T))
    else:
        poynting = np.ones(N)

    # ------------------------------------------------------------------
    # Build h_i and define the inner iteration update
    # ------------------------------------------------------------------
    if candidate_phase == 'liquid':
        # Liquid candidate, vapor trial.
        gamma_z = np.asarray(activity_model.gammas(T, z), dtype=float)
        # h_i = ln(z_i gamma_i(z) p_sat_i phi_sat_i e^P / p)
        h = (np.log(np.maximum(z, 1e-300))
             + np.log(np.maximum(gamma_z, 1e-300))
             + np.log(psat * phi_sat * poynting / p))
        trial_phase = 'vapor'

        def _eval_trial_log_phi(Y_norm):
            """Return ln phi_i^V(Y_norm)."""
            try:
                rho_v = eos.density_from_pressure(p, T, Y_norm,
                                                    phase_hint='vapor')
                ln_phi_v = np.asarray(eos.ln_phi(rho_v, T, Y_norm),
                                        dtype=float)
                return ln_phi_v
            except Exception:
                return None

    else:  # candidate_phase == 'vapor'
        # Vapor candidate, liquid trial.
        try:
            rho_z = eos.density_from_pressure(p, T, z, phase_hint='vapor')
            ln_phi_z = np.asarray(eos.ln_phi(rho_z, T, z), dtype=float)
        except Exception as e:
            raise RuntimeError(
                f"EOS failed at z={z}, T={T}, p={p}: {e}"
            )
        # h_i = ln(z_i phi_i^V(z) p / (p_sat_i phi_sat_i e^P))
        h = (np.log(np.maximum(z, 1e-300))
             + ln_phi_z
             + np.log(p / (psat * phi_sat * poynting)))
        trial_phase = 'liquid'

        def _eval_trial_log_phi(Y_norm):
            """Return ln gamma_i^L(Y_norm)."""
            try:
                gamma_Y = np.asarray(activity_model.gammas(T, Y_norm),
                                       dtype=float)
                return np.log(np.maximum(gamma_Y, 1e-300))
            except Exception:
                return None

    # ------------------------------------------------------------------
    # Build trial list
    # ------------------------------------------------------------------
    trials: List[np.ndarray] = []
    if trial_initials is not None:
        for Y0 in trial_initials:
            arr = np.asarray(Y0, dtype=float)
            arr = arr / arr.sum()
            trials.append(arr)
    if include_default_trials:
        eps = 1.0 / max(100.0, 10.0 * N)
        for i in range(N):
            Y = np.full(N, eps / max(N - 1, 1))
            Y[i] = 1.0 - eps
            trials.append(Y / Y.sum())
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

    # ------------------------------------------------------------------
    # Iteration loop
    # ------------------------------------------------------------------
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
            ln_phi_Y = _eval_trial_log_phi(Y_norm)
            if ln_phi_Y is None:
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
        # Reject trivial (Y_norm ~= z) -- can happen if the phase types
        # happen to give matching fugacities at z (e.g. at saturation).
        if float(np.max(np.abs(Y_norm - z))) < 1e-4:
            continue
        tpd = -np.log(s)
        n_stationary += 1
        if tpd < tpd_min:
            tpd_min = tpd
            Y_at_min = Y_norm.copy()

    if n_stationary == 0:
        return CrossPhaseStabilityResult(
            stable=True, tpd_min=0.0, Y_min=z.copy(),
            candidate_phase=candidate_phase, trial_phase=trial_phase,
            n_stationary=0, iterations_total=iter_total,
            trials_evaluated=trials_evaluated,
        )
    return CrossPhaseStabilityResult(
        stable=bool(tpd_min > -tpd_tol),
        tpd_min=float(tpd_min),
        Y_min=Y_at_min,
        candidate_phase=candidate_phase,
        trial_phase=trial_phase,
        n_stationary=n_stationary,
        iterations_total=iter_total,
        trials_evaluated=trials_evaluated,
    )
