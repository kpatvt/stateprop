"""Mixture critical-point solver for the Helmholtz/GERG EOS family (v0.9.11).

Heidemann-Khalil (1980) / Michelsen (1982) formulation, same structure as
stateprop.cubic.critical but with the EOS-specific A^{res} matrix
assembled from the analytic composition derivatives added in v0.9.9.

At the mixture critical point:
    R1 = lambda_min(B) = 0         -- spinodal at fixed (T, V)
    R2 = u^T (dB/ds)|_{s=0} u = 0  -- third-order directional condition
where B = I + sqrt(n_i n_j) A^res_{ij} / (RT), n_tot = 1, and u is the
null eigenvector.

The A^res matrix is the composition Hessian of the residual Helmholtz
energy n * alpha^r, which is symmetric by Schwarz:

    A_ij^{res}/(RT) = d^2(n * alpha^r)/(dn_i dn_j)|_{T, V}

From chain-rule expansion using our v0.9.9 primitives:

    A_ij/(RT) = rho * dlnphi_drho[i] + dlnphi_dx[i,j] - sum_k x_k * dlnphi_dx[i,k]
              + (dp/dn_j) / p - 1

where the second line is d(ln Z)/dn_j, which cancels the asymmetric part
of d(ln phi_i)/dn_j to give a symmetric Hessian on A^{res} = RT*(lnphi + ln Z).

Restrictions:
- Ignores volume translation -- not applicable here (Helmholtz EOS has no
  volume translation analog).

References:
- R.A. Heidemann and A.M. Khalil, AIChE J. 26, 769 (1980).
- M.L. Michelsen and R.A. Heidemann, AIChE J. 27, 521 (1981).
"""
import numpy as np

from .properties import (
    pressure,
    dlnphi_dx_at_rho, dlnphi_drho_at_x,
    dp_dx_at_rho, dp_drho_T,
)


# ---------------------------------------------------------------------------
# Analytic A^{res} matrix
# ---------------------------------------------------------------------------

def _A_residual_matrix(T, V, n, mixture):
    """Compute A_ij^{res} = d^2 (n*alpha^r) / dn_i dn_j |_{T, V, n_{k!=i,j}}.

    Symmetric by Schwarz's theorem. Assembled from the analytic primitives
    via the chain rule explained in this module's docstring.

    Parameters
    ----------
    T : float         temperature [K]
    V : float         molar volume [m^3/mol]  (V = 1/rho since n_tot = 1)
    n : array (N,)    composition (mole fractions; this function assumes
                      n_tot = 1 so n == x)
    mixture : Mixture

    Returns
    -------
    A : ndarray (N, N)  symmetric residual Helmholtz Hessian in [J/mol]
    """
    x = np.asarray(n, dtype=float)
    rho = 1.0 / V
    R = mixture.components[0].fluid.R
    RT = R * T

    # Composition derivatives at fixed (T, rho)
    dphi_drho = dlnphi_drho_at_x(rho, T, x, mixture)      # (N,)
    dphi_dx = dlnphi_dx_at_rho(rho, T, x, mixture)        # (N, N)
    sum_xdphi = np.sum(x * dphi_dx, axis=1)                # Σ_k x_k dphi[i,k]

    # d(ln phi_i)/dn_j = ρ * dphi_drho[i] + dphi_dx[i,j] - sum_xdphi[i]
    dlnphi_dn = rho * dphi_drho[:, None] + dphi_dx - sum_xdphi[:, None]

    # d(ln Z)/dn_j.  Z = pV/(nRT) = p/(rho RT) at n_tot=1
    # d(ln Z)/dn_j = (1/p) dp/dn_j - 1  (at n_tot=1)
    # dp/dn_j at fixed (T, V, n_{k≠j}) = dp/drho|_{T,x} * (1/V)
    #                                   + Σ_k dp/dx_k|_{T,rho} * (δ_kj - x_k)
    dp_drho = dp_drho_T(rho, T, x, mixture)
    dp_dx = dp_dx_at_rho(rho, T, x, mixture)                 # (N,)
    sum_x_dpdx = float(np.sum(x * dp_dx))
    dp_dn = dp_drho * rho + dp_dx - sum_x_dpdx                # (N,)
    p = pressure(rho, T, x, mixture)
    dlnZ_dn = dp_dn / p - 1.0                                 # (N,)

    # A_ij/(RT) = d(ln phi_i)/dn_j + d(ln Z)/dn_j  (the -ln Z term from ln phi
    # is cancelled by adding d(ln Z)/dn_j back to recover ∂²(n α^r)/∂n_i∂n_j)
    A_over_RT = dlnphi_dn + dlnZ_dn[None, :]
    # Enforce exact symmetry (round-off cleanup)
    A_over_RT = 0.5 * (A_over_RT + A_over_RT.T)
    return RT * A_over_RT


def _B_matrix(T, V, n, mixture):
    """Michelsen's scaled matrix B = I + sqrt(n_i n_j) A^{res}_{ij} / (RT).

    At the mixture critical point, one eigenvalue of B is exactly zero.
    """
    R = mixture.components[0].fluid.R
    A_res = _A_residual_matrix(T, V, n, mixture)
    sqrt_n = np.sqrt(np.maximum(n, 0.0))
    return np.eye(len(n)) + np.outer(sqrt_n, sqrt_n) * A_res / (R * T)


# ---------------------------------------------------------------------------
# Critical residuals
# ---------------------------------------------------------------------------

def _critical_residuals(T, V, z, mixture, eps_perturb=None):
    """Return (R1, R2, u) at (T, V) and composition z (n_tot = 1).

    R1 = lambda_min(B)           -- spinodal
    R2 = (u^T B(n + eps u) u - u^T B(n - eps u) u) / (2 eps)
                                  -- third-order condition (directional FD)
    """
    n = np.asarray(z, dtype=float)
    B = _B_matrix(T, V, n, mixture)
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = int(np.argmin(eigvals))
    lam = float(eigvals[idx])
    u = eigvecs[:, idx]

    if eps_perturb is None:
        u_max = float(np.max(np.abs(u)))
        eps_perturb = min(0.1 * float(np.min(n)) / max(u_max, 1e-30), 1e-4)
    n_p = n + eps_perturb * u
    n_m = n - eps_perturb * u
    while np.any(n_p <= 1e-12) or np.any(n_m <= 1e-12):
        eps_perturb *= 0.5
        n_p = n + eps_perturb * u
        n_m = n - eps_perturb * u
        if eps_perturb < 1e-12:
            raise RuntimeError("Cannot perturb along null eigenvector safely.")

    B_p = _B_matrix(T, V, n_p, mixture)
    B_m = _B_matrix(T, V, n_m, mixture)
    Q_p = float(u @ B_p @ u)
    Q_m = float(u @ B_m @ u)
    dQ = (Q_p - Q_m) / (2.0 * eps_perturb)
    return lam, dQ, u


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def critical_point(z, mixture, T_init=None, V_init=None,
                   tol=1e-8, maxiter=80, step_cap=0.15, verbose=False,
                   sanity_check=True):
    """Find the mixture critical point at composition z using Heidemann-Khalil.

    Same interface as stateprop.cubic.critical.critical_point; the Helmholtz
    mixture version uses analytic composition derivatives from v0.9.9 to
    assemble the A^{res} Hessian. The (T, V) Newton loop uses FD Jacobian
    on the residual pair (R1, R2).

    Parameters
    ----------
    z : array (N,)     Overall composition (will be renormalized).
    mixture : Mixture  Helmholtz/GERG mixture.
    T_init, V_init : float, optional
        Initial guesses. Defaults: T_init = x-weighted Tc,
        V_init = 1/rho_r (pseudo-critical molar volume from reducing function).
    tol : float        Convergence on max(|R1|, |R2|).
    maxiter : int      Maximum Newton iterations.
    step_cap : float   Max fractional step on T or V per iteration.
    sanity_check : bool
        If True (default), verify the converged V_c is within [0.3x, 3x] of
        the pure-component-weighted V_c average, and retry from a tighter
        initial guess if not. This guards against H-K's occasional convergence
        to secondary (non-VLE) critical solutions, which is a known issue for
        certain binary systems like CH4-N2, CO2-N2.

    Returns
    -------
    dict with keys: T_c, p_c, V_c, rho_c, u, iterations, residual

    Notes
    -----
    The H-K formulation can converge to secondary critical solutions for
    some binaries (especially systems involving N2 as the smaller component).
    When `sanity_check=True`, the solver retries from an initial guess
    consistent with pure-component critical volumes. If that also fails the
    sanity test, the result is returned with a `suspicious=True` flag so
    callers can decide whether to trust it.
    """
    z = np.asarray(z, dtype=float)
    z = z / z.sum()
    # Pure component critical volume average (used for sanity checking)
    V_pure = np.array([1.0 / c.fluid.rho_c for c in mixture.components])
    V_pure_avg = float(np.dot(z, V_pure))

    def _run_solver(T0, V0):
        T = float(T0); V = float(V0)
        for it in range(maxiter):
            R1, R2, u = _critical_residuals(T, V, z, mixture)
            if verbose:
                print(f"  it {it}: T={T:.4f}, V={V:.4e}, R1={R1:.3e}, R2={R2:.3e}")
            if max(abs(R1), abs(R2)) < tol:
                return T, V, u, it + 1, max(abs(R1), abs(R2)), True

            hT = max(abs(T) * 1e-5, 1e-3)
            hV = max(abs(V) * 1e-5, 1e-10)
            R1_Tp, R2_Tp, _ = _critical_residuals(T + hT, V, z, mixture)
            R1_Tm, R2_Tm, _ = _critical_residuals(T - hT, V, z, mixture)
            R1_Vp, R2_Vp, _ = _critical_residuals(T, V + hV, z, mixture)
            R1_Vm, R2_Vm, _ = _critical_residuals(T, V - hV, z, mixture)

            J = np.array([
                [(R1_Tp - R1_Tm) / (2 * hT), (R1_Vp - R1_Vm) / (2 * hV)],
                [(R2_Tp - R2_Tm) / (2 * hT), (R2_Vp - R2_Vm) / (2 * hV)],
            ])
            try:
                delta = np.linalg.solve(J, -np.array([R1, R2]))
            except np.linalg.LinAlgError:
                return T, V, u, it + 1, max(abs(R1), abs(R2)), False

            frac_T = abs(delta[0]) / abs(T)
            frac_V = abs(delta[1]) / abs(V)
            if max(frac_T, frac_V) > step_cap:
                delta = delta * (step_cap / max(frac_T, frac_V))

            T = T + delta[0]
            V = V + delta[1]
            if V <= 0:
                V = 0.5 * V_pure_avg
        return T, V, u, maxiter, max(abs(R1), abs(R2)), False

    # Default initial guesses
    if T_init is None:
        T_init = float(sum(z[i] * mixture.components[i].fluid.T_c
                           for i in range(len(z))))
    if V_init is None:
        Tr, rho_r = mixture.reduce(z)
        V_init = 1.0 / rho_r if rho_r > 0 else V_pure_avg

    # First attempt
    T, V, u, n_it, res, ok = _run_solver(T_init, V_init)
    if not ok:
        raise RuntimeError(
            f"critical_point did not converge in {maxiter} iterations: "
            f"T={T}, V={V}, residual={res:.2e}")

    suspicious = False
    if sanity_check:
        # V_c should be within [0.3x, 3x] of pure-weighted V_c
        if V < 0.3 * V_pure_avg or V > 3.0 * V_pure_avg:
            if verbose:
                print(f"  V_c = {V*1e6:.1f} cm^3/mol out of [0.3, 3.0] x "
                      f"{V_pure_avg*1e6:.1f} cm^3/mol; retrying...")
            # Retry from pure-V-averaged guess
            T2, V2, u2, n_it2, res2, ok2 = _run_solver(T_init, V_pure_avg)
            if ok2 and 0.3 * V_pure_avg <= V2 <= 3.0 * V_pure_avg:
                T, V, u, n_it, res = T2, V2, u2, n_it + n_it2, res2
            else:
                # Try one more initial guess: slightly lower T
                T3, V3, u3, n_it3, res3, ok3 = _run_solver(0.95 * T_init, V_pure_avg)
                if ok3 and 0.3 * V_pure_avg <= V3 <= 3.0 * V_pure_avg:
                    T, V, u, n_it, res = T3, V3, u3, n_it + n_it2 + n_it3, res3
                else:
                    suspicious = True

    rho = 1.0 / V
    p = pressure(rho, T, z, mixture)
    return {
        "T_c": T, "p_c": p, "V_c": V, "rho_c": rho,
        "u": u,
        "iterations": n_it,
        "residual": res,
        "suspicious": suspicious,
    }


# ---------------------------------------------------------------------------
# Multistart with physical filtering (v0.9.12)
# ---------------------------------------------------------------------------
#
# The H-K equations R1=R2=0 typically have multiple roots in (T, V) space for
# a given composition: the physical VLE critical plus secondary stationary
# points of the residual Helmholtz surface. A grid-based scan of CH4-N2 50/50
# finds six distinct stationary points, only one of which is the physical
# VLE critical. The secondary roots all have unphysical Z_c (> 0.4 or so,
# or even negative p at convergence) but otherwise satisfy the math exactly.
#
# This multistart solver runs H-K from a diverse set of initial guesses,
# clusters the converged solutions, and scores each cluster by physical
# plausibility. Scoring uses two signals:
#
#   1) Compressibility factor Z_c = p_c V_c / (R T_c).  For non-polar and
#      weakly polar mixtures Z_c is consistently in [0.24, 0.33]. The pure
#      components in any binary have Z_c values in this range by
#      definition -- it's a near-universal property of the principle of
#      corresponding states. Secondary H-K roots routinely produce
#      Z_c = 0.4-0.9 or negative p_c; these are filtered out.
#
#   2) V_c proximity to pure-component-weighted average. Real mixture
#      critical volumes typically lie within ~2x the Kay-rule Σ z_i V_c_i.
#      Secondary roots often produce V_c that's 3-5x this value.
#
# The scoring function is a smooth Gaussian on Z_c centered at 0.29 with
# width 0.05, plus a Gaussian on V_c with logarithmic scale. Candidates
# with negative pressure or Z_c outside [0.15, 0.45] are rejected outright.


def _score_candidate(T, V, z, mixture, V_pure_avg, Zc_target=0.29,
                     Zc_sigma=0.08, Vc_log_sigma=0.7, use_tpd=True):
    """Physical-plausibility score for a converged H-K stationary point.

    Higher is better. Returns -inf for non-physical candidates (negative p,
    Z_c far outside typical range).

    Scoring combines three signals:
      1) Gaussian on Z_c centered at 0.29 (principle of corresponding states)
      2) Log-Gaussian on V_c / V_pure_avg (Kay-rule sanity)
      3) If use_tpd=True, a TPD marginal-stability VETO: at a true critical,
         |S-1| from the Michelsen stability test should be small (<0.1).
         Candidates with |S-1| > 0.3 are hard-rejected as non-critical;
         smaller values don't affect the score. This is a robustness veto,
         not a fine-grained tiebreaker, because the Wilson K-factor init
         used inside the TPD test is not always close enough to the
         critical direction for |S-1| to be tight.
    """
    rho = 1.0 / V
    try:
        p = pressure(rho, T, z, mixture)
    except Exception:
        return -float('inf')
    if p <= 0 or not np.isfinite(p):
        return -float('inf')
    R = mixture.components[0].fluid.R
    Zc = p * V / (R * T)
    # Hard reject: Z_c outside the plausible range.
    if Zc < 0.13 or Zc > 0.55:
        return -float('inf')
    # Soft Gaussian scoring
    s_Zc = np.exp(-0.5 * ((Zc - Zc_target) / Zc_sigma) ** 2)
    s_Vc = np.exp(-0.5 * (np.log(V / V_pure_avg) / Vc_log_sigma) ** 2)
    # TPD veto (hard reject only)
    if use_tpd:
        try:
            from .stability import stability_test_TPD
            _, _, Sm1 = stability_test_TPD(z, T, p, mixture,
                                           tol=1e-10, maxiter=30)
            if abs(Sm1) > 0.3:
                return -float('inf')
        except Exception:
            pass  # TPD failed -- don't penalize
    return float(s_Zc * s_Vc)


def _run_hk_newton(T0, V0, z, mixture, tol, maxiter, step_cap):
    """Run Heidemann-Khalil Newton from (T0, V0). Returns (T, V, u, iters, res, ok)."""
    T, V = float(T0), float(V0)
    R1 = R2 = float('nan'); u = None
    for it in range(maxiter):
        try:
            R1, R2, u = _critical_residuals(T, V, z, mixture)
        except Exception:
            return T, V, u, it + 1, float('nan'), False
        if max(abs(R1), abs(R2)) < tol:
            return T, V, u, it + 1, max(abs(R1), abs(R2)), True
        hT = max(abs(T) * 1e-5, 1e-3)
        hV = max(abs(V) * 1e-5, 1e-10)
        try:
            R1_Tp, R2_Tp, _ = _critical_residuals(T + hT, V, z, mixture)
            R1_Tm, R2_Tm, _ = _critical_residuals(T - hT, V, z, mixture)
            R1_Vp, R2_Vp, _ = _critical_residuals(T, V + hV, z, mixture)
            R1_Vm, R2_Vm, _ = _critical_residuals(T, V - hV, z, mixture)
            J = np.array([
                [(R1_Tp - R1_Tm) / (2 * hT), (R1_Vp - R1_Vm) / (2 * hV)],
                [(R2_Tp - R2_Tm) / (2 * hT), (R2_Vp - R2_Vm) / (2 * hV)],
            ])
            delta = np.linalg.solve(J, -np.array([R1, R2]))
        except Exception:
            return T, V, u, it + 1, max(abs(R1), abs(R2)), False
        frac = max(abs(delta[0]) / abs(T), abs(delta[1]) / abs(V))
        if frac > step_cap:
            delta = delta * (step_cap / frac)
        T = T + delta[0]
        V = V + delta[1]
        if V <= 0:
            return T, V, u, it + 1, float('nan'), False
    return T, V, u, maxiter, max(abs(R1), abs(R2)), False


def _homotopy_critical(z_target, mixture, i_start, n_steps, tol, maxiter, step_cap,
                       T_seed=None, V_seed=None, max_V_jump=0.25):
    """Homotopy from pure component i_start to z_target with branch tracking.

    Walk composition z_0 = pure(i_start) -> z_target using adaptive steps,
    solving H-K at each step starting from the previous converged (T, V).
    Between steps we monitor V (and T) continuity: if the solution makes
    an unphysically large jump (>max_V_jump relative change), we reject
    the step, halve the composition step, and retry. This keeps the walk
    on the physical critical branch even when secondary roots sit nearby
    and the naive Newton basin is too narrow.

    Returns (T, V, u, iterations_total, residual, ok).
    """
    N = len(z_target)
    eps = 1e-4
    z0 = np.full(N, eps / (N - 1))
    z0[i_start] = 1.0 - eps
    if T_seed is None:
        T_seed = mixture.components[i_start].fluid.T_c
    if V_seed is None:
        V_seed = 1.0 / mixture.components[i_start].fluid.rho_c

    T, V = float(T_seed), float(V_seed)
    total_iters = 0
    # Converge at z0
    T, V, u, it, res, ok = _run_hk_newton(T, V, z0, mixture, tol, maxiter, step_cap)
    total_iters += it
    if not ok:
        return T, V, None, total_iters, res, False

    z_target_arr = np.asarray(z_target, dtype=float)
    # Adaptive composition stepping with branch-jump detection
    alpha = 0.0
    dalpha = 1.0 / n_steps
    min_dalpha = 1.0 / (32 * n_steps)    # 5 halvings max
    max_halvings_per_step = 6

    while alpha < 1.0 - 1e-12:
        step = min(dalpha, 1.0 - alpha)
        z_next = (1.0 - (alpha + step)) * z0 + (alpha + step) * z_target_arr
        z_next = z_next / z_next.sum()

        T_new, V_new, u_new, it_new, res_new, ok_new = _run_hk_newton(
            T, V, z_next, mixture, tol, maxiter, step_cap)
        total_iters += it_new

        # Reject step if: didn't converge, or jumped branches
        dV_rel = abs(V_new - V) / V if V > 0 else float('inf')
        dT_rel = abs(T_new - T) / T if T > 0 else float('inf')
        rejected = (not ok_new) or (dV_rel > max_V_jump) or (dT_rel > max_V_jump)

        if rejected:
            # Halve and retry
            if step <= min_dalpha:
                return T, V, u, total_iters, res_new, False
            dalpha = step * 0.5
            continue

        # Accept step
        T, V, u = T_new, V_new, u_new
        res = res_new
        alpha += step
        # Gentle step-size growth after success
        dalpha = min(dalpha * 1.3, 1.0 / n_steps)

    return T, V, u, total_iters, res, True


def critical_point_multistart(z, mixture, tol=1e-8, maxiter=40, step_cap=0.15,
                              n_starts=33, verbose=False, return_all=False,
                              use_homotopy=False, homotopy_steps=10):
    """Multistart + physical-filter critical point solver for Helmholtz mixtures.

    Runs Heidemann-Khalil Newton from a diverse set of initial guesses,
    clusters converged solutions, and returns the candidate with the
    highest physical-plausibility score.

    This addresses H-K's known failure mode: R1 = R2 = 0 has multiple roots,
    only one of which is the physical VLE critical. Secondary roots satisfy
    the math exactly but have unphysical Z_c or V_c.

    Parameters
    ----------
    z : array (N,)     overall composition (will be renormalized)
    mixture : Mixture  Helmholtz/GERG mixture
    tol : float        convergence tolerance on max(|R1|, |R2|)
    maxiter : int      Newton iterations per start
    step_cap : float   maximum fractional step on T or V per iteration
    n_starts : int
        Minimum number of initial guesses. The default pipeline generates
        ~15-17 candidates covering physically-motivated single points, each
        pure component's critical, a geometric V-grid sweep, and T
        perturbations. If n_starts exceeds the default set size, extra
        random candidates are added. Typical useful range: 10-25.
    verbose : bool     print scoring and candidate details
    return_all : bool
        If True, also return the list of all scored candidates, useful for
        diagnosing difficult systems.

    Returns
    -------
    dict with keys: T_c, p_c, V_c, rho_c, u, iterations, residual, score,
                    n_candidates_found, suspicious
        (plus `all_candidates` if return_all=True)
    """
    z = np.asarray(z, dtype=float)
    z = z / z.sum()

    # Pure-component reference values
    T_c_pure = np.array([c.fluid.T_c for c in mixture.components])
    V_c_pure = np.array([1.0 / c.fluid.rho_c for c in mixture.components])
    T_pseudo = float(z @ T_c_pure)
    V_pure_avg = float(z @ V_c_pure)

    # GERG reducing function
    try:
        Tr, rho_r = mixture.reduce(z)
        V_reducing = 1.0 / rho_r if rho_r > 0 else V_pure_avg
    except Exception:
        Tr, V_reducing = T_pseudo, V_pure_avg

    # Candidate initial (T, V) pairs. The H-K equations have multiple roots
    # in (T, V) space, and basins of attraction for the physical VLE critical
    # can be tiny. We use a multi-layer strategy combining physical heuristics
    # with grid coverage.
    candidates = []

    # Layer 1: physically-motivated single points
    candidates.append((T_pseudo, V_pure_avg))   # Kay's rule
    candidates.append((Tr, V_reducing))          # GERG reducing function
    candidates.append((T_pseudo, V_reducing))
    candidates.append((Tr, V_pure_avg))

    # Layer 2: pure component (T_c_i, V_c_i). Essential for dilute mixtures.
    for i in range(len(z)):
        candidates.append((T_c_pure[i], V_c_pure[i]))

    # Layer 3: pairwise midpoints (T_c_i + T_c_j)/2 and (V_c_i + V_c_j)/2
    # These catch the common case where the physical mixture critical sits
    # BETWEEN the pure-component criticals in the (T, V) plane.
    N = len(z)
    for i in range(N):
        for j in range(i + 1, N):
            T_mid = 0.5 * (T_c_pure[i] + T_c_pure[j])
            V_mid = 0.5 * (V_c_pure[i] + V_c_pure[j])
            candidates.append((T_mid, V_mid))

    # Layer 4: Dense 2D T-V grid near the pseudo-critical. The physical VLE
    # critical can have a very small basin of attraction (seen in CH4-N2,
    # CH4-ethane). A tight grid near T_pseudo x V_pure_avg is the
    # highest-yield zone; we then add a wider grid for coverage.
    T_tight_lo = 0.90 * T_pseudo
    T_tight_hi = 1.10 * T_pseudo
    V_tight_lo = 0.60 * V_pure_avg
    V_tight_hi = 1.60 * V_pure_avg
    T_tight = np.linspace(T_tight_lo, T_tight_hi, 5)
    V_tight = np.geomspace(V_tight_lo, V_tight_hi, 5)
    for T0 in T_tight:
        for V0 in V_tight:
            candidates.append((float(T0), float(V0)))

    # Wider 2D grid for outlier coverage
    T_grid_lo = 0.85 * T_c_pure.min()
    T_grid_hi = 1.05 * T_c_pure.max()
    V_grid_lo = 0.35 * min(V_c_pure.min(), V_pure_avg)
    V_grid_hi = 1.8 * max(V_c_pure.max(), V_pure_avg)
    T_sweep = np.linspace(T_grid_lo, T_grid_hi, 4)
    V_sweep = np.geomspace(V_grid_lo, V_grid_hi, 4)
    for T0 in T_sweep:
        for V0 in V_sweep:
            candidates.append((float(T0), float(V0)))

    # Optional extra random starts
    if n_starts > len(candidates):
        rng = np.random.default_rng(42)
        for _ in range(n_starts - len(candidates)):
            T0 = T_pseudo * (0.85 + 0.3 * rng.random())
            V0 = V_pure_avg * (0.4 + 1.2 * rng.random())
            candidates.append((T0, V0))

    # Run H-K from each
    raw_solutions = []

    # Layer 0 (first priority): homotopy walks from each pure component to z.
    # This traces the critical locus continuously and is far more reliable
    # than single-shot Newton when the physical VLE critical has a small
    # basin of attraction. Skip the homotopy walk only for very dilute
    # compositions where it would degenerate to "already at pure".
    if use_homotopy:
        # Find the dominant component and any secondary components with
        # meaningful presence; walk from each to get multiple homotopy paths
        N = len(z)
        # Start from each component with z_i > 0.02 (or all if binary)
        start_comps = [i for i in range(N) if z[i] > 0.02] if N > 2 else list(range(N))
        for i_start in start_comps:
            # Skip homotopy if z is already essentially pure_i
            if z[i_start] > 0.99:
                continue
            T_h, V_h, u_h, it_h, res_h, ok_h = _homotopy_critical(
                z, mixture, i_start, homotopy_steps, tol, maxiter, step_cap)
            if ok_h:
                raw_solutions.append({
                    "T_c": T_h, "V_c": V_h, "u": u_h, "iterations": it_h,
                    "residual": res_h, "source": f"homotopy_from_{i_start}",
                })
            elif verbose:
                print(f"  homotopy from comp {i_start} failed "
                      f"(last T={T_h:.2f}, V={V_h*1e6:.2f})")

    for T0, V0 in candidates:
        T, V, u, it, res, ok = _run_hk_newton(T0, V0, z, mixture, tol, maxiter, step_cap)
        if ok:
            raw_solutions.append({
                "T_c": T, "V_c": V, "u": u, "iterations": it, "residual": res,
                "source": "multistart",
            })

    if not raw_solutions:
        raise RuntimeError(
            "critical_point_multistart: no candidate converged")

    # Deduplicate: cluster by relative (T, V) proximity
    unique = []
    for s in raw_solutions:
        new = True
        for u in unique:
            if (abs(s["T_c"] - u["T_c"]) / u["T_c"] < 0.01
                    and abs(s["V_c"] - u["V_c"]) / u["V_c"] < 0.02):
                new = False
                break
        if new:
            unique.append(s)

    # Score each
    for s in unique:
        s["score"] = _score_candidate(s["T_c"], s["V_c"], z, mixture, V_pure_avg)

    # Sort by score (highest first)
    unique.sort(key=lambda s: s["score"], reverse=True)

    if verbose:
        print(f"Multistart found {len(unique)} unique candidates "
              f"(from {len(raw_solutions)} converged):")
        for s in unique:
            rho = 1.0 / s["V_c"]
            p = pressure(rho, s["T_c"], z, mixture)
            R = mixture.components[0].fluid.R
            Zc = p * s["V_c"] / (R * s["T_c"])
            print(f"  T={s['T_c']:6.2f}K  V={s['V_c']*1e6:6.2f}cm3/mol  "
                  f"p={p/1e6:6.3f}MPa  Zc={Zc:.3f}  score={s['score']:.3f}")

    best = unique[0]
    rho = 1.0 / best["V_c"]
    p = pressure(rho, best["T_c"], z, mixture)
    result = {
        "T_c": best["T_c"],
        "p_c": p,
        "V_c": best["V_c"],
        "rho_c": rho,
        "u": best["u"],
        "iterations": best["iterations"],
        "residual": best["residual"],
        "score": best["score"],
        "n_candidates_found": len(unique),
        "suspicious": best["score"] < 0.1,   # Low score = uncertain result
    }
    if return_all:
        # Add p_c and Z_c info to each candidate for user inspection
        all_cands = []
        for s in unique:
            r = 1.0 / s["V_c"]
            p_s = pressure(r, s["T_c"], z, mixture)
            R = mixture.components[0].fluid.R
            all_cands.append({
                "T_c": s["T_c"], "V_c": s["V_c"], "p_c": p_s,
                "Z_c": p_s * s["V_c"] / (R * s["T_c"]),
                "score": s["score"],
                "residual": s["residual"],
            })
        result["all_candidates"] = all_cands
    return result
