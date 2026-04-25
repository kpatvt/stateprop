"""Phase envelope tracing for Helmholtz/GERG EOS mixtures (v0.9.13).

Structural port of stateprop.cubic.envelope to the Helmholtz mixture EOS.
Traces the bubble/dew envelope at fixed composition z by seeding from the
mixture critical point (from v0.9.12 `critical_point_multistart`) and
stepping outward along the envelope tangent in both directions. This
avoids the critical-point turning-point problem that Michelsen-style
low-p-seeded tracing runs into.

Uses the v0.9.9 analytic composition derivatives and v0.9.10 T/p
derivatives of ln phi for the Jacobian (via FD on the residual vector
for simplicity; the primitives are all analytic underneath).

Algorithm:
1. Call critical_point_multistart(z, mixture) to get (T_c, p_c, V_c) and
   the null eigenvector u of the residual Helmholtz Hessian at the critical.
2. For each of direction in {+1, -1}:
   a. Multi-strategy seeding: try eigenvector perturbation at decreasing
      `crit_offset` values; if all fail, fall back to Wilson-seeded point
      just below the critical.
   b. Adaptive continuation: quadratic predictor when history >= 3 points
      and extrapolation is within span; linear tangent predictor otherwise.
   c. Adaptive beta-switch (bubble=0 vs dew=1) based on z-weighted ln K
      asymmetry with hysteresis to prevent chattering near the critical.
   d. Step control from predictor-corrector agreement: tight agreement
      -> grow step; poor agreement -> shrink.
3. Combine: [dew reversed] + [critical] + [bubble] gives the full envelope
   from low-p dew to low-p bubble.

Variables at each point: X = (ln K_1..N, ln T, ln p).
Residuals (N+2):
    R_i = ln K_i - (ln phi_i^L - ln phi_i^V), i=1..N
    R_N = Sum(y) - 1 (bubble, beta=0) or Sum(x) - 1 (dew, beta=1)
    R_{N+1} = X[spec_idx] - spec_target

Every point recorded satisfies the Rachford-Rice residual for its
labeled branch (beta=0 or 1) to ~1e-9.

Robustness notes:
- The near-critical portion of the envelope is reliably traced.
- Bubble-side trace (z as liquid) typically works well down to low (T, p).
- Dew-side trace can have numerical issues when K-factors are close to
  1: (Sum(z/K) - 1) becomes insensitive to K errors. For quantitatively
  reliable points use `envelope_point(T, p, z, mixture)` with targets.
"""
import numpy as np

from .properties import (
    ln_phi, density_from_pressure,
    dlnphi_dx_at_p, dlnphi_dp_at_T, dlnphi_dT_at_p,
)
from .critical import critical_point_multistart


# ---------------------------------------------------------------------------
# Core residuals and Jacobian
# ---------------------------------------------------------------------------

def _envelope_residuals(X, beta, z, spec_idx, spec_val, mixture):
    """Return N+2 residuals at state X = (ln K, ln T, ln p)."""
    N = len(z)
    lnK = X[:N]
    K = np.exp(lnK)
    T = np.exp(X[N])
    p = np.exp(X[N + 1])

    if beta == 0:
        x = z
        y_un = K * z
        y_sum = float(y_un.sum())
        y = y_un / y_sum
        rr_residual = y_sum - 1.0
    else:
        y = z
        x_un = z / K
        x_sum = float(x_un.sum())
        x = x_un / x_sum
        rr_residual = x_sum - 1.0

    rho_L = density_from_pressure(p, T, x, mixture, phase_hint='liquid')
    rho_V = density_from_pressure(p, T, y, mixture, phase_hint='vapor')
    lnphi_L = ln_phi(rho_L, T, x, mixture)
    lnphi_V = ln_phi(rho_V, T, y, mixture)

    R = np.empty(N + 2)
    R[:N] = lnK - (lnphi_L - lnphi_V)
    R[N] = rr_residual
    R[N + 1] = X[spec_idx] - spec_val
    return R


def _envelope_jacobian_fd(X, beta, z, spec_idx, spec_val, mixture, eps=1e-6):
    """Central-difference Jacobian.

    We use FD here for simplicity. All underlying primitives (ln_phi,
    density_from_pressure) are analytic, so this only adds numerical noise
    at ~1e-8 relative, which is well below typical envelope tolerances.
    """
    dim = len(X)
    J = np.empty((dim, dim))
    for k in range(dim):
        X_p = X.copy(); X_m = X.copy()
        h = eps * max(abs(X[k]), 1.0)
        X_p[k] += h; X_m[k] -= h
        R_p = _envelope_residuals(X_p, beta, z, spec_idx, spec_val, mixture)
        R_m = _envelope_residuals(X_m, beta, z, spec_idx, spec_val, mixture)
        J[:, k] = (R_p - R_m) / (2.0 * h)
    return J


def _envelope_jacobian_analytic(X, beta, z, spec_idx, spec_val, mixture):
    """Analytic (N+2) x (N+2) Jacobian of the Helmholtz envelope residuals.

    v0.9.18 port of the cubic analytic envelope Jacobian (v0.9.17) to the
    Helmholtz/GERG family. Built from the v0.9.9 composition derivatives
    (`dlnphi_dx_at_p`) and v0.9.10 T/p derivatives of ln phi
    (`dlnphi_dp_at_T`, `dlnphi_dT_at_p`).

    Variables: X = (ln K_1, ..., ln K_N, ln T, ln p).
    Residuals (N+2):
        R_i      = ln K_i - (lnphi_L_i - lnphi_V_i),   i=1..N
        R_{N+1}  = Sum(K*z) - 1    (beta=0)  or  Sum(z/K) - 1    (beta=1)
        R_{N+2}  = X[spec_idx] - spec_val

    Jacobian structure (N+2 columns: lnK_j for j=0..N-1, lnT, lnp):

    For beta=0 (bubble, x=z fixed, y=K*z/Sum(K*z)):
      Column j (lnK_j):
        dy_m/dlnK_j = y_m * (delta_mj - y_j)
        dR_i/dlnK_j = delta_ij + Sum_m (dlnphi_V_i/dy_m) * dy_m/dlnK_j
        dR_{N+1}/dlnK_j = K_j * z_j
      Column N (lnT):  dR_i/dlnT = -(dlnphi_L_i/dT - dlnphi_V_i/dT) * T
      Column N+1 (lnp): dR_i/dlnp = -(dlnphi_L_i/dp - dlnphi_V_i/dp) * p
      dR_{N+1}/dlnT = dR_{N+1}/dlnp = 0
      dR_{N+2}/dX_k = delta_{k, spec_idx}

    For beta=1 (dew, y=z fixed, x=(z/K)/Sum(z/K)):  roles of L/V swapped.
      dx_m/dlnK_j = x_m * (x_j - delta_mj)
    """
    N = len(z)
    lnK = X[:N]
    K = np.exp(lnK)
    T = float(np.exp(X[N]))
    p = float(np.exp(X[N + 1]))

    if beta == 0:
        x = z
        y_un = K * z
        y_sum = float(y_un.sum())
        y = y_un / y_sum
    else:
        y = z
        x_un = z / K
        x_sum = float(x_un.sum())
        x = x_un / x_sum

    # Derivatives always needed: phase-specific T and p derivatives
    dlnphi_L_dT = dlnphi_dT_at_p(p, T, x, mixture, phase_hint='liquid')
    dlnphi_V_dT = dlnphi_dT_at_p(p, T, y, mixture, phase_hint='vapor')
    dlnphi_L_dp = dlnphi_dp_at_T(p, T, x, mixture, phase_hint='liquid')
    dlnphi_V_dp = dlnphi_dp_at_T(p, T, y, mixture, phase_hint='vapor')

    J = np.zeros((N + 2, N + 2))

    if beta == 0:
        # bubble: x=z fixed, y depends on K
        dlnphi_V_dy = dlnphi_dx_at_p(p, T, y, mixture, phase_hint='vapor')  # (N, N)
        dy_dlnK = np.diag(y) - np.outer(y, y)                                # (N, N)
        J[:N, :N] = np.eye(N) + dlnphi_V_dy @ dy_dlnK
        J[N, :N] = K * z                                                     # d(Sum(K*z))/dlnK_j
    else:
        # dew: y=z fixed, x depends on K
        dlnphi_L_dx = dlnphi_dx_at_p(p, T, x, mixture, phase_hint='liquid')  # (N, N)
        dx_dlnK = np.outer(x, x) - np.diag(x)                                # (N, N)
        J[:N, :N] = np.eye(N) - dlnphi_L_dx @ dx_dlnK
        J[N, :N] = -z / K                                                    # d(Sum(z/K))/dlnK_j

    # T and p columns
    J[:N, N] = -(dlnphi_L_dT - dlnphi_V_dT) * T
    J[:N, N + 1] = -(dlnphi_L_dp - dlnphi_V_dp) * p
    # R_{N+1} independent of T, p at fixed K
    J[N, N] = 0.0
    J[N, N + 1] = 0.0
    # R_{N+2} = X[spec_idx] - spec_val
    J[N + 1, spec_idx] = 1.0
    return J


def _converge_envelope_point(X0, beta, z, spec_idx, spec_val, mixture,
                              tol=1e-9, maxiter=30, step_cap=0.5,
                              use_analytic_jac=False):
    """Newton-Raphson to convergence at a single envelope point.

    v0.9.18: when use_analytic_jac=True, builds the (N+2) x (N+2)
    Jacobian analytically from the Helmholtz derivative primitives
    (v0.9.9 / v0.9.10). ~2-3x faster per iteration for N>=5.

    Default is FD (use_analytic_jac=False) for the same reason as cubic
    v0.9.17: the tracer's near-critical seed convergence relies on
    controlled FD noise to escape the trivial-solution basin (K=1).
    For single-point corrector calls (envelope_point) and in-tracer
    correctors, the analytic path is a strict improvement and can be
    enabled. Falls back to FD on NotImplementedError or RuntimeError
    from the derivative primitives.
    """
    X = X0.copy()
    for it in range(maxiter):
        R = _envelope_residuals(X, beta, z, spec_idx, spec_val, mixture)
        res_norm = float(np.max(np.abs(R)))
        if res_norm < tol:
            return X, it
        if use_analytic_jac:
            try:
                J = _envelope_jacobian_analytic(X, beta, z, spec_idx,
                                                 spec_val, mixture)
            except (NotImplementedError, RuntimeError):
                J = _envelope_jacobian_fd(X, beta, z, spec_idx,
                                           spec_val, mixture)
        else:
            J = _envelope_jacobian_fd(X, beta, z, spec_idx, spec_val, mixture)
        try:
            dX = -np.linalg.solve(J, R)
        except np.linalg.LinAlgError:
            raise RuntimeError(
                f"Singular Jacobian at envelope point (iter={it})"
            )
        dX_max = float(np.max(np.abs(dX)))
        if dX_max > step_cap:
            dX = dX * (step_cap / dX_max)
        X = X + dX
    raise RuntimeError(
        f"Envelope Newton did not converge after {maxiter} iters; "
        f"final residual {res_norm:.3e}"
    )


def _wilson_K(T, p, mixture):
    """Wilson K-factor correlation for Helmholtz mixture components.

    Each component has T_c, p_c, acentric_factor in its underlying
    `.fluid` attributes.
    """
    N = len(mixture.components)
    K = np.empty(N)
    for i in range(N):
        f = mixture.components[i].fluid
        K[i] = (f.p_c / p) * np.exp(
            5.373 * (1.0 + f.acentric_factor) * (1.0 - f.T_c / T)
        )
    return K


def envelope_point(T, p, z, mixture, beta=0, max_iter=20,
                   use_analytic_jac=True):
    """Converge a single envelope point at given (T, p, z). Wilson-seeded.

    beta=0: bubble (z = liquid, solve for vapor K-factors)
    beta=1: dew    (z = vapor,  solve for liquid K-factors)

    v0.9.18: the single-point corrector uses the analytic (N+2)x(N+2)
    Jacobian by default, for ~2-3x speedup per iteration on N>=5
    mixtures. Wilson-seeded states start far from the trivial K=1 basin,
    so the analytic path is robust here. Set use_analytic_jac=False to
    force FD.
    """
    z = np.asarray(z, dtype=float); z = z / z.sum()
    N = len(z)
    K_init = _wilson_K(T, p, mixture)
    X0 = np.concatenate([np.log(K_init), [np.log(T), np.log(p)]])
    spec_idx = N
    spec_val = float(np.log(T))
    X, iters = _converge_envelope_point(
        X0, beta, z, spec_idx, spec_val, mixture, maxiter=max_iter,
        use_analytic_jac=use_analytic_jac,
    )
    return {
        "T": float(np.exp(X[N])),
        "p": float(np.exp(X[N + 1])),
        "K": np.exp(X[:N]),
        "iterations": iters,
    }


# ---------------------------------------------------------------------------
# Continuation helpers (EOS-agnostic)
# ---------------------------------------------------------------------------

def _best_beta(K, z, current_beta=None, hysteresis=0.15):
    """Choose beta in {0, 1} based on K-factor magnitude asymmetry.

    Criterion: z-weighted average a = Sum(z_i * ln K_i).
    If a > 0, K's lean positive -> beta=0 (bubble). Else beta=1 (dew).
    Hysteresis prevents rapid toggling near the critical where a ~ 0.
    """
    a = float(np.dot(z, np.log(K)))
    if current_beta is None:
        return 0 if a >= 0 else 1
    if current_beta == 0:
        return 1 if a < -hysteresis else 0
    else:
        return 0 if a > hysteresis else 1


def _quadratic_predictor(X_hist, s_hist, ds_next):
    """Quadratic Lagrange extrapolation using last 3 converged points."""
    s0, s1, s2 = s_hist
    X0, X1, X2 = X_hist
    s = s2 + ds_next
    L0 = (s - s1) * (s - s2) / ((s0 - s1) * (s0 - s2))
    L1 = (s - s0) * (s - s2) / ((s1 - s0) * (s1 - s2))
    L2 = (s - s0) * (s - s1) / ((s2 - s0) * (s2 - s1))
    return L0 * X0 + L1 * X1 + L2 * X2


# ---------------------------------------------------------------------------
# Critical-seeded envelope tracer
# ---------------------------------------------------------------------------

def trace_envelope(z, mixture,
                    crit=None,
                    p_min=1e3,
                    T_min=50.0, T_max=1500.0,
                    p_max=1e10,
                    max_points_per_branch=100,
                    step_init=0.04,
                    step_max=0.10,
                    crit_offset=0.03,
                    verbose=False,
                    use_analytic_jac_corrector=False):
    """Trace the Helmholtz phase envelope at fixed composition z, seeded
    from the mixture critical point.

    Parameters
    ----------
    z : array             overall composition (will be renormalized)
    mixture : Mixture     Helmholtz/GERG mixture
    crit : dict, optional
        Output of critical_point_multistart(z, mixture). If None, computed.
    p_min, p_max : float  pressure bounds [Pa]
    T_min, T_max : float  temperature bounds [K]
    max_points_per_branch : int
    step_init : float     initial arclength step in log-space variables
    step_max : float      upper cap on step size
    crit_offset : float   initial eigenvector-perturbation magnitude
    verbose : bool

    Returns
    -------
    dict with keys T, p, K, branch (0=bubble, 1=dew, -1=critical),
    critical, n_points.
    """
    z = np.asarray(z, dtype=float); z = z / z.sum()
    N = len(z)

    if crit is None:
        crit = critical_point_multistart(z, mixture)
    T_c = crit["T_c"]
    p_c = crit["p_c"]
    if verbose:
        print(f"  Critical: T={T_c:.3f} K, p={p_c/1e6:.4f} MPa")

    X_crit = np.zeros(N + 2)
    X_crit[N] = np.log(T_c)
    X_crit[N + 1] = np.log(p_c)

    u = crit["u"]
    u = u / np.max(np.abs(u))
    if verbose:
        print(f"  Critical eigenvector u: {u}")

    all_T = [T_c]
    all_p = [p_c]
    all_K = [np.ones(N)]
    all_branch = [-1]

    for direction in [+1.0, -1.0]:
        X_converged = None
        beta = 0

        # Strategy 1 & 2: eigenvector perturbation with shrinking offset
        for offset_frac in [1.0, 0.5, 0.25, 0.1]:
            off = crit_offset * offset_frac
            X_try = X_crit.copy()
            X_try[:N] = direction * off * u
            K_seed = np.exp(X_try[:N])
            beta_try = _best_beta(K_seed, z)
            try:
                X_converged, _ = _converge_envelope_point(
                    X_try, beta_try, z, N, float(X_try[N]),
                    mixture, maxiter=30,
                )
                beta = beta_try
                if verbose:
                    K0 = np.exp(X_converged[:N])
                    print(f"  Direction {direction:+.0f}: eigenvector seed at offset={off:.3f} "
                          f"converged to T={np.exp(X_converged[N]):.3f}K, "
                          f"p={np.exp(X_converged[N+1])/1e6:.4f}MPa, "
                          f"K range=[{K0.min():.4f}, {K0.max():.4f}], beta={beta}")
                break
            except RuntimeError:
                continue

        # Strategy 3: Wilson seed fallback
        if X_converged is None:
            T_seed = T_c + direction * 5.0
            p_seed = p_c * 0.9
            K_seed = _wilson_K(T_seed, p_seed, mixture)
            X_try = np.concatenate([np.log(K_seed), [np.log(T_seed), np.log(p_seed)]])
            beta_try = _best_beta(K_seed, z)
            try:
                X_converged, _ = _converge_envelope_point(
                    X_try, beta_try, z, N, float(X_try[N]),
                    mixture, maxiter=40,
                )
                beta = beta_try
                if verbose:
                    print(f"  Direction {direction:+.0f}: Wilson fallback at T={T_seed:.2f}K "
                          f"converged to T={np.exp(X_converged[N]):.3f}K, "
                          f"p={np.exp(X_converged[N+1])/1e6:.4f}MPa, beta={beta}")
            except RuntimeError as e:
                if verbose:
                    print(f"  Direction {direction:+.0f}: Wilson fallback failed ({e}); skipping")
                continue

        X = X_converged
        spec_idx = N
        spec_val = float(X[N])

        branch_Ts = [float(np.exp(X[N]))]
        branch_ps = [float(np.exp(X[N + 1]))]
        branch_Ks = [np.exp(X[:N]).copy()]
        branch_tags = [beta]

        prev_tangent = np.zeros(N + 2)
        prev_tangent[:N] = direction * u
        prev_tangent = prev_tangent / np.linalg.norm(prev_tangent)
        step = step_init

        X_hist = [X.copy()]
        s_hist = [0.0]
        s_cum = 0.0

        for pt in range(max_points_per_branch):
            # Compute new tangent
            try:
                J = _envelope_jacobian_fd(X, beta, z, spec_idx, spec_val, mixture)
            except RuntimeError:
                # Density solver failed at the current state -- likely crossed
                # into supercritical region. Terminate this branch.
                if verbose:
                    print(f"  direction {direction:+.0f} pt {pt}: density solver failed at current state, terminating branch")
                break
            e = np.zeros(N + 2); e[N + 1] = 1.0
            try:
                sens = np.linalg.solve(J, e)
            except np.linalg.LinAlgError:
                if verbose:
                    print(f"  direction {direction:+.0f} pt {pt}: Jacobian singular, stopping")
                break
            t = sens / np.linalg.norm(sens)
            if float(np.dot(t, prev_tangent)) < 0:
                t = -t
            prev_tangent = t

            spec_idx_new = int(np.argmax(np.abs(t)))

            use_quadratic = False
            if len(X_hist) >= 3:
                span = s_hist[-1] - s_hist[-3]
                if step < span:
                    use_quadratic = True

            if use_quadratic:
                ds_next = step
                X_pred = _quadratic_predictor(X_hist[-3:], s_hist[-3:], ds_next)
                spec_val_new = float(X_pred[spec_idx_new])
                dX_predicted_norm = np.linalg.norm(X_pred - X)
            else:
                dX = step * t
                X_pred = X + dX
                spec_val_new = float(X[spec_idx_new] + dX[spec_idx_new])
                dX_predicted_norm = np.linalg.norm(dX)

            try:
                X_new, iters = _converge_envelope_point(
                    X_pred, beta, z, spec_idx_new, spec_val_new, mixture,
                    maxiter=15,
                    use_analytic_jac=use_analytic_jac_corrector,
                )
            except RuntimeError:
                step *= 0.5
                if step < 1e-5:
                    if verbose:
                        print(f"  direction {direction:+.0f} pt {pt}: step collapsed")
                    break
                continue

            corrector_jump = np.linalg.norm(X_new - X)
            if corrector_jump > 3.0 * max(dX_predicted_norm, 1e-3):
                step *= 0.5
                if step < 1e-6:
                    if verbose:
                        print(f"  direction {direction:+.0f} pt {pt}: rejected big jump")
                    break
                continue

            if len(X_hist) >= 2:
                prev_Tp_jump = np.linalg.norm(X_hist[-1][N:] - X_hist[-2][N:])
                new_Tp_jump = np.linalg.norm(X_new[N:] - X[N:])
                threshold = 3.0 if len(X_hist) >= 4 else 5.0
                if new_Tp_jump > threshold * max(prev_Tp_jump, 0.005):
                    step *= 0.5
                    if step < 1e-6:
                        if verbose:
                            print(f"  direction {direction:+.0f} pt {pt}: T,p jump rejected")
                        break
                    continue

            # Accept this point
            X = X_new
            spec_idx = spec_idx_new
            spec_val = spec_val_new

            ds_actual = float(np.linalg.norm(X - X_hist[-1]))
            s_cum += ds_actual
            X_hist.append(X.copy())
            s_hist.append(s_cum)
            if len(X_hist) > 3:
                X_hist.pop(0); s_hist.pop(0)

            K_new = np.exp(X[:N])
            branch_Ts.append(float(np.exp(X[N])))
            branch_ps.append(float(np.exp(X[N + 1])))
            branch_Ks.append(K_new.copy())
            branch_tags.append(beta)

            best_beta = _best_beta(K_new, z, current_beta=beta)
            if best_beta != beta:
                if verbose:
                    a = float(np.dot(z, np.log(K_new)))
                    print(f"  direction {direction:+.0f} pt {pt}: beta switch "
                          f"{beta}->{best_beta} (z-wtd ln K = {a:+.3f})")
                beta = best_beta

            # Step adaptation based on predictor-corrector agreement
            corrector_correction = np.linalg.norm(X_new - X_pred)
            predictor_size = max(dX_predicted_norm, 1e-6)
            ratio = corrector_correction / predictor_size
            if ratio < 0.1:
                step = min(step * 1.5, step_max)
            elif ratio < 0.3:
                step = min(step * 1.2, step_max)
            elif ratio < 0.6:
                pass
            else:
                step *= 0.7

            T_cur = np.exp(X[N])
            p_cur = np.exp(X[N + 1])
            if T_cur < T_min or T_cur > T_max or p_cur < p_min or p_cur > p_max:
                if verbose:
                    print(f"  direction {direction:+.0f} pt {pt}: hit bound "
                          f"T={T_cur:.1f}, p={p_cur/1e3:.1f}kPa, stopping")
                break

            if verbose and pt % 10 == 0:
                print(f"  direction {direction:+.0f} pt {pt}: T={T_cur:.2f}K, "
                      f"p={p_cur/1e6:.4f}MPa, beta={beta}, iters={iters}, step={step:.3f}")

        if direction < 0:
            all_T = branch_Ts[::-1] + all_T
            all_p = branch_ps[::-1] + all_p
            all_K = branch_Ks[::-1] + all_K
            all_branch = branch_tags[::-1] + all_branch
        else:
            all_T = all_T + branch_Ts
            all_p = all_p + branch_ps
            all_K = all_K + branch_Ks
            all_branch = all_branch + branch_tags

    return {
        "T": np.array(all_T),
        "p": np.array(all_p),
        "K": np.array(all_K),
        "branch": np.array(all_branch),
        "critical": crit,
        "n_points": len(all_T),
    }
