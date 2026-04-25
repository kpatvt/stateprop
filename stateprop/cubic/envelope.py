"""Phase envelope tracing for cubic EOS mixtures.

Traces the bubble/dew envelope at fixed composition z by seeding from the
mixture critical point and stepping outward along the envelope tangent in
both directions. This avoids the critical-point turning-point problem
that Michelsen-style low-p-seeded tracing runs into.

Algorithm:
1. Call critical_point(z, mixture) to get (T_c, p_c, V_c) exactly.
2. At the critical, use the null eigenvector u of the A^res matrix as
   the composition-perturbation direction for the first step.
3. For each of direction in {+1, -1}:
   a. Multi-strategy seeding: try eigenvector perturbation at decreasing
      `crit_offset` values; if all fail, fall back to Wilson-seeded point
      just below the critical.
   b. Adaptive continuation: quadratic predictor when history >= 3 points
      and extrapolation is within span; linear tangent predictor otherwise.
   c. Adaptive beta-switch (bubble=0 vs dew=1) based on z-weighted ln K
      asymmetry with hysteresis to prevent chattering near the critical.
   d. Step control from predictor-corrector agreement: tight agreement
      -> grow step; poor agreement -> shrink.
   e. Continuity checks reject corrections that jump suspiciously far in
      the full X-space or in the (T, p) subspace.
4. Combine: [dew points reversed] + [critical] + [bubble points] gives
   the full envelope from low-p dew to low-p bubble.

Variables at each point: X = (ln K_1..N, ln T, ln p).
Residuals (N+2):
    R_i = ln K_i - (ln phi_i^L - ln phi_i^V), i=1..N
    R_N = Sigma(y) - 1 (bubble) or Sigma(x) - 1 (dew)
    R_{N+1} = X[spec_idx] - spec_target

Every point recorded in the result satisfies the Rachford-Rice residual
for its labeled branch (beta=0 or 1) to ~1e-9.
"""
import numpy as np

from .mixture import CubicMixture
from .critical import critical_point


# ---------------------------------------------------------------------------
# Core residuals and Jacobian (shared with single-point solve)
# ---------------------------------------------------------------------------

def _envelope_residuals(X, beta, z, spec_idx, spec_val, mixture):
    """Return N+2 residuals at state X."""
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

    rho_L = mixture.density_from_pressure(p, T, x, phase_hint='liquid')
    rho_V = mixture.density_from_pressure(p, T, y, phase_hint='vapor')
    lnphi_L = mixture.ln_phi(rho_L, T, x)
    lnphi_V = mixture.ln_phi(rho_V, T, y)

    R = np.empty(N + 2)
    R[:N] = lnK - (lnphi_L - lnphi_V)
    R[N] = rr_residual
    R[N + 1] = X[spec_idx] - spec_val
    return R


def _envelope_jacobian_fd(X, beta, z, spec_idx, spec_val, mixture, eps=1e-6):
    """Central-difference Jacobian."""
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
    """Analytic (N+2) x (N+2) Jacobian of the envelope residuals.

    Built from the v0.9.8 / v0.9.10 cubic analytic derivatives of ln phi:
    `dlnphi_dxk_at_p`, `dlnphi_dp_at_T`, `dlnphi_dT_at_p`. As of v0.9.16
    these support Peneloux volume-shifted mixtures too.

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

    # Derivatives we always need: phase-specific T and p derivatives.
    dlnphi_L_dT = mixture.dlnphi_dT_at_p(p, T, x, phase_hint='liquid')
    dlnphi_V_dT = mixture.dlnphi_dT_at_p(p, T, y, phase_hint='vapor')
    dlnphi_L_dp = mixture.dlnphi_dp_at_T(p, T, x, phase_hint='liquid')
    dlnphi_V_dp = mixture.dlnphi_dp_at_T(p, T, y, phase_hint='vapor')

    J = np.zeros((N + 2, N + 2))

    if beta == 0:
        # bubble: x=z fixed (no composition Jacobian on liquid side);
        # y depends on K.
        dlnphi_V_dy = mixture.dlnphi_dxk_at_p(p, T, y, phase_hint='vapor')  # (N, N)
        dy_dlnK = np.diag(y) - np.outer(y, y)                                # (N, N)
        # dR_i/dlnK_j = delta_ij + sum_m (dlnphi_V_dy @ dy_dlnK)_ij
        # (-lnphi_V in R -> +dlnphi_V, but we have R_i = lnK - (lnphi_L - lnphi_V)
        #  so dR_i/dy_m = dlnphi_V/dy_m, and dy_m/dlnK_j gives the final chain rule.
        #  Wait: dR_i/dlnK_j = d(lnK_i)/dlnK_j - 0 + dlnphi_V_i/dy_m * dy_m/dlnK_j)
        J[:N, :N] = np.eye(N) + dlnphi_V_dy @ dy_dlnK
        J[N, :N] = K * z                                                     # d(Sum(K*z))/dlnK_j = K_j z_j
    else:
        # dew: y=z fixed; x depends on K.
        dlnphi_L_dx = mixture.dlnphi_dxk_at_p(p, T, x, phase_hint='liquid')  # (N, N)
        dx_dlnK = np.outer(x, x) - np.diag(x)                                # (N, N)
        # R_i = lnK_i - (lnphi_L - lnphi_V), lnphi_L depends on x(K), lnphi_V fixed in y.
        # dR_i/dlnK_j = delta_ij - (dlnphi_L_dx @ dx_dlnK)_ij
        J[:N, :N] = np.eye(N) - dlnphi_L_dx @ dx_dlnK
        J[N, :N] = -z / K                                                    # d(Sum(z/K))/dlnK_j = -z_j/K_j

    # T and p columns are the same structure for both branches
    J[:N, N] = -(dlnphi_L_dT - dlnphi_V_dT) * T
    J[:N, N + 1] = -(dlnphi_L_dp - dlnphi_V_dp) * p
    # R_{N+1} is independent of T and p at fixed K
    J[N, N] = 0.0
    J[N, N + 1] = 0.0
    # R_{N+2} = X[spec_idx] - spec_val
    J[N + 1, spec_idx] = 1.0
    return J


def _converge_envelope_point(X0, beta, z, spec_idx, spec_val, mixture,
                              tol=1e-9, maxiter=30, step_cap=0.5,
                              use_analytic_jac=False):
    """Newton-Raphson to convergence at a single envelope point.

    v0.9.17: when use_analytic_jac=True, builds the (N+2) x (N+2)
    Jacobian analytically from the cubic derivative primitives
    (v0.9.8 / v0.9.10 / v0.9.16) rather than via central-difference FD.
    This is ~5x faster per iteration when N is large because the FD
    path requires 2(N+2) residual evaluations per Jacobian.

    The default is FD (use_analytic_jac=False) because the tracer's
    near-critical seed convergence depends on controlled imprecision
    from FD noise to escape the trivial-solution basin (K = 1). For
    bulk-state corrector calls in a pre-seeded tracer, analytic is a
    strict improvement and can be enabled. `envelope_point` (single
    Wilson-seeded point far from the trivial root) also benefits from
    the analytic path.

    Falls back to FD on NotImplementedError or RuntimeError from the
    derivative primitives.
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
    """Wilson K-factor correlation."""
    N = mixture.N
    K = np.empty(N)
    for i in range(N):
        c = mixture.components[i]
        K[i] = (c.p_c / p) * np.exp(
            5.373 * (1.0 + c.acentric_factor) * (1.0 - c.T_c / T)
        )
    return K


def envelope_point(T, p, z, mixture, beta=0, max_iter=20,
                   use_analytic_jac=True):
    """Converge a single envelope point at given (T, p, z). Wilson-seeded.

    beta=0: bubble (z = liquid, solve for vapor K-factors)
    beta=1: dew    (z = vapor,  solve for liquid K-factors)

    v0.9.17: single-point corrector uses the analytic (N+2)x(N+2)
    Jacobian by default, for ~5x speedup per iteration. Wilson-seeded
    states start far from the trivial K=1 basin, so the analytic path
    is robust here. Set use_analytic_jac=False to force FD.
    """
    z = np.asarray(z, dtype=float); z = z / z.sum()
    N = mixture.N
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
# Tangent at the critical point
# ---------------------------------------------------------------------------

def _tangent_at(X, beta, z, mixture, spec_idx=None):
    """Compute unit tangent to the envelope at state X.

    The envelope is the 1D manifold defined by the first N+1 residuals
    (fugacity + Rachford-Rice). Its tangent is the null vector of the
    (N+1)x(N+2) Jacobian. We compute it via bordered inversion: augment
    with a spec equation, solve for dX/d(spec_val), and normalize.

    If spec_idx is None, choose the variable with smallest sensitivity
    (gives best-conditioned system).
    """
    N = len(z)
    # Pick spec_idx by trial if not given: start with ln p
    if spec_idx is None:
        spec_idx = N + 1
    spec_val = float(X[spec_idx])
    J = _envelope_jacobian_fd(X, beta, z, spec_idx, spec_val, mixture)
    e = np.zeros(N + 2)
    e[N + 1] = 1.0    # d/d(spec_val) of the spec residual is -1 after flip... wait
    # R_{N+1} = X[spec_idx] - spec_val, so dR_{N+1}/d(spec_val) = -1
    # Solving J dX/d(spec_val) = -dR/d(spec_val) = e_{N+1}? Let me redo.
    # We want the null vector of rows 0..N of J (the non-spec residuals).
    # Equivalently: for infinitesimal change ds in spec_val, dX/ds satisfies
    # R(X + dX/ds * ds, spec_val + ds) = 0, so sum(dR/dX_k * dX_k/ds) + dR/d(spec_val) = 0.
    # The only row with dR/d(spec_val) != 0 is row N+1, which has value -1.
    # So J * (dX/ds) = e_{N+1} where e_{N+1} = (0,0,...,1).
    tangent = np.linalg.solve(J, e)
    norm = np.linalg.norm(tangent)
    return tangent / norm


# ---------------------------------------------------------------------------
# Critical-seeded envelope tracer
# ---------------------------------------------------------------------------

def _best_beta(K, z, current_beta=None, hysteresis=0.15):
    """Choose beta in {0, 1} based on K-factor magnitude asymmetry.

    At the envelope both Sum(K*z) and Sum(z/K) equal 1, so comparing them
    is meaningless. The relevant question is which formulation has a
    better-conditioned Jacobian in the NEIGHBORHOOD of the current point.

    When most ln K_i are positive (K > 1), small K values (K ~ 1) cause
    z/K to be close to z and Sum(z/K)-1 becomes insensitive to K errors
    near 1 -- beta=1 is ill-conditioned, use beta=0.
    Conversely, when most ln K_i are negative (K < 1), beta=1 is better.

    Criterion: compute weighted ln K asymmetry a = Sum(z_i * ln K_i).
    If a > 0, K's lean positive -> beta=0. Else beta=1. Hysteresis
    prevents rapid toggling near the critical where a ~ 0.

    current_beta : optional int
        If given, only switch when |a| > hysteresis (prevents chattering).
    hysteresis : float
        Threshold in ln K space. For z-weighted average: if currently beta=0
        and a > -hysteresis, stay at 0; switch only if a < -hysteresis.
    """
    a = float(np.dot(z, np.log(K)))
    if current_beta is None:
        return 0 if a >= 0 else 1
    # With hysteresis
    if current_beta == 0:
        # Switch to 1 only if K clearly leans negative
        return 1 if a < -hysteresis else 0
    else:
        # Switch to 0 only if K clearly leans positive
        return 0 if a > hysteresis else 1


def _quadratic_predictor(X_hist, s_hist, ds_next):
    """Quadratic Lagrange extrapolation using last 3 converged points.

    X_hist : list of 3 converged X vectors, oldest first
    s_hist : list of 3 arc-length values (cumulative), matching X_hist
    ds_next : arc-length step from X_hist[-1] to predict
    Returns: predicted X at s_hist[-1] + ds_next
    """
    s0, s1, s2 = s_hist
    X0, X1, X2 = X_hist
    s = s2 + ds_next
    # Lagrange basis polynomials evaluated at s
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
                    step_init=0.04,          # smaller default for robustness
                    step_max=0.10,
                    crit_offset=0.03,        # smaller default seed offset
                    verbose=False,
                    use_analytic_jac_corrector=False):
    """Trace the phase envelope at fixed composition z, seeded from critical.

    Starts at the mixture critical point (computed via v0.5.0 critical_point
    solver) and walks outward along the envelope tangent in two directions.
    This avoids the critical-point turning-point problem that plagues
    low-p-seeded Michelsen continuation.

    **Robustness caveats:**

    The near-critical portion of the envelope (say within T_c +/- 30 K and
    p_c / 2 to p_c) is reliably traced. Further from the critical:

    - The bubble-side trace (z as liquid) typically works well down to low
      pressure and temperature.
    - The dew-side trace (z as vapor) has numerical issues when K-factors
      are close to 1: the (Sum(z/K) - 1) residual becomes insensitive to
      K-factor errors, and the tracer can wander to spurious states.

    For quantitatively reliable points, use `envelope_point(T, p, z, mixture)`
    with specific (T, p) targets. Use `trace_envelope` for visualization of
    the near-critical region or as a qualitative scan.

    Parameters
    ----------
    z : array
        Overall composition.
    mixture : CubicMixture
    crit : dict, optional
        Output of critical_point(z, mixture). If None, computed here.
    p_min, p_max : float
        Pressure bounds [Pa] -- tracer stops when crossed.
    T_min, T_max : float
        Temperature bounds [K].
    max_points_per_branch : int
    step_init : float
        Initial arc-length step size in ln-variables. Smaller values
        (0.03-0.05) give smoother envelopes but take more points; larger
        (0.1+) is faster but more prone to jumps.
    step_max : float
        Upper cap on step size.
    crit_offset : float
        How far to step off the exact critical along the instability
        eigenvector for the first point.
    verbose : bool
    use_analytic_jac_corrector : bool, default False
        v0.9.17: when True, the arc-length corrector uses the analytic
        (N+2)x(N+2) Jacobian from the cubic derivative primitives instead
        of central-difference FD. This is ~3-5x faster per corrector call
        and tightens fugacity closure by the final iteration. Kept off by
        default because the seed step (first converged point on each
        direction) still benefits from FD imprecision to escape the
        trivial-solution basin near the critical; the corrector, which
        operates on pre-converged envelope points, has no such issue.

    Returns
    -------
    dict with keys:
        T : array of temperatures [K]
        p : array of pressures [Pa]
        K : (n_points, N) array of K-factors
        branch : array of ints, 0=bubble (z=liquid), 1=dew (z=vapor), -1=critical
        critical : dict (same as critical_point output)
        n_points : int
    """
    z = np.asarray(z, dtype=float); z = z / z.sum()
    N = mixture.N

    # 1. Compute critical point if not provided
    if crit is None:
        crit = critical_point(z, mixture)
    T_c = crit["T_c"]
    p_c = crit["p_c"]
    if verbose:
        print(f"  Critical: T={T_c:.3f} K, p={p_c/1e6:.4f} MPa")

    # At exact critical, ln K_i = 0 for all i. Build the X vector there.
    X_crit = np.zeros(N + 2)
    X_crit[N] = np.log(T_c)
    X_crit[N + 1] = np.log(p_c)

    # Starting direction at the critical is the null eigenvector u of the
    # critical-point A^res matrix. This is physically the composition-
    # perturbation direction along which phase instability first appears.
    # In X-space, the tangent at the critical has ln K components
    # proportional to u (and vanishing T, p components at lowest order).
    u = crit["u"]
    # Normalize so max|u_i| = 1 -- gives consistent step magnitudes
    u = u / np.max(np.abs(u))

    if verbose:
        print(f"  Critical eigenvector u: {u}")

    # Trace each direction. Branch labeling:
    # - direction_sign = +1 : the side where ln K_1 > 0 initially (bubble-like: y > x in heavy comp)
    #   Actually there's no universal convention; we'll just label the two
    #   halves as 0 (bubble-convention) and 1 (dew-convention) based on
    #   which side has Sigma(K*z) > 1 initially.
    branches = []        # 0 or 1
    all_T = [T_c]
    all_p = [p_c]
    all_K = [np.ones(N)]
    all_branch = [-1]    # -1 = critical

    for direction in [+1.0, -1.0]:
        # Seed strategy (in order of preference):
        # 1. Eigenvector perturbation at `crit_offset`: X = X_crit + dir*offset*u
        # 2. Same with progressively smaller offsets (halve up to 4x)
        # 3. Wilson-seeded point at T = T_c - 5, p = p_c * 0.9 (or 1.1 for dir=-1)
        #
        # The eigenvector approach is preferred when it works (clean
        # continuation from the critical), but fails on some compositions
        # where the u-direction is not well-aligned with the envelope tangent
        # at finite offset. The Wilson fallback gives a reliable "somewhere
        # on the envelope near the critical" seed.
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
            # Perturb T away from T_c, keep p close to p_c
            T_seed = T_c + direction * 5.0
            p_seed = p_c * 0.9
            K_seed = _wilson_K(T_seed, p_seed, mixture)
            X_try = np.concatenate([np.log(K_seed), [np.log(T_seed), np.log(p_seed)]])
            # Pick beta from Wilson K-factor magnitudes
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
                    print(f"  Direction {direction:+.0f}: Wilson fallback failed ({e}); skipping branch")
                continue

        X = X_converged
        spec_idx = N
        spec_val = float(X[N])

        # Record this seed
        branch_Ts = [float(np.exp(X[N]))]
        branch_ps = [float(np.exp(X[N + 1]))]
        branch_Ks = [np.exp(X[:N]).copy()]
        branch_tags = [beta]

        # Continuation: arc-length stepping outward from critical
        # Initial tangent: ln-K direction is direction*u, T,p slots zero.
        # This is the first-order tangent at the critical; the sign-continuity
        # check will flip subsequent tangents to align with this.
        prev_tangent = np.zeros(N + 2)
        prev_tangent[:N] = direction * u
        prev_tangent = prev_tangent / np.linalg.norm(prev_tangent)
        step = step_init

        # Rolling history for quadratic predictor (oldest first)
        X_hist = [X.copy()]
        s_hist = [0.0]           # cumulative arc length from seed
        s_cum = 0.0

        for pt in range(max_points_per_branch):
            # Compute new tangent
            J = _envelope_jacobian_fd(X, beta, z, spec_idx, spec_val, mixture)
            e = np.zeros(N + 2); e[N + 1] = 1.0
            try:
                sens = np.linalg.solve(J, e)
            except np.linalg.LinAlgError:
                if verbose:
                    print(f"  direction {direction:+.0f} pt {pt}: Jacobian singular, stopping")
                break
            t = sens / np.linalg.norm(sens)
            # Sign continuity
            if float(np.dot(t, prev_tangent)) < 0:
                t = -t
            prev_tangent = t

            # Select new spec: variable with largest tangent component
            spec_idx_new = int(np.argmax(np.abs(t)))

            # Predictor: use quadratic extrapolation when we have enough
            # history (3 points) AND extrapolation is within the span of
            # the history. Otherwise fall back to linear tangent predictor.
            use_quadratic = False
            if len(X_hist) >= 3:
                span = s_hist[-1] - s_hist[-3]
                # Only use quadratic when extrapolating LESS than the span;
                # otherwise the fit is unreliable.
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

            # Corrector
            try:
                X_new, iters = _converge_envelope_point(
                    X_pred, beta, z, spec_idx_new, spec_val_new, mixture,
                    maxiter=15,
                    use_analytic_jac=use_analytic_jac_corrector,
                )
            except RuntimeError:
                # Shrink step and retry
                step *= 0.5
                if step < 1e-5:
                    if verbose:
                        print(f"  direction {direction:+.0f} pt {pt}: step collapsed")
                    break
                continue

            # Sanity: reject steps that jump much further than the predictor
            # intended. Quadratic predictor should already track the envelope
            # well, so a tighter threshold is justified.
            corrector_jump = np.linalg.norm(X_new - X)
            if corrector_jump > 3.0 * max(dX_predicted_norm, 1e-3):
                step *= 0.5
                if step < 1e-6:
                    if verbose:
                        print(f"  direction {direction:+.0f} pt {pt}: rejected big jump, step collapsed")
                    break
                continue

            # Additional continuity check: reject corrections that produce a
            # (T, p) change more than 3x the (T, p) change of the last
            # successful step. This catches cases where the corrector
            # converged to a distant envelope point rather than the nearby
            # continuation. Only apply once we have at least one prior step.
            if len(X_hist) >= 2:
                prev_Tp_jump = np.linalg.norm(X_hist[-1][N:] - X_hist[-2][N:])
                new_Tp_jump = np.linalg.norm(X_new[N:] - X[N:])
                # Reject corrections that make a sudden large (T, p) jump
                # compared to the prior step. Looser early when step is
                # still calibrating.
                threshold = 3.0 if len(X_hist) >= 4 else 5.0
                if new_Tp_jump > threshold * max(prev_Tp_jump, 0.005):
                    step *= 0.5
                    if step < 1e-6:
                        if verbose:
                            print(f"  direction {direction:+.0f} pt {pt}: rejected T,p jump (new={new_Tp_jump:.3f}, prev={prev_Tp_jump:.3f}), step collapsed")
                        break
                    continue

            # Accept this point
            X = X_new
            spec_idx = spec_idx_new
            spec_val = spec_val_new

            # Update arc length (Euclidean norm of actual step in X-space)
            ds_actual = float(np.linalg.norm(X - X_hist[-1]))
            s_cum += ds_actual
            X_hist.append(X.copy())
            s_hist.append(s_cum)
            # Keep only last 3 for quadratic predictor
            if len(X_hist) > 3:
                X_hist.pop(0); s_hist.pop(0)

            # Record this point FIRST with the beta that was used to
            # converge it (so the RR residual matches the label).
            K_new = np.exp(X[:N])
            branch_Ts.append(float(np.exp(X[N])))
            branch_ps.append(float(np.exp(X[N + 1])))
            branch_Ks.append(K_new.copy())
            branch_tags.append(beta)

            # THEN decide whether to switch beta for the NEXT iteration,
            # with hysteresis to prevent chattering near the critical
            # where K-factor asymmetry is near zero.
            best_beta = _best_beta(K_new, z, current_beta=beta)
            if best_beta != beta:
                if verbose:
                    a = float(np.dot(z, np.log(K_new)))
                    print(f"  direction {direction:+.0f} pt {pt}: beta switch {beta}->{best_beta} "
                          f"(z-wtd ln K = {a:+.3f})")
                beta = best_beta

            # Adapt step size based on predictor-corrector AGREEMENT.
            # When the corrector lands close to the predictor, the model
            # (linear tangent or quadratic) tracks the envelope well and we
            # can grow the step. When they disagree, shrink.
            #
            # This is more stable than iteration-count heuristics which
            # can oscillate: easy convergence grows step, next step
            # overshoots, gets rejected, step halves, repeat forever.
            corrector_correction = np.linalg.norm(X_new - X_pred)
            predictor_size = max(dX_predicted_norm, 1e-6)
            ratio = corrector_correction / predictor_size
            if ratio < 0.1:
                # Predictor was excellent; grow aggressively
                step = min(step * 1.5, step_max)
            elif ratio < 0.3:
                step = min(step * 1.2, step_max)
            elif ratio < 0.6:
                pass  # keep step
            else:
                step *= 0.7

            # Termination: hit bounds
            T_cur = np.exp(X[N])
            p_cur = np.exp(X[N + 1])
            if T_cur < T_min or T_cur > T_max or p_cur < p_min or p_cur > p_max:
                if verbose:
                    print(f"  direction {direction:+.0f} pt {pt}: hit bound T={T_cur:.1f}, p={p_cur/1e3:.1f}kPa, stopping")
                break

            if verbose and pt % 10 == 0:
                print(f"  direction {direction:+.0f} pt {pt}: T={T_cur:.2f}K, p={p_cur/1e6:.4f}MPa, "
                      f"beta={beta}, iters={iters}, step={step:.3f}")

        # Append this branch to results. If direction == -1, reverse so that
        # the full envelope reads smoothly from low-p dew -> critical -> low-p bubble.
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

