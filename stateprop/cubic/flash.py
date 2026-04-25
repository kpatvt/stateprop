"""
Phase-equilibrium flash calculations for cubic-EOS mixtures.

This module provides:
  - Rachford-Rice (generic, reused from Helmholtz mixture)
  - Michelsen tangent-plane stability analysis, cubic-adapted
  - PT flash (two-phase + single-phase, with stability pre-check)

The algorithms are essentially the same as for the Helmholtz-mixture stack;
only the underlying ln_phi and density_from_pressure come from the cubic
mixture object. We duplicate the core iteration logic here rather than
refactoring the Helmholtz flash code, keeping the two paths independent.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


# -------------------------------------------------------------------------
# Reuse Rachford-Rice from the Helmholtz path
# -------------------------------------------------------------------------
from stateprop.mixture.flash import rachford_rice


@dataclass
class CubicFlashResult:
    """Result dataclass from a cubic mixture flash."""
    phase: str           # 'vapor', 'liquid', 'two_phase', 'supercritical'
    T: float
    p: float
    beta: Optional[float]   # vapor fraction; None for single-phase
    x: np.ndarray           # liquid composition (= z if single-phase liquid/vapor)
    y: np.ndarray           # vapor composition  (= z if single-phase)
    z: np.ndarray
    rho: float              # overall molar density
    rho_L: Optional[float]  # liquid density (2-phase only)
    rho_V: Optional[float]  # vapor density
    h: float                # NOT COMPUTED yet; 0.0 placeholder
    s: float                # NOT COMPUTED yet; 0.0 placeholder
    iterations: int
    K: Optional[np.ndarray] # K-factors at convergence


# -------------------------------------------------------------------------
# SS + Broyden hybrid solver, shared between flash and stability test
# -------------------------------------------------------------------------
#
# Both the cubic two-phase PT flash and the cubic Michelsen stability test
# solve a fixed-point equation of the form:
#
#     u_new = g(u)              (SS update)
#
# where u is either ln K (flash) or ln W (stability). Equivalently, they
# drive the residual F(u) = u - g(u) to zero. Pure successive substitution
# is linearly convergent at rate ~|d g / d u|; for strongly non-ideal
# systems this rate approaches 1 and SS needs 25+ iterations to converge.
#
# Broyden's "good" method maintains a rank-1 secant approximation of the
# inverse Jacobian H ≈ (∂F/∂u)^-1. Each iter costs the same as one SS step
# (one residual evaluation), but convergence becomes super-linear after a
# short warm-up phase. Initial H = I makes the first Broyden step IS a
# pure SS step (since -H @ F = -F = u_old - g(u_old) - u_old + g(u_old)
# wait let me redo: -F = -(u - g(u)) = g(u) - u, so u_new = u + (-F) =
# g(u), the SS update). Zero setup overhead at the SS->Broyden handoff.
#
# This is the same accelerator added to the Helmholtz mixture flash in
# v0.9.5 / v0.9.6; ported here for the cubic path in v0.9.7.

_SS_WARMUP = 4          # SS iters before switching to Broyden


def _ss_broyden_solve(u0, residual_fn, ss_step_fn, tol, maxiter):
    """Solve F(u) = 0 via SS warm-up then Broyden's "good" method.

    Parameters
    ----------
    u0 : ndarray, shape (N,)
        Initial guess for u (typically ln K from Wilson, or ln W from
        the stability test seed).
    residual_fn : callable u -> (F, aux) | None
        Returns (F, aux) where F is the residual vector and aux is any
        extra state the caller wants to retain at convergence. Returns
        None to signal a transient failure (e.g. density solve diverged).
    ss_step_fn : callable (u, aux) -> u_new
        Computes the pure-SS update from the current iterate's state.
        Used both during the warm-up phase and as a fallback when a
        Broyden step would otherwise stall.
    tol : float
        Convergence threshold on max |F|.
    maxiter : int
        Total iteration cap.

    Returns
    -------
    (u, aux, niter, status) where status is one of:
        "converged"  : ||F||_inf < tol
        "maxiter"    : exhausted budget without converging
        "failed"     : residual_fn returned None twice in a row
    """
    N = u0.shape[0]
    u = u0.copy()
    aux = None

    # ---- SS warm-up ----
    for it in range(min(_SS_WARMUP, maxiter)):
        res = residual_fn(u)
        if res is None:
            return u, aux, it, "failed"
        F, aux = res
        if np.max(np.abs(F)) < tol:
            return u, aux, it + 1, "converged"
        u = ss_step_fn(u, aux)

    # ---- Broyden phase ----
    # Re-evaluate F at the post-SS-update u to get a state consistent for
    # the secant updates.
    res = residual_fn(u)
    if res is None:
        return u, aux, _SS_WARMUP, "failed"
    F, aux = res
    if np.max(np.abs(F)) < tol:
        return u, aux, _SS_WARMUP + 1, "converged"

    H = np.eye(N)                     # inverse Jacobian estimate
    F_prev = F.copy()
    u_prev = u.copy()
    for it in range(_SS_WARMUP + 1, maxiter):
        # Broyden step: delta = -H @ F (with damping)
        delta = -H @ F_prev
        max_step = np.max(np.abs(delta))
        if max_step > 1.0:
            delta = delta / max_step
        u = u_prev + delta
        res = residual_fn(u)
        if res is None:
            # Broyden step took us into a bad region. Fall back to one SS
            # step from the last good state and reset H.
            u = ss_step_fn(u_prev, aux)
            res = residual_fn(u)
            if res is None:
                return u, aux, it, "failed"
            H = np.eye(N)
        F, aux = res
        if np.max(np.abs(F)) < tol:
            return u, aux, it + 1, "converged"
        # Broyden's "good" rank-1 update: H_new = H + (s - Hy) sH / (sH y)
        s = u - u_prev
        y_vec = F - F_prev
        Hy = H @ y_vec
        sH = s @ H
        denom = s @ Hy
        if abs(denom) > 1e-30:
            H = H + np.outer(s - Hy, sH) / denom
        u_prev = u.copy()
        F_prev = F.copy()

    return u, aux, maxiter, "maxiter"


# -------------------------------------------------------------------------
# Michelsen TPD stability analysis
# -------------------------------------------------------------------------

def stability_test_TPD(z, T, p, mixture, tol=1e-8, maxiter=50, verbose=False):
    """Michelsen tangent-plane stability test for a cubic mixture.

    Returns (stable, K_best, max_S_minus_1).

    See stateprop.mixture.stability.stability_test_TPD for algorithm details;
    the only difference here is that ln_phi and density_from_pressure come
    from the cubic mixture object.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()
    N = mixture.N

    # Reference feed: Gibbs-minimizing density
    try:
        rho_z_vap = mixture.density_from_pressure(p, T, z, phase_hint='vapor')
    except RuntimeError:
        rho_z_vap = None
    try:
        rho_z_liq = mixture.density_from_pressure(p, T, z, phase_hint='liquid')
    except RuntimeError:
        rho_z_liq = None
    if rho_z_vap is None and rho_z_liq is None:
        raise RuntimeError("no density solution for feed")
    if rho_z_vap is None:
        rho_z = rho_z_liq
    elif rho_z_liq is None:
        rho_z = rho_z_vap
    elif abs(rho_z_vap - rho_z_liq) / max(rho_z_vap, rho_z_liq) < 1e-4:
        rho_z = rho_z_vap     # single root
    else:
        # Pick lower Gibbs energy (g = ln phi + ln x + ln p for each component;
        # for equal composition z in both phases, the feed's chemical potential
        # is simply R T * (ln z + ln phi + ln p), so compare sum x (ln phi))
        lpv = mixture.ln_phi(rho_z_vap, T, z)
        lpl = mixture.ln_phi(rho_z_liq, T, z)
        # Gibbs excess = sum x * ln phi (p, T cancel out since same z)
        g_v = float(np.dot(z, lpv))
        g_l = float(np.dot(z, lpl))
        rho_z = rho_z_vap if g_v < g_l else rho_z_liq

    lnphi_z = mixture.ln_phi(rho_z, T, z)
    d = np.log(z + 1e-300) + lnphi_z

    K_wilson = mixture.wilson_K(T, p)

    max_S_minus_1 = -np.inf
    best_K = K_wilson.copy()
    unstable = False

    for direction in ("vapor", "liquid"):
        if direction == "vapor":
            W = z * K_wilson.copy()
        else:
            W = z / K_wilson.copy()
        phase_hint = direction

        trivial = False
        converged = False

        # Trial-phase iteration: SS warm-up then Broyden's "good" method
        # on ln W. Same pattern as flash's _ss_broyden_solve, but with the
        # extra trivial-solution check (y ~= z, S ~= 1) interleaved in
        # both phases. Helps strongly non-ideal systems where pure SS
        # would otherwise need 25+ iters; small overhead on mild systems.

        def _residual(ln_W):
            W_curr = np.exp(ln_W)
            S_ = W_curr.sum()
            if S_ <= 0 or not np.all(np.isfinite(W_curr)):
                return None
            y_ = W_curr / S_
            try:
                rho_W_ = mixture.density_from_pressure(p, T, y_, phase_hint=phase_hint)
            except RuntimeError:
                return None
            lnphi_W_ = mixture.ln_phi(rho_W_, T, y_)
            F_ = ln_W - (d - lnphi_W_)
            return F_, (S_, y_, rho_W_, lnphi_W_)

        def _ss_step(ln_W, aux):
            S_, y_, rho_W_, lnphi_W_ = aux
            return d - lnphi_W_

        # Inline the SS+Broyden phases here rather than using the helper,
        # because we need the trivial-solution check at every iteration
        # (the helper has no hooks for that).
        ln_W = np.log(W + 1e-300)

        # SS warm-up
        for it in range(min(_SS_WARMUP, maxiter)):
            W = np.exp(ln_W)
            S = W.sum()
            if S <= 0 or not np.all(np.isfinite(W)):
                break
            y = W / S
            if np.max(np.abs(y - z)) < 1e-5 and abs(S - 1.0) < 1e-4:
                trivial = True
                break
            res = _residual(ln_W)
            if res is None:
                break
            F, aux = res
            if np.max(np.abs(F)) < 1e-9:
                converged = True
                break
            ln_W = _ss_step(ln_W, aux)

        # Broyden phase
        if not converged and not trivial:
            res = _residual(ln_W)
            if res is None:
                pass
            else:
                F, aux = res
                if np.max(np.abs(F)) < 1e-9:
                    converged = True
                else:
                    H = np.eye(len(z))
                    F_prev = F.copy()
                    ln_W_prev = ln_W.copy()
                    for it in range(_SS_WARMUP + 1, maxiter):
                        W = np.exp(ln_W)
                        S = W.sum()
                        if S > 0 and np.all(np.isfinite(W)):
                            y_check = W / S
                            if (np.max(np.abs(y_check - z)) < 1e-5
                                    and abs(S - 1.0) < 1e-4):
                                trivial = True
                                break
                        delta = -H @ F_prev
                        max_step = np.max(np.abs(delta))
                        if max_step > 1.0:
                            delta = delta / max_step
                        ln_W = ln_W_prev + delta
                        res = _residual(ln_W)
                        if res is None:
                            ln_W = _ss_step(ln_W_prev, aux)
                            res = _residual(ln_W)
                            if res is None:
                                break
                            H = np.eye(len(z))
                        F, aux = res
                        if np.max(np.abs(F)) < 1e-9:
                            converged = True
                            break
                        s = ln_W - ln_W_prev
                        y_vec = F - F_prev
                        Hy = H @ y_vec
                        sH = s @ H
                        denom = s @ Hy
                        if abs(denom) > 1e-30:
                            H = H + np.outer(s - Hy, sH) / denom
                        ln_W_prev = ln_W.copy()
                        F_prev = F.copy()

        W = np.exp(ln_W)

        if trivial:
            if verbose:
                print(f"  [{direction}] trivial")
            continue

        S = W.sum()
        if not np.isfinite(S) or S <= 0:
            continue

        y_final = W / S
        if np.max(np.abs(y_final - z)) < 1e-5:
            continue

        S_minus_1 = S - 1.0
        if verbose:
            print(f"  [{direction}] nontrivial: S={S:.4f}, converged={converged}")

        if S_minus_1 > max_S_minus_1:
            max_S_minus_1 = S_minus_1
            if direction == "vapor":
                best_K = W / (z + 1e-300)
            else:
                best_K = (z + 1e-300) / W

        if S_minus_1 > tol:
            unstable = True

    if max_S_minus_1 == -np.inf:
        max_S_minus_1 = 0.0

    return (not unstable), best_K, max_S_minus_1


# -------------------------------------------------------------------------
# PT flash
# -------------------------------------------------------------------------

def flash_pt(p, T, z, mixture, K_init=None, check_stability=True,
             tol=1e-10, maxiter=100):
    """Isothermal-isobaric flash for a cubic mixture at (p, T, z)."""
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    # Stability: determine whether two-phase
    if check_stability:
        try:
            stable, K_stab, _ = stability_test_TPD(z, T, p, mixture)
        except RuntimeError:
            stable = True
            K_stab = None

        if stable:
            # Single phase: pick the right root
            try:
                rho_v = mixture.density_from_pressure(p, T, z, phase_hint='vapor')
            except RuntimeError:
                rho_v = None
            try:
                rho_l = mixture.density_from_pressure(p, T, z, phase_hint='liquid')
            except RuntimeError:
                rho_l = None

            if rho_v is None and rho_l is None:
                raise RuntimeError("no density root at single-phase state")
            if rho_v is None:
                rho = rho_l
                label = 'liquid'
            elif rho_l is None:
                rho = rho_v
                label = 'vapor'
            elif abs(rho_v - rho_l) / max(rho_v, rho_l) < 1e-4:
                rho = rho_v
                # single root case -- supercritical if T > Tc_pseudo else label by density
                Tc_pseudo, rho_c_pseudo = mixture.reduce(z)
                if T > Tc_pseudo:
                    label = 'supercritical'
                else:
                    label = 'vapor' if rho < 0.5 * rho_c_pseudo else 'liquid'
            else:
                # Two distinct roots -- Gibbs minimum
                lpv = mixture.ln_phi(rho_v, T, z)
                lpl = mixture.ln_phi(rho_l, T, z)
                if float(np.dot(z, lpv)) < float(np.dot(z, lpl)):
                    rho = rho_v; label = 'vapor'
                else:
                    rho = rho_l; label = 'liquid'

            # Fill caloric properties
            cal = mixture.caloric(rho, T, z, p=p)
            return CubicFlashResult(
                phase=label, T=T, p=p, beta=None,
                x=z.copy(), y=z.copy(), z=z, rho=rho,
                rho_L=None, rho_V=None,
                h=cal["h"], s=cal["s"], iterations=0, K=None,
            )
        K_init = K_stab

    # Two-phase SS+Broyden iteration on K-factors. The hybrid converges
    # in ~5-10 iters for typical cubic-EOS mixtures vs ~15-30 for pure SS
    # on strongly non-ideal systems; same accelerator pattern as the
    # Helmholtz mixture flash (added in v0.9.5).
    if K_init is None:
        K = mixture.wilson_K(T, p)
    else:
        K = K_init.copy()

    def _residual(ln_K):
        K_curr = np.exp(ln_K)
        beta_, _ = rachford_rice(z, K_curr)
        denom = 1.0 + beta_ * (K_curr - 1.0)
        x_ = z / denom; x_ /= x_.sum()
        y_ = K_curr * x_; y_ /= y_.sum()
        try:
            rho_L_ = mixture.density_from_pressure(p, T, x_, phase_hint='liquid')
            rho_V_ = mixture.density_from_pressure(p, T, y_, phase_hint='vapor')
        except RuntimeError:
            return None
        lnphi_L_ = mixture.ln_phi(rho_L_, T, x_)
        lnphi_V_ = mixture.ln_phi(rho_V_, T, y_)
        F_ = ln_K - (lnphi_L_ - lnphi_V_)
        # aux carries the converged-state values the caller needs
        return F_, (beta_, x_, y_, rho_L_, rho_V_, lnphi_L_, lnphi_V_)

    def _ss_step(ln_K, aux):
        # SS update on ln K: ln K_new = lnphi_L - lnphi_V
        # (which makes the residual zero in one step if lnphi were
        # composition-independent; in practice contracts at rate
        # ~|d lnphi / d ln K|).
        beta_, x_, y_, rho_L_, rho_V_, lnphi_L_, lnphi_V_ = aux
        return lnphi_L_ - lnphi_V_

    ln_K, aux, it, status = _ss_broyden_solve(
        np.log(K), _residual, _ss_step, tol, maxiter
    )
    if status == "failed":
        raise RuntimeError(
            f"flash iteration failed: density solve diverged at "
            f"T={T}, p={p}, z={z}"
        )
    K = np.exp(ln_K)
    # Helper returns aux from the converged residual evaluation, so we
    # already have all the per-phase state -- no need to recompute.
    beta, x, y, rho_L, rho_V, lnphi_L, lnphi_V = aux

    # Check for single-phase collapse: if x and y very similar, revert to single phase
    if np.max(np.abs(y - x)) < 1e-5:
        # Flash wanted to collapse; stability must have been wrong
        # Fall back to a single-phase state
        Tc_pseudo, rho_c_pseudo = mixture.reduce(z)
        rho = rho_L if beta < 0.5 else rho_V
        label = 'vapor' if rho < 0.5 * rho_c_pseudo else 'liquid'
        cal = mixture.caloric(rho, T, z, p=p)
        return CubicFlashResult(
            phase=label, T=T, p=p, beta=None,
            x=z.copy(), y=z.copy(), z=z, rho=rho,
            rho_L=None, rho_V=None,
            h=cal["h"], s=cal["s"], iterations=it, K=K,
        )

    # Overall density via volume-weighted
    v_avg = beta / rho_V + (1.0 - beta) / rho_L
    rho_avg = 1.0 / v_avg

    # Mixture caloric: mole-fraction-weighted combination of the two phases
    # h_mix = beta * h_V + (1 - beta) * h_L
    # s_mix = beta * s_V + (1 - beta) * s_L
    # Each phase is evaluated at its own composition and density but at the
    # same (T, p).
    cal_L = mixture.caloric(rho_L, T, x, p=p)
    cal_V = mixture.caloric(rho_V, T, y, p=p)
    h_mix = beta * cal_V["h"] + (1.0 - beta) * cal_L["h"]
    s_mix = beta * cal_V["s"] + (1.0 - beta) * cal_L["s"]

    return CubicFlashResult(
        phase="two_phase", T=T, p=p, beta=beta,
        x=x, y=y, z=z, rho=rho_avg,
        rho_L=rho_L, rho_V=rho_V,
        h=h_mix, s=s_mix, iterations=it, K=K,
    )


# -------------------------------------------------------------------------
# Newton-Raphson flash with analytic Jacobian (v0.9.8)
# -------------------------------------------------------------------------
#
# Uses the analytic d(ln phi)/d x_k Jacobian (added in v0.9.8 to
# CubicMixture.dlnphi_dxk_at_p) to build the full N x N Newton Jacobian
# for the ln-K residual:
#
#     F_i(ln K) = ln K_i - (lnphi_L_i(x(K)) - lnphi_V_i(y(K)))
#
# Convergence is quadratic, typically 4-6 iterations from a Wilson
# starting estimate vs 8-15 for Broyden. Per-iter cost is ~2x Broyden's
# (one analytic Jacobian build per iter, where Broyden uses a rank-1
# secant update), so walltime is roughly tied or slightly slower than
# Broyden for the typical 2-5 component flash.
#
# Use this when:
#   - You need guaranteed quadratic convergence (e.g. inside a trust-region
#     solver, or when stepping along a phase envelope at small dT)
#   - The problem is large (N >> 5), where Broyden's secant approximation
#     fills in slowly enough that full-Newton wins on iter count
#   - You're doing sensitivity analysis and want the converged Jacobian
#     for free
#
# Use the default flash_pt() (SS+Broyden) when you just want a flash.

def newton_flash_pt(p, T, z, mixture, K_init=None, check_stability=True,
                    tol=1e-9, maxiter=30):
    """Newton-Raphson PT flash on ln K with full analytic Jacobian.

    Same interface as flash_pt; returns CubicFlashResult or raises if
    not in the two-phase region. The Jacobian is assembled from
    mixture.dlnphi_dxk_at_p (the analytic composition derivative of
    ln phi at fixed T, p) chained through Rachford-Rice's beta(K)
    via implicit differentiation:

        d(beta)/dK_j = (z_j / D_j^2) / sum_m z_m (K_m-1)^2 / D_m^2
        D_m = 1 + beta * (K_m - 1)
        d(x_m)/dK_j = -z_m * [d(beta)/dK_j * (K_m-1) + beta * delta_mj] / D_m^2
        d(y_m)/dK_j = delta_mj * x_m + K_m * d(x_m)/dK_j

    The flash Jacobian is then:

        J[i, j] = delta_ij + K_j * sum_m {dlnphi_V_dy[i,m] * dy_dK[m,j]
                                       - dlnphi_L_dx[i,m] * dx_dK[m,j]}

    where the K_j factor converts d/dK_j to d/d(ln K_j) (since the
    Newton update is on ln K). Step damping caps ||delta_lnK||_inf at 1.0
    to prevent overshoot when far from the solution.
    """
    z = np.asarray(z, dtype=np.float64); z = z / z.sum()
    N = len(z)

    # Stability pre-check (mirrors flash_pt's behavior)
    if check_stability and K_init is None:
        stable, K_stab, _ = stability_test_TPD(z, T, p, mixture)
        if stable:
            # Single phase -- delegate to the existing single-phase
            # classification path in flash_pt by calling it directly.
            return flash_pt(p, T, z, mixture, K_init=None, check_stability=True,
                            tol=tol, maxiter=maxiter)
        K = K_stab
    elif K_init is not None:
        K = K_init.copy()
    else:
        K = mixture.wilson_K(T, p)

    for it in range(maxiter):
        beta, _ = rachford_rice(z, K)
        if not (0 < beta < 1):
            # Outside the two-phase region given current K. Fall back to
            # SS+Broyden which has more robust handling of these cases.
            return flash_pt(p, T, z, mixture, K_init=K, check_stability=False,
                            tol=tol, maxiter=maxiter)
        D = 1.0 + beta * (K - 1.0)
        x = z / D; x = x / x.sum()
        y = K * (z / D); y = y / y.sum()

        try:
            rho_L = mixture.density_from_pressure(p, T, x, phase_hint='liquid')
            rho_V = mixture.density_from_pressure(p, T, y, phase_hint='vapor')
        except RuntimeError:
            return flash_pt(p, T, z, mixture, K_init=K, check_stability=False,
                            tol=tol, maxiter=maxiter)

        lnphi_L = mixture.ln_phi(rho_L, T, x)
        lnphi_V = mixture.ln_phi(rho_V, T, y)
        F = np.log(K) - (lnphi_L - lnphi_V)
        if np.max(np.abs(F)) < tol:
            v_avg = beta / rho_V + (1.0 - beta) / rho_L
            cal_L = mixture.caloric(rho_L, T, x, p=p)
            cal_V = mixture.caloric(rho_V, T, y, p=p)
            return CubicFlashResult(
                phase="two_phase", T=T, p=p, beta=beta,
                x=x, y=y, z=z, rho=1.0/v_avg,
                rho_L=rho_L, rho_V=rho_V,
                h=beta*cal_V["h"] + (1-beta)*cal_L["h"],
                s=beta*cal_V["s"] + (1-beta)*cal_L["s"],
                iterations=it+1, K=K,
            )

        # Build analytic Jacobian.
        # Step 1: dbeta/dK from implicit RR differentiation
        denom = np.sum(z * (K - 1.0)**2 / D**2)
        dbeta_dK = (z / D**2) / denom
        # Step 2: dx/dK, dy/dK from chain rule through RR
        # dx[m, j] = -z_m * (dbeta_dK[j]*(K_m - 1) + beta*delta_mj) / D_m^2
        dx_dK = -z[:, None] * (np.outer(K - 1.0, dbeta_dK) + beta * np.eye(N)) / (D[:, None] ** 2)
        # dy[m, j] = delta_mj * x_m + K_m * dx[m, j]
        dy_dK = np.eye(N) * x[:, None] + K[:, None] * dx_dK
        # Step 3: analytic dlnphi/dx for both phases
        try:
            dlnphi_L_dx = mixture.dlnphi_dxk_at_p(p, T, x, phase_hint='liquid')
            dlnphi_V_dy = mixture.dlnphi_dxk_at_p(p, T, y, phase_hint='vapor')
        except (RuntimeError, NotImplementedError):
            return flash_pt(p, T, z, mixture, K_init=K, check_stability=False,
                            tol=tol, maxiter=maxiter)
        # Step 4: assemble. Newton step is on ln K, so dF/d(ln K_j) = K_j * dF/dK_j.
        # dF_i/dK_j = delta_ij/K_j + (something / K_j... no, let me redo)
        # F_i = ln K_i - (lnphi_L_i - lnphi_V_i)
        # d/d(ln K_j) [ln K_i] = delta_ij
        # d/d(ln K_j) [lnphi_X_i] = K_j * d(lnphi_X_i)/dK_j
        #                        = K_j * sum_m dlnphi_X_i/dx_m * dx_m/dK_j   (X=L)
        # Same for V with y instead of x.
        J = np.eye(N) + (dlnphi_V_dy @ dy_dK - dlnphi_L_dx @ dx_dK) * K

        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return flash_pt(p, T, z, mixture, K_init=K, check_stability=False,
                            tol=tol, maxiter=maxiter)
        # Step damping
        max_step = np.max(np.abs(delta))
        if max_step > 1.0:
            delta = delta / max_step
        K = np.exp(np.log(K) + delta)

    # Did not converge in maxiter; fall back to SS+Broyden from current K
    return flash_pt(p, T, z, mixture, K_init=K, check_stability=False,
                    tol=tol, maxiter=80)


# -------------------------------------------------------------------------
# Bubble- and dew-point solvers
# -------------------------------------------------------------------------

def _bubble_residual_at(T, p, z, mixture, y_init=None, maxiter=30, tol=1e-10):
    """Inner iteration for bubble point: at (T, p), iterate to self-consistency
    between K = phi_L/phi_V and vapor composition y = z*K / sum(z*K).

    Returns (S, K, y) where S = sum(z*K) at convergence. S=1 marks the
    bubble line.

    Raises RuntimeError("trivial") if K collapses to ~1 (above the dome).
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if y_init is None:
        K = mixture.wilson_K(T, p)
        y = z * K
        y = y / y.sum()
    else:
        y = y_init.copy() / y_init.sum()
        K = y / np.maximum(z, 1e-300)

    last_ln_K = np.log(np.maximum(K, 1e-300))

    for _ in range(maxiter):
        try:
            rho_L = mixture.density_from_pressure(p, T, z, phase_hint='liquid')
        except RuntimeError:
            rho_L = mixture.density_from_pressure(p, T, z, phase_hint='vapor')
        try:
            rho_V = mixture.density_from_pressure(p, T, y, phase_hint='vapor')
        except RuntimeError:
            rho_V = mixture.density_from_pressure(p, T, y, phase_hint='liquid')

        lnphi_L = mixture.ln_phi(rho_L, T, z)
        lnphi_V = mixture.ln_phi(rho_V, T, y)
        ln_K_new = lnphi_L - lnphi_V
        K_new = np.exp(ln_K_new)

        if np.max(np.abs(ln_K_new)) < 1e-3:
            raise RuntimeError("trivial")

        S = float(np.dot(z, K_new))
        if S <= 0 or not np.isfinite(S):
            raise RuntimeError("bad S")
        y_new = z * K_new / S
        y_new = y_new / y_new.sum()

        if np.max(np.abs(ln_K_new - last_ln_K)) < tol:
            return S, K_new, y_new

        last_ln_K = ln_K_new
        K = K_new
        y = y_new

    return S, K_new, y_new


def _dew_residual_at(T, p, z, mixture, x_init=None, maxiter=30, tol=1e-10):
    """Inner iteration for dew point. Dew residual: S = sum(z/K) = 1."""
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if x_init is None:
        K = mixture.wilson_K(T, p)
        x = z / K
        x = x / x.sum()
    else:
        x = x_init.copy() / x_init.sum()
        K = np.maximum(z, 1e-300) / np.maximum(x, 1e-300)

    last_ln_K = np.log(np.maximum(K, 1e-300))

    for _ in range(maxiter):
        try:
            rho_L = mixture.density_from_pressure(p, T, x, phase_hint='liquid')
        except RuntimeError:
            rho_L = mixture.density_from_pressure(p, T, x, phase_hint='vapor')
        try:
            rho_V = mixture.density_from_pressure(p, T, z, phase_hint='vapor')
        except RuntimeError:
            rho_V = mixture.density_from_pressure(p, T, z, phase_hint='liquid')

        lnphi_L = mixture.ln_phi(rho_L, T, x)
        lnphi_V = mixture.ln_phi(rho_V, T, z)
        ln_K_new = lnphi_L - lnphi_V
        K_new = np.exp(ln_K_new)

        if np.max(np.abs(ln_K_new)) < 1e-3:
            raise RuntimeError("trivial")

        S = float(np.dot(z, 1.0 / K_new))
        if S <= 0 or not np.isfinite(S):
            raise RuntimeError("bad S")
        x_new = (z / K_new) / S
        x_new = x_new / x_new.sum()

        if np.max(np.abs(ln_K_new - last_ln_K)) < tol:
            return S, K_new, x_new

        last_ln_K = ln_K_new
        x = x_new

    return S, K_new, x_new


def bubble_point_p(T, z, mixture, p_init=None, tol=1e-8, maxiter=60):
    """Bubble-point pressure for a cubic mixture at temperature T.

    Uses Michelsen pressure-correction: p_new = p * S where S = sum(z*K) at
    self-consistent (y, K). Converges quadratically.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if p_init is None:
        # Wilson bubble-p: sum(z_i A_i) where A_i = p_c_i * exp(5.373(1+w)(1-Tc/T))
        p_init = 0.0
        for i, c in enumerate(mixture.components):
            p_init += z[i] * c.p_c * np.exp(
                5.373 * (1.0 + c.acentric_factor) * (1.0 - c.T_c / T)
            )
        p_init = max(p_init, 1e3)

    p = p_init
    y_last = None
    f_resid = float("nan")

    for it in range(maxiter):
        if p < 1.0:    # pressure below 1 Pa is unphysical
            raise RuntimeError(
                f"bubble_point_p: iteration collapsed below 1 Pa at T={T}; "
                f"no physical bubble point exists for z={z.tolist()}."
            )
        try:
            S, K, y = _bubble_residual_at(T, p, z, mixture, y_init=y_last)
        except RuntimeError as e:
            if str(e) == "trivial":
                p = p * 0.5
                y_last = None
                continue
            elif str(e) == "bad S":
                p = p * 0.8
                y_last = None
                continue
            raise
        y_last = y

        f_resid = S - 1.0
        if abs(f_resid) < tol:
            rho_L = mixture.density_from_pressure(p, T, z, phase_hint='liquid')
            rho_V = mixture.density_from_pressure(p, T, y, phase_hint='vapor')
            cal = mixture.caloric(rho_L, T, z, p=p)  # feed z is in liquid phase at beta=0
            return CubicFlashResult(
                phase="bubble", T=T, p=p, beta=0.0,
                x=z.copy(), y=y, z=z, rho=rho_L,
                rho_L=rho_L, rho_V=rho_V,
                h=cal["h"], s=cal["s"], iterations=it + 1, K=K,
            )

        # Pressure-correction: p_new = p * S
        p_new = p * S
        if p_new > 5.0 * p:
            p_new = 5.0 * p
        elif p_new < 0.2 * p:
            p_new = 0.2 * p
        p = p_new

    raise RuntimeError(
        f"bubble_point_p did not converge: T={T}, final p={p}, residual={f_resid:.3e}"
    )


def dew_point_p(T, z, mixture, p_init=None, tol=1e-8, maxiter=60):
    """Dew-point pressure for a cubic mixture at temperature T.

    Uses Michelsen pressure-correction: p_new = p / S where S = sum(z/K).
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if p_init is None:
        # Wilson dew-p: 1/sum(z_i/A_i)
        inv = 0.0
        for i, c in enumerate(mixture.components):
            A_i = c.p_c * np.exp(
                5.373 * (1.0 + c.acentric_factor) * (1.0 - c.T_c / T)
            )
            inv += z[i] / A_i
        p_init = max(1.0 / inv, 1e3)

    p = p_init
    x_last = None
    f_resid = float("nan")

    for it in range(maxiter):
        if p < 1.0:
            raise RuntimeError(
                f"dew_point_p: iteration collapsed below 1 Pa at T={T}; "
                f"no physical dew point exists for z={z.tolist()}."
            )
        try:
            S, K, x = _dew_residual_at(T, p, z, mixture, x_init=x_last)
        except RuntimeError as e:
            if str(e) == "trivial":
                p = p * 0.5
                x_last = None
                continue
            elif str(e) == "bad S":
                p = p * 1.5
                x_last = None
                continue
            raise
        x_last = x

        f_resid = S - 1.0
        if abs(f_resid) < tol:
            rho_L = mixture.density_from_pressure(p, T, x, phase_hint='liquid')
            rho_V = mixture.density_from_pressure(p, T, z, phase_hint='vapor')
            cal = mixture.caloric(rho_V, T, z, p=p)  # feed z is in vapor phase at beta=1
            return CubicFlashResult(
                phase="dew", T=T, p=p, beta=1.0,
                x=x, y=z.copy(), z=z, rho=rho_V,
                rho_L=rho_L, rho_V=rho_V,
                h=cal["h"], s=cal["s"], iterations=it + 1, K=K,
            )

        p_new = p / S
        if p_new > 5.0 * p:
            p_new = 5.0 * p
        elif p_new < 0.2 * p:
            p_new = 0.2 * p
        p = p_new

    raise RuntimeError(
        f"dew_point_p did not converge: T={T}, final p={p}, residual={f_resid:.3e}"
    )


def bubble_point_T(p, z, mixture, T_init=None, tol=1e-8, maxiter=60):
    """Bubble-point temperature at pressure p.

    Uses bracketed secant in ln(S) vs 1/T with bisection fallback.
    Rejects trivial solutions above the dome.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    # Wilson-based initial bracket
    def wilson_bracket():
        # Solve Wilson bubble: sum(z_i A_i) = p, for T
        # A_i(T) = p_c_i * exp(5.373(1+omega)(1 - Tc/T)), sum z*A = p
        # Just bisect over T
        lo, hi = 30.0, 2000.0
        def f(T):
            s = 0.0
            for i, c in enumerate(mixture.components):
                s += z[i] * c.p_c * np.exp(
                    5.373 * (1.0 + c.acentric_factor) * (1.0 - c.T_c / T)
                )
            return s - p
        if f(lo) * f(hi) > 0:
            return None
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if f(mid) * f(lo) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 0.01:
                break
        return 0.5 * (lo + hi)

    if T_init is None:
        T_init = wilson_bracket() or float(np.dot(z, mixture.T_c))

    def eval_S(T, y_hint):
        try:
            S, K, y = _bubble_residual_at(T, p, z, mixture, y_init=y_hint)
            return S, K, y, "ok"
        except RuntimeError as e:
            return None, None, None, "trivial" if str(e) == "trivial" else "bad"

    # Establish bracket
    T = T_init
    S, K, y, status = eval_S(T, None)
    T_lo = T_hi = None
    S_lo = S_hi = None
    K_lo = K_hi = None
    y_lo = y_hi = None

    if status == "ok":
        if S < 1.0:
            T_lo, S_lo, K_lo, y_lo = T, S, K, y
        else:
            T_hi, S_hi, K_hi, y_hi = T, S, K, y

    if status != "ok":
        T_probe = T_init
        for _ in range(20):
            T_probe *= 0.8
            if T_probe < 30.0:
                break
            S, K, y, status = eval_S(T_probe, None)
            if status == "ok":
                if S < 1.0:
                    T_lo, S_lo, K_lo, y_lo = T_probe, S, K, y
                else:
                    T_hi, S_hi, K_hi, y_hi = T_probe, S, K, y
                break
        if T_lo is None and T_hi is None:
            raise RuntimeError(f"bubble_point_T: no non-trivial T at p={p}")

    if T_lo is None:
        T_probe = T_hi * 0.9
        for _ in range(30):
            S, K, y, status = eval_S(T_probe, y_hi)
            if status == "ok" and S < 1.0:
                T_lo, S_lo, K_lo, y_lo = T_probe, S, K, y
                break
            T_probe *= 0.9
            if T_probe < 30.0:
                break
        if T_lo is None:
            raise RuntimeError(f"bubble_point_T: could not find T_lo for p={p}")

    if T_hi is None:
        T_probe = T_lo * 1.1
        for _ in range(30):
            S, K, y, status = eval_S(T_probe, y_lo)
            if status == "ok" and S > 1.0:
                T_hi, S_hi, K_hi, y_hi = T_probe, S, K, y
                break
            if status == "trivial":
                T_probe = 0.5 * (T_lo + T_probe)
                continue
            T_probe *= 1.1
            if T_probe > 3000.0:
                break
        if T_hi is None:
            raise RuntimeError(
                f"bubble_point_T: no bubble point exists at p={p} for z={z.tolist()}"
            )

    for it in range(maxiter):
        lnS_lo = np.log(S_lo); lnS_hi = np.log(S_hi)
        if lnS_hi == lnS_lo:
            T_new = 0.5 * (T_lo + T_hi)
        else:
            frac = -lnS_lo / (lnS_hi - lnS_lo)
            frac = max(0.1, min(0.9, frac))
            inv_T_new = (1 - frac) / T_lo + frac / T_hi
            T_new = 1.0 / inv_T_new

        y_hint = y_lo if S_lo > 0 else y_hi
        S_new, K_new, y_new, status = eval_S(T_new, y_hint)
        if status == "trivial":
            T_hi = T_new
            S_hi = max(S_hi, 1.001)
            continue
        if status == "bad":
            T_new = 0.5 * (T_lo + T_hi)
            S_new, K_new, y_new, status = eval_S(T_new, y_hint)
            if status != "ok":
                raise RuntimeError(f"bubble_point_T: failed at T={T_new}")

        f_resid = S_new - 1.0
        if abs(f_resid) < tol:
            rho_L = mixture.density_from_pressure(p, T_new, z, phase_hint='liquid')
            rho_V = mixture.density_from_pressure(p, T_new, y_new, phase_hint='vapor')
            cal = mixture.caloric(rho_L, T_new, z, p=p)
            return CubicFlashResult(
                phase="bubble", T=T_new, p=p, beta=0.0,
                x=z.copy(), y=y_new, z=z, rho=rho_L,
                rho_L=rho_L, rho_V=rho_V,
                h=cal["h"], s=cal["s"], iterations=it + 1, K=K_new,
            )
        if S_new < 1.0:
            T_lo, S_lo, K_lo, y_lo = T_new, S_new, K_new, y_new
        else:
            T_hi, S_hi, K_hi, y_hi = T_new, S_new, K_new, y_new

    raise RuntimeError(f"bubble_point_T did not converge: p={p}")


def dew_point_T(p, z, mixture, T_init=None, tol=1e-8, maxiter=60):
    """Dew-point temperature at pressure p. Mirror of bubble_point_T.

    For dew: S(T) = sum(z/K) is high at low T, low at high T. So T_lo has
    S>1, T_hi has S<1.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if T_init is None:
        # Wilson dew-T: 1 / sum(z_i / A_i) = p  with A_i(T) depending on T
        lo, hi = 30.0, 2000.0
        def f(T):
            s = 0.0
            for i, c in enumerate(mixture.components):
                A_i = c.p_c * np.exp(
                    5.373 * (1.0 + c.acentric_factor) * (1.0 - c.T_c / T)
                )
                s += z[i] / A_i
            return 1.0 / s - p
        if f(lo) * f(hi) > 0:
            T_init = float(np.dot(z, mixture.T_c))
        else:
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if f(mid) * f(lo) > 0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 0.01:
                    break
            T_init = 0.5 * (lo + hi)

    def eval_S(T, x_hint):
        try:
            S, K, x = _dew_residual_at(T, p, z, mixture, x_init=x_hint)
            return S, K, x, "ok"
        except RuntimeError as e:
            return None, None, None, "trivial" if str(e) == "trivial" else "bad"

    T = T_init
    S, K, x, status = eval_S(T, None)
    T_lo = T_hi = None
    S_lo = S_hi = None
    K_lo = K_hi = None
    x_lo = x_hi = None

    if status == "ok":
        if S > 1.0:
            T_lo, S_lo, K_lo, x_lo = T, S, K, x
        else:
            T_hi, S_hi, K_hi, x_hi = T, S, K, x

    if status != "ok":
        T_probe = T_init
        for _ in range(20):
            T_probe *= 0.8
            if T_probe < 30.0:
                break
            S, K, x, status = eval_S(T_probe, None)
            if status == "ok":
                if S > 1.0:
                    T_lo, S_lo, K_lo, x_lo = T_probe, S, K, x
                else:
                    T_hi, S_hi, K_hi, x_hi = T_probe, S, K, x
                break
        if T_lo is None and T_hi is None:
            raise RuntimeError(f"dew_point_T: no non-trivial T at p={p}")

    if T_lo is None:
        T_probe = T_hi * 0.9
        for _ in range(30):
            S, K, x, status = eval_S(T_probe, x_hi)
            if status == "ok" and S > 1.0:
                T_lo, S_lo, K_lo, x_lo = T_probe, S, K, x
                break
            T_probe *= 0.9
            if T_probe < 30.0:
                break
        if T_lo is None:
            raise RuntimeError(f"dew_point_T: could not find T_lo for p={p}")

    if T_hi is None:
        T_probe = T_lo * 1.1
        for _ in range(30):
            S, K, x, status = eval_S(T_probe, x_lo)
            if status == "ok" and S < 1.0:
                T_hi, S_hi, K_hi, x_hi = T_probe, S, K, x
                break
            if status == "trivial":
                T_probe = 0.5 * (T_lo + T_probe)
                continue
            T_probe *= 1.1
            if T_probe > 3000.0:
                break
        if T_hi is None:
            raise RuntimeError(f"dew_point_T: no dew point exists at p={p}")

    for it in range(maxiter):
        lnS_lo = np.log(S_lo); lnS_hi = np.log(S_hi)
        if lnS_hi == lnS_lo:
            T_new = 0.5 * (T_lo + T_hi)
        else:
            frac = -lnS_lo / (lnS_hi - lnS_lo)
            frac = max(0.1, min(0.9, frac))
            inv_T_new = (1 - frac) / T_lo + frac / T_hi
            T_new = 1.0 / inv_T_new

        x_hint = x_lo if S_lo > 0 else x_hi
        S_new, K_new, x_new, status = eval_S(T_new, x_hint)
        if status == "trivial":
            T_hi = T_new
            S_hi = min(S_hi, 0.999)
            continue
        if status == "bad":
            T_new = 0.5 * (T_lo + T_hi)
            S_new, K_new, x_new, status = eval_S(T_new, x_hint)
            if status != "ok":
                raise RuntimeError(f"dew_point_T: failed at T={T_new}")

        f_resid = S_new - 1.0
        if abs(f_resid) < tol:
            rho_L = mixture.density_from_pressure(p, T_new, x_new, phase_hint='liquid')
            rho_V = mixture.density_from_pressure(p, T_new, z, phase_hint='vapor')
            cal = mixture.caloric(rho_V, T_new, z, p=p)
            return CubicFlashResult(
                phase="dew", T=T_new, p=p, beta=1.0,
                x=x_new, y=z.copy(), z=z, rho=rho_V,
                rho_L=rho_L, rho_V=rho_V,
                h=cal["h"], s=cal["s"], iterations=it + 1, K=K_new,
            )
        if S_new > 1.0:
            T_lo, S_lo, K_lo, x_lo = T_new, S_new, K_new, x_new
        else:
            T_hi, S_hi, K_hi, x_hi = T_new, S_new, K_new, x_new

    raise RuntimeError(f"dew_point_T did not converge: p={p}")


# -------------------------------------------------------------------------
# Bubble and dew point solvers
# -------------------------------------------------------------------------

def _bubble_residual(T, p, z, mixture, y_init=None, maxiter=30, tol=1e-10):
    """Inner iteration for bubble point at fixed (T, p).

    Returns (S, K, y) where S = sum(z*K). Raises RuntimeError on trivial
    convergence (K->1 everywhere).
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()
    if y_init is None:
        K = mixture.wilson_K(T, p)
        y = z * K
        y = y / y.sum()
    else:
        y = y_init.copy() / y_init.sum()
        K = y / np.maximum(z, 1e-300)

    last_lnK = np.log(np.maximum(K, 1e-300))
    for it in range(maxiter):
        try:
            rho_L = mixture.density_from_pressure(p, T, z, phase_hint='liquid')
        except RuntimeError:
            rho_L = mixture.density_from_pressure(p, T, z, phase_hint='vapor')
        lnphi_L = mixture.ln_phi(rho_L, T, z)

        try:
            rho_V = mixture.density_from_pressure(p, T, y, phase_hint='vapor')
        except RuntimeError:
            rho_V = mixture.density_from_pressure(p, T, y, phase_hint='liquid')
        lnphi_V = mixture.ln_phi(rho_V, T, y)

        ln_K_new = lnphi_L - lnphi_V
        K_new = np.exp(ln_K_new)
        if np.max(np.abs(ln_K_new)) < 1e-3:
            raise RuntimeError("trivial")

        S = float(np.dot(z, K_new))
        if not np.isfinite(S) or S <= 0:
            raise RuntimeError("bad S")
        y_new = z * K_new / S
        y_new = y_new / y_new.sum()

        if np.max(np.abs(ln_K_new - last_lnK)) < tol:
            return S, K_new, y_new
        last_lnK = ln_K_new
        y = y_new
        K = K_new
    return S, K_new, y_new


def _dew_residual(T, p, z, mixture, x_init=None, maxiter=30, tol=1e-10):
    """Inner iteration for dew point at fixed (T, p). Returns (S, K, x) with S=sum(z/K)."""
    z = np.asarray(z, dtype=np.float64); z = z/z.sum()
    if x_init is None:
        K = mixture.wilson_K(T, p)
        x = z / K; x = x / x.sum()
    else:
        x = x_init.copy() / x_init.sum()
        K = z / np.maximum(x, 1e-300)

    last_lnK = np.log(np.maximum(K, 1e-300))
    for it in range(maxiter):
        try:
            rho_L = mixture.density_from_pressure(p, T, x, phase_hint='liquid')
        except RuntimeError:
            rho_L = mixture.density_from_pressure(p, T, x, phase_hint='vapor')
        lnphi_L = mixture.ln_phi(rho_L, T, x)

        try:
            rho_V = mixture.density_from_pressure(p, T, z, phase_hint='vapor')
        except RuntimeError:
            rho_V = mixture.density_from_pressure(p, T, z, phase_hint='liquid')
        lnphi_V = mixture.ln_phi(rho_V, T, z)

        ln_K_new = lnphi_L - lnphi_V
        K_new = np.exp(ln_K_new)
        if np.max(np.abs(ln_K_new)) < 1e-3:
            raise RuntimeError("trivial")
        S = float(np.dot(z, 1.0 / K_new))
        if not np.isfinite(S) or S <= 0:
            raise RuntimeError("bad S")
        x_new = (z / K_new) / S
        x_new = x_new / x_new.sum()
        if np.max(np.abs(ln_K_new - last_lnK)) < tol:
            return S, K_new, x_new
        last_lnK = ln_K_new
        x = x_new
        K = K_new
    return S, K_new, x_new


def bubble_point_p(T, z, mixture, p_init=None, tol=1e-8, maxiter=60):
    """Bubble-point pressure at temperature T via pressure correction p_new = p*S."""
    z = np.asarray(z, dtype=np.float64); z = z/z.sum()
    if p_init is None:
        # Wilson-based bubble-p: sum(z * K_Wilson(T, p)) = 1, where K_i = A_i/p
        # => p = sum(z_i * A_i)
        A = np.array([c.p_c * np.exp(5.373*(1+c.acentric_factor)*(1 - c.T_c/T))
                      for c in mixture.components])
        p_init = float(np.dot(z, A))
    p = max(p_init, 1.0)
    y_last = None
    f_resid = float('nan')
    for it in range(maxiter):
        try:
            S, K, y = _bubble_residual(T, p, z, mixture, y_init=y_last)
        except RuntimeError as e:
            if str(e) == 'trivial':
                p *= 0.5; y_last = None; continue
            elif str(e) == 'bad S':
                p *= 0.8; y_last = None; continue
            raise
        y_last = y
        f_resid = S - 1.0
        if abs(f_resid) < tol:
            rho_L = mixture.density_from_pressure(p, T, z, phase_hint='liquid')
            rho_V = mixture.density_from_pressure(p, T, y, phase_hint='vapor')
            cal = mixture.caloric(rho_L, T, z, p=p)
            return CubicFlashResult(
                phase='bubble', T=T, p=p, beta=0.0,
                x=z.copy(), y=y, z=z, rho=rho_L,
                rho_L=rho_L, rho_V=rho_V,
                h=cal["h"], s=cal["s"], iterations=it+1, K=K,
            )
        p_new = p * S
        if p_new > 5*p: p_new = 5*p
        elif p_new < 0.2*p: p_new = 0.2*p
        p = p_new
    raise RuntimeError(f"bubble_point_p did not converge: T={T}, p={p}, S-1={f_resid:.2e}")


def dew_point_p(T, z, mixture, p_init=None, tol=1e-8, maxiter=60):
    """Dew-point pressure at temperature T via pressure correction p_new = p/S."""
    z = np.asarray(z, dtype=np.float64); z = z/z.sum()
    if p_init is None:
        A = np.array([c.p_c * np.exp(5.373*(1+c.acentric_factor)*(1 - c.T_c/T))
                      for c in mixture.components])
        p_init = 1.0 / float(np.dot(z, 1.0/A))
    p = max(p_init, 1.0)
    x_last = None
    f_resid = float('nan')
    for it in range(maxiter):
        try:
            S, K, x = _dew_residual(T, p, z, mixture, x_init=x_last)
        except RuntimeError as e:
            if str(e) == 'trivial':
                p *= 0.5; x_last = None; continue
            elif str(e) == 'bad S':
                p *= 1.5; x_last = None; continue
            raise
        x_last = x
        f_resid = S - 1.0
        if abs(f_resid) < tol:
            rho_L = mixture.density_from_pressure(p, T, x, phase_hint='liquid')
            rho_V = mixture.density_from_pressure(p, T, z, phase_hint='vapor')
            cal = mixture.caloric(rho_V, T, z, p=p)
            return CubicFlashResult(
                phase='dew', T=T, p=p, beta=1.0,
                x=x, y=z.copy(), z=z, rho=rho_V,
                rho_L=rho_L, rho_V=rho_V,
                h=cal["h"], s=cal["s"], iterations=it+1, K=K,
            )
        p_new = p / S
        if p_new > 5*p: p_new = 5*p
        elif p_new < 0.2*p: p_new = 0.2*p
        p = p_new
    raise RuntimeError(f"dew_point_p did not converge: T={T}, p={p}, S-1={f_resid:.2e}")


# ---------------------------------------------------------------------------
# Newton-Raphson bubble/dew solvers (v0.9.15)
# ---------------------------------------------------------------------------
#
# Quadratic-convergence bubble/dew solvers using the analytic Jacobian built
# from the cubic analytic composition derivatives (v0.9.8) and T/p
# derivatives of ln phi (v0.9.10) implemented on CubicMixture.
#
# Formulation mirrors the Helmholtz Newton solvers in
# stateprop.mixture.flash (v0.9.14): unknowns X = (ln K_1..N, ln p) [or ln T],
# residuals (N+1 total):
#
#   R_i        = ln K_i - (ln phi_i^L(z) - ln phi_i^V(y)), i=1..N
#   R_{N+1}    = Sum(K z) - 1  (bubble) or Sum(z/K) - 1 (dew)
#
# The Jacobian uses dlnphi_dxk_at_p, dlnphi_dp_at_T, dlnphi_dT_at_p from
# CubicMixture. Because these three methods raise NotImplementedError for
# Peneloux volume-shifted mixtures (no derivatives implemented through the
# c_mix shift chain rule), each Newton solver falls back to SS on that
# error type, and also on singular Jacobian or mid-iteration density
# failure. This makes Newton at worst as robust as SS.


def _newton_cubic_bubble_residual_jac_p(lnK, lnp, T, z, mixture):
    """Residual and analytic Jacobian for cubic bubble_point_p Newton.
    Unknowns: X = (ln K_1..N, ln p).  Fixed: T, z.
    """
    z = np.asarray(z, dtype=np.float64)
    N = len(z)
    K = np.exp(lnK)
    p = float(np.exp(lnp))

    Kz = K * z
    S = float(Kz.sum())
    y = Kz / S

    try:
        rho_L = mixture.density_from_pressure(p, T, z, phase_hint='liquid')
    except RuntimeError:
        rho_L = mixture.density_from_pressure(p, T, z, phase_hint='vapor')
    try:
        rho_V = mixture.density_from_pressure(p, T, y, phase_hint='vapor')
    except RuntimeError:
        rho_V = mixture.density_from_pressure(p, T, y, phase_hint='liquid')
    lnphi_L = mixture.ln_phi(rho_L, T, z)
    lnphi_V = mixture.ln_phi(rho_V, T, y)

    R = np.empty(N + 1)
    R[:N] = lnK - (lnphi_L - lnphi_V)
    R[N] = S - 1.0

    dy_dlnK = np.diag(y) - np.outer(y, y)
    dlnphi_V_dy = mixture.dlnphi_dxk_at_p(p, T, y, phase_hint='vapor')
    dlnphi_L_dp = mixture.dlnphi_dp_at_T(p, T, z, phase_hint='liquid')
    dlnphi_V_dp = mixture.dlnphi_dp_at_T(p, T, y, phase_hint='vapor')

    J = np.zeros((N + 1, N + 1))
    J[:N, :N] = np.eye(N) + dlnphi_V_dy @ dy_dlnK
    J[:N, N] = -(dlnphi_L_dp - dlnphi_V_dp) * p
    J[N, :N] = K * z
    J[N, N] = 0.0
    return R, J, K, y


def _newton_cubic_bubble_residual_jac_T(lnK, lnT, p, z, mixture):
    """Residual and analytic Jacobian for cubic bubble_point_T Newton.
    Unknowns: X = (ln K_1..N, ln T).  Fixed: p, z.
    """
    z = np.asarray(z, dtype=np.float64)
    N = len(z)
    K = np.exp(lnK)
    T = float(np.exp(lnT))

    Kz = K * z
    S = float(Kz.sum())
    y = Kz / S

    try:
        rho_L = mixture.density_from_pressure(p, T, z, phase_hint='liquid')
    except RuntimeError:
        rho_L = mixture.density_from_pressure(p, T, z, phase_hint='vapor')
    try:
        rho_V = mixture.density_from_pressure(p, T, y, phase_hint='vapor')
    except RuntimeError:
        rho_V = mixture.density_from_pressure(p, T, y, phase_hint='liquid')
    lnphi_L = mixture.ln_phi(rho_L, T, z)
    lnphi_V = mixture.ln_phi(rho_V, T, y)

    R = np.empty(N + 1)
    R[:N] = lnK - (lnphi_L - lnphi_V)
    R[N] = S - 1.0

    dy_dlnK = np.diag(y) - np.outer(y, y)
    dlnphi_V_dy = mixture.dlnphi_dxk_at_p(p, T, y, phase_hint='vapor')
    dlnphi_L_dT = mixture.dlnphi_dT_at_p(p, T, z, phase_hint='liquid')
    dlnphi_V_dT = mixture.dlnphi_dT_at_p(p, T, y, phase_hint='vapor')

    J = np.zeros((N + 1, N + 1))
    J[:N, :N] = np.eye(N) + dlnphi_V_dy @ dy_dlnK
    J[:N, N] = -(dlnphi_L_dT - dlnphi_V_dT) * T
    J[N, :N] = K * z
    J[N, N] = 0.0
    return R, J, K, y


def _newton_cubic_dew_residual_jac_p(lnK, lnp, T, z, mixture):
    """Residual and analytic Jacobian for cubic dew_point_p Newton.
    Unknowns: X = (ln K_1..N, ln p).  Fixed: T, z (vapor).
    """
    z = np.asarray(z, dtype=np.float64)
    N = len(z)
    K = np.exp(lnK)
    p = float(np.exp(lnp))

    zK = z / K
    W = float(zK.sum())
    x = zK / W

    try:
        rho_V = mixture.density_from_pressure(p, T, z, phase_hint='vapor')
    except RuntimeError:
        rho_V = mixture.density_from_pressure(p, T, z, phase_hint='liquid')
    try:
        rho_L = mixture.density_from_pressure(p, T, x, phase_hint='liquid')
    except RuntimeError:
        rho_L = mixture.density_from_pressure(p, T, x, phase_hint='vapor')
    lnphi_V = mixture.ln_phi(rho_V, T, z)
    lnphi_L = mixture.ln_phi(rho_L, T, x)

    R = np.empty(N + 1)
    R[:N] = lnK - (lnphi_L - lnphi_V)
    R[N] = W - 1.0

    dx_dlnK = np.outer(x, x) - np.diag(x)
    dlnphi_L_dx = mixture.dlnphi_dxk_at_p(p, T, x, phase_hint='liquid')
    dlnphi_L_dp = mixture.dlnphi_dp_at_T(p, T, x, phase_hint='liquid')
    dlnphi_V_dp = mixture.dlnphi_dp_at_T(p, T, z, phase_hint='vapor')

    J = np.zeros((N + 1, N + 1))
    J[:N, :N] = np.eye(N) - dlnphi_L_dx @ dx_dlnK
    J[:N, N] = -(dlnphi_L_dp - dlnphi_V_dp) * p
    J[N, :N] = -z / K
    J[N, N] = 0.0
    return R, J, K, x


def _newton_cubic_dew_residual_jac_T(lnK, lnT, p, z, mixture):
    """Residual and analytic Jacobian for cubic dew_point_T Newton."""
    z = np.asarray(z, dtype=np.float64)
    N = len(z)
    K = np.exp(lnK)
    T = float(np.exp(lnT))

    zK = z / K
    W = float(zK.sum())
    x = zK / W

    try:
        rho_V = mixture.density_from_pressure(p, T, z, phase_hint='vapor')
    except RuntimeError:
        rho_V = mixture.density_from_pressure(p, T, z, phase_hint='liquid')
    try:
        rho_L = mixture.density_from_pressure(p, T, x, phase_hint='liquid')
    except RuntimeError:
        rho_L = mixture.density_from_pressure(p, T, x, phase_hint='vapor')
    lnphi_V = mixture.ln_phi(rho_V, T, z)
    lnphi_L = mixture.ln_phi(rho_L, T, x)

    R = np.empty(N + 1)
    R[:N] = lnK - (lnphi_L - lnphi_V)
    R[N] = W - 1.0

    dx_dlnK = np.outer(x, x) - np.diag(x)
    dlnphi_L_dx = mixture.dlnphi_dxk_at_p(p, T, x, phase_hint='liquid')
    dlnphi_L_dT = mixture.dlnphi_dT_at_p(p, T, x, phase_hint='liquid')
    dlnphi_V_dT = mixture.dlnphi_dT_at_p(p, T, z, phase_hint='vapor')

    J = np.zeros((N + 1, N + 1))
    J[:N, :N] = np.eye(N) - dlnphi_L_dx @ dx_dlnK
    J[:N, N] = -(dlnphi_L_dT - dlnphi_V_dT) * T
    J[N, :N] = -z / K
    J[N, N] = 0.0
    return R, J, K, x


def newton_bubble_point_p(T, z, mixture, p_init=None, tol=1e-10, maxiter=25,
                          step_cap=0.5):
    """Newton-Raphson bubble-point pressure for a cubic mixture at fixed T.

    Uses v0.9.8 (composition) + v0.9.10 (p) analytic Jacobian blocks from
    CubicMixture for quadratic convergence. Falls back to SS on singular
    Jacobian, density-solver failure, or NotImplementedError (e.g. on
    Peneloux volume-shifted mixtures where the analytic derivative chain
    is not yet wired through the volume shift).

    Parameters
    ----------
    T : float
    z : array (N,)
    mixture : CubicMixture
    p_init : float or None     initial p; Wilson estimate if None
    tol : float                max |R_i| at convergence
    maxiter : int
    step_cap : float           max |dX_k| per iteration in log-space

    Returns
    -------
    CubicFlashResult with phase='bubble', beta=0.
    """
    z = np.asarray(z, dtype=np.float64); z = z / z.sum()
    N = len(z)

    if p_init is None:
        A = np.array([c.p_c * np.exp(5.373 * (1 + c.acentric_factor) * (1 - c.T_c / T))
                      for c in mixture.components])
        p_init = max(float(np.dot(z, A)), 1e3)

    p = float(p_init)
    K_init = mixture.wilson_K(T, p)
    lnK = np.log(K_init)
    lnp = np.log(p)

    for it in range(maxiter):
        try:
            R, J, K, y = _newton_cubic_bubble_residual_jac_p(lnK, lnp, T, z, mixture)
        except (RuntimeError, NotImplementedError):
            return bubble_point_p(T, z, mixture, p_init=float(np.exp(lnp)),
                                  tol=tol, maxiter=maxiter + 20)

        res = float(np.max(np.abs(R)))
        if res < tol:
            p_final = float(np.exp(lnp))
            try:
                rho_L = mixture.density_from_pressure(p_final, T, z, phase_hint='liquid')
            except RuntimeError:
                rho_L = mixture.density_from_pressure(p_final, T, z, phase_hint='vapor')
            try:
                rho_V = mixture.density_from_pressure(p_final, T, y, phase_hint='vapor')
            except RuntimeError:
                rho_V = mixture.density_from_pressure(p_final, T, y, phase_hint='liquid')
            cal = mixture.caloric(rho_L, T, z, p=p_final)
            return CubicFlashResult(
                phase='bubble', T=T, p=p_final, beta=0.0,
                x=z.copy(), y=y, z=z, rho=rho_L,
                rho_L=rho_L, rho_V=rho_V,
                h=cal["h"], s=cal["s"], iterations=it + 1, K=K,
            )

        try:
            dX = -np.linalg.solve(J, R)
        except np.linalg.LinAlgError:
            return bubble_point_p(T, z, mixture, p_init=float(np.exp(lnp)),
                                  tol=tol, maxiter=maxiter + 20)

        dX_max = float(np.max(np.abs(dX)))
        if dX_max > step_cap:
            dX = dX * (step_cap / dX_max)

        lnK = lnK + dX[:N]
        lnp = lnp + dX[N]

    return bubble_point_p(T, z, mixture, p_init=float(np.exp(lnp)),
                         tol=tol, maxiter=maxiter + 30)


def newton_bubble_point_T(p, z, mixture, T_init=None, tol=1e-10, maxiter=25,
                          step_cap=0.5):
    """Newton-Raphson bubble-point temperature for a cubic mixture at fixed p."""
    z = np.asarray(z, dtype=np.float64); z = z / z.sum()
    N = len(z)

    if T_init is None:
        T_init = float(np.dot(z, [c.T_c for c in mixture.components]))

    T = float(T_init)
    K_init = mixture.wilson_K(T, p)
    lnK = np.log(K_init)
    lnT = np.log(T)

    for it in range(maxiter):
        try:
            R, J, K, y = _newton_cubic_bubble_residual_jac_T(lnK, lnT, p, z, mixture)
        except (RuntimeError, NotImplementedError):
            # Pass original T_init, not the (possibly diverged) Newton iterate
            return bubble_point_T(p, z, mixture, T_init=T_init,
                                  tol=tol, maxiter=maxiter + 20)

        res = float(np.max(np.abs(R)))
        if res < tol:
            T_final = float(np.exp(lnT))
            try:
                rho_L = mixture.density_from_pressure(p, T_final, z, phase_hint='liquid')
            except RuntimeError:
                rho_L = mixture.density_from_pressure(p, T_final, z, phase_hint='vapor')
            try:
                rho_V = mixture.density_from_pressure(p, T_final, y, phase_hint='vapor')
            except RuntimeError:
                rho_V = mixture.density_from_pressure(p, T_final, y, phase_hint='liquid')
            cal = mixture.caloric(rho_L, T_final, z, p=p)
            return CubicFlashResult(
                phase='bubble', T=T_final, p=p, beta=0.0,
                x=z.copy(), y=y, z=z, rho=rho_L,
                rho_L=rho_L, rho_V=rho_V,
                h=cal["h"], s=cal["s"], iterations=it + 1, K=K,
            )

        try:
            dX = -np.linalg.solve(J, R)
        except np.linalg.LinAlgError:
            return bubble_point_T(p, z, mixture, T_init=T_init,
                                  tol=tol, maxiter=maxiter + 20)

        dX_max = float(np.max(np.abs(dX)))
        if dX_max > step_cap:
            dX = dX * (step_cap / dX_max)

        lnK = lnK + dX[:N]
        lnT = lnT + dX[N]

    return bubble_point_T(p, z, mixture, T_init=T_init,
                         tol=tol, maxiter=maxiter + 30)


def newton_dew_point_p(T, z, mixture, p_init=None, tol=1e-10, maxiter=25,
                       step_cap=0.5):
    """Newton-Raphson dew-point pressure for a cubic mixture at fixed T."""
    z = np.asarray(z, dtype=np.float64); z = z / z.sum()
    N = len(z)

    if p_init is None:
        A = np.array([c.p_c * np.exp(5.373 * (1 + c.acentric_factor) * (1 - c.T_c / T))
                      for c in mixture.components])
        p_init = max(1.0 / float(np.dot(z, 1.0 / A)), 1e3)

    p = float(p_init)
    K_init = mixture.wilson_K(T, p)
    lnK = np.log(K_init)
    lnp = np.log(p)

    for it in range(maxiter):
        try:
            R, J, K, x = _newton_cubic_dew_residual_jac_p(lnK, lnp, T, z, mixture)
        except (RuntimeError, NotImplementedError):
            return dew_point_p(T, z, mixture, p_init=float(np.exp(lnp)),
                               tol=tol, maxiter=maxiter + 20)

        res = float(np.max(np.abs(R)))
        if res < tol:
            p_final = float(np.exp(lnp))
            try:
                rho_L = mixture.density_from_pressure(p_final, T, x, phase_hint='liquid')
            except RuntimeError:
                rho_L = mixture.density_from_pressure(p_final, T, x, phase_hint='vapor')
            try:
                rho_V = mixture.density_from_pressure(p_final, T, z, phase_hint='vapor')
            except RuntimeError:
                rho_V = mixture.density_from_pressure(p_final, T, z, phase_hint='liquid')
            cal = mixture.caloric(rho_V, T, z, p=p_final)
            return CubicFlashResult(
                phase='dew', T=T, p=p_final, beta=1.0,
                x=x, y=z.copy(), z=z, rho=rho_V,
                rho_L=rho_L, rho_V=rho_V,
                h=cal["h"], s=cal["s"], iterations=it + 1, K=K,
            )

        try:
            dX = -np.linalg.solve(J, R)
        except np.linalg.LinAlgError:
            return dew_point_p(T, z, mixture, p_init=float(np.exp(lnp)),
                               tol=tol, maxiter=maxiter + 20)

        dX_max = float(np.max(np.abs(dX)))
        if dX_max > step_cap:
            dX = dX * (step_cap / dX_max)

        lnK = lnK + dX[:N]
        lnp = lnp + dX[N]

    return dew_point_p(T, z, mixture, p_init=float(np.exp(lnp)),
                      tol=tol, maxiter=maxiter + 30)


def newton_dew_point_T(p, z, mixture, T_init=None, tol=1e-10, maxiter=25,
                       step_cap=0.5):
    """Newton-Raphson dew-point temperature for a cubic mixture at fixed p."""
    z = np.asarray(z, dtype=np.float64); z = z / z.sum()
    N = len(z)

    if T_init is None:
        T_init = float(np.dot(z, [c.T_c for c in mixture.components]))

    T = float(T_init)
    K_init = mixture.wilson_K(T, p)
    lnK = np.log(K_init)
    lnT = np.log(T)

    for it in range(maxiter):
        try:
            R, J, K, x = _newton_cubic_dew_residual_jac_T(lnK, lnT, p, z, mixture)
        except (RuntimeError, NotImplementedError):
            return dew_point_T(p, z, mixture, T_init=float(np.exp(lnT)),
                               tol=tol, maxiter=maxiter + 20)

        res = float(np.max(np.abs(R)))
        if res < tol:
            T_final = float(np.exp(lnT))
            try:
                rho_L = mixture.density_from_pressure(p, T_final, x, phase_hint='liquid')
            except RuntimeError:
                rho_L = mixture.density_from_pressure(p, T_final, x, phase_hint='vapor')
            try:
                rho_V = mixture.density_from_pressure(p, T_final, z, phase_hint='vapor')
            except RuntimeError:
                rho_V = mixture.density_from_pressure(p, T_final, z, phase_hint='liquid')
            cal = mixture.caloric(rho_V, T_final, z, p=p)
            return CubicFlashResult(
                phase='dew', T=T_final, p=p, beta=1.0,
                x=x, y=z.copy(), z=z, rho=rho_V,
                rho_L=rho_L, rho_V=rho_V,
                h=cal["h"], s=cal["s"], iterations=it + 1, K=K,
            )

        try:
            dX = -np.linalg.solve(J, R)
        except np.linalg.LinAlgError:
            return dew_point_T(p, z, mixture, T_init=float(np.exp(lnT)),
                               tol=tol, maxiter=maxiter + 20)

        dX_max = float(np.max(np.abs(dX)))
        if dX_max > step_cap:
            dX = dX * (step_cap / dX_max)

        lnK = lnK + dX[:N]
        lnT = lnT + dX[N]

    return dew_point_T(p, z, mixture, T_init=float(np.exp(lnT)),
                      tol=tol, maxiter=maxiter + 30)


# -------------------------------------------------------------------------
# State-function flashes for cubic mixtures
#
# Pattern: outer 1-D Newton-secant in the state variable (T or p) with
# bracketed bisection fallback; inner flash_pt at each iterate. The
# Helmholtz-mixture versions in stateprop.mixture.flash follow the same
# pattern and are thoroughly tested.
# -------------------------------------------------------------------------

def _safe_step(step, T_or_p, frac_cap=0.2):
    """Cap a Newton step so we don't move more than frac_cap of the current T or p."""
    cap = frac_cap * abs(T_or_p)
    if abs(step) > cap:
        step = cap * np.sign(step)
    return step


def flash_ph(p, h_target, z, mixture, T_init=None, tol=1e-5, maxiter=60):
    """PH flash for a cubic mixture: given (p, h_target, z), find T (and phase).

    Outer 1-D secant in T; inner PT flash at each iterate. Since mixture
    enthalpy increases monotonically with T (at fixed p) in the single-phase
    region and varies continuously through the two-phase dome (discontinuous
    *derivative* but not value), secant converges quickly.

    Returns a CubicFlashResult at the solved (T, p).
    """
    z = np.asarray(z, dtype=np.float64); z = z / z.sum()

    if T_init is None:
        T_init = float(np.dot(z, mixture.T_c))
    T = max(float(T_init), 50.0)

    h_scale = max(1.0, abs(h_target))
    # Maintain a bracket once discovered
    T_lo = T_hi = None
    h_lo = h_hi = None
    last_T = last_dh = None   # for secant

    for it in range(maxiter):
        r = flash_pt(p, T, z, mixture, tol=1e-9)
        dh = r.h - h_target
        if abs(dh) < tol * h_scale:
            return r

        # Update bracket
        if dh < 0:
            if T_lo is None or T > T_lo:
                T_lo, h_lo = T, r.h
        else:
            if T_hi is None or T < T_hi:
                T_hi, h_hi = T, r.h

        # Compute dh/dT via small step (= Cp of mixture)
        dT = max(0.05, 0.001 * T)
        r2 = flash_pt(p, T + dT, z, mixture, tol=1e-9)
        cp_est = (r2.h - r.h) / dT

        if cp_est <= 0 or not np.isfinite(cp_est):
            # Bad derivative; fall back to bisection if bracket available
            if T_lo is not None and T_hi is not None:
                T = 0.5 * (T_lo + T_hi)
                continue
            T = T * 1.1 if dh < 0 else T * 0.9
            continue

        step = -dh / cp_est
        step = _safe_step(step, T, frac_cap=0.2)
        T_new = T + step
        if T_new <= 0:
            T_new = 0.5 * T

        # If we have a bracket, project T_new into it
        if T_lo is not None and T_hi is not None:
            if not (min(T_lo, T_hi) <= T_new <= max(T_lo, T_hi)):
                T_new = 0.5 * (T_lo + T_hi)

        last_T = T
        T = T_new

    raise RuntimeError(
        f"flash_ph did not converge (p={p}, h_target={h_target}); final T={T}"
    )


def flash_ps(p, s_target, z, mixture, T_init=None, tol=1e-5, maxiter=60):
    """PS flash: given (p, s_target, z), find T.

    Outer secant in T, using (dS/dT)_p = Cp/T as the derivative estimate.
    """
    z = np.asarray(z, dtype=np.float64); z = z / z.sum()

    if T_init is None:
        T_init = float(np.dot(z, mixture.T_c))
    T = max(float(T_init), 50.0)

    s_scale = max(1.0, abs(s_target))
    T_lo = T_hi = None

    for it in range(maxiter):
        r = flash_pt(p, T, z, mixture, tol=1e-9)
        ds = r.s - s_target
        if abs(ds) < tol * s_scale:
            return r

        if ds < 0:
            if T_lo is None or T > T_lo:
                T_lo = T
        else:
            if T_hi is None or T < T_hi:
                T_hi = T

        dT = max(0.05, 0.001 * T)
        r2 = flash_pt(p, T + dT, z, mixture, tol=1e-9)
        dsdT_est = (r2.s - r.s) / dT

        if dsdT_est <= 0 or not np.isfinite(dsdT_est):
            if T_lo is not None and T_hi is not None:
                T = 0.5 * (T_lo + T_hi)
                continue
            T = T * 1.1 if ds < 0 else T * 0.9
            continue

        step = -ds / dsdT_est
        step = _safe_step(step, T, frac_cap=0.2)
        T_new = T + step
        if T_new <= 0:
            T_new = 0.5 * T
        if T_lo is not None and T_hi is not None:
            if not (min(T_lo, T_hi) <= T_new <= max(T_lo, T_hi)):
                T_new = 0.5 * (T_lo + T_hi)
        T = T_new

    raise RuntimeError(
        f"flash_ps did not converge (p={p}, s_target={s_target}); final T={T}"
    )


def flash_th(T, h_target, z, mixture, p_init=None, tol=1e-5, maxiter=60):
    """TH flash: given (T, h_target, z), find p.

    Note: at subcritical T, h vs p is multi-valued inside the 2-phase band.
    We iterate on ln(p) with secant, seeding from a single-phase initial
    guess (critical-pressure-scale).

    dh/d(ln p) at constant T is NOT simply Cp; but a numeric derivative via
    small ln p step is used.
    """
    z = np.asarray(z, dtype=np.float64); z = z / z.sum()

    if p_init is None:
        p_init = float(np.dot(z, mixture.p_c))   # mole-average pc
    ln_p = np.log(max(float(p_init), 1e3))

    h_scale = max(1.0, abs(h_target))
    ln_p_lo = ln_p_hi = None

    for it in range(maxiter):
        p = np.exp(ln_p)
        r = flash_pt(p, T, z, mixture, tol=1e-9)
        dh = r.h - h_target
        if abs(dh) < tol * h_scale:
            return r

        if dh < 0:
            if ln_p_lo is None or ln_p < ln_p_lo:
                ln_p_lo = ln_p   # higher p = lower h in many regions... but depends
        # Actually h vs p is nuanced: in gas h ~ weakly decreasing with p, in
        # liquid ~ weakly increasing. Skip bracket maintenance for TH.

        dln_p = 0.01
        r2 = flash_pt(np.exp(ln_p + dln_p), T, z, mixture, tol=1e-9)
        dh_dlnp = (r2.h - r.h) / dln_p

        if abs(dh_dlnp) < 1e-9:
            ln_p += 0.2 * np.sign(-dh)
            continue

        step = -dh / dh_dlnp
        if abs(step) > 0.5:
            step = 0.5 * np.sign(step)
        ln_p += step
        ln_p = max(min(ln_p, np.log(1e10)), np.log(1.0))   # bound between 1 Pa and 10 GPa

    raise RuntimeError(
        f"flash_th did not converge (T={T}, h_target={h_target}); final p={np.exp(ln_p)}"
    )


def flash_ts(T, s_target, z, mixture, p_init=None, tol=1e-5, maxiter=60):
    """TS flash: given (T, s_target, z), find p.

    Outer secant in ln(p); inner PT flash at each iterate.
    """
    z = np.asarray(z, dtype=np.float64); z = z / z.sum()

    if p_init is None:
        p_init = float(np.dot(z, mixture.p_c))
    ln_p = np.log(max(float(p_init), 1e3))

    s_scale = max(1.0, abs(s_target))

    for it in range(maxiter):
        p = np.exp(ln_p)
        r = flash_pt(p, T, z, mixture, tol=1e-9)
        ds = r.s - s_target
        if abs(ds) < tol * s_scale:
            return r

        dln_p = 0.01
        r2 = flash_pt(np.exp(ln_p + dln_p), T, z, mixture, tol=1e-9)
        ds_dlnp = (r2.s - r.s) / dln_p

        if abs(ds_dlnp) < 1e-9:
            ln_p += 0.2 * np.sign(-ds)
            continue

        step = -ds / ds_dlnp
        if abs(step) > 0.5:
            step = 0.5 * np.sign(step)
        ln_p += step
        ln_p = max(min(ln_p, np.log(1e10)), np.log(1.0))

    raise RuntimeError(
        f"flash_ts did not converge (T={T}, s_target={s_target}); final p={np.exp(ln_p)}"
    )


# ---------------------------------------------------------------------------
# TV and UV flashes -- natural-variable flashes for dynamic simulation
# ---------------------------------------------------------------------------
#
# These mirror the Helmholtz-mixture versions in stateprop/mixture/flash.py.
# The inner flash_tv iterates on pressure at fixed T to match a target
# molar volume; the outer flash_uv iterates on T to match internal energy.

def flash_tv(T, v_target, z, mixture, p_init=None, tol=1e-8, maxiter=60):
    """TV flash for a cubic mixture.

    Given (T, v_target, z), find the pressure such that the bulk mixture
    molar volume equals v_target. Returns a CubicFlashResult.

    Algorithm: secant in ln(p) with bracket expansion and stagnation
    detection (to handle the density-resolution limit in dense-liquid
    regions where flash_pt's internal density solver is near its own
    numerical floor).
    """
    z = np.asarray(z, dtype=np.float64); z = z / z.sum()

    R = 8.314472   # J/(mol K) -- cubic uses a fixed R by convention

    # Initial guess: ideal-gas p for vapor; moderate 1 MPa for liquid
    v_ref_vapor = R * T / 1e5
    if p_init is None:
        if v_target < 0.01 * v_ref_vapor:
            p_init = 1e6
        else:
            p_init = R * T / v_target
    p = max(float(p_init), 1.0)

    def v_residual(p_val):
        r = flash_pt(p_val, T, z, mixture, check_stability=False, tol=1e-10)
        return 1.0 / r.rho - v_target, r

    res1, r1 = v_residual(p)
    if abs(res1) < tol * abs(v_target):
        return r1
    p2 = p * 2.0 if res1 > 0 else p * 0.5
    res2, r2 = v_residual(p2)
    if abs(res2) < tol * abs(v_target):
        return r2

    best_res, best_r = (res1, r1) if abs(res1) <= abs(res2) else (res2, r2)

    # Bracket search
    expand = 0
    while res1 * res2 > 0 and expand < 20:
        if abs(res2) < abs(res1):
            p, res1, r1 = p2, res2, r2
            p2 = p2 * 2.0 if res2 > 0 else p2 * 0.5
        else:
            p2 = p2 * 2.0 if res2 > 0 else p2 * 0.5
        res2, r2 = v_residual(p2)
        if abs(res2) < abs(best_res):
            best_res, best_r = res2, r2
        if abs(res2) < tol * abs(v_target):
            return r2
        expand += 1

    prev_best_res = abs(best_res)
    stagnation_count = 0
    for it in range(maxiter):
        lnp = np.log(p); lnp2 = np.log(p2)
        if abs(res2 - res1) < 1e-30:
            lnp_new = 0.5 * (lnp + lnp2)
        else:
            lnp_new = lnp2 - res2 * (lnp2 - lnp) / (res2 - res1)
        if res1 * res2 < 0:
            lo, hi = (lnp, lnp2) if lnp < lnp2 else (lnp2, lnp)
            if not (lo <= lnp_new <= hi):
                lnp_new = 0.5 * (lo + hi)
        p_new = float(np.exp(lnp_new))
        res_new, r_new = v_residual(p_new)
        if abs(res_new) < abs(best_res):
            best_res, best_r = res_new, r_new
        if abs(res_new) < tol * abs(v_target):
            return r_new

        if abs(best_res) >= prev_best_res * 0.9:
            stagnation_count += 1
            if stagnation_count >= 3 and abs(best_res) < 1e-5 * abs(v_target):
                return best_r
        else:
            stagnation_count = 0
        prev_best_res = abs(best_res)

        if res1 * res2 < 0:
            if res_new * res1 < 0:
                p2, res2, r2 = p_new, res_new, r_new
            else:
                p, res1, r1 = p_new, res_new, r_new
        else:
            p, res1, r1 = p2, res2, r2
            p2, res2, r2 = p_new, res_new, r_new

    if abs(best_res) < 1e-4 * abs(v_target):
        return best_r
    raise RuntimeError(
        f"cubic flash_tv did not converge (T={T}, v={v_target}, "
        f"last p={p2:.3e}, last v={1.0/r2.rho:.3e}, "
        f"best |res|={abs(best_res):.3e})"
    )


def flash_uv(u_target, v_target, z, mixture,
             T_init=None, tol=1e-6, maxiter=40):
    """UV flash for a cubic mixture.

    Given (u_target, v_target, z), find (T, p, phase). Outer 1-D Newton on
    T with (du/dT)_v ~ cv_mix approximated by finite difference through
    flash_tv.

    The CubicFlashResult stores h and rho (and p); u is recovered via
    u = h - p/rho at the converged state.
    """
    z = np.asarray(z, dtype=np.float64); z = z / z.sum()

    if T_init is None:
        T_init = float(np.dot(z, mixture.T_c))
    T = max(float(T_init), 50.0)

    u_scale = max(1.0, abs(u_target))
    best_diff = np.inf
    best_result = None
    last_p = None

    for it in range(maxiter):
        r = flash_tv(T, v_target, z, mixture, p_init=last_p, tol=1e-10)
        u_calc = r.h - r.p / r.rho
        diff = u_calc - u_target

        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_result = r

        if abs(diff) < tol * u_scale:
            return r

        last_p = r.p
        dT = max(0.01, 1e-4 * T)
        try:
            r2 = flash_tv(T + dT, v_target, z, mixture, p_init=last_p, tol=1e-10)
            u2 = r2.h - r2.p / r2.rho
            cv_est = (u2 - u_calc) / dT
        except RuntimeError:
            r2 = flash_tv(T - dT, v_target, z, mixture, p_init=last_p, tol=1e-10)
            u2 = r2.h - r2.p / r2.rho
            cv_est = (u_calc - u2) / dT

        if cv_est <= 0 or not np.isfinite(cv_est):
            T = T * (1.05 if diff < 0 else 0.95)
            continue

        step = -diff / cv_est
        if abs(step) > 0.2 * T:
            step = 0.2 * T * np.sign(step)
        T_new = T + step
        if T_new <= 0:
            T_new = 0.5 * T
        T = T_new

    if best_result is not None and best_diff < tol * u_scale * 100:
        return best_result
    raise RuntimeError(
        f"cubic flash_uv did not converge (u={u_target}, v={v_target}, "
        f"last T={T}, last |du|={best_diff:.3e})"
    )


# -----------------------------------------------------------------------
# PV flash and Pα / Tα flash modes (v0.9.56)
# -----------------------------------------------------------------------


def flash_pv(p, v_target, z, mixture, T_init=None, tol=1e-8, maxiter=60):
    """PV flash for a cubic mixture.

    Given (p, v_target, z), find the temperature T such that the bulk
    mixture molar volume (1/rho) at fixed p equals v_target. Symmetric
    counterpart of `flash_tv` (which solves for p at fixed T).

    Useful for:
    - Throttling-valve and aftercooler analysis where pressure and
      total volume of the inventory are specified.
    - Tank/cylinder calculations: known pressure, known geometric
      volume, known total moles -> molar volume, find equilibrium T.

    Algorithm: secant in ln(T) with bracket. Exterior branch (vapor at
    high T, liquid at low T) gives v_total monotonically decreasing in
    T at fixed p; the two-phase region produces a discontinuity in
    dv/dT but the value remains continuous. Bracket-projecting secant
    handles both regimes robustly.
    """
    z = np.asarray(z, dtype=np.float64); z = z / z.sum()

    R = 8.314472

    if T_init is None:
        # Heuristic: ideal-gas T at this v gives the vapor branch start;
        # for low v, start near critical T weighted by composition.
        T_ig = p * v_target / R
        if T_ig > 50.0 and T_ig < 2000.0:
            T_init = T_ig
        else:
            T_init = float(np.dot(z, mixture.T_c)) * 0.9
    T = max(float(T_init), 50.0)

    def v_residual(T_val):
        r = flash_pt(p, T_val, z, mixture, check_stability=False, tol=1e-10)
        return 1.0 / r.rho - v_target, r

    res1, r1 = v_residual(T)
    if abs(res1) < tol * abs(v_target):
        return r1
    # Initial second point
    T2 = T * 1.5 if res1 < 0 else T * 0.7
    T2 = max(T2, 50.0)
    res2, r2 = v_residual(T2)
    if abs(res2) < tol * abs(v_target):
        return r2

    best_res, best_r = (res1, r1) if abs(res1) <= abs(res2) else (res2, r2)

    # Bracket search by expanding T2
    expand = 0
    while res1 * res2 > 0 and expand < 20:
        if abs(res2) < abs(res1):
            T, res1, r1 = T2, res2, r2
            T2 = T2 * 1.5 if res2 < 0 else T2 * 0.7
        else:
            T2 = T2 * 1.5 if res2 < 0 else T2 * 0.7
        T2 = max(T2, 50.0)
        res2, r2 = v_residual(T2)
        if abs(res2) < abs(best_res):
            best_res, best_r = res2, r2
        if abs(res2) < tol * abs(v_target):
            return r2
        expand += 1

    prev_best = abs(best_res)
    stagnation = 0
    for it in range(maxiter):
        lnT, lnT2 = np.log(T), np.log(T2)
        if abs(res2 - res1) < 1e-30:
            lnT_new = 0.5 * (lnT + lnT2)
        else:
            lnT_new = lnT2 - res2 * (lnT2 - lnT) / (res2 - res1)
        if res1 * res2 < 0:
            lo, hi = (lnT, lnT2) if lnT < lnT2 else (lnT2, lnT)
            if not (lo <= lnT_new <= hi):
                lnT_new = 0.5 * (lo + hi)
        T_new = float(np.exp(lnT_new))
        T_new = max(T_new, 50.0)
        res_new, r_new = v_residual(T_new)
        if abs(res_new) < abs(best_res):
            best_res, best_r = res_new, r_new
        if abs(res_new) < tol * abs(v_target):
            return r_new

        if abs(best_res) >= prev_best * 0.9:
            stagnation += 1
            if stagnation >= 3 and abs(best_res) < 1e-5 * abs(v_target):
                return best_r
        else:
            stagnation = 0
        prev_best = abs(best_res)

        if res1 * res2 < 0:
            if res_new * res1 < 0:
                T2, res2, r2 = T_new, res_new, r_new
            else:
                T, res1, r1 = T_new, res_new, r_new
        else:
            T, res1, r1 = T2, res2, r2
            T2, res2, r2 = T_new, res_new, r_new

    if abs(best_res) < 1e-4 * abs(v_target):
        return best_r
    raise RuntimeError(
        f"cubic flash_pv did not converge (p={p}, v={v_target}, "
        f"last T={T2:.3e}, last v={1.0/r2.rho:.3e}, "
        f"best |res|={abs(best_res):.3e})"
    )


def flash_p_alpha(p, alpha, z, mixture, T_init=None, tol=1e-7, maxiter=60):
    """Specified-vapor-fraction flash at fixed pressure.

    Given (p, alpha, z) where alpha is the desired vapor mole fraction
    (0 = bubble point, 1 = dew point, anything in between is a partial
    vaporization at fixed p), find the temperature T at which the
    isothermal flash yields beta = alpha.

    Special cases:
      alpha = 0  -> equivalent to bubble_point_T (calls newton_bubble_point_T)
      alpha = 1  -> equivalent to dew_point_T   (calls newton_dew_point_T)
      0 < alpha < 1 -> internal secant on T using flash_pt residuals

    Use cases:
      - Phase envelope construction at constant pressure (sweep alpha).
      - Sizing flash drums for partial vaporization streams.
      - Constructing T-x-y diagrams: alpha = 0 gives bubble curve in
        terms of x, alpha = 1 gives dew curve in terms of y.

    The algorithm brackets the solution between the bubble T (where
    beta = 0) and the dew T (where beta = 1), then secants on T using
    the residual r.beta - alpha. The bracket is a guaranteed solution
    interval since beta increases monotonically with T at fixed p
    inside the two-phase dome.
    """
    z = np.asarray(z, dtype=np.float64); z = z / z.sum()

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    # Special endpoints reduce to bubble/dew solvers
    if alpha == 0.0:
        return newton_bubble_point_T(p, z, mixture, T_init=T_init,
                                        tol=tol, maxiter=maxiter)
    if alpha == 1.0:
        return newton_dew_point_T(p, z, mixture, T_init=T_init,
                                     tol=tol, maxiter=maxiter)

    # Find bubble T and dew T to establish bracket
    try:
        bub = newton_bubble_point_T(p, z, mixture, T_init=T_init, tol=1e-8)
        T_bub = bub.T
    except Exception as e:
        raise RuntimeError(f"flash_p_alpha: bubble T failed at p={p}: {e}")
    try:
        dew = newton_dew_point_T(p, z, mixture, T_init=T_bub * 1.05, tol=1e-8)
        T_dew = dew.T
    except Exception as e:
        raise RuntimeError(f"flash_p_alpha: dew T failed at p={p}: {e}")

    if T_dew <= T_bub:
        raise RuntimeError(
            f"flash_p_alpha: degenerate envelope (T_bub={T_bub}, T_dew={T_dew})"
        )

    def beta_residual(T_val):
        r = flash_pt(p, T_val, z, mixture, check_stability=False, tol=1e-10)
        beta = r.beta if r.beta is not None else (1.0 if r.phase == 'vapor' else 0.0)
        return beta - alpha, r

    # Initial linear interpolation in T between T_bub and T_dew, weighted
    # by alpha (good first guess for many systems).
    T_lo, T_hi = T_bub, T_dew
    T = T_lo + alpha * (T_hi - T_lo)
    res, r = beta_residual(T)
    if abs(res) < tol:
        return r

    # Maintain bracket [T_lo, T_hi] with res < 0 at T_lo, res > 0 at T_hi
    T_a, res_a = T_lo, -alpha           # at bubble, beta = 0, res = -alpha
    T_b, res_b = T_hi, 1.0 - alpha      # at dew,    beta = 1, res = 1-alpha
    # Insert current point into the bracket
    if res < 0:
        T_a, res_a = T, res
    else:
        T_b, res_b = T, res

    last_T = T
    for it in range(maxiter):
        # Secant in T using the bracket endpoints
        if abs(res_b - res_a) < 1e-30:
            T_new = 0.5 * (T_a + T_b)
        else:
            T_new = T_b - res_b * (T_b - T_a) / (res_b - res_a)
        # Project into bracket
        if not (min(T_a, T_b) < T_new < max(T_a, T_b)):
            T_new = 0.5 * (T_a + T_b)

        res_new, r_new = beta_residual(T_new)
        if abs(res_new) < tol:
            return r_new

        if res_new < 0:
            T_a, res_a = T_new, res_new
        else:
            T_b, res_b = T_new, res_new

        # Stagnation guard
        if abs(T_new - last_T) < 1e-10 * T_new:
            return r_new
        last_T = T_new

    raise RuntimeError(
        f"cubic flash_p_alpha did not converge (p={p}, alpha={alpha}, "
        f"T={T_new:.3e}, |res|={abs(res_new):.3e}); bracket=[{T_a:.3f}, "
        f"{T_b:.3f}]"
    )


def flash_t_alpha(T, alpha, z, mixture, p_init=None, tol=1e-7, maxiter=60):
    """Specified-vapor-fraction flash at fixed temperature.

    Given (T, alpha, z) where alpha is the desired vapor mole fraction,
    find the pressure p at which the isothermal flash yields beta =
    alpha.

    Special cases:
      alpha = 0  -> bubble pressure (calls newton_bubble_point_p)
      alpha = 1  -> dew pressure   (calls newton_dew_point_p)
      0 < alpha < 1 -> internal secant on ln(p)

    Use cases:
      - Phase envelope at constant temperature.
      - Constructing P-x-y diagrams.
      - Determining condensation pressure for partial-condensation
        process design.

    Algorithm: bracket [p_dew, p_bubble] (note: bubble p > dew p at
    fixed T, opposite ordering from temperature flash), secant on
    ln(p) using residual beta(p) - alpha. Inside the dome,
    beta DECREASES as p increases (compression liquefies vapor).
    """
    z = np.asarray(z, dtype=np.float64); z = z / z.sum()

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    if alpha == 0.0:
        return newton_bubble_point_p(T, z, mixture, p_init=p_init,
                                        tol=tol, maxiter=maxiter)
    if alpha == 1.0:
        return newton_dew_point_p(T, z, mixture, p_init=p_init,
                                     tol=tol, maxiter=maxiter)

    try:
        bub = newton_bubble_point_p(T, z, mixture, p_init=p_init, tol=1e-8)
        p_bub = bub.p
    except Exception as e:
        raise RuntimeError(f"flash_t_alpha: bubble p failed at T={T}: {e}")
    try:
        dew = newton_dew_point_p(T, z, mixture, p_init=p_bub * 0.5, tol=1e-8)
        p_dew = dew.p
    except Exception as e:
        raise RuntimeError(f"flash_t_alpha: dew p failed at T={T}: {e}")

    if p_dew >= p_bub:
        raise RuntimeError(
            f"flash_t_alpha: degenerate envelope (p_bub={p_bub}, p_dew={p_dew})"
        )

    def beta_residual(p_val):
        r = flash_pt(p_val, T, z, mixture, check_stability=False, tol=1e-10)
        beta = r.beta if r.beta is not None else (1.0 if r.phase == 'vapor' else 0.0)
        return beta - alpha, r

    # Bracket: at p = p_bub, beta=0 (res=-alpha); at p = p_dew, beta=1
    # (res=1-alpha). beta is monotonically decreasing in p; bracket is
    # [p_dew, p_bub] with res(p_bub) < 0, res(p_dew) > 0.
    p_a, res_a = p_dew, 1.0 - alpha     # higher residual at lower p
    p_b, res_b = p_bub, -alpha
    # Linear-in-alpha guess in ln(p)
    lnp_dew, lnp_bub = np.log(p_dew), np.log(p_bub)
    p = float(np.exp(lnp_dew + alpha * (lnp_bub - lnp_dew)))
    res, r = beta_residual(p)
    if abs(res) < tol:
        return r

    # Insert current point into bracket
    if res > 0:
        p_a, res_a = p, res
    else:
        p_b, res_b = p, res

    last_p = p
    for it in range(maxiter):
        lnpa, lnpb = np.log(p_a), np.log(p_b)
        if abs(res_b - res_a) < 1e-30:
            lnp_new = 0.5 * (lnpa + lnpb)
        else:
            lnp_new = lnpb - res_b * (lnpb - lnpa) / (res_b - res_a)
        if not (min(lnpa, lnpb) < lnp_new < max(lnpa, lnpb)):
            lnp_new = 0.5 * (lnpa + lnpb)
        p_new = float(np.exp(lnp_new))

        res_new, r_new = beta_residual(p_new)
        if abs(res_new) < tol:
            return r_new

        if res_new > 0:
            p_a, res_a = p_new, res_new
        else:
            p_b, res_b = p_new, res_new

        if abs(p_new - last_p) < 1e-10 * p_new:
            return r_new
        last_p = p_new

    raise RuntimeError(
        f"cubic flash_t_alpha did not converge (T={T}, alpha={alpha}, "
        f"p={p_new:.3e}, |res|={abs(res_new):.3e}); bracket=[{p_a:.3e}, "
        f"{p_b:.3e}]"
    )
