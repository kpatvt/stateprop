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

        for it in range(maxiter):
            S = W.sum()
            if S <= 0 or not np.all(np.isfinite(W)):
                break
            y = W / S

            if np.max(np.abs(y - z)) < 1e-5 and abs(S - 1.0) < 1e-4:
                trivial = True
                break

            try:
                rho_W = mixture.density_from_pressure(p, T, y, phase_hint=phase_hint)
            except RuntimeError:
                break
            lnphi_W = mixture.ln_phi(rho_W, T, y)

            ln_W_new = d - lnphi_W
            W_new = np.exp(ln_W_new)

            if np.max(np.abs(np.log(W_new + 1e-300) - np.log(W + 1e-300))) < 1e-9:
                W = W_new
                converged = True
                break
            W = W_new

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

    # Two-phase SS iteration on K-factors
    if K_init is None:
        K = mixture.wilson_K(T, p)
    else:
        K = K_init.copy()

    for it in range(maxiter):
        # Rachford-Rice for beta
        beta, converged_rr = rachford_rice(z, K)

        # Compositions at current beta
        x = z / (1.0 + beta * (K - 1.0))
        y = K * x

        # Normalize
        x = x / x.sum()
        y = y / y.sum()

        # Densities
        try:
            rho_L = mixture.density_from_pressure(p, T, x, phase_hint='liquid')
            rho_V = mixture.density_from_pressure(p, T, y, phase_hint='vapor')
        except RuntimeError as e:
            raise RuntimeError(f"density failure in flash iteration: {e}")

        lnphi_L = mixture.ln_phi(rho_L, T, x)
        lnphi_V = mixture.ln_phi(rho_V, T, y)

        # K update: K_new = phi_L / phi_V = exp(lnphi_L - lnphi_V)
        ln_K_new = lnphi_L - lnphi_V
        K_new = np.exp(ln_K_new)

        dK_max = np.max(np.abs(np.log(K_new + 1e-300) - np.log(K + 1e-300)))
        K = K_new

        if dK_max < tol:
            break

    # Final
    beta, _ = rachford_rice(z, K)
    x = z / (1.0 + beta * (K - 1.0))
    y = K * x
    x = x / x.sum()
    y = y / y.sum()
    rho_L = mixture.density_from_pressure(p, T, x, phase_hint='liquid')
    rho_V = mixture.density_from_pressure(p, T, y, phase_hint='vapor')

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
            h=cal["h"], s=cal["s"], iterations=it + 1, K=K,
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
        h=h_mix, s=s_mix, iterations=it + 1, K=K,
    )


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
