"""
Michelsen tangent-plane stability analysis.

Given a feed composition z at (T, p), determine whether the single-phase
assumption is thermodynamically stable. If not, return K-factors suitable
for initializing a two-phase flash.

Algorithm (Michelsen 1982):

  Define the tangent-plane distance function
    tm(W) = sum_i W_i * [ln(W_i) + ln(phi_i(W)) - d_i] + 1 - sum_i W_i

  where W is a trial phase (not normalized), and
    d_i = ln(z_i) + ln(phi_i(z))    (evaluated once for the feed)

  The mixture is unstable iff there exists W such that the reduced tpd
    tpd(W) = tm(W) / (sum_i W_i)  <  0.

  Converting to trial mole amounts W with y_i = W_i / sum(W), the
  iteration is successive substitution on W:
    ln(W_i^new) = d_i - ln(phi_i(W))

  Then we check:
    - Convergence to trivial solution (W -> z): stable for that trial
    - Convergence to nontrivial min with tpd < 0: unstable

We run two trials: one vapor-like (W = z * K_Wilson), one liquid-like
(W = z / K_Wilson). Unstable if either trial finds a negative tpd.
"""
import numpy as np
from .properties import ln_phi, density_from_pressure


def wilson_K(T, p, mixture):
    """Wilson K-factor correlation.

    K_i = (p_c_i / p) * exp(5.373 * (1 + omega_i) * (1 - T_c_i/T))

    Uses the pure-component T_c, p_c, and omega (acentric factor).
    Falls back to omega = 0 (equivalent to Raoult's law with vapor pressure
    from Clausius-Clapeyron) if a component lacks acentric factor data.
    """
    K = np.zeros(mixture.N)
    for i, comp in enumerate(mixture.components):
        fl = comp.fluid
        Tc_i = fl.T_c
        pc_i = fl.p_c
        # Try to read acentric factor from fluid data; else 0
        omega = getattr(fl, "acentric_factor", 0.0)
        K[i] = (pc_i / p) * np.exp(5.373 * (1.0 + omega) * (1.0 - Tc_i / T))
    return K


def _tpd_reduced(W, d, rho, T, mixture):
    """Compute tangent-plane distance (reduced) at trial W."""
    S = W.sum()
    y = W / S
    rho_trial = rho  # will be replaced by actual density of trial
    # Density of trial phase at (T, p) -- but we have rho, not p, here.
    # Strategy: the caller passes the trial density from a preceding step.
    lnphi_W = ln_phi(rho_trial, T, y, mixture)
    # tpd_reduced = 1/S * [ sum W_i (ln W_i + ln phi_i(y) - d_i) ] + 1 - 1
    #             = sum y_i (ln W_i + ln phi_i(y) - d_i)
    tpd = 0.0
    for i in range(len(W)):
        if W[i] <= 0:
            continue
        tpd += y[i] * (np.log(W[i]) + lnphi_W[i] - d[i])
    return tpd


def stability_test_TPD(z, T, p, mixture, tol=1e-8, maxiter=50, verbose=False):
    """Michelsen tangent-plane stability analysis.

    The mixture is **unstable** as a single phase at (T, p, z) if and only if
    there exists a trial composition y such that the tangent-plane distance
    function is negative:

        tpd(y) = sum_i y_i [ln y_i + ln phi_i(y) - ln z_i - ln phi_i(z)]  <  0

    Michelsen's method searches for stationary points of tpd using an
    auxiliary trial mole-number vector W. Defining d_i = ln z_i + ln phi_i(z),
    the successive-substitution iteration is

        ln W_i^new = d_i - ln phi_i(y),    y = W / sum(W)

    At a CONVERGED stationary point, every W_i satisfies ln W_i + ln phi_i(y)
    - d_i = 0, which gives (via the modified TPD definition) the compact
    stability indicator

        tm* = 1 - sum(W) = 1 - S

    So the sign of (S - 1) at convergence determines stability from that trial:

        S > 1  -> unstable (tpd < 0 in a neighborhood of y)
        S < 1  -> stable from that trial
        S ~ 1  -> marginally stable / on the phase boundary

    We run two trials from Wilson K-factor initialization: a vapor-like trial
    (W = z * K_Wilson) and a liquid-like trial (W = z / K_Wilson). The
    mixture is unstable iff EITHER trial converges to a nontrivial stationary
    point with S > 1 + tol.

    Parameters
    ----------
    z : array (N,)     feed composition (mole fractions)
    T, p : floats      state conditions
    mixture : Mixture
    tol : float        tolerance on (S - 1) to declare instability
    maxiter : int
    verbose : bool     print diagnostic info per trial

    Returns
    -------
    stable : bool      True if the feed is stable as single phase
    K_best : ndarray   K-factors for initializing a flash if unstable
    S_minus_1 : float  max of (S-1) across both trials; > tol means unstable.
                        (Historically called a "tpd-like" indicator.)
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()
    N = mixture.N

    # Reference: Gibbs-minimizing density at feed composition.
    # Wrap in try/except: at conditions far from any phase boundary the
    # opposite branch may not exist (e.g. compressed liquid has no real
    # vapor root) and Newton fails. Accept whichever branch converged.
    try:
        rho_z_vap = density_from_pressure(p, T, z, mixture, phase_hint="vapor")
    except RuntimeError:
        rho_z_vap = None
    try:
        rho_z_liq = density_from_pressure(p, T, z, mixture, phase_hint="liquid")
    except RuntimeError:
        rho_z_liq = None
    if rho_z_vap is None and rho_z_liq is None:
        raise RuntimeError(
            f"stability_test_TPD: no density branch converged at p={p}, T={T}"
        )
    if rho_z_vap is None:
        rho_z = rho_z_liq
    elif rho_z_liq is None:
        rho_z = rho_z_vap
    elif abs(rho_z_vap - rho_z_liq) / max(rho_z_vap, rho_z_liq) < 1e-4:
        rho_z = rho_z_vap
    else:
        from .properties import _pure_caloric
        gv = _pure_caloric(rho_z_vap, T, mixture, z)
        gl = _pure_caloric(rho_z_liq, T, mixture, z)
        g_vap = gv["h"] - T * gv["s"]
        g_liq = gl["h"] - T * gl["s"]
        rho_z = rho_z_vap if g_vap < g_liq else rho_z_liq

    lnphi_z = ln_phi(rho_z, T, z, mixture)
    d = np.log(z + 1e-300) + lnphi_z     # component-wise reference

    K_wilson = wilson_K(T, p, mixture)

    # Track the max (S-1) across trials; > tol means unstable
    max_S_minus_1 = -np.inf
    best_K = K_wilson.copy()
    unstable = False

    for direction in ("vapor", "liquid"):
        if direction == "vapor":
            W = z * K_wilson.copy()
        else:
            W = z / K_wilson.copy()
        phase_hint = direction
        # Warm-start density cache for this trial direction. Across trial-phase
        # iterations the composition y barely moves once we're past the first
        # 2-3 iters, so the previous density is an excellent initial guess for
        # the next density solve. This typically cuts each density solve from
        # 5 Newton iters cold-start to 1-2 warm-started.
        rho_W_prev = None

        trivial = False
        converged = False

        # Inner-loop solver: SS warm-up then Broyden's "good" method on ln W.
        # Same accelerator pattern as the flash _successive_substitution: SS
        # is robust globally but converges only linearly (rate ~|d lnphi/dW|);
        # Broyden's secant updates achieve super-linear convergence at the
        # same per-iter cost (one residual evaluation), so the trial-phase
        # iteration drops from typical ~15-30 SS iters to ~6-10 hybrid iters
        # for strongly non-ideal systems. Identity initialization H^0 = I
        # makes the first Broyden step IS a SS step (since the SS update
        # is exactly W_new = exp(d - lnphi_W) = exp(ln_W - F)), so there is
        # no extra setup cost for the switchover.
        SS_WARMUP = 4
        N = len(z)

        def _residual(W_curr):
            """Returns (F, ln_W, y, rho_W, lnphi_W) at the current trial-phase
            estimate W_curr, or None if density solve fails or W_curr is
            unphysical. F = ln_W - (d - lnphi_W) is the residual to drive
            to zero; the SS update is the special case W_new = exp(ln_W - F).
            """
            nonlocal rho_W_prev
            S_ = W_curr.sum()
            if S_ <= 0 or not np.all(np.isfinite(W_curr)):
                return None
            y_ = W_curr / S_
            try:
                rho_W_ = density_from_pressure(p, T, y_, mixture,
                                               phase_hint=phase_hint,
                                               rho_init=rho_W_prev)
            except RuntimeError:
                return None
            rho_W_prev = rho_W_
            lnphi_W_ = ln_phi(rho_W_, T, y_, mixture)
            ln_W_ = np.log(W_curr + 1e-300)
            F_ = ln_W_ - (d - lnphi_W_)
            return F_, ln_W_, y_, rho_W_, lnphi_W_

        # ---- SS warm-up phase ----
        for it in range(min(SS_WARMUP, maxiter)):
            S = W.sum()
            if S <= 0 or not np.all(np.isfinite(W)):
                break
            y = W / S

            # Trivial-solution check: converging back to z
            if np.max(np.abs(y - z)) < 1e-5 and abs(S - 1.0) < 1e-4:
                trivial = True
                break

            res = _residual(W)
            if res is None:
                break
            F, ln_W, y, rho_W, lnphi_W = res

            if np.max(np.abs(F)) < 1e-9:
                converged = True
                break

            # SS update: W_new = exp(d - lnphi_W) (equivalent to ln_W - F)
            W = np.exp(d - lnphi_W)

        # ---- Broyden phase ----
        # Continue from where SS warm-up left off. Need F and ln_W consistent
        # at the current W; re-evaluate (one extra call but keeps math clean).
        if not converged and not trivial:
            res = _residual(W)
            if res is None:
                # Fall through to break-out path below; W carries last good state
                pass
            else:
                F, ln_W, y, rho_W, lnphi_W = res
                if np.max(np.abs(F)) < 1e-9:
                    converged = True
                else:
                    H = np.eye(N)
                    F_prev = F.copy()
                    ln_W_prev = ln_W.copy()
                    for it in range(SS_WARMUP, maxiter):
                        # Trivial-solution check on current W
                        S = W.sum()
                        if S > 0 and np.all(np.isfinite(W)):
                            y_check = W / S
                            if np.max(np.abs(y_check - z)) < 1e-5 and abs(S - 1.0) < 1e-4:
                                trivial = True
                                break

                        # Broyden step: delta = -H @ F (with damping)
                        delta = -H @ F_prev
                        max_step = np.max(np.abs(delta))
                        if max_step > 1.0:
                            delta = delta / max_step
                        ln_W = ln_W_prev + delta
                        W = np.exp(ln_W)
                        res = _residual(W)
                        if res is None:
                            # Fall back to one SS step from last good state and
                            # reset H to identity. If SS step also fails, give
                            # up on this trial direction.
                            W = np.exp(d - lnphi_W)
                            res = _residual(W)
                            if res is None:
                                break
                            H = np.eye(N)
                        F, ln_W, y, rho_W, lnphi_W = res
                        if np.max(np.abs(F)) < 1e-9:
                            converged = True
                            break
                        # Broyden's "good" rank-1 update of H
                        s = ln_W - ln_W_prev
                        y_vec = F - F_prev
                        Hy = H @ y_vec
                        sH = s @ H
                        denom = s @ Hy
                        if abs(denom) > 1e-30:
                            H = H + np.outer(s - Hy, sH) / denom
                        ln_W_prev = ln_W.copy()
                        F_prev = F.copy()

        if trivial:
            if verbose:
                print(f"  [{direction}] trivial convergence to feed")
            continue

        S = W.sum()
        if not np.isfinite(S) or S <= 0:
            if verbose:
                print(f"  [{direction}] iteration blew up")
            continue

        y_final = W / S
        if np.max(np.abs(y_final - z)) < 1e-5:
            if verbose:
                print(f"  [{direction}] y ~ z (trivial), S={S:.6f}")
            continue

        S_minus_1 = S - 1.0
        if verbose:
            print(f"  [{direction}] nontrivial stationary point: "
                  f"S = {S:.6f}, S-1 = {S_minus_1:.3e}, y = {y_final}, "
                  f"converged={converged}")

        if S_minus_1 > max_S_minus_1:
            max_S_minus_1 = S_minus_1
            # K-factor convention for subsequent flash:
            # For a vapor-like trial, y_final is the vapor composition candidate;
            # for liquid-like, y_final is the liquid candidate. Flash's
            # successive substitution will refine either way.
            if direction == "vapor":
                best_K = W / (z + 1e-300)
            else:
                best_K = (z + 1e-300) / W

        if S_minus_1 > tol:
            unstable = True

    if max_S_minus_1 == -np.inf:
        max_S_minus_1 = 0.0     # both trials were trivial -> stable-indeterminate; report 0

    if verbose:
        print(f"  verdict: stable={not unstable}, max(S-1)={max_S_minus_1:.3e}")

    return (not unstable), best_K, max_S_minus_1
