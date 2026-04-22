"""
Multicomponent flash algorithms.

Core primitive: isothermal-isobaric PT flash via Rachford-Rice + successive
substitution, with Wilson K-factor initialization and Michelsen stability
analysis.

State-function flashes (PH, PS, TH, TS) wrap PT via an outer 1-D Newton on
the missing variable.

Vapor-fraction flashes:
  - Tbeta (flash_tbeta): given T and target vapor fraction beta_t, find p
  - Pbeta (flash_pbeta): given p and target vapor fraction beta_t, find T
  Special cases: beta=0 (bubble point), beta=1 (dew point)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from .properties import ln_phi, pressure, density_from_pressure, enthalpy, entropy, _pure_caloric
from .stability import stability_test_TPD, wilson_K


@dataclass
class MixtureFlashResult:
    """Outcome of a multicomponent flash.

    Fields that are None indicate "not applicable" (e.g. quality is None
    for single-phase results).
    """
    phase: str                # 'liquid', 'vapor', 'two_phase', 'supercritical'
    T: float                  # temperature [K]
    p: float                  # pressure [Pa]
    beta: Optional[float]     # molar vapor fraction (None if single-phase)
    x: np.ndarray             # liquid-phase composition (feed composition if vapor)
    y: np.ndarray             # vapor-phase composition (feed composition if liquid)
    z: np.ndarray             # feed composition
    rho: float                # bulk density [mol/m^3] (mixture-averaged if two-phase)
    rho_L: Optional[float]    # liquid density (None if single-phase)
    rho_V: Optional[float]    # vapor density (None if single-phase)
    h: float                  # molar enthalpy [J/mol]
    s: float                  # molar entropy [J/(mol K)]
    iterations: int           # outer-loop iterations
    K: Optional[np.ndarray]   # converged K-factors (None if single-phase)


# ---------------------------------------------------------------------------
# Rachford-Rice
# ---------------------------------------------------------------------------

def rachford_rice(z, K, tol=1e-12, maxiter=100):
    """Solve the Rachford-Rice equation for beta:

        sum_i  z_i (K_i - 1) / (1 + beta (K_i - 1))  =  0

    Returns (beta, converged_flag). If all K > 1 or all K < 1, returns
    beta clipped to the physical range and converged=False (single phase).
    """
    K = np.asarray(K, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    Km1 = K - 1.0

    # Physical bracket: beta_min = 1/(1 - K_max), beta_max = 1/(1 - K_min)
    Kmax = K.max()
    Kmin = K.min()
    if Kmax <= 1.0:
        return 0.0, False   # all K <= 1 -> liquid only
    if Kmin >= 1.0:
        return 1.0, False   # all K >= 1 -> vapor only

    beta_lo = 1.0 / (1.0 - Kmax) + 1e-12
    beta_hi = 1.0 / (1.0 - Kmin) - 1e-12
    # Clip bracket to [0, 1] for thermodynamically meaningful range
    beta_lo = max(beta_lo, 0.0)
    beta_hi = min(beta_hi, 1.0)
    if beta_hi <= beta_lo:
        # Degenerate bracket
        return 0.5, False

    def f(beta):
        return np.sum(z * Km1 / (1.0 + beta * Km1))

    def fp(beta):
        return -np.sum(z * Km1**2 / (1.0 + beta * Km1)**2)

    # Check endpoint signs
    f_lo = f(beta_lo)
    f_hi = f(beta_hi)
    if f_lo * f_hi > 0:
        # No sign change -> both phases implausible; report clipped
        if f_lo > 0:
            return beta_hi, False
        return beta_lo, False

    # Bracketed Newton with bisection safeguard
    beta = 0.5 * (beta_lo + beta_hi)
    for it in range(maxiter):
        fv = f(beta)
        if abs(fv) < tol:
            return beta, True
        fd = fp(beta)
        # Newton step
        if fd != 0:
            beta_new = beta - fv / fd
        else:
            beta_new = 0.5 * (beta_lo + beta_hi)
        # Safeguard: if Newton leaves bracket, fall back to bisection
        if beta_new <= beta_lo or beta_new >= beta_hi:
            beta_new = 0.5 * (beta_lo + beta_hi)
        # Update bracket from sign
        if fv > 0:
            beta_lo = beta
        else:
            beta_hi = beta
        beta = beta_new
        if beta_hi - beta_lo < tol:
            return beta, True

    return beta, abs(f(beta)) < 1e-6


# ---------------------------------------------------------------------------
# PT flash
# ---------------------------------------------------------------------------

def _successive_substitution(z, K_init, T, p, mixture,
                             tol=1e-9, maxiter=80):
    """Successive-substitution iterations for PT flash.

    Updates K from fugacity equality: K_i^new = phi_i_L / phi_i_V.
    """
    z = np.asarray(z, dtype=np.float64)
    K = np.asarray(K_init, dtype=np.float64).copy()

    for it in range(maxiter):
        beta, rr_ok = rachford_rice(z, K)
        if not rr_ok:
            # Single phase -- return with beta at extremum
            return None, K, beta, it

        # Liquid and vapor compositions
        denom = 1.0 + beta * (K - 1.0)
        x = z / denom
        y = K * x
        # Normalize (should already sum to 1)
        x = x / x.sum()
        y = y / y.sum()

        # Solve densities for each phase
        try:
            rho_L = density_from_pressure(p, T, x, mixture, phase_hint="liquid")
            rho_V = density_from_pressure(p, T, y, mixture, phase_hint="vapor")
        except RuntimeError:
            return None, K, beta, it

        # Fugacity coefficients
        lnphi_L = ln_phi(rho_L, T, x, mixture)
        lnphi_V = ln_phi(rho_V, T, y, mixture)

        # Update K: K_i = phi_i_L / phi_i_V = exp(ln phi_L - ln phi_V)
        ln_K_new = lnphi_L - lnphi_V
        K_new = np.exp(ln_K_new)

        # Convergence: max |ln K_new - ln K| small
        if np.max(np.abs(ln_K_new - np.log(K))) < tol:
            return (x, y, rho_L, rho_V), K_new, beta, it + 1

        K = K_new

    # Did not converge -- return last state anyway
    return (x, y, rho_L, rho_V), K, beta, maxiter


def flash_pt(p, T, z, mixture, K_init=None, check_stability=True,
             tol=1e-9, maxiter=80):
    """Isothermal-isobaric PT flash.

    Parameters
    ----------
    p : pressure [Pa]
    T : temperature [K]
    z : feed composition (length N)
    mixture : Mixture
    K_init : optional initial K-factors; otherwise uses Wilson
    check_stability : run Michelsen stability check first to classify phase count

    Returns
    -------
    MixtureFlashResult
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()
    N = mixture.N

    # Stability analysis (unless skipped)
    if K_init is None and check_stability:
        stable, K_stab, min_tpd = stability_test_TPD(z, T, p, mixture)
        if stable:
            # Single phase -- decide which (vapor or liquid) by density.
            # Wrap density solves in try/except: at conditions far from any
            # phase boundary (e.g. compressed liquid), the opposite branch
            # may not exist physically and Newton will fail to converge.
            try:
                rho_v = density_from_pressure(p, T, z, mixture, phase_hint="vapor")
            except RuntimeError:
                rho_v = None
            try:
                rho_l = density_from_pressure(p, T, z, mixture, phase_hint="liquid")
            except RuntimeError:
                rho_l = None
            if rho_v is None and rho_l is None:
                raise RuntimeError(
                    f"flash_pt (stable branch): no density branch converged "
                    f"at p={p}, T={T}"
                )
            if rho_v is None:
                rho = rho_l; phase_label = "liquid"
            elif rho_l is None:
                rho = rho_v; phase_label = "vapor"
            elif abs(rho_v - rho_l) / max(rho_v, rho_l) < 1e-4:
                if T > np.dot(z, mixture.T_c):
                    phase_label = "supercritical"
                else:
                    phase_label = "vapor" if rho_v < 0.5 * mixture.reduce(z)[1] else "liquid"
                rho = rho_v
            else:
                gv = _pure_caloric(rho_v, T, mixture, z)
                gl = _pure_caloric(rho_l, T, mixture, z)
                if gv["h"] - T * gv["s"] < gl["h"] - T * gl["s"]:
                    rho = rho_v
                    phase_label = "vapor"
                else:
                    rho = rho_l
                    phase_label = "liquid"
            caloric = _pure_caloric(rho, T, mixture, z)
            return MixtureFlashResult(
                phase=phase_label, T=T, p=p, beta=None,
                x=z.copy(), y=z.copy(), z=z, rho=rho,
                rho_L=None, rho_V=None,
                h=caloric["h"], s=caloric["s"],
                iterations=0, K=None,
            )
        K_init = K_stab

    if K_init is None:
        K_init = wilson_K(T, p, mixture)

    # Successive substitution
    result = _successive_substitution(z, K_init, T, p, mixture, tol=tol, maxiter=maxiter)
    compositions, K, beta, niter = result

    if compositions is None or beta <= 0 or beta >= 1:
        # Successive substitution converged to the trivial solution
        # (beta outside [0,1] or all-identical K values). This means the
        # stability test flagged the feed as unstable but the trial
        # initialization wasn't a true split -- classically this happens
        # near the phase boundary or when the stability test used only
        # one of the two trial phases. Fall through to single-phase
        # evaluation rather than recursing (which would loop forever).
        # We try BOTH density branches and pick lower Gibbs; if one branch
        # doesn't exist physically (e.g. vapor branch above bubble curve),
        # the corresponding density solver will fail to converge and we
        # accept whichever branch did converge.
        try:
            rho_v = density_from_pressure(p, T, z, mixture, phase_hint="vapor")
        except RuntimeError:
            rho_v = None
        try:
            rho_l = density_from_pressure(p, T, z, mixture, phase_hint="liquid")
        except RuntimeError:
            rho_l = None
        if rho_v is None and rho_l is None:
            raise RuntimeError(
                f"flash_pt: no density branch converged at p={p}, T={T}"
            )
        if rho_v is None:
            rho = rho_l; phase_label = "liquid"
        elif rho_l is None:
            rho = rho_v; phase_label = "vapor"
        elif abs(rho_v - rho_l) / max(rho_v, rho_l) < 1e-4:
            if T > np.dot(z, mixture.T_c):
                phase_label = "supercritical"
            else:
                phase_label = "vapor" if rho_v < 0.5 * mixture.reduce(z)[1] else "liquid"
            rho = rho_v
        else:
            gv = _pure_caloric(rho_v, T, mixture, z)
            gl = _pure_caloric(rho_l, T, mixture, z)
            if gv["h"] - T * gv["s"] < gl["h"] - T * gl["s"]:
                rho = rho_v; phase_label = "vapor"
            else:
                rho = rho_l; phase_label = "liquid"
        caloric = _pure_caloric(rho, T, mixture, z)
        return MixtureFlashResult(
            phase=phase_label, T=T, p=p, beta=None,
            x=z.copy(), y=z.copy(), z=z, rho=rho,
            rho_L=None, rho_V=None,
            h=caloric["h"], s=caloric["s"],
            iterations=niter, K=K,
        )

    x, y, rho_L, rho_V = compositions
    # Overall density: 1/rho = beta/rho_V + (1-beta)/rho_L
    v_avg = beta / rho_V + (1.0 - beta) / rho_L
    rho_avg = 1.0 / v_avg
    # Overall enthalpy and entropy
    h_L = enthalpy(rho_L, T, x, mixture)
    h_V = enthalpy(rho_V, T, y, mixture)
    s_L = entropy(rho_L, T, x, mixture)
    s_V = entropy(rho_V, T, y, mixture)
    h = beta * h_V + (1.0 - beta) * h_L
    s = beta * s_V + (1.0 - beta) * s_L

    return MixtureFlashResult(
        phase="two_phase", T=T, p=p, beta=beta,
        x=x, y=y, z=z, rho=rho_avg,
        rho_L=rho_L, rho_V=rho_V,
        h=h, s=s,
        iterations=niter, K=K,
    )


# ---------------------------------------------------------------------------
# Vapor-fraction flashes: Tbeta (find p) and Pbeta (find T)
# ---------------------------------------------------------------------------

def _wilson_bubble_dew_p(T, z, mixture):
    """Estimate bubble and dew pressures at temperature T via Wilson K-factors.

    Wilson: K_i = (p_c_i / p) * exp(5.373 * (1 + omega_i) * (1 - T_c_i/T))
             = A_i / p,  with A_i = p_c_i * exp(...)

    Bubble (beta=0, x=z): sum(z_i * K_i) = 1  -->  p_bub = sum(z_i * A_i)
    Dew    (beta=1, y=z): sum(z_i / K_i) = 1  -->  p_dew = 1 / sum(z_i / A_i)
    """
    N = mixture.N
    A = np.zeros(N)
    for i, comp in enumerate(mixture.components):
        fl = comp.fluid
        omega = getattr(fl, "acentric_factor", 0.0)
        A[i] = fl.p_c * np.exp(5.373 * (1.0 + omega) * (1.0 - fl.T_c / T))
    p_bub = float(np.dot(z, A))
    p_dew = 1.0 / float(np.dot(z, 1.0 / A))
    return p_bub, p_dew


def flash_tbeta(T, beta_target, z, mixture,
                p_init=None, tol=1e-7, maxiter=40):
    """Given T and target vapor fraction beta, find p.

    Special cases:
      - beta_target = 0: bubble-point pressure
      - beta_target = 1: dew-point pressure

    Algorithm
    ---------
    Outer secant/Newton iteration on ln(p), with bracket-based safeguards to
    handle non-monotonic beta(p) near-critical behavior. The initial guess
    uses Wilson-based bubble/dew interpolation in ln(p).

    Because beta(p) is typically monotonic (decreasing in p) on the
    sub-critical side of the bubble/dew dome, Newton converges quickly from
    a decent guess. Near the mixture critical point beta(p) can turn around;
    we safeguard with bisection when the sign-change bracket is known.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    # Wilson-based initial guess: interpolate in ln(p) between bubble and dew
    # based on beta_target.
    p_bub, p_dew = _wilson_bubble_dew_p(T, z, mixture)
    if p_init is None:
        beta_clamped = max(min(beta_target, 1.0), 0.0)
        p_init = np.exp((1.0 - beta_clamped) * np.log(p_bub) + beta_clamped * np.log(p_dew))
        p_init = max(p_init, 1e3)

    # Track last two points for secant fallback
    def beta_at(p):
        r = flash_pt(p, T, z, mixture, check_stability=False, tol=tol * 0.1)
        if r.phase == "two_phase":
            return r.beta, r
        # Single-phase: decide whether we're on the "vapor" side (beta=1) or
        # "liquid" side (beta=0) of the phase boundary. Compare the density
        # to the pseudocritical density: below ~0.5*rho_pc we're vapor-like.
        rho_pc = mixture.reduce(z)[1]
        if r.rho < 0.7 * rho_pc:
            return 1.0, r    # vapor-like
        else:
            return 0.0, r    # liquid-like

    p = p_init
    # Keep a bracket: (p_lo, beta_lo - target, p_hi, beta_hi - target) with opposite signs
    # We populate the bracket as we go.
    bracket = None   # (p_lo, resid_lo, p_hi, resid_hi)

    for it in range(maxiter):
        beta_actual, result = beta_at(p)
        residual = beta_actual - beta_target
        if abs(residual) < tol:
            return result

        # Update bracket
        if bracket is None:
            bracket = [p, residual, None, None]
        else:
            # Insert p into bracket, maintaining sign opposition
            p_lo, r_lo, p_hi, r_hi = bracket
            if p_hi is None:
                if np.sign(residual) != np.sign(r_lo):
                    bracket = [p_lo, r_lo, p, residual]
                else:
                    # same sign, replace with closer guess
                    bracket = [p, residual, None, None]
            else:
                # Bracket exists -- refine
                if np.sign(residual) == np.sign(r_lo):
                    bracket = [p, residual, p_hi, r_hi]
                else:
                    bracket = [p_lo, r_lo, p, residual]

        # Numerical derivative d beta / d ln p via forward FD
        dlnp = 1e-4
        p_hi_fd = p * np.exp(dlnp)
        beta_hi_fd, _ = beta_at(p_hi_fd)
        dbeta_dlnp = (beta_hi_fd - beta_actual) / dlnp

        if abs(dbeta_dlnp) < 1e-12:
            # No sensitivity -- use bracket or move away
            if bracket[2] is not None:
                p = 0.5 * (bracket[0] + bracket[2])
                continue
            p *= 1.5 if residual > 0 else 1.0 / 1.5
            continue

        dlnp_step = -residual / dbeta_dlnp

        # Damp
        if abs(dlnp_step) > 0.5:
            dlnp_step = 0.5 * np.sign(dlnp_step)

        p_newton = p * np.exp(dlnp_step)

        # If we have a bracket and Newton wants to leave it, bisect instead
        if bracket[2] is not None:
            lo = min(bracket[0], bracket[2])
            hi = max(bracket[0], bracket[2])
            if p_newton <= lo or p_newton >= hi:
                p_newton = np.sqrt(lo * hi)   # geometric mean (bisection in ln p)

        p = p_newton

    raise RuntimeError(f"flash_tbeta did not converge (T={T}, beta={beta_target})")


def _wilson_bubble_dew_T(p, z, mixture, T_lo=50.0, T_hi=1000.0):
    """Estimate bubble and dew temperatures at pressure p via Wilson K-factors.

    Wilson: ln K_i = ln(p_c_i/p) + 5.373*(1+omega_i)*(1 - T_c_i/T)

    At bubble (x=z): sum(z_i * K_i) = 1
    At dew    (y=z): sum(z_i / K_i) = 1

    We solve each via Newton on T.
    """
    N = mixture.N

    def sum_K_minus_1(T, inverse=False):
        s = 0.0
        for i, comp in enumerate(mixture.components):
            fl = comp.fluid
            omega = getattr(fl, "acentric_factor", 0.0)
            lnK = np.log(fl.p_c / p) + 5.373 * (1.0 + omega) * (1.0 - fl.T_c / T)
            if inverse:
                s += z[i] * np.exp(-lnK)
            else:
                s += z[i] * np.exp(lnK)
        return s - 1.0

    # Bisection for T_bubble
    T_bub = _bisect(lambda T: sum_K_minus_1(T, False), T_lo, T_hi)
    T_dew = _bisect(lambda T: sum_K_minus_1(T, True),  T_lo, T_hi)
    return T_bub, T_dew


def _bisect(f, lo, hi, tol=1e-6, maxiter=80):
    """Simple robust bisection. Returns mid-point if no sign change."""
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo * f_hi > 0:
        # No sign change -- return geometric mean as crude estimate
        return np.sqrt(lo * hi)
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol or hi - lo < tol:
            return mid
        if f_mid * f_lo > 0:
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return 0.5 * (lo + hi)


def flash_pbeta(p, beta_target, z, mixture,
                T_init=None, tol=1e-7, maxiter=40):
    """Given p and target vapor fraction beta, find T.

    Uses Wilson-based bubble/dew T estimation for initial guess, then
    secant/Newton with bracketing on the outer loop.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if T_init is None:
        # Interpolate between bubble-T and dew-T from Wilson
        try:
            T_bub, T_dew = _wilson_bubble_dew_T(p, z, mixture)
            beta_clamped = max(min(beta_target, 1.0), 0.0)
            # Linear interpolation in 1/T (since vapor pressure is ~exp(-A/T))
            inv_T_init = (1.0 - beta_clamped) / T_bub + beta_clamped / T_dew
            T_init = 1.0 / inv_T_init
        except Exception:
            T_init = float(np.dot(z, mixture.T_c))

    def beta_at(T):
        r = flash_pt(p, T, z, mixture, check_stability=False, tol=tol * 0.1)
        if r.phase == "two_phase":
            return r.beta, r
        rho_pc = mixture.reduce(z)[1]
        if r.rho < 0.7 * rho_pc:
            return 1.0, r    # vapor-like
        else:
            return 0.0, r    # liquid-like

    T = T_init
    bracket = None   # [T_lo, resid_lo, T_hi, resid_hi]

    for it in range(maxiter):
        beta_actual, result = beta_at(T)
        residual = beta_actual - beta_target
        if abs(residual) < tol:
            return result

        # Update bracket
        if bracket is None:
            bracket = [T, residual, None, None]
        else:
            T_lo, r_lo, T_hi, r_hi = bracket
            if T_hi is None:
                if np.sign(residual) != np.sign(r_lo):
                    bracket = [T_lo, r_lo, T, residual]
                else:
                    bracket = [T, residual, None, None]
            else:
                if np.sign(residual) == np.sign(r_lo):
                    bracket = [T, residual, T_hi, r_hi]
                else:
                    bracket = [T_lo, r_lo, T, residual]

        # Numerical derivative d beta / d (1/T)  -- since beta(T) is more
        # linear in 1/T than in T for most systems
        dinvT = 1e-5 / T
        T_pert = 1.0 / (1.0/T + dinvT)
        beta_pert, _ = beta_at(T_pert)
        dbeta_dinvT = (beta_pert - beta_actual) / dinvT

        if abs(dbeta_dinvT) < 1e-12:
            if bracket[2] is not None:
                T = 0.5 * (bracket[0] + bracket[2])
                continue
            T *= 0.95 if residual > 0 else 1.05
            continue

        dinvT_step = -residual / dbeta_dinvT
        inv_T_new = 1.0 / T + dinvT_step

        # Damp
        if abs(dinvT_step) > 0.3 / T:
            dinvT_step = 0.3 / T * np.sign(dinvT_step)
            inv_T_new = 1.0 / T + dinvT_step

        if inv_T_new <= 0:
            inv_T_new = 0.5 / T
        T_newton = 1.0 / inv_T_new

        # Bisection safeguard
        if bracket[2] is not None:
            lo = min(bracket[0], bracket[2])
            hi = max(bracket[0], bracket[2])
            if T_newton <= lo or T_newton >= hi:
                T_newton = 0.5 * (lo + hi)

        T = T_newton

    raise RuntimeError(f"flash_pbeta did not converge (p={p}, beta={beta_target})")


# Convenience: bubble/dew points
# ---------------------------------------------------------------------------
# Bubble and dew point solvers
# ---------------------------------------------------------------------------
#
# At a bubble point (beta=0, x=z, incipient vapor y):
#     x_i * phi_L_i(T,p,x)  =  y_i * phi_V_i(T,p,y)       [fugacity equality]
#     sum(y_i) = 1                                        [mole fraction constraint]
#
# Equivalent formulation using K-factors K_i = y_i/x_i = phi_L_i / phi_V_i:
#     K_i = phi_L_i(T,p,z) / phi_V_i(T,p,y)
#     sum(z_i * K_i) = 1                                  [bubble point equation]
#     y_i = z_i * K_i
#
# At a dew point (beta=1, y=z, incipient liquid x):
#     sum(z_i / K_i) = 1                                  [dew point equation]
#     x_i = z_i / K_i
#
# ALGORITHM (Michelsen-style, successive substitution with pressure correction):
#
# At fixed T, solve for (p, y) at the bubble line:
#   1. Guess p from Wilson bubble-p; guess y from Wilson K-factors
#   2. Loop:
#        a. Compute phi_L at (T, p, z).
#        b. Compute phi_V at (T, p, y).
#        c. K_new = phi_L / phi_V
#        d. S = sum(z * K_new)
#        e. p_new = p * S       (the key pressure-correction step)
#        f. y_new = z * K_new / S
#      Converge when |S - 1| < tol_f AND ||ln K_new - ln K|| < tol_K
#
# For bubble_point_T (given p, find T), use outer Newton on T enclosing
# bubble_point_p (which gives p(T)). Adjust T until p(T) matches target.
# Alternatively, use a similar inner iteration but with T as the updated
# variable instead of p; the latter is simpler and is used here.
#
# The pressure-correction p_new = p * S converges quadratically near the
# solution because K ~ 1/p at first order, so dS/dlnp ~ -S, giving a
# Newton-like step.


def _bubble_residual_at(T, p, z, mixture, y_init=None, maxiter=30, tol_inner=1e-10):
    """Evaluate the bubble-point residual S(T, p) = sum(z * K) at a given state.

    Self-consistently iterates K and y at fixed (T, p), until y and K stop changing.
    Returns (S, K, y), or raises RuntimeError if iteration fails or converges to
    a trivial (K ~ 1) fixed point.

    The reported S can be used by an outer loop as a function of T (for fixed p)
    or p (for fixed T) to locate the bubble line where S = 1.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if y_init is None:
        K = wilson_K(T, p, mixture)
        y = z * K
        y = y / y.sum()
    else:
        y = y_init.copy()
        y = y / y.sum()
        K = y / np.maximum(z, 1e-300)

    last_ln_K = np.log(np.maximum(K, 1e-300))

    for it in range(maxiter):
        try:
            rho_L = density_from_pressure(p, T, z, mixture, phase_hint="liquid")
        except RuntimeError:
            rho_L = density_from_pressure(p, T, z, mixture, phase_hint="vapor")
        lnphi_L = ln_phi(rho_L, T, z, mixture)

        try:
            rho_V = density_from_pressure(p, T, y, mixture, phase_hint="vapor")
        except RuntimeError:
            rho_V = density_from_pressure(p, T, y, mixture, phase_hint="liquid")
        lnphi_V = ln_phi(rho_V, T, y, mixture)

        ln_K_new = lnphi_L - lnphi_V
        K_new = np.exp(ln_K_new)

        # Trivial-solution check
        if np.max(np.abs(ln_K_new)) < 1e-3:
            raise RuntimeError("trivial")

        S = float(np.dot(z, K_new))
        if S <= 0 or not np.isfinite(S):
            raise RuntimeError("bad S")
        y_new = (z * K_new) / S
        y_new = y_new / y_new.sum()

        if np.max(np.abs(ln_K_new - last_ln_K)) < tol_inner:
            return S, K_new, y_new

        last_ln_K = ln_K_new
        K = K_new
        y = y_new

    return S, K_new, y_new


def _dew_residual_at(T, p, z, mixture, x_init=None, maxiter=30, tol_inner=1e-10):
    """Dew-point residual S(T, p) = sum(z / K). Returns (S, K, x)."""
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if x_init is None:
        K = wilson_K(T, p, mixture)
        x = z / K
        x = x / x.sum()
    else:
        x = x_init.copy()
        x = x / x.sum()
        K = np.maximum(z, 1e-300) / np.maximum(x, 1e-300)

    last_ln_K = np.log(np.maximum(K, 1e-300))

    for it in range(maxiter):
        try:
            rho_L = density_from_pressure(p, T, x, mixture, phase_hint="liquid")
        except RuntimeError:
            rho_L = density_from_pressure(p, T, x, mixture, phase_hint="vapor")
        lnphi_L = ln_phi(rho_L, T, x, mixture)

        try:
            rho_V = density_from_pressure(p, T, z, mixture, phase_hint="vapor")
        except RuntimeError:
            rho_V = density_from_pressure(p, T, z, mixture, phase_hint="liquid")
        lnphi_V = ln_phi(rho_V, T, z, mixture)

        ln_K_new = lnphi_L - lnphi_V
        K_new = np.exp(ln_K_new)

        if np.max(np.abs(ln_K_new)) < 1e-3:
            raise RuntimeError("trivial")

        S = float(np.dot(z, 1.0 / K_new))
        if S <= 0 or not np.isfinite(S):
            raise RuntimeError("bad S")
        x_new = (z / K_new) / S
        x_new = x_new / x_new.sum()

        if np.max(np.abs(ln_K_new - last_ln_K)) < tol_inner:
            return S, K_new, x_new

        last_ln_K = ln_K_new
        K = K_new
        x = x_new

    return S, K_new, x_new


def bubble_point_p(T, z, mixture, p_init=None, tol=1e-8, maxiter=60):
    """Bubble-point pressure: find p at which the feed z is just at the bubble
    of its two-phase envelope at temperature T.

    Uses Michelsen-style pressure-correction iteration: p_new = p * S where
    S = sum(z * K(T, p)) and K comes from self-consistent inner iteration of
    the vapor composition y.

    Parameters
    ----------
    T : float           temperature [K]
    z : array (N,)      feed composition (mole fractions)
    mixture : Mixture
    p_init : float or None
        Initial pressure guess. If None, uses Wilson-based estimate.
    tol : float
    maxiter : int

    Returns
    -------
    MixtureFlashResult with phase='bubble', beta=0, x=z, y=incipient vapor.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if p_init is None:
        p_bub_wilson, _ = _wilson_bubble_dew_p(T, z, mixture)
        p_init = max(p_bub_wilson, 1e4)

    p = p_init
    y_last = None
    f_resid = float("nan")

    for it in range(maxiter):
        try:
            S, K, y = _bubble_residual_at(T, p, z, mixture, y_init=y_last)
        except RuntimeError as e:
            if str(e) == "trivial":
                # We're above the dome; back off to lower p
                p = p * 0.5
                y_last = None
                continue
            elif str(e) == "bad S":
                p = p * 0.8
                y_last = None
                continue
            else:
                raise
        y_last = y

        f_resid = S - 1.0
        if abs(f_resid) < tol:
            # Converged -- assemble result
            try:
                rho_L = density_from_pressure(p, T, z, mixture, phase_hint="liquid")
            except RuntimeError:
                rho_L = density_from_pressure(p, T, z, mixture, phase_hint="vapor")
            try:
                rho_V = density_from_pressure(p, T, y, mixture, phase_hint="vapor")
            except RuntimeError:
                rho_V = density_from_pressure(p, T, y, mixture, phase_hint="liquid")
            h = enthalpy(rho_L, T, z, mixture)
            s = entropy(rho_L, T, z, mixture)
            return MixtureFlashResult(
                phase="bubble", T=T, p=p, beta=0.0,
                x=z.copy(), y=y, z=z, rho=rho_L,
                rho_L=rho_L, rho_V=rho_V,
                h=h, s=s, iterations=it + 1, K=K,
            )

        # Michelsen pressure correction: p_new = p * S
        p_new = p * S
        # Damp aggressively if S is far from 1
        if p_new > 5.0 * p:
            p_new = 5.0 * p
        elif p_new < 0.2 * p:
            p_new = 0.2 * p
        p = p_new

    raise RuntimeError(
        f"bubble_point_p did not converge: T={T}, final p={p}, residual={f_resid:.3e}"
    )


def bubble_point_T(p, z, mixture, T_init=None, tol=1e-8, maxiter=60):
    """Bubble-point temperature: find T at which the feed z is at the bubble
    of its two-phase envelope at pressure p.

    Algorithm
    ---------
    Outer bracketed Newton in T enclosing the inner (y, K) iteration.

    We first establish a bracket [T_lo, T_hi] where S(T_lo) < 1 (liquid-rich
    side) and S(T_hi) > 1 (vapor-rich side). The Wilson-based initial
    estimate is usually between them. Newton steps are constrained to
    stay within the bracket; bisection fallback is used if Newton leaves.

    Trivial-solution evaluations (when T is above the mixture dome) are
    handled by reducing the upper bound.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if T_init is None:
        T_bub_wilson, _ = _wilson_bubble_dew_T(p, z, mixture)
        T_init = T_bub_wilson

    # Establish initial bracket by probing around T_init
    # S(T) increases with T (at low T, K~0; at high T, K~1 from the trivial solution)
    # For the *physical* bubble line, S crosses 1 from below as T increases.

    def eval_S(T, y_hint):
        try:
            S, K, y = _bubble_residual_at(T, p, z, mixture, y_init=y_hint)
            return S, K, y, "ok"
        except RuntimeError as e:
            if str(e) == "trivial":
                return None, None, None, "trivial"
            return None, None, None, "bad"

    # Find T_lo with S < 1 and T_hi with S > 1
    T = T_init
    S, K, y, status = eval_S(T, None)

    T_lo = None; T_hi = None; S_lo = None; S_hi = None; K_lo = None; K_hi = None; y_lo = None; y_hi = None

    if status == "ok":
        if S < 1.0:
            T_lo = T; S_lo = S; K_lo = K; y_lo = y
        else:
            T_hi = T; S_hi = S; K_hi = K; y_hi = y
    # If initial guess is trivial/bad, scan lower T's
    if status != "ok":
        # Walk T down from T_init toward the triple point, searching for non-trivial region
        T_probe = T_init
        for _ in range(20):
            T_probe *= 0.8
            if T_probe < 40.0:
                break
            S, K, y, status = eval_S(T_probe, None)
            if status == "ok":
                if S < 1.0:
                    T_lo = T_probe; S_lo = S; K_lo = K; y_lo = y
                else:
                    T_hi = T_probe; S_hi = S; K_hi = K; y_hi = y
                break
        if T_lo is None and T_hi is None:
            raise RuntimeError(
                f"bubble_point_T: could not find any valid (non-trivial) T "
                f"for p={p}. The system may be beyond the dome at this pressure."
            )

    # Extend bracket if needed
    # Find T_lo (S<1)
    if T_lo is None:
        T_probe = T_hi * 0.9
        for _ in range(30):
            S, K, y, status = eval_S(T_probe, y_hi)
            if status == "ok" and S < 1.0:
                T_lo = T_probe; S_lo = S; K_lo = K; y_lo = y
                break
            T_probe *= 0.9
            if T_probe < 40.0:
                break
        if T_lo is None:
            raise RuntimeError(
                f"bubble_point_T: no bubble point exists at p={p} Pa for z={z.tolist()}. "
                f"All probed temperatures gave S > 1 (feed is above its bubble line "
                f"at this pressure). Try a higher pressure."
            )

    # Find T_hi (S>1)
    if T_hi is None:
        T_probe = T_lo * 1.1
        for _ in range(30):
            S, K, y, status = eval_S(T_probe, y_lo)
            if status == "ok" and S > 1.0:
                T_hi = T_probe; S_hi = S; K_hi = K; y_hi = y
                break
            if status == "trivial":
                # T_probe is above the dome; upper bound is here but we need to back off
                T_probe = 0.5 * (T_lo + T_probe)
                continue
            T_probe *= 1.1
            if T_probe > 2000.0:
                break
        if T_hi is None:
            raise RuntimeError(
                f"bubble_point_T: could not find T_hi with S>1 for p={p}"
            )

    # Bracketed Newton-bisection iteration
    for it in range(maxiter):
        # Pick next T: try Newton-like step based on secant between endpoints
        # In ln(S) vs 1/T:  ln S ~ A/T + B, so 1/T_new = 1/T_lo + (-ln(S_lo)) * (1/T_hi - 1/T_lo)/(ln(S_hi)-ln(S_lo))
        lnS_lo = np.log(S_lo)
        lnS_hi = np.log(S_hi)
        if lnS_hi == lnS_lo:
            T_new = 0.5 * (T_lo + T_hi)
        else:
            # Interpolate in 1/T targeting ln(S) = 0 (i.e. S = 1)
            frac = -lnS_lo / (lnS_hi - lnS_lo)
            # Clamp frac to [0.1, 0.9] so bisection if near bracket edge
            frac = max(0.1, min(0.9, frac))
            inv_T_new = (1 - frac) / T_lo + frac / T_hi
            T_new = 1.0 / inv_T_new

        # Closer-y hint
        y_hint = y_lo if S_lo > 0 else y_hi
        S_new, K_new, y_new, status = eval_S(T_new, y_hint)

        if status == "trivial":
            # Above dome; tighten upper bracket
            T_hi = T_new
            # Use a small S > 1 to keep the bracket valid (bisect-only behavior)
            S_hi = max(S_hi, 1.001)
            continue
        if status == "bad":
            T_new = 0.5 * (T_lo + T_hi)
            S_new, K_new, y_new, status = eval_S(T_new, y_hint)
            if status != "ok":
                raise RuntimeError(f"bubble_point_T: failed at T={T_new}, p={p}")

        f_resid = S_new - 1.0
        if abs(f_resid) < tol:
            # Converged
            try:
                rho_L = density_from_pressure(p, T_new, z, mixture, phase_hint="liquid")
            except RuntimeError:
                rho_L = density_from_pressure(p, T_new, z, mixture, phase_hint="vapor")
            try:
                rho_V = density_from_pressure(p, T_new, y_new, mixture, phase_hint="vapor")
            except RuntimeError:
                rho_V = density_from_pressure(p, T_new, y_new, mixture, phase_hint="liquid")
            h = enthalpy(rho_L, T_new, z, mixture)
            s = entropy(rho_L, T_new, z, mixture)
            return MixtureFlashResult(
                phase="bubble", T=T_new, p=p, beta=0.0,
                x=z.copy(), y=y_new, z=z, rho=rho_L,
                rho_L=rho_L, rho_V=rho_V,
                h=h, s=s, iterations=it + 1, K=K_new,
            )

        # Update bracket
        if S_new < 1.0:
            T_lo = T_new; S_lo = S_new; K_lo = K_new; y_lo = y_new
        else:
            T_hi = T_new; S_hi = S_new; K_hi = K_new; y_hi = y_new

    raise RuntimeError(
        f"bubble_point_T did not converge: p={p}, bracket=[{T_lo}, {T_hi}], "
        f"S_lo={S_lo}, S_hi={S_hi}"
    )


def dew_point_p(T, z, mixture, p_init=None, tol=1e-8, maxiter=60):
    """Dew-point pressure: find p at which feed z is on the dew line at T."""
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if p_init is None:
        _, p_dew_wilson = _wilson_bubble_dew_p(T, z, mixture)
        p_init = max(p_dew_wilson, 1e3)

    p = p_init
    x_last = None
    f_resid = float("nan")

    for it in range(maxiter):
        try:
            S, K, x = _dew_residual_at(T, p, z, mixture, x_init=x_last)
        except RuntimeError as e:
            if str(e) == "trivial":
                # Above dome -- decrease p
                p = p * 0.5
                x_last = None
                continue
            elif str(e) == "bad S":
                p = p * 1.5
                x_last = None
                continue
            else:
                raise
        x_last = x

        f_resid = S - 1.0
        if abs(f_resid) < tol:
            try:
                rho_L = density_from_pressure(p, T, x, mixture, phase_hint="liquid")
            except RuntimeError:
                rho_L = density_from_pressure(p, T, x, mixture, phase_hint="vapor")
            try:
                rho_V = density_from_pressure(p, T, z, mixture, phase_hint="vapor")
            except RuntimeError:
                rho_V = density_from_pressure(p, T, z, mixture, phase_hint="liquid")
            h = enthalpy(rho_V, T, z, mixture)
            s = entropy(rho_V, T, z, mixture)
            return MixtureFlashResult(
                phase="dew", T=T, p=p, beta=1.0,
                x=x, y=z.copy(), z=z, rho=rho_V,
                rho_L=rho_L, rho_V=rho_V,
                h=h, s=s, iterations=it + 1, K=K,
            )

        # Dew pressure correction: p_new = p / S
        p_new = p / S
        if p_new > 5.0 * p:
            p_new = 5.0 * p
        elif p_new < 0.2 * p:
            p_new = 0.2 * p
        p = p_new

    raise RuntimeError(
        f"dew_point_p did not converge: T={T}, final p={p}, residual={f_resid:.3e}"
    )


def dew_point_T(p, z, mixture, T_init=None, tol=1e-8, maxiter=60):
    """Dew-point temperature: find T at which feed z is on the dew line at p.

    Uses a bracketed Newton-bisection search similar to bubble_point_T.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if T_init is None:
        _, T_dew_wilson = _wilson_bubble_dew_T(p, z, mixture)
        T_init = T_dew_wilson

    def eval_S(T, x_hint):
        try:
            S, K, x = _dew_residual_at(T, p, z, mixture, x_init=x_hint)
            return S, K, x, "ok"
        except RuntimeError as e:
            if str(e) == "trivial":
                return None, None, None, "trivial"
            return None, None, None, "bad"

    T = T_init
    S, K, x, status = eval_S(T, None)

    # For dew: S(T) = sum(z/K) is high at low T (K small), low at high T (K large)
    # So we want T_lo where S > 1 and T_hi where S < 1 (opposite of bubble)
    T_lo = None; T_hi = None; S_lo = None; S_hi = None; K_lo = None; K_hi = None; x_lo = None; x_hi = None

    if status == "ok":
        if S > 1.0:
            T_lo = T; S_lo = S; K_lo = K; x_lo = x
        else:
            T_hi = T; S_hi = S; K_hi = K; x_hi = x

    if status != "ok":
        T_probe = T_init
        for _ in range(20):
            T_probe *= 0.8
            if T_probe < 40.0:
                break
            S, K, x, status = eval_S(T_probe, None)
            if status == "ok":
                if S > 1.0:
                    T_lo = T_probe; S_lo = S; K_lo = K; x_lo = x
                else:
                    T_hi = T_probe; S_hi = S; K_hi = K; x_hi = x
                break
        if T_lo is None and T_hi is None:
            raise RuntimeError(
                f"dew_point_T: could not find any valid (non-trivial) T for p={p}"
            )

    if T_lo is None:
        T_probe = T_hi * 0.9
        for _ in range(30):
            S, K, x, status = eval_S(T_probe, x_hi)
            if status == "ok" and S > 1.0:
                T_lo = T_probe; S_lo = S; K_lo = K; x_lo = x
                break
            T_probe *= 0.9
            if T_probe < 40.0:
                break
        if T_lo is None:
            raise RuntimeError(f"dew_point_T: could not find T_lo with S>1 for p={p}")

    if T_hi is None:
        T_probe = T_lo * 1.1
        for _ in range(30):
            S, K, x, status = eval_S(T_probe, x_lo)
            if status == "ok" and S < 1.0:
                T_hi = T_probe; S_hi = S; K_hi = K; x_hi = x
                break
            if status == "trivial":
                T_probe = 0.5 * (T_lo + T_probe)
                continue
            T_probe *= 1.1
            if T_probe > 2000.0:
                break
        if T_hi is None:
            raise RuntimeError(f"dew_point_T: could not find T_hi with S<1 for p={p}")

    for it in range(maxiter):
        lnS_lo = np.log(S_lo)
        lnS_hi = np.log(S_hi)
        if lnS_hi == lnS_lo:
            T_new = 0.5 * (T_lo + T_hi)
        else:
            # Same interpolation idea: target ln(S)=0
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
                raise RuntimeError(f"dew_point_T: failed at T={T_new}, p={p}")

        f_resid = S_new - 1.0
        if abs(f_resid) < tol:
            try:
                rho_L = density_from_pressure(p, T_new, x_new, mixture, phase_hint="liquid")
            except RuntimeError:
                rho_L = density_from_pressure(p, T_new, x_new, mixture, phase_hint="vapor")
            try:
                rho_V = density_from_pressure(p, T_new, z, mixture, phase_hint="vapor")
            except RuntimeError:
                rho_V = density_from_pressure(p, T_new, z, mixture, phase_hint="liquid")
            h = enthalpy(rho_V, T_new, z, mixture)
            s = entropy(rho_V, T_new, z, mixture)
            return MixtureFlashResult(
                phase="dew", T=T_new, p=p, beta=1.0,
                x=x_new, y=z.copy(), z=z, rho=rho_V,
                rho_L=rho_L, rho_V=rho_V,
                h=h, s=s, iterations=it + 1, K=K_new,
            )

        if S_new > 1.0:
            T_lo = T_new; S_lo = S_new; K_lo = K_new; x_lo = x_new
        else:
            T_hi = T_new; S_hi = S_new; K_hi = K_new; x_hi = x_new

    raise RuntimeError(
        f"dew_point_T did not converge: p={p}, bracket=[{T_lo}, {T_hi}]"
    )


# ---------------------------------------------------------------------------
# State-function flashes (PH, PS, TH, TS) via outer Newton
# ---------------------------------------------------------------------------

def flash_ph(p, h_target, z, mixture, T_init=None, tol=1e-6, maxiter=40):
    """PH flash: given p and h, find T (and phase).

    Outer 1-D Newton in T; inner PT flash at each iterate; target h.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if T_init is None:
        T_init = float(np.dot(z, mixture.T_c))  # mole-average Tc as crude guess

    T = T_init
    for it in range(maxiter):
        result = flash_pt(p, T, z, mixture, tol=1e-9)
        dh = result.h - h_target
        if abs(dh) < tol * max(1.0, abs(h_target)):
            return result
        # Derivative dh/dT at constant p (~ cp of mixture)
        dT = 0.1
        result2 = flash_pt(p, T + dT, z, mixture, tol=1e-9)
        cp_est = (result2.h - result.h) / dT
        if abs(cp_est) < 1e-6:
            T += 1.0 * np.sign(-dh)
            continue
        step = -dh / cp_est
        # Damp
        if abs(step) > 0.2 * T:
            step = 0.2 * T * np.sign(step)
        T_new = T + step
        if T_new <= 0:
            T_new = 0.5 * T
        T = T_new

    raise RuntimeError(f"flash_ph did not converge (p={p}, h={h_target})")


def flash_ps(p, s_target, z, mixture, T_init=None, tol=1e-6, maxiter=40):
    """PS flash: given p and s, find T (and phase)."""
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if T_init is None:
        T_init = float(np.dot(z, mixture.T_c))

    T = T_init
    for it in range(maxiter):
        result = flash_pt(p, T, z, mixture, tol=1e-9)
        ds = result.s - s_target
        if abs(ds) < tol * max(1.0, abs(s_target)):
            return result
        dT = 0.1
        result2 = flash_pt(p, T + dT, z, mixture, tol=1e-9)
        cp_T_est = (result2.s - result.s) / dT   # dS/dT at const p = cp/T
        if abs(cp_T_est) < 1e-9:
            T += 1.0 * np.sign(-ds)
            continue
        step = -ds / cp_T_est
        if abs(step) > 0.2 * T:
            step = 0.2 * T * np.sign(step)
        T_new = T + step
        if T_new <= 0:
            T_new = 0.5 * T
        T = T_new

    raise RuntimeError(f"flash_ps did not converge (p={p}, s={s_target})")


def flash_th(T, h_target, z, mixture, p_init=None, tol=1e-6, maxiter=40):
    """TH flash: given T and h, find p (and phase).

    Note: at subcritical T, h inside the two-phase band is multi-valued with
    respect to pressure. This implementation does a monotonic search assuming
    either single-phase or two-phase; it will lock onto the first converged
    root. For compressed-liquid cases, pass p_init above the bubble pressure.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if p_init is None:
        p_init = 1e5  # 1 bar default

    p = p_init
    for it in range(maxiter):
        result = flash_pt(p, T, z, mixture, tol=1e-9)
        dh = result.h - h_target
        if abs(dh) < tol * max(1.0, abs(h_target)):
            return result
        dlnp = 1e-3
        result2 = flash_pt(p * np.exp(dlnp), T, z, mixture, tol=1e-9)
        dh_dlnp = (result2.h - result.h) / dlnp
        if abs(dh_dlnp) < 1e-6:
            p *= 1.2 if dh < 0 else 0.8
            continue
        step = -dh / dh_dlnp
        if abs(step) > 0.5:
            step = 0.5 * np.sign(step)
        p = p * np.exp(step)

    raise RuntimeError(f"flash_th did not converge (T={T}, h={h_target})")


def flash_ts(T, s_target, z, mixture, p_init=None, tol=1e-6, maxiter=40):
    """TS flash: given T and s, find p (and phase)."""
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if p_init is None:
        p_init = 1e5

    p = p_init
    for it in range(maxiter):
        result = flash_pt(p, T, z, mixture, tol=1e-9)
        ds = result.s - s_target
        if abs(ds) < tol * max(1.0, abs(s_target)):
            return result
        dlnp = 1e-3
        result2 = flash_pt(p * np.exp(dlnp), T, z, mixture, tol=1e-9)
        ds_dlnp = (result2.s - result.s) / dlnp
        if abs(ds_dlnp) < 1e-9:
            p *= 1.2 if ds > 0 else 0.8
            continue
        step = -ds / ds_dlnp
        if abs(step) > 0.5:
            step = 0.5 * np.sign(step)
        p = p * np.exp(step)

    raise RuntimeError(f"flash_ts did not converge (T={T}, s={s_target})")


# ---------------------------------------------------------------------------
# TV and UV flashes -- the natural-variable flashes used in dynamic simulation
# ---------------------------------------------------------------------------
#
# For dynamic (time-stepping) simulation, the conserved quantities in a
# constant-volume control mass are internal energy U, volume V, and moles
# n_i.  After advancing U and V by one time step we need to recover the
# intensive thermodynamic state, i.e. solve
#
#     u(T, v, z) = u*,          v = V/n = 1/rho
#
# for the temperature T (and, if two-phase, the vapor fraction beta).
# This is the UV flash.  The TV flash is the simpler sub-problem: given
# (T, v, z), determine whether the bulk state is single-phase or two-phase
# and return the consistent pressure, densities, and compositions.
#
# Design (matches PH/PS pattern in this module):
#   * flash_tv:  inner pressure solve such that 1/rho_mix(T, p) = v_target.
#                Uses secant on ln(p), scanning from a vapor-side starting
#                point towards the liquid side.
#   * flash_uv:  outer Newton on T with du/dT evaluated by finite difference
#                through flash_tv.

def flash_tv(T, v_target, z, mixture, p_init=None, tol=1e-8, maxiter=60):
    """TV flash: given (T, v_target, z), find the pressure such that the
    bulk mixture molar volume equals v_target.

    In a two-phase region, the outer phase-fraction beta adjusts so that
    the volume-weighted average matches. Returns a MixtureFlashResult.

    Parameters
    ----------
    T : float                    Temperature [K]
    v_target : float             Target molar volume [m^3/mol]  (= 1/rho_target)
    z : array-like               Feed composition (mole fractions, sums to 1)
    mixture : Mixture            Mixture object
    p_init : float, optional     Initial pressure guess [Pa]. If None, uses
                                 ideal-gas estimate p = RT/v.
    tol : float                  Relative tolerance on v (default 1e-8)
    maxiter : int                Maximum outer iterations

    Notes
    -----
    Algorithm: secant in ln(p). At each iterate run flash_pt(p, T, z) and
    compute v_calc = 1/rho_bulk, then solve v_calc(ln p) = v_target by
    bracketing secant with bisection fallback.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()
    rho_target = 1.0 / v_target

    R_first = mixture.components[0].fluid.R

    # Initial guess: ideal-gas p = RT/v works well for vapor/supercritical.
    # For liquid-side v (small v, high rho), the ideal-gas estimate can land
    # on a far-too-high pressure where flash_pt's internal density solves
    # sometimes struggle. Detect liquid-side targets by comparing v_target
    # to a crude vapor-side reference (ideal gas at 1 atm): if v_target is
    # much smaller than that, use a moderate starting pressure (1 MPa)
    # that's more likely to land on a well-conditioned density root.
    R = R_first
    v_ref_vapor = R * T / 1e5   # vapor-side v at 1 atm
    if p_init is None:
        if v_target < 0.01 * v_ref_vapor:
            # Liquid-side target: start at 1 MPa (moderate pressure where
            # the liquid root of density_from_pressure is well-defined).
            p_init = 1e6
        else:
            p_init = R * T / v_target
    p = max(float(p_init), 1.0)

    def v_residual(p_val):
        r = flash_pt(p_val, T, z, mixture, check_stability=False, tol=1e-10)
        return 1.0 / r.rho - v_target, r

    # Initial probe
    res1, r1 = v_residual(p)
    if abs(res1) < tol * abs(v_target):
        return r1

    # Get a second point for secant. Pressure moves v in the opposite
    # direction (higher p -> smaller v); so if res1 > 0 (v_calc too big)
    # increase p, else decrease p.
    p2 = p * 2.0 if res1 > 0 else p * 0.5
    res2, r2 = v_residual(p2)
    if abs(res2) < tol * abs(v_target):
        return r2

    # Track the iterate with smallest |residual| so we can return it if
    # we stagnate near the answer (at dense-liquid states, flash_pt's own
    # density resolution limits how tightly we can satisfy v_target).
    best_res, best_r = (res1, r1) if abs(res1) <= abs(res2) else (res2, r2)

    # Bracket search: expand if same-sign
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

    # Now we have either a bracket or similar points; do secant with
    # bracket-preserving fallback
    prev_best_res = abs(best_res)
    stagnation_count = 0
    for it in range(maxiter):
        # Secant step in ln(p)
        lnp = np.log(p); lnp2 = np.log(p2)
        if abs(res2 - res1) < 1e-30:
            # Nearly flat -- midpoint
            lnp_new = 0.5 * (lnp + lnp2)
        else:
            lnp_new = lnp2 - res2 * (lnp2 - lnp) / (res2 - res1)
        # Safeguard: if outside bracket (when bracket exists), bisect
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

        # Stagnation: if best_res isn't improving, we've hit the density
        # resolution limit of flash_pt. Return the best iterate if it's
        # within a reasonable relative tolerance.
        if abs(best_res) >= prev_best_res * 0.9:
            stagnation_count += 1
            if stagnation_count >= 3 and abs(best_res) < 1e-5 * abs(v_target):
                return best_r
        else:
            stagnation_count = 0
        prev_best_res = abs(best_res)

        # Shift for next iteration: keep the point that maintains the
        # bracket (if one exists), otherwise drop the oldest.
        if res1 * res2 < 0:
            # Bracket exists; keep the side that makes a new bracket
            if res_new * res1 < 0:
                p2, res2, r2 = p_new, res_new, r_new
            else:
                p, res1, r1 = p_new, res_new, r_new
        else:
            # No bracket; slide: (p, p2) -> (p2, p_new)
            p, res1, r1 = p2, res2, r2
            p2, res2, r2 = p_new, res_new, r_new

    # Loop exhausted. If the best iterate is within a loose relative
    # tolerance (1e-4), return it with a note; otherwise raise.
    if abs(best_res) < 1e-4 * abs(v_target):
        return best_r
    raise RuntimeError(
        f"flash_tv did not converge (T={T}, v={v_target}, "
        f"last p={p2:.3e}, last v={1.0/r2.rho:.3e}, "
        f"best |res|={abs(best_res):.3e})"
    )


def flash_uv(u_target, v_target, z, mixture,
             T_init=None, tol=1e-6, maxiter=40):
    """UV flash: given (u_target, v_target, z), find (T, p, phase).

    The internal energy u is recovered from h, p, rho via u = h - p/rho,
    since MixtureFlashResult stores h and rho.

    Parameters
    ----------
    u_target : float             Target molar internal energy [J/mol]
    v_target : float             Target molar volume [m^3/mol]
    z : array-like               Feed composition
    mixture : Mixture            Mixture object
    T_init : float, optional     Initial temperature [K]. If None, uses
                                 mole-weighted component Tc.
    tol : float                  Relative tolerance on u
    maxiter : int                Maximum outer Newton iterations

    Returns
    -------
    MixtureFlashResult
        The standard flash result; use `result.h - result.p / result.rho`
        to recover u at the converged state.

    Notes
    -----
    Outer 1-D Newton on T:
        f(T) = u(T, v_target) - u_target
        f'(T) ~ cv_mix (approximated by finite difference through flash_tv)
    At each T we solve flash_tv to enforce the volume constraint.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z / z.sum()

    if T_init is None:
        T_init = float(np.dot(z, mixture.T_c))
    T = max(float(T_init), 50.0)

    u_scale = max(1.0, abs(u_target))

    # For damped Newton: track best result so far in case of divergence
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

        # Derivative: du/dT at constant v ~ cv_mix. Finite-difference via a
        # second flash_tv at T + dT; reuse the converged pressure as init
        # to make the inner solve cheap.
        last_p = r.p
        dT = max(0.01, 1e-4 * T)
        try:
            r2 = flash_tv(T + dT, v_target, z, mixture, p_init=last_p, tol=1e-10)
            u2 = r2.h - r2.p / r2.rho
            cv_est = (u2 - u_calc) / dT
        except RuntimeError:
            # Backward FD fallback
            r2 = flash_tv(T - dT, v_target, z, mixture, p_init=last_p, tol=1e-10)
            u2 = r2.h - r2.p / r2.rho
            cv_est = (u_calc - u2) / dT

        if cv_est <= 0 or not np.isfinite(cv_est):
            # Unphysical derivative; take a conservative bisection-style step
            T = T * (1.05 if diff < 0 else 0.95)
            continue

        step = -diff / cv_est
        # Damp: cap at 20% of T
        if abs(step) > 0.2 * T:
            step = 0.2 * T * np.sign(step)
        T_new = T + step
        if T_new <= 0:
            T_new = 0.5 * T
        T = T_new

    # If outer loop exhausted but we made progress, return the best result
    if best_result is not None and best_diff < tol * u_scale * 100:
        return best_result
    raise RuntimeError(
        f"flash_uv did not converge (u={u_target}, v={v_target}, "
        f"last T={T}, last |du|={best_diff:.3e})"
    )
