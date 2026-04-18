"""
Phase-equilibrium utilities.

For a pure component, vapor-liquid equilibrium is the solution of

    p(rho_L, T) = p(rho_V, T)                    (equal pressures)
    g(rho_L, T) = g(rho_V, T)                    (equal Gibbs energies)

We solve this 2-variable Newton system in (rho_L, rho_V) at given T.

Since the Maxwell (equal-area) criterion is equivalent to equal chemical
potentials, the residual function F takes the form:

    F_1 = p(rho_L, T) - p(rho_V, T)
    F_2 = mu(rho_L, T) - mu(rho_V, T)
        = R T [ alpha_r(rho_L) + Z_L - 1 - ln(rho_L)
              -(alpha_r(rho_V) + Z_V - 1 - ln(rho_V)) ]
        = R T [ alpha_r(rho_L) - alpha_r(rho_V) + Z_L - Z_V - ln(rho_L/rho_V) ]

The Jacobian is built from (dp/drho)_T and d(ln phi)/drho (both already
available analytically via the kernel derivatives).

A density root-finder for given (p, T) is also provided.
"""
import numpy as np

from .core import alpha_r_derivs
from .properties import _pressure_kernel, _dp_drho_T_kernel


def _mu_over_RT(rho, T, R, rho_c, T_c,
                pn, pd, pt, en, ed, et, ec,
                gn, gd, gt, ge, geps, gb, ggam,
                na, naa, nb, nB, nC, nD, nA, nbeta):
    """Return (Ar + Z - 1 - ln(rho / ref)). Additive constants cancel in
    differences mu_L - mu_V.
    """
    delta = rho / rho_c
    tau = T_c / T
    Ar, Ar_d, _, _, _, _ = alpha_r_derivs(delta, tau,
                                           pn, pd, pt,
                                           en, ed, et, ec,
                                           gn, gd, gt, ge, geps, gb, ggam,
                                           na, naa, nb, nB, nC, nD, nA, nbeta)
    Z = 1.0 + delta * Ar_d
    return Ar + (Z - 1.0) - np.log(rho)


def _density_newton(p_target, T, rho_init, fluid, tol=1e-10, maxiter=60):
    """Solve p(rho, T) = p_target at fixed T using a safeguarded Newton."""
    args = fluid.pack()[:25]  # residual part only for pressure
    rho = float(rho_init)
    for _ in range(maxiter):
        p = _pressure_kernel(rho, T, *args)
        dpdrho = _dp_drho_T_kernel(rho, T, *args)
        if dpdrho <= 0.0:
            # in unstable region -- step away toward stability
            rho *= 1.1 if p < p_target else 0.9
            continue
        step = (p - p_target) / dpdrho
        # backtracking to keep rho positive and within rho_max
        rho_new = rho - step
        trials = 0
        while (rho_new <= 0.0 or rho_new > fluid.rho_max) and trials < 20:
            step *= 0.5
            rho_new = rho - step
            trials += 1
        rho = rho_new
        if abs(step) < tol * max(1.0, abs(rho)):
            return rho
    return rho


def density_from_pressure(p, T, fluid, phase="auto"):
    """Find density (mol/m^3) satisfying p(rho, T) = p at the requested phase.

    Parameters
    ----------
    phase : {'auto', 'vapor', 'liquid'}
        - 'vapor':  start from ideal-gas estimate rho = p/(RT)
        - 'liquid': start from the saturated-liquid density at T (from the
                    ancillary), which is an excellent guess for any liquid
                    state at p > p_sat(T).
        - 'auto':   if T >= T_c or p < p_sat, return vapor; else liquid
    """
    if phase == "vapor":
        rho0 = p / (fluid.R * T)
    elif phase == "liquid":
        # For subcritical T, use rho_L_sat(T) as the seed -- much better than
        # a generic dense-liquid guess. For supercritical, fall back to a
        # high-density initial guess.
        if T < fluid.T_c:
            try:
                rL, _, _ = saturation_pT(T, fluid)
                rho0 = rL
            except Exception:
                rho0 = fluid.rho_c * 2.5
        else:
            rho0 = fluid.rho_c * 2.5
    else:  # auto
        if T >= fluid.T_c:
            rho0 = p / (fluid.R * T)
        else:
            try:
                rL, rV, _ = saturation_pT(T, fluid)
                rho0 = rL if p > _pressure_kernel(rL, T, *fluid.pack()[:25]) * 0.999 else rV
            except Exception:
                rho0 = fluid.rho_c * 2.5 if p > fluid.p_c else p / (fluid.R * T)
    return _density_newton(p, T, rho0, fluid)


def _ancillary_densities(T, fluid):
    """Saturation-density initial guesses for the Newton solver.

    If the fluid's JSON provides ancillary equations under the key
    ``'ancillary'``, those are used. Otherwise we fall back to a generic
    corresponding-states-style estimate. The Newton solver is tolerant of
    moderately bad initial guesses as long as they're on the right side of
    rho_c.
    """
    T_c = fluid.T_c
    rho_c = fluid.rho_c
    tau = 1.0 - T / T_c
    if tau <= 0.0:
        return rho_c, rho_c

    anc = fluid._raw.get("ancillary", {})
    # A few different published ancillary forms are in common use. We try in
    # order: (a) explicit form flag; (b) heuristic on the sign/magnitude of
    # the sum; (c) generic fallback.
    if "rho_L" in anc and "rho_V" in anc:
        nL = np.asarray(anc["rho_L"]["n"])
        tL = np.asarray(anc["rho_L"]["t"])
        nV = np.asarray(anc["rho_V"]["n"])
        tV = np.asarray(anc["rho_V"]["t"])
        form = anc.get("form", "auto")

        sL = float(np.sum(nL * tau ** tL))
        sV = float(np.sum(nV * tau ** tV))

        if form == "exp_exp":
            # Both liquid and vapor are rho/rho_c = exp(sum)
            rL = rho_c * np.exp(sL)
            rV = rho_c * np.exp(sV)
        elif form == "poly_exp":
            # Liquid: rho/rho_c = 1 + sum;    vapor: rho/rho_c = exp(sum)
            rL = rho_c * (1.0 + sL)
            rV = rho_c * np.exp(sV)
        else:  # auto
            # Liquid is the hard one: pick whichever of (1+sL) or exp(sL) gives
            # rho_L > rho_c (the physical branch)
            rL_poly = rho_c * (1.0 + sL)
            rL_exp = rho_c * np.exp(sL)
            rL = rL_poly if rL_poly > rho_c else rL_exp
            rV = rho_c * np.exp(sV)

        # Safety: nudge onto the right side of rho_c if the ancillary was mis-applied
        if rL <= rho_c:
            rL = rho_c * (1.0 + 1.75 * tau ** 0.35 + 0.75 * tau)
        if rV >= rho_c or rV <= 0:
            rV = rho_c * np.exp(-2.0 * tau ** 0.35 - 5.0 * tau)
        return rL, rV

    # Generic fallback (Rackett/Riedel-inspired)
    rL = rho_c * (1.0 + 1.75 * tau ** 0.35 + 0.75 * tau)
    rV = rho_c * np.exp(-2.0 * tau ** 0.35 - 5.0 * tau)
    return rL, rV


def saturation_pT(T, fluid, tol=1e-9, maxiter=80):
    """Solve pure-component VLE at temperature T.

    Returns (rho_L, rho_V, p_sat).

    Algorithm
    ---------
    We iterate on the saturation pressure rather than densities directly,
    because the two density scales (rho_L ~ 10^4, rho_V ~ 10^1) differ by
    orders of magnitude and make a joint-density Newton stiff.

    Outer loop (Newton on p):
        Given p, solve p(rho_L, T) = p on the liquid branch and
              p(rho_V, T) = p on the vapor branch.
        Residual:  F(p) = g(rho_L, T) - g(rho_V, T)           [equal Gibbs]
        dF/dp    =  1/rho_L - 1/rho_V                         [Maxwell]

    This is numerically tame: each inner density solve is a 1-D Newton on
    a monotonic (single-branch) p(rho) curve.
    """
    if T >= fluid.T_c:
        raise ValueError(f"T={T} is at or above the critical temperature "
                         f"T_c={fluid.T_c}; no saturation state exists.")
    if T < fluid.T_min:
        raise ValueError(f"T={T} below validity Tmin={fluid.T_min}")

    args = fluid.pack()[:25]
    R = fluid.R

    # Initial densities via ancillary
    rho_L, rho_V = _ancillary_densities(T, fluid)

    # Initial pressure guess: use the vapor-side pressure (which the ancillary
    # gets nearly right), bounded away from 0 and p_c.
    pL_init = _pressure_kernel(rho_L, T, *args)
    pV_init = _pressure_kernel(rho_V, T, *args)
    # Vapor branch is usually accurate; liquid can be far off. Prefer vapor.
    p = pV_init if 0 < pV_init < fluid.p_c else 0.5 * fluid.p_c

    for it in range(maxiter):
        # Liquid-branch density at this p: use current rho_L as start
        rho_L = _solve_density_branch(p, T, rho_L, fluid, "liquid", args)
        # Vapor-branch density at this p
        rho_V = _solve_density_branch(p, T, rho_V, fluid, "vapor", args)

        # Residual: F(p) = ln(phi_L) - ln(phi_V) at equilibrium = 0
        F = _ln_phi_diff(p, T, rho_L, rho_V, fluid, args)

        if abs(F) < tol:
            return rho_L, rho_V, p

        # Exact derivative dF/dp at fixed T:
        # Holding rho on each branch at the constraint p(rho,T) = p,
        #   d(ln phi)/dp|_T = (Z - 1)/p
        # (Standard result; derivation: d ln(phi)/dp = (V - V_ideal)/(RT) = (Z-1)/p.)
        # So  dF/dp = (Z_L - Z_V)/p.
        ZL = p / (rho_L * R * T)
        ZV = p / (rho_V * R * T)
        dF_dp = (ZL - ZV) / p
        if dF_dp == 0.0:
            return rho_L, rho_V, p
        dp = -F / dF_dp

        # Bound the step: don't cross 0 or p_c
        p_new = p + dp
        if p_new <= 0.0:
            p_new = 0.5 * p
        if p_new >= fluid.p_c:
            p_new = 0.5 * (p + fluid.p_c)
        p = p_new

    raise RuntimeError(f"Saturation solver did not converge at T={T} "
                       f"(final |F|={abs(F):.3e}, p={p:.3e} Pa)")


def _ln_phi_diff(p, T, rho_L, rho_V, fluid, args):
    """Compute ln(phi_L) - ln(phi_V) at given state."""
    R = fluid.R
    ZL = p / (rho_L * R * T)
    ZV = p / (rho_V * R * T)
    deltaL = rho_L / fluid.rho_c
    deltaV = rho_V / fluid.rho_c
    tau = fluid.T_c / T
    ArL = alpha_r_derivs(deltaL, tau, *args[3:])[0]
    ArV = alpha_r_derivs(deltaV, tau, *args[3:])[0]
    lnphiL = ArL + (ZL - 1.0) - np.log(ZL)
    lnphiV = ArV + (ZV - 1.0) - np.log(ZV)
    return lnphiL - lnphiV


def _solve_density_branch(p_target, T, rho_start, fluid, branch, args,
                          tol=1e-12, maxiter=100):
    """1-D Newton for rho satisfying p(rho, T) = p_target on a given branch.

    ``branch`` is 'liquid' (rho > rho_c, mechanically stable) or 'vapor'
    (rho < rho_c, mechanically stable). We safeguard by clipping rho away
    from the spinodal (where dp/drho = 0).
    """
    rho_c = fluid.rho_c
    rho = float(rho_start)

    for _ in range(maxiter):
        p = _pressure_kernel(rho, T, *args)
        dp = _dp_drho_T_kernel(rho, T, *args)
        if dp <= 0.0:
            # Inside the mechanical spinodal -- push toward the stable branch
            if branch == "liquid":
                rho *= 1.05
            else:
                rho *= 0.95
            continue
        step = (p - p_target) / dp
        rho_new = rho - step
        # Keep on the correct side of rho_c (with a small margin)
        if branch == "liquid" and rho_new <= rho_c * 1.001:
            rho_new = 0.5 * (rho + rho_c * 1.05)
        if branch == "vapor" and rho_new >= rho_c * 0.999:
            rho_new = 0.5 * (rho + rho_c * 0.95)
        if rho_new <= 0.0:
            rho_new = 0.5 * rho
        if abs(rho_new - rho) < tol * max(1.0, abs(rho)):
            return rho_new
        rho = rho_new
    return rho
