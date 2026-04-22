"""
Flash algorithms for single-component fluids.

Supported flashes
-----------------
    flash_pt(p, T, fluid)          -- pressure/temperature  -> state
    flash_ph(p, h, fluid)          -- pressure/enthalpy     -> state
    flash_ps(p, s, fluid)          -- pressure/entropy      -> state
    flash_th(T, h, fluid)          -- temperature/enthalpy  -> state
    flash_ts(T, s, fluid)          -- temperature/entropy   -> state
    flash_tv(T, v, fluid)          -- temperature/volume    -> state
    flash_uv(u, v, fluid)          -- energy/volume (natural variables) -> state

All flashes return a ``FlashResult`` dataclass with fields:

    phase     -- 'liquid', 'vapor', 'two_phase', or 'supercritical'
    T, p      -- temperature [K], pressure [Pa]
    rho       -- density [mol/m^3] (mass-average if two-phase)
    rho_L     -- liquid density (None if single-phase)
    rho_V     -- vapor density (None if single-phase)
    quality   -- vapor mass fraction in [0,1] (None if single-phase)
    u, h, s   -- internal energy [J/mol], enthalpy [J/mol], entropy [J/(mol K)]
    cp, cv, w -- caloric and acoustic properties (None inside the two-phase region)

Conventions
-----------
When (p, h), (p, s), (T, h), or (T, s) lies strictly inside the two-phase
envelope:
  - The saturation pressure/temperature is used as the state's p, T
  - rho is the mass-average density (using quality as weight)
  - cv, cp, w are set to None (they are singular on the coexistence line)
When the state is right on the saturation boundary, a single-phase result
is returned with quality = 0.0 or 1.0 as appropriate.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from . import properties as props
from .saturation import saturation_pT, density_from_pressure


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FlashResult:
    """Thermodynamic state returned by any flash call."""
    phase: str
    T: float
    p: float
    rho: float
    u: float
    h: float
    s: float
    rho_L: Optional[float] = None
    rho_V: Optional[float] = None
    quality: Optional[float] = None
    cp: Optional[float] = None
    cv: Optional[float] = None
    w: Optional[float] = None

    def __repr__(self):
        lines = [f"FlashResult(phase={self.phase!r}"]
        lines.append(f"    T = {self.T:.4f} K")
        lines.append(f"    p = {self.p:.4e} Pa = {self.p*1e-6:.5f} MPa")
        lines.append(f"    rho = {self.rho:.4f} mol/m^3")
        if self.quality is not None:
            lines.append(f"    quality = {self.quality:.6f}")
            lines.append(f"    rho_L = {self.rho_L:.2f}, rho_V = {self.rho_V:.4f}")
        lines.append(f"    u = {self.u:.3f} J/mol")
        lines.append(f"    h = {self.h:.3f} J/mol")
        lines.append(f"    s = {self.s:.4f} J/(mol K)")
        if self.cp is not None:
            lines.append(f"    cp = {self.cp:.3f}, cv = {self.cv:.3f} J/(mol K)")
            lines.append(f"    w  = {self.w:.3f} m/s" if self.w is not None else "")
        return "\n".join(lines) + ")"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _single_phase_result(rho, T, fluid, phase_label):
    """Build a FlashResult for a single-phase state at known (rho, T)."""
    p = props.pressure(rho, T, fluid)
    u = props.internal_energy(rho, T, fluid)
    h = props.enthalpy(rho, T, fluid)
    s = props.entropy(rho, T, fluid)
    cv = props.cv(rho, T, fluid)
    cp = props.cp(rho, T, fluid)
    try:
        w = props.speed_of_sound(rho, T, fluid)
    except ValueError:
        w = None   # molar_mass missing
    return FlashResult(
        phase=phase_label, T=T, p=p, rho=rho,
        u=u, h=h, s=s, cv=cv, cp=cp, w=w,
    )


def _two_phase_result(p, T, rho_L, rho_V, quality, fluid):
    """Build a FlashResult for a two-phase (VLE) state.

    ``quality`` is the vapor mole fraction in [0, 1].
    """
    # Extensive properties scale with mole-fraction averages
    u_L = props.internal_energy(rho_L, T, fluid)
    u_V = props.internal_energy(rho_V, T, fluid)
    h_L = props.enthalpy(rho_L, T, fluid)
    h_V = props.enthalpy(rho_V, T, fluid)
    s_L = props.entropy(rho_L, T, fluid)
    s_V = props.entropy(rho_V, T, fluid)

    x = quality
    u = (1.0 - x) * u_L + x * u_V
    h = (1.0 - x) * h_L + x * h_V
    s = (1.0 - x) * s_L + x * s_V
    # Specific-volume average gives overall density:
    #   v = (1-x) v_L + x v_V    =>    rho = 1/v
    v_L, v_V = 1.0 / rho_L, 1.0 / rho_V
    rho = 1.0 / ((1.0 - x) * v_L + x * v_V)
    return FlashResult(
        phase="two_phase", T=T, p=p, rho=rho,
        u=u, h=h, s=s,
        rho_L=rho_L, rho_V=rho_V, quality=x,
        cp=None, cv=None, w=None,   # singular across the two-phase line
    )


def _T_sat_from_p(p, fluid, T_init=None, tol=1e-9, maxiter=60):
    """Invert p_sat(T) = p.  Returns (T_sat, rho_L, rho_V).

    Uses a safeguarded Newton iteration with the Clausius-Clapeyron
    derivative: d p_sat / d T = (s_V - s_L) / (v_V - v_L).
    """
    if p >= fluid.p_c:
        raise ValueError(f"p = {p} Pa >= p_c = {fluid.p_c} Pa; no saturation.")
    if p <= 0.0:
        raise ValueError(f"p = {p} Pa must be positive for saturation.")

    # Initial guess: linear interpolation in ln(p_sat) between triple and critical
    if T_init is None:
        # Clausius-Clapeyron-like log-linear interpolation
        #   ln(p_sat) ~ A - B / T
        # Anchored at triple (T_t, p_t) and critical (T_c, p_c).
        T_t = max(fluid.T_triple, fluid.T_min + 1.0)
        p_t = max(fluid.p_triple, 1.0)
        if p_t >= fluid.p_c:
            T = 0.5 * (T_t + fluid.T_c)
        else:
            # ln(p) = a + b/T, solve for T given p
            lnpc = np.log(fluid.p_c)
            lnpt = np.log(p_t)
            b = (lnpc - lnpt) / (1.0 / T_t - 1.0 / fluid.T_c)
            a = lnpc - b / fluid.T_c
            T = b / (np.log(p) - a)
            # Clip into valid range
            T = max(T_t + 1e-3, min(T, fluid.T_c - 1e-3))
    else:
        T = T_init

    for _ in range(maxiter):
        rho_L, rho_V, p_calc = saturation_pT(T, fluid)
        if abs(p_calc - p) < tol * max(1.0, p):
            return T, rho_L, rho_V
        # Clausius-Clapeyron slope:
        #   d p_sat / d T = (h_V - h_L) / (T * (v_V - v_L))
        # Use the enthalpy form (equivalent to entropy form via g_L = g_V).
        h_L = props.enthalpy(rho_L, T, fluid)
        h_V = props.enthalpy(rho_V, T, fluid)
        dpdT = (h_V - h_L) / (T * (1.0 / rho_V - 1.0 / rho_L))
        if dpdT <= 0.0:
            # Shouldn't happen physically; bail out with bisection step
            T = 0.5 * (T + fluid.T_c)
            continue
        dT = (p - p_calc) / dpdT
        # Damp step to stay in (T_triple, T_c)
        T_new = T + dT
        T_new = max(fluid.T_triple + 1e-4, min(T_new, fluid.T_c - 1e-4))
        T = T_new

    raise RuntimeError(f"T_sat inversion did not converge for p = {p} Pa.")


# ---------------------------------------------------------------------------
# PT flash
# ---------------------------------------------------------------------------

def flash_pt(p, T, fluid, phase_hint=None):
    """PT flash for a single-component fluid.

    Parameters
    ----------
    p : float              pressure [Pa]
    T : float              temperature [K]
    fluid : Fluid
    phase_hint : str or None
        Optional. 'liquid', 'vapor' or 'supercritical'. If omitted, chosen
        automatically. For a subcritical isotherm, a (p, T) pair lying on the
        saturation line is ambiguous between liquid and vapor; this argument
        lets you select the branch.

    Returns
    -------
    FlashResult
    """
    if T >= fluid.T_c:
        rho = density_from_pressure(p, T, fluid, phase="vapor")
        return _single_phase_result(rho, T, fluid, "supercritical")

    # Subcritical: compare p to p_sat(T)
    try:
        rho_L, rho_V, p_sat = saturation_pT(T, fluid)
    except Exception:
        # Could not solve saturation (e.g. at T_triple boundary); just try vapor
        rho = density_from_pressure(p, T, fluid, phase="vapor")
        return _single_phase_result(rho, T, fluid, "vapor")

    rel_gap = abs(p - p_sat) / p_sat
    if rel_gap < 1e-9 and phase_hint is None:
        # Exactly on the saturation line -- ambiguous; default to liquid
        # (caller should pass phase_hint or use a flash with quality)
        return _single_phase_result(rho_L, T, fluid, "liquid")

    if phase_hint == "liquid":
        rho = density_from_pressure(p, T, fluid, phase="liquid")
        return _single_phase_result(rho, T, fluid, "liquid")
    if phase_hint == "vapor":
        rho = density_from_pressure(p, T, fluid, phase="vapor")
        return _single_phase_result(rho, T, fluid, "vapor")

    # Auto: liquid if above saturation pressure, vapor if below
    if p > p_sat:
        rho = density_from_pressure(p, T, fluid, phase="liquid")
        return _single_phase_result(rho, T, fluid, "liquid")
    else:
        rho = density_from_pressure(p, T, fluid, phase="vapor")
        return _single_phase_result(rho, T, fluid, "vapor")


# ---------------------------------------------------------------------------
# PH and PS flashes
# ---------------------------------------------------------------------------

def _isobaric_newton(p, target, prop_fn, fluid, T_init, phase,
                     tol=1e-8, maxiter=80):
    """Newton iteration on a property along an isobar.

    Solves  prop_fn(p, T) = target  for T at fixed p, single-phase.

    Uses exact dh/dT|p = cp  or  ds/dT|p = cp/T.

    Step is damped and T is clipped into [T_min, 10*T_c] to stay robust even
    with a poor initial guess.
    """
    T_lo = fluid.T_min if fluid.T_min > 0 else 1.0
    T_hi = 10.0 * fluid.T_c
    # Also don't cross T_c from above/below without care in subcritical mode
    if phase == "liquid":
        T_hi = min(T_hi, fluid.T_c)
    elif phase == "vapor" and p >= fluid.p_c:
        pass   # allow crossing T_c
    # Allow T to move freely within [T_lo, T_hi]; Newton handles subcritical
    # liquid and vapor separately via `phase`.

    T = float(T_init)
    T = max(T_lo + 1e-3, min(T, T_hi - 1e-3))
    last_diff = np.inf
    for _ in range(maxiter):
        rho = density_from_pressure(p, T, fluid, phase=phase)
        val = prop_fn(rho, T, fluid)
        diff = val - target
        if abs(diff) < tol * max(1.0, abs(target)):
            return rho, T
        cp = props.cp(rho, T, fluid)
        if cp is None or cp <= 0 or np.isnan(cp):
            # Nudge and retry
            T = T * 1.05
            continue
        deriv = cp if prop_fn is props.enthalpy else cp / T
        dT = -diff / deriv
        # Damp step size to at most 20% of current T (or 50 K, whichever smaller)
        max_step = min(abs(T) * 0.2, 50.0)
        if abs(dT) > max_step:
            dT = np.sign(dT) * max_step
        T_new = T + dT
        # Clip into valid T range
        if T_new <= T_lo:
            T_new = 0.5 * (T + T_lo)
        if T_new >= T_hi:
            T_new = 0.5 * (T + T_hi)
        T = T_new
        last_diff = diff
    raise RuntimeError(f"Isobaric Newton did not converge "
                       f"(p={p}, target={target}, last T={T}, residual={last_diff})")


def flash_ph(p, h_target, fluid, T_init=None):
    """PH flash.

    Given pressure and molar enthalpy, return the complete state.

    Algorithm
    ---------
    1. If p >= p_c:  single-phase everywhere (no phase boundary crossed as T
       changes at constant p >= p_c). Newton on h(T) with phase='auto' so the
       density solver picks the appropriate branch at each iterate.
    2. If p <  p_c:  compute saturation at p, get h_L, h_V.
         a. If h_target < h_L:  subcritical liquid -- Newton with liquid phase.
         b. If h_target > h_V:  subcritical vapor  -- Newton with vapor phase.
         c. Else:               two-phase with quality = (h - h_L)/(h_V - h_L).
    """
    if p >= fluid.p_c:
        # Single-phase above critical pressure: no dome to cross.
        # Use 'auto' phase -- density solver picks the right branch.
        if T_init is None:
            # Rough initial guess: bisect-bracket on T_c.
            # If h_target is small, we're in the liquid-like regime (T < T_c);
            # if large, gas-like (T > T_c). We don't know h(T_c) a priori but
            # can use this heuristic:
            T_init = fluid.T_c
        rho, T = _isobaric_newton(p, h_target, props.enthalpy, fluid,
                                  T_init, phase="auto")
        label = "supercritical" if T >= fluid.T_c else "liquid"
        return _single_phase_result(rho, T, fluid, label)

    # Subcritical: identify region using saturation enthalpies
    T_sat, rho_L_sat, rho_V_sat = _T_sat_from_p(p, fluid)
    h_L = props.enthalpy(rho_L_sat, T_sat, fluid)
    h_V = props.enthalpy(rho_V_sat, T_sat, fluid)

    if h_target <= h_L:
        # Subcooled liquid
        T0 = T_init if T_init is not None else T_sat * 0.98
        rho, T = _isobaric_newton(p, h_target, props.enthalpy, fluid,
                                  T0, phase="liquid")
        return _single_phase_result(rho, T, fluid, "liquid")
    elif h_target >= h_V:
        # Superheated vapor
        T0 = T_init if T_init is not None else T_sat * 1.05
        rho, T = _isobaric_newton(p, h_target, props.enthalpy, fluid,
                                  T0, phase="vapor")
        return _single_phase_result(rho, T, fluid, "vapor")
    else:
        # Two-phase
        x = (h_target - h_L) / (h_V - h_L)
        return _two_phase_result(p, T_sat, rho_L_sat, rho_V_sat, x, fluid)


def flash_ps(p, s_target, fluid, T_init=None):
    """PS flash (same structure as PH)."""
    if p >= fluid.p_c:
        if T_init is None:
            T_init = fluid.T_c
        rho, T = _isobaric_newton(p, s_target, props.entropy, fluid,
                                  T_init, phase="auto")
        label = "supercritical" if T >= fluid.T_c else "liquid"
        return _single_phase_result(rho, T, fluid, label)

    T_sat, rho_L_sat, rho_V_sat = _T_sat_from_p(p, fluid)
    s_L = props.entropy(rho_L_sat, T_sat, fluid)
    s_V = props.entropy(rho_V_sat, T_sat, fluid)

    if s_target <= s_L:
        T0 = T_init if T_init is not None else T_sat * 0.98
        rho, T = _isobaric_newton(p, s_target, props.entropy, fluid,
                                  T0, phase="liquid")
        return _single_phase_result(rho, T, fluid, "liquid")
    elif s_target >= s_V:
        T0 = T_init if T_init is not None else T_sat * 1.05
        rho, T = _isobaric_newton(p, s_target, props.entropy, fluid,
                                  T0, phase="vapor")
        return _single_phase_result(rho, T, fluid, "vapor")
    else:
        x = (s_target - s_L) / (s_V - s_L)
        return _two_phase_result(p, T_sat, rho_L_sat, rho_V_sat, x, fluid)


# ---------------------------------------------------------------------------
# TV and UV flashes
# ---------------------------------------------------------------------------

def flash_tv(T, v, fluid):
    """TV flash.  v is molar volume [m^3/mol]; rho = 1/v.

    For a pure fluid this is trivial, except that if T < T_c and
    rho lies between rho_V_sat and rho_L_sat, the state is two-phase.
    """
    rho = 1.0 / v

    if T >= fluid.T_c:
        return _single_phase_result(rho, T, fluid, "supercritical")

    # Subcritical: check if inside the dome
    rho_L, rho_V, p_sat = saturation_pT(T, fluid)
    if rho_V < rho < rho_L:
        # Two-phase: quality from lever rule on volume
        v_L = 1.0 / rho_L
        v_V = 1.0 / rho_V
        x = (v - v_L) / (v_V - v_L)
        return _two_phase_result(p_sat, T, rho_L, rho_V, x, fluid)

    label = "liquid" if rho >= rho_L else "vapor"
    return _single_phase_result(rho, T, fluid, label)


def flash_uv(u_target, v, fluid, T_init=None, tol=1e-8, maxiter=60):
    """UV flash -- natural-variable flash used in dynamic simulation.

    Solves u(T, v) = u_target at fixed v.
    Uses du/dT|v = cv as the Newton derivative.
    """
    rho = 1.0 / v

    if T_init is None:
        # A crude but usually adequate guess: ideal-gas with cv ~ 2.5 R
        # u_ig(T) - u_ig(T_ref) = cv * (T - T_ref)
        # We don't have u_ig(T_ref), but at moderate T near T_c this is OK.
        T_init = fluid.T_c

    T = float(T_init)
    for _ in range(maxiter):
        # First check if at this (T, v) we're inside the dome
        if T < fluid.T_c:
            rho_L, rho_V, p_sat = saturation_pT(T, fluid)
            if rho_V < rho < rho_L:
                # Inside dome: u varies linearly with quality
                v_L, v_V = 1.0 / rho_L, 1.0 / rho_V
                x = (v - v_L) / (v_V - v_L)
                u_L = props.internal_energy(rho_L, T, fluid)
                u_V = props.internal_energy(rho_V, T, fluid)
                u_mix = (1 - x) * u_L + x * u_V
                diff = u_mix - u_target
                # du_mix / dT |v  has contributions from both legs; use
                # Clausius-Clapeyron-style temperature sensitivity.
                # Cheap finite difference:
                dT_probe = 1e-3 * T
                rL2, rV2, _ = saturation_pT(T + dT_probe, fluid)
                if rV2 < rho < rL2:
                    v_L2, v_V2 = 1.0 / rL2, 1.0 / rV2
                    x2 = (v - v_L2) / (v_V2 - v_L2)
                    u_L2 = props.internal_energy(rL2, T + dT_probe, fluid)
                    u_V2 = props.internal_energy(rV2, T + dT_probe, fluid)
                    u_mix2 = (1 - x2) * u_L2 + x2 * u_V2
                    du_dT = (u_mix2 - u_mix) / dT_probe
                else:
                    du_dT = props.cv(rho, T, fluid)  # fallback
                if abs(diff) < tol * max(1.0, abs(u_target)):
                    return _two_phase_result(p_sat, T, rho_L, rho_V, x, fluid)
                T -= diff / du_dT
                continue
        # Single-phase Newton on u at constant v (i.e., at constant rho)
        u = props.internal_energy(rho, T, fluid)
        cv = props.cv(rho, T, fluid)
        diff = u - u_target
        if abs(diff) < tol * max(1.0, abs(u_target)):
            if T >= fluid.T_c:
                return _single_phase_result(rho, T, fluid, "supercritical")
            # Is it liquid or vapor branch?
            rho_L, rho_V, _ = saturation_pT(T, fluid)
            label = "liquid" if rho >= rho_L else "vapor"
            return _single_phase_result(rho, T, fluid, label)
        T_new = T - diff / cv
        if T_new <= 0:
            T_new = 0.5 * T
        T = T_new

    raise RuntimeError(f"UV flash did not converge (u={u_target}, v={v})")


# ---------------------------------------------------------------------------
# TH and TS flashes
# ---------------------------------------------------------------------------
#
# At fixed T, we search for the density rho that satisfies
# h(rho, T) = h_target  or  s(rho, T) = s_target.
#
# Analytic derivatives at fixed T (derived from the Helmholtz formulation):
#
#   (dh/drho)_T = [ RT * tau * alpha_r_dt                   ... from u part
#                   + RT * (alpha_r_d + delta * alpha_r_dd) ... from p/rho part
#                 ] / rho_c
#
#   (ds/drho)_T = -(1/rho^2) * (dp/dT)_rho     -- Maxwell relation
#
# Both are cheap to evaluate from quantities we already compute in the
# _all_props_kernel, so we reuse that.


def _isothermal_newton(T, target, prop_fn, deriv_fn, fluid,
                       rho_init, rho_min, rho_max,
                       tol=1e-9, maxiter=80):
    """Newton in rho at fixed T, clipped to [rho_min, rho_max].

    ``prop_fn(rho, T, fluid)`` returns the property value (h or s).
    ``deriv_fn(rho, T, fluid)`` returns (d prop / d rho)_T.

    Uses a damped Newton step capped at 30% of rho and clipped into the
    bracket.  If the sign of the residual indicates the guess is outside
    the bracket, we pull it in.
    """
    rho = float(rho_init)
    rho = max(rho_min * 1.0001, min(rho, rho_max * 0.9999))
    last_diff = np.inf
    for _ in range(maxiter):
        val = prop_fn(rho, T, fluid)
        diff = val - target
        if abs(diff) < tol * max(1.0, abs(target)):
            return rho
        d = deriv_fn(rho, T, fluid)
        if d == 0.0 or not np.isfinite(d):
            # Perturb and retry
            rho *= 1.01
            continue
        drho = -diff / d
        # Damp step
        max_step = 0.30 * abs(rho)
        if abs(drho) > max_step:
            drho = np.sign(drho) * max_step
        rho_new = rho + drho
        # Bound to branch
        if rho_new <= rho_min:
            rho_new = 0.5 * (rho + rho_min)
        if rho_new >= rho_max:
            rho_new = 0.5 * (rho + rho_max)
        rho = rho_new
        last_diff = diff
    raise RuntimeError(
        f"Isothermal Newton in rho did not converge "
        f"(T={T}, target={target}, last rho={rho}, residual={last_diff})"
    )


def _dh_drho_T(rho, T, fluid):
    """(dh/drho)_T computed from the Helmholtz derivatives.

    Derivation:  h = RT*tau*(a0_t + ar_t) + RT*(1 + delta*ar_d)
                 where delta = rho/rho_c, tau = T_c/T.
    At fixed T (fixed tau):
        dh/d(delta) = RT*tau*ar_dt + RT*(ar_d + delta*ar_dd)
        dh/drho     = (1/rho_c) * dh/d(delta)
    """
    # Pull alpha_r derivatives directly from the full property kernel
    # Using a minimal call path -- we compute the residual derivatives
    # via alpha_r_derivs and build from there.
    from .core import alpha_r_derivs
    R, rho_c, T_c = fluid.R, fluid.rho_c, fluid.T_c
    delta = rho / rho_c
    tau = T_c / T
    pack = fluid.pack()
    res_args = pack[3:41]
    _, Ar_d, _, Ar_dd, _, Ar_dt = alpha_r_derivs(delta, tau, *res_args)
    dh_ddelta = R * T * tau * Ar_dt + R * T * (Ar_d + delta * Ar_dd)
    return dh_ddelta / rho_c


def _ds_drho_T(rho, T, fluid):
    """(ds/drho)_T from the Maxwell relation.

        (ds/dv)_T = (dp/dT)_v     (intensive form of dA = -s dT - p dv)
        (ds/drho)_T = -(1/rho^2) * (dp/dT)_rho
    """
    dpdT = props.dp_dT_rho(rho, T, fluid)
    return -dpdT / (rho * rho)


def flash_th(T, h_target, fluid, phase_hint=None, p_hint=None):
    """TH flash: find pressure (and phase) at given temperature and molar enthalpy.

    At fixed T there can be **multiple** single-phase states with the same h
    (different p on the liquid vs vapor branch).  For subcritical T:

      - Compressed liquid has h increasing from h_L_sat as p increases above p_sat.
      - Superheated vapor has h near h_V_sat, slightly varying with p below p_sat.

    So h in [h_L_sat, h_V_sat] is genuinely ambiguous between two-phase and
    (often) compressed liquid.  This function resolves ambiguity as follows:

      - If h < h_L_sat: two-phase (h is below the liquid-saturation enthalpy;
        no stable single-phase state has h this low at this T).
      - If h > h_V_sat: superheated vapor (unambiguous).
      - If h_L_sat <= h <= h_V_sat:
          * If ``phase_hint='liquid'``: solve compressed-liquid branch.
          * If ``phase_hint='two_phase'`` (or None, default): return two-phase
            at p_sat with lever-rule quality.
          * If ``phase_hint='vapor'``: solve vapor branch (rarely useful; h in
            this range is above the ideal-gas-like vapor values for most fluids).

    For supercritical T, no classification is needed; Newton in rho at fixed T.

    Parameters
    ----------
    T          : float  temperature [K]
    h_target   : float  molar enthalpy [J/mol]
    fluid      : Fluid
    phase_hint : {'liquid', 'vapor', 'two_phase', None}
        Branch selection for ambiguous (subcritical) cases.
    p_hint     : float or None
        Initial pressure guess for supercritical Newton.

    Returns
    -------
    FlashResult
    """
    if T >= fluid.T_c:
        if p_hint is None:
            p_hint = max(fluid.p_triple, 1e5)
        rho_init = p_hint / (fluid.R * T)
        rho_min = 1e-8 * fluid.rho_c
        rho_max = 50.0 * fluid.rho_c
        rho = _isothermal_newton(T, h_target,
                                 props.enthalpy, _dh_drho_T, fluid,
                                 rho_init, rho_min, rho_max)
        return _single_phase_result(rho, T, fluid, "supercritical")

    # Subcritical: classify via saturation enthalpies
    rho_L_sat, rho_V_sat, p_sat = saturation_pT(T, fluid)
    h_L = props.enthalpy(rho_L_sat, T, fluid)
    h_V = props.enthalpy(rho_V_sat, T, fluid)

    if h_target < h_L:
        # Below saturated-liquid enthalpy: only two-phase (with very low x) or
        # subcooled-metastable liquid. Conventional result: two-phase would be
        # at p_sat with x < 0, which is unphysical; this case actually means
        # "two-phase with x=0 plus extra cooling", i.e. no valid state at T.
        # But for round-trip robustness we try the compressed-liquid branch
        # (p slightly above p_sat), which may have h slightly below h_L_sat
        # due to the p*v term sign -- unusual but possible for fluids with
        # negative thermal expansion.
        try:
            rho_init = rho_L_sat * 1.001
            rho = _isothermal_newton(T, h_target,
                                     props.enthalpy, _dh_drho_T, fluid,
                                     rho_init,
                                     rho_L_sat * 0.9999,    # just below sat for tiny h differences
                                     50.0 * fluid.rho_c)
            return _single_phase_result(rho, T, fluid, "liquid")
        except RuntimeError:
            # Fall back to two-phase with clipped quality
            return _two_phase_result(p_sat, T, rho_L_sat, rho_V_sat, 0.0, fluid)

    if h_target > h_V:
        # Superheated vapor (unambiguous)
        rho_init = rho_V_sat * 0.5
        rho = _isothermal_newton(T, h_target,
                                 props.enthalpy, _dh_drho_T, fluid,
                                 rho_init,
                                 1e-8 * fluid.rho_c,
                                 rho_V_sat)
        return _single_phase_result(rho, T, fluid, "vapor")

    # h_L_sat <= h <= h_V_sat: ambiguous.  Resolve by phase_hint.
    if phase_hint == "liquid":
        # Compressed liquid: p > p_sat, rho > rho_L_sat
        rho_init = rho_L_sat * 1.01
        rho = _isothermal_newton(T, h_target,
                                 props.enthalpy, _dh_drho_T, fluid,
                                 rho_init, rho_L_sat, 50.0 * fluid.rho_c)
        return _single_phase_result(rho, T, fluid, "liquid")
    if phase_hint == "vapor":
        rho_init = rho_V_sat * 0.5
        rho = _isothermal_newton(T, h_target,
                                 props.enthalpy, _dh_drho_T, fluid,
                                 rho_init,
                                 1e-8 * fluid.rho_c,
                                 rho_V_sat)
        return _single_phase_result(rho, T, fluid, "vapor")
    # Default: two-phase at p_sat with lever-rule quality
    x = (h_target - h_L) / (h_V - h_L)
    return _two_phase_result(p_sat, T, rho_L_sat, rho_V_sat, x, fluid)


def flash_ts(T, s_target, fluid, phase_hint=None, p_hint=None):
    """TS flash: find pressure (and phase) at given temperature and molar entropy.

    Same classification logic as ``flash_th``, applied to entropy.
    Entropy is generally better-behaved for phase classification than enthalpy:
    compressed liquid has s just below s_L_sat (decreasing with p), and
    superheated vapor has s above s_V_sat (decreasing with p from a high value
    at low p).  So the "unambiguous" regions are s < s_L (compressed liquid)
    and s > s_V (superheated vapor), and inside [s_L, s_V] is two-phase.
    """
    if T >= fluid.T_c:
        if p_hint is None:
            p_hint = max(fluid.p_triple, 1e5)
        rho_init = p_hint / (fluid.R * T)
        rho_min = 1e-8 * fluid.rho_c
        rho_max = 50.0 * fluid.rho_c
        rho = _isothermal_newton(T, s_target,
                                 props.entropy, _ds_drho_T, fluid,
                                 rho_init, rho_min, rho_max)
        return _single_phase_result(rho, T, fluid, "supercritical")

    rho_L_sat, rho_V_sat, p_sat = saturation_pT(T, fluid)
    s_L = props.entropy(rho_L_sat, T, fluid)
    s_V = props.entropy(rho_V_sat, T, fluid)

    if s_target < s_L:
        # Compressed liquid: (ds/dp)_T = -(dv/dT)_p < 0 typically, so s < s_L
        # corresponds to p > p_sat.  Newton in rho > rho_L_sat.
        rho_init = rho_L_sat * 1.01
        rho = _isothermal_newton(T, s_target,
                                 props.entropy, _ds_drho_T, fluid,
                                 rho_init, rho_L_sat, 50.0 * fluid.rho_c)
        return _single_phase_result(rho, T, fluid, "liquid")
    if s_target > s_V:
        # Superheated vapor at p < p_sat (s increases as p decreases)
        rho_init = rho_V_sat * 0.5
        rho = _isothermal_newton(T, s_target,
                                 props.entropy, _ds_drho_T, fluid,
                                 rho_init,
                                 1e-8 * fluid.rho_c,
                                 rho_V_sat)
        return _single_phase_result(rho, T, fluid, "vapor")
    # Two-phase
    x = (s_target - s_L) / (s_V - s_L)
    return _two_phase_result(p_sat, T, rho_L_sat, rho_V_sat, x, fluid)
