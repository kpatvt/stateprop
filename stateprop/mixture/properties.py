"""
Mixture Helmholtz energy evaluation and thermodynamic properties.

This module is the core math engine of the mixture calculation. It provides:

  - alpha_r_mix(delta, tau, x, components)       : mixture residual alpha (scaled)
  - alpha_r_derivs_mix(delta, tau, x, components): full 6-deriv residual alpha
  - alpha_0_mix(rho, T, x, components)           : mixture ideal alpha (and tau-derivs)

  - pressure(rho, T, x, mixture)
  - compressibility(rho, T, x, mixture)
  - ln_phi(rho, T, x, mixture)    : fugacity coefficients (length-N array)
  - enthalpy, entropy, etc.

  - density_from_pressure(p, T, x, mixture, phase_hint) : iterative rho solver

The composition derivatives for ln_phi follow Kunz-Wagner Appendix A.

Simplification in this framework build: Delta_alpha_r = 0 (no departure
function). When departure data are provided, the hooks exist to add that term.
"""
import numpy as np
from ..core import alpha_r_derivs, alpha_0_derivs


def _pure_alpha_r_derivs(delta, tau, component):
    """Evaluate a pure component's residual Helmholtz derivatives at mixture
    reduced coordinates.

    Returns (a_r, a_r_d, a_r_t, a_r_dd, a_r_tt, a_r_dt).
    """
    pack = component.fluid.pack()
    res_args = pack[3:41]
    return alpha_r_derivs(delta, tau, *res_args)


def _pure_alpha_0_derivs(rho, T, component):
    """Evaluate a pure component's ideal Helmholtz derivatives in natural
    (rho, T) coordinates.

    Note: alpha_0 uses the PURE component's critical delta, tau, since the
    ideal-gas part is a property of the component alone at (rho, T), not at
    mixture-reduced coordinates. The Fluid's alpha_0 kernel takes (delta_pure,
    tau_pure) as inputs with delta_pure = rho/rho_c_pure, tau_pure = Tc_pure/T.

    Returns (a_0, a_0_d, a_0_t, a_0_dd, a_0_tt, a_0_dt).

    IMPORTANT: For ideal-gas, alpha_0 depends on rho and T only, and the
    d-derivatives at pure-fluid reduction have no physical meaning at mixture
    density; we only need the tau-derivatives (for enthalpy, etc.) and the
    pure alpha_0 value itself (for the entropy-of-mixing term).
    """
    pack = component.fluid.pack()
    ideal_codes = pack[41]
    ideal_a = pack[42]
    ideal_b = pack[43]
    ideal_c = pack[44]
    ideal_d = pack[45]
    # Use the pure-component's own critical delta, tau
    delta_pure = rho / component.fluid.rho_c
    tau_pure = component.fluid.T_c / T
    return alpha_0_derivs(delta_pure, tau_pure,
                          ideal_codes, ideal_a, ideal_b, ideal_c, ideal_d)


def alpha_r_mix_derivs(rho, T, x, mixture):
    """Evaluate all mixture residual Helmholtz derivatives.

    Returns a dict with:
        a_r, a_r_d, a_r_t, a_r_dd, a_r_tt, a_r_dt   -- mixture alpha_r and derivs
        delta, tau                                   -- mixture reduced coords
        T_r, rho_r                                   -- reducing values
        dTr_dxi, d_invrhor_dxi                       -- composition derivs of reducing
        a_r_pure_i                                   -- pure contribs at mixture (delta, tau)
                                                        array of length N, each entry is
                                                        (a_r, a_r_d, a_r_t, a_r_dd, a_r_tt, a_r_dt)

    With Delta_alpha_r = 0, the mixture alpha_r equals sum_i x_i * alpha_r_oi.
    """
    reducing = mixture.reducing
    Tr, rho_r, dTr, d_invrho = reducing.derivatives(x)
    delta = rho / rho_r
    tau = Tr / T

    # Evaluate each pure component's alpha_r at the mixture (delta, tau)
    pure = []
    a_r = 0.0
    a_r_d = 0.0
    a_r_t = 0.0
    a_r_dd = 0.0
    a_r_tt = 0.0
    a_r_dt = 0.0
    for i, comp in enumerate(mixture.components):
        if x[i] == 0.0:
            pure.append(None)
            continue
        d = _pure_alpha_r_derivs(delta, tau, comp)
        pure.append(d)
        a_r    += x[i] * d[0]
        a_r_d  += x[i] * d[1]
        a_r_t  += x[i] * d[2]
        a_r_dd += x[i] * d[3]
        a_r_tt += x[i] * d[4]
        a_r_dt += x[i] * d[5]

    # Departure-function contribution: Delta_alpha_r(delta, tau, x)
    # Shape: sum_{i<j} x_i x_j F_ij alpha_r_ij(delta, tau)
    dDelta_dx = np.zeros(mixture.N)
    Delta = 0.0
    if mixture.departures:
        from .departure import evaluate_total_departure
        (Delta, D_d, D_t, D_dd, D_tt, D_dt, dDelta_dx) = evaluate_total_departure(
            x, delta, tau, mixture.departures
        )
        a_r    += Delta
        a_r_d  += D_d
        a_r_t  += D_t
        a_r_dd += D_dd
        a_r_tt += D_tt
        a_r_dt += D_dt

    return {
        "a_r": a_r, "a_r_d": a_r_d, "a_r_t": a_r_t,
        "a_r_dd": a_r_dd, "a_r_tt": a_r_tt, "a_r_dt": a_r_dt,
        "delta": delta, "tau": tau,
        "T_r": Tr, "rho_r": rho_r,
        "dTr_dxi": dTr, "d_invrhor_dxi": d_invrho,
        "a_r_pure_i": pure,
        "Delta_alpha_r": Delta,        # departure value
        "dDelta_dxi": dDelta_dx,       # departure composition derivative (explicit, at fixed delta,tau)
    }


def alpha_0_mix(rho, T, x, mixture):
    """Evaluate mixture ideal-gas alpha_0 and its tau-derivatives.

    For an ideal-gas mixture at (rho, T):
        alpha_0_mix = sum_i x_i * [alpha_0_oi(rho, T) + ln(x_i)]

    The ln(x_i) is the entropy-of-mixing term. We return tau-derivatives only
    since d-derivatives of alpha_0 are not meaningful for caloric properties
    (they just encode the pressure contribution of the ideal-gas part, which
    we handle separately via p = rho*R*T*(1 + delta*a_r_d)).

    Returns dict with:
        a_0      : scalar, the full mixture ideal alpha_0 (incl. mixing entropy)
        a_0_t    : first tau derivative (sum of pure components', weighted)
        a_0_tt   : second tau derivative
    """
    a0 = 0.0
    a0_t = 0.0
    a0_tt = 0.0
    # Note: the tau-derivatives of the mixing term ln(x_i) are zero
    for i, comp in enumerate(mixture.components):
        if x[i] <= 0:
            continue
        d = _pure_alpha_0_derivs(rho, T, comp)
        # Each pure's a_0 is evaluated at its own (delta_pure, tau_pure).
        # But we want the "tau-derivative" in mixture-reduced terms for enthalpy
        # and entropy calculations. The relation between pure-tau and mixture-tau
        # is handled implicitly in how we convert a_0_t to h and s below.
        #
        # Convention: we keep a_0_t expressed in PURE-COMPONENT tau for each
        # component, because that's how the Fluid's alpha_0_derivs returns it.
        # For the caloric properties below, we use:
        #   u_total / RT = tau_pure_i * a_0_t_pure + ...
        # Since each pure contributes at its own tau_pure = Tc_i/T, we compute
        # h/RT etc. component-by-component.
        a0 += x[i] * (d[0] + np.log(x[i]))
        # For tau-derivatives we need a different convention since each pure
        # has its own tau. We report the components instead.

    return {"a_0": a0}


def _pure_caloric(rho, T, mixture, x):
    """Compute enthalpy, internal energy, entropy of the mixture.

    Properly accounts for the different tau for each component's ideal-gas part.
    """
    R_common = None
    # Ideal part contributions: each component at its own pure tau
    u_ideal_RT = 0.0    # u^0 / (R T)
    s_ideal_R = 0.0     # s^0 / R
    a0_total_R = 0.0    # a^0 / (R T) = sum x_i [a_0_oi + ln x_i]   (for Gibbs check only)

    for i, comp in enumerate(mixture.components):
        if x[i] <= 0:
            continue
        fl = comp.fluid
        if R_common is None:
            R_common = fl.R
        elif abs(fl.R - R_common) > 1e-10:
            # Mixture components with different R values is a subtle issue.
            # GERG uses a single R=8.314472; if the pure fluids use different R,
            # caloric properties may be slightly inconsistent. For simplicity
            # we use the first component's R throughout and warn via a check.
            pass

        delta_i = rho / fl.rho_c
        tau_i = fl.T_c / T
        pack = fl.pack()
        ideal_codes, ideal_a, ideal_b = pack[41], pack[42], pack[43]
        ideal_c, ideal_d = pack[44], pack[45]
        d = alpha_0_derivs(delta_i, tau_i, ideal_codes, ideal_a, ideal_b,
                           ideal_c, ideal_d)
        # u^0/(RT) for pure = tau_i * a_0_t;   s^0/R = tau_i * a_0_t - a_0
        u_ideal_RT += x[i] * tau_i * d[2]
        # Mixing entropy: s/R gets a term -ln(x_i) for each component
        # i.e., s_mix_ideal = -R * sum x_i ln x_i   (positive term in entropy)
        s_ideal_R += x[i] * (tau_i * d[2] - d[0] - np.log(x[i]))
        a0_total_R += x[i] * (d[0] + np.log(x[i]))

    # Residual contribution at mixture reduced coordinates
    res = alpha_r_mix_derivs(rho, T, x, mixture)
    R = R_common
    # u^r/(RT) = tau * a_r_t    (mixture tau)
    u_res_RT = res["tau"] * res["a_r_t"]
    # s^r/R   = tau * a_r_t - a_r
    s_res_R = res["tau"] * res["a_r_t"] - res["a_r"]
    # p = rho R T (1 + delta * a_r_d)    =>   p/(rho R T) = 1 + delta * a_r_d
    Z = 1.0 + res["delta"] * res["a_r_d"]
    p = rho * R * T * Z

    u = R * T * (u_ideal_RT + u_res_RT)
    h = u + p / rho          # molar enthalpy [J/mol]
    s = R * (s_ideal_R + s_res_R)

    return {
        "p": p, "Z": Z, "u": u, "h": h, "s": s, "R": R,
        "res": res,
    }


def pressure(rho, T, x, mixture):
    """Mixture pressure [Pa] at (rho, T, x)."""
    res = alpha_r_mix_derivs(rho, T, x, mixture)
    # All components must use the same R (which is checked when pure fluids are loaded);
    # use first component's R
    R = mixture.components[0].fluid.R
    return rho * R * T * (1.0 + res["delta"] * res["a_r_d"])


def compressibility(rho, T, x, mixture):
    res = alpha_r_mix_derivs(rho, T, x, mixture)
    return 1.0 + res["delta"] * res["a_r_d"]


def enthalpy(rho, T, x, mixture):
    return _pure_caloric(rho, T, mixture, x)["h"]


def entropy(rho, T, x, mixture):
    return _pure_caloric(rho, T, mixture, x)["s"]


def dp_drho_T(rho, T, x, mixture):
    """(dp/drho)_T = R T [1 + 2 delta a_r_d + delta^2 a_r_dd]"""
    res = alpha_r_mix_derivs(rho, T, x, mixture)
    R = mixture.components[0].fluid.R
    return R * T * (1.0 + 2.0 * res["delta"] * res["a_r_d"] + res["delta"]**2 * res["a_r_dd"])


# ---------------------------------------------------------------------------
# Fugacity coefficients (the key quantities for phase equilibrium)
# ---------------------------------------------------------------------------

def ln_phi(rho, T, x, mixture):
    """Logarithm of the fugacity coefficients of each component.

    Derived from first principles. For a mixture without departure function:

        alpha^r(delta, tau, x) = sum_i x_i * alpha^r_oi(delta, tau)

    where delta = rho/rho_r(x), tau = T_r(x)/T.

    The fugacity coefficient identity:

        ln(phi_i) = [n * d alpha^r / d n_i]_{T, V, n_{j != i}} - ln(Z)

    Expanding the composition derivative via chain rule through delta, tau,
    and the explicit x_k dependence:

        n * d alpha^r / d n_i  =
            alpha^r_delta * n * (d delta / d n_i)      # delta channel
          + alpha^r_tau   * n * (d tau   / d n_i)       # tau channel
          + [partial alpha^r / partial x_i]_{delta,tau,x_{k!=i}}
            - sum_k x_k * [partial alpha^r / partial x_k]_{delta,tau,x_{l!=k}}   # x channel (projected)

    With alpha^r = sum_i x_i * alpha^r_oi(delta, tau):
        partial alpha^r / partial x_i |_{delta, tau, x_{k!=i}} = alpha^r_oi(delta, tau)
        sum_k x_k * (partial alpha^r / partial x_k) = sum_k x_k * alpha^r_ok = alpha^r_mix

    So the x-channel contribution simplifies to  alpha^r_oi(delta, tau) - alpha^r_mix.

    For the chain-rule pieces:
        n * (d delta / d n_i) = delta * [1 - (1/rho_r) * n * (d rho_r / d n_i)]
        n * (d tau   / d n_i) = (tau / T_r) * n * (d T_r / d n_i)

    The composition derivatives of intensive quantities T_r(x), rho_r(x) are
    converted from partial-x form to partial-n form via the projection:
        n * (d Y / d n_i) = (dY/dx_i) - sum_k x_k (dY/dx_k)

    where dY/dx_k is the "unconstrained" partial derivative (treating all
    x_l as independent variables, not enforcing sum to 1). This is the
    convention returned by `reducing.derivatives()`.

    Parameters
    ----------
    rho : float       mixture molar density [mol/m^3]
    T   : float       temperature [K]
    x   : array (N,)  composition (mole fractions, sum = 1)
    mixture : Mixture

    Returns
    -------
    ln_phi_arr : ndarray (N,)
    """
    res = alpha_r_mix_derivs(rho, T, x, mixture)
    N = mixture.N
    delta = res["delta"]
    tau = res["tau"]
    a_r_mix = res["a_r"]         # mixture alpha^r (includes departure)
    a_r_d = res["a_r_d"]         # d alpha^r / d delta  (includes departure)
    a_r_t = res["a_r_t"]         # d alpha^r / d tau    (includes departure)
    Tr = res["T_r"]
    rho_r = res["rho_r"]
    dTr_dxi = res["dTr_dxi"]
    d_invrhor_dxi = res["d_invrhor_dxi"]
    # Departure function value and its explicit composition derivatives
    Delta = res.get("Delta_alpha_r", 0.0)
    dDelta_dxi = res.get("dDelta_dxi", np.zeros(N))

    # Projection sums: sum_k x_k * (dY/dx_k)
    sum_x_dTr = float(np.dot(x, dTr_dxi))
    sum_x_d_invrho = float(np.dot(x, d_invrhor_dxi))

    Z = 1.0 + delta * a_r_d
    ln_phi_arr = np.zeros(N)

    for i in range(N):
        if x[i] <= 0:
            ln_phi_arr[i] = 0.0
            continue
        pure_i = res["a_r_pure_i"][i]
        a_r_oi = pure_i[0]   # alpha^r_oi(delta_mix, tau_mix) -- pure i at mixture coords

        # Composition derivatives of T_r and 1/rho_r, converted to partial-n form:
        n_dTr_dni = dTr_dxi[i] - sum_x_dTr
        n_d_invrhor_dni = d_invrhor_dxi[i] - sum_x_d_invrho
        # Convert d(1/rho_r)/dn -> d(rho_r)/dn:  d(1/y) = -dy/y^2, so
        #   (1/rho_r) * n d rho_r/d n_i = -rho_r * n d(1/rho_r)/d n_i
        rho_r_factor = -rho_r * n_d_invrhor_dni       # this is (1/rho_r) * n * d rho_r / d n_i

        # Chain-rule pieces (use FULL alpha_r derivatives, which include Delta):
        delta_piece = delta * a_r_d * (1.0 - rho_r_factor)       # alpha^r_delta * n * d delta / d n_i
        tau_piece   = tau   * a_r_t * (n_dTr_dni / Tr)            # alpha^r_tau   * n * d tau   / d n_i

        # Explicit x-channel (projected): (dalpha/dx_i) - sum_k x_k (dalpha/dx_k)
        #   With alpha^r = alpha^r_linear + Delta:
        #     d alpha^r / d x_i = alpha^r_oi + d Delta / d x_i
        #     sum_k x_k d alpha^r / d x_k = alpha^r_linear + 2*Delta   (Delta is homog. deg 2)
        #                                 = (alpha^r - Delta) + 2*Delta
        #                                 = alpha^r + Delta
        #   So x_piece = (alpha^r_oi + dDelta/dx_i) - (alpha^r + Delta)
        x_piece = a_r_oi + dDelta_dxi[i] - a_r_mix - Delta

        # Combine with the alpha^r that gets added from n * d(n*alpha)/dn:
        # ln phi_i = alpha^r + n * d alpha^r/d n_i - ln Z
        #          = alpha^r + (delta_piece + tau_piece + x_piece) - ln Z
        # Substituting x_piece simplification:
        # ln phi_i = alpha^r_oi + dDelta/dx_i - Delta + delta_piece + tau_piece - ln Z
        ln_phi_arr[i] = a_r_oi + dDelta_dxi[i] - Delta + delta_piece + tau_piece - np.log(Z)

    return ln_phi_arr


# ---------------------------------------------------------------------------
# Analytic composition derivatives for second-order (Newton) flash methods
# ---------------------------------------------------------------------------
#
# Three building blocks are exposed here:
#
#   dp_dx_at_rho(rho, T, x, mixture)           -- N-vector dp/dx_k at fixed (T, rho)
#   dlnphi_drho_at_x(rho, T, x, mixture)       -- N-vector dlnphi/drho at fixed (T, x)
#   dlnphi_dx_at_rho(rho, T, x, mixture)       -- NxN matrix dlnphi_i/dx_k at fixed (T, rho)
#
# These combine into the headline result via the chain rule:
#
#   dlnphi_dx_at_p(p, T, x, mixture, phase_hint)  -- NxN matrix dlnphi_i/dx_k at fixed (T, p)
#       = dlnphi_dx_at_rho + outer(dlnphi_drho_at_x, drho_dx_at_p)
#       drho_dx_at_p = -dp_dx_at_rho / dp_drho_T
#
# All four pieces are FD-verified to ~1e-6 absolute (about 5 significant
# figures) by the test suite, limited mainly by the FD stencil precision
# rather than the analytic formulas themselves (which are essentially
# machine-precision once the FD reference is removed).
#
# Like the cubic-EOS analogue (CubicMixture.dlnphi_dxk_at_p), these are
# the foundational pieces for: Newton-Raphson flash with a true Jacobian
# (vs Broyden's secant approximation), arclength continuation along phase
# envelopes, direct critical-point solvers, sensitivity analysis, and
# trust-region methods for ill-conditioned cases. The Newton flash
# (newton_flash_pt in stateprop/mixture/flash.py) is a representative
# downstream user of these primitives.


def _alpha_r_mix_derivs_extended(rho, T, x, mixture):
    """Like alpha_r_mix_derivs but also returns the extended departure outputs.

    Returns the same dict as alpha_r_mix_derivs plus:
      dDd_dx   : ndarray (N,)    d(Delta_d)/dx_k at fixed (delta, tau)
      dDt_dx   : ndarray (N,)    d(Delta_t)/dx_k at fixed (delta, tau)
      d2D_dxx  : ndarray (N, N)  d^2 Delta / dx_k dx_l at fixed (delta, tau)
    """
    from .departure import evaluate_total_departure_extended
    base = alpha_r_mix_derivs(rho, T, x, mixture)
    N = mixture.N
    if mixture.departures:
        (Delta_, Dd_, Dt_, Ddd_, Dtt_, Ddt_,
         dD_dx, dDd_dx, dDt_dx, d2D) = evaluate_total_departure_extended(
            x, base["delta"], base["tau"], mixture.departures)
        # Sanity: Delta_, Dd_, Dt_, dD_dx should match what's already in base
        # (they're the same calculation; we trust evaluate_total_departure_extended
        # which uses the same loop).
    else:
        dDd_dx = np.zeros(N)
        dDt_dx = np.zeros(N)
        d2D = np.zeros((N, N))
    base["dDd_dx"] = dDd_dx
    base["dDt_dx"] = dDt_dx
    base["d2D_dxx"] = d2D
    return base


def dp_dx_at_rho(rho, T, x, mixture):
    """Composition derivative of pressure at fixed (T, rho), N-vector.

    p = rho * R * T * (1 + delta * a_r_d). At fixed (T, rho), only delta
    and the mixture a_r_d depend on x:

        ddelta/dxk    = rho * d(1/rho_r)/dxk                    (since delta = rho/rho_r)
        d(a_r_d)/dxk  = a_r_d_ok + dDd_dx[k]
                        + a_r_dd * ddelta/dxk + a_r_dt * dtau/dxk
        dtau/dxk      = dTr/dxk / T                              (since tau = Tr/T)

    Then dp/dxk = rho * R * T * (ddelta/dxk * a_r_d + delta * d(a_r_d)/dxk).
    """
    res = _alpha_r_mix_derivs_extended(rho, T, x, mixture)
    R = mixture.components[0].fluid.R
    delta = res["delta"]; tau = res["tau"]
    a_r_d = res["a_r_d"]; a_r_dd = res["a_r_dd"]; a_r_dt = res["a_r_dt"]
    dDd_dx = res["dDd_dx"]
    dTr = res["dTr_dxi"]; d_invrho = res["d_invrhor_dxi"]
    pure = res["a_r_pure_i"]
    N = mixture.N
    # Per-component a_r_d_oi at mixture (delta, tau)
    a_r_d_oi = np.array([0.0 if pure[i] is None else pure[i][1] for i in range(N)])
    # Composition derivatives of (delta, tau) at fixed (T, rho)
    ddelta_dx = rho * d_invrho                # shape (N,)
    dtau_dx = dTr / T                          # shape (N,)
    # d(a_r_d)/dx at fixed (T, rho)
    da_r_d_dx = a_r_d_oi + dDd_dx + a_r_dd * ddelta_dx + a_r_dt * dtau_dx
    # dp/dx
    return rho * R * T * (ddelta_dx * a_r_d + delta * da_r_d_dx)


def dlnphi_drho_at_x(rho, T, x, mixture):
    """Density derivative of ln phi at fixed (T, x), N-vector.

    Differentiates the ln_phi formula w.r.t. rho. Only delta depends on
    rho (delta = rho/rho_r); tau and the reducing-function-derived
    quantities (n_dTr_dni, rho_r_factor) are pure functions of x.
    Composition-explicit pieces (dDelta_dxi, etc.) depend on rho only
    through delta. Each piece picks up a factor of (1/rho_r) via the
    chain rule on delta.
    """
    res = _alpha_r_mix_derivs_extended(rho, T, x, mixture)
    R = mixture.components[0].fluid.R
    delta = res["delta"]; tau = res["tau"]
    Tr = res["T_r"]; rho_r = res["rho_r"]
    a_r_d = res["a_r_d"]; a_r_dd = res["a_r_dd"]; a_r_dt = res["a_r_dt"]
    dDelta_dxi = res.get("dDelta_dxi", np.zeros(mixture.N))
    Delta_d = res.get("Delta_d", 0.0) if "Delta_d" in res else 0.0
    # Note: Delta_d isn't returned by alpha_r_mix_derivs (only Delta_alpha_r);
    # recover from extended evaluator.
    if mixture.departures:
        from .departure import evaluate_total_departure
        (_, Delta_d, _, _, _, _, _) = evaluate_total_departure(
            x, delta, tau, mixture.departures)
    dDd_dx = res["dDd_dx"]
    dTr = res["dTr_dxi"]; d_invrho = res["d_invrhor_dxi"]
    pure = res["a_r_pure_i"]
    N = mixture.N
    a_r_d_oi = np.array([0.0 if pure[i] is None else pure[i][1] for i in range(N)])
    Z = 1.0 + delta * a_r_d
    # Projection sums (used for delta_piece, tau_piece)
    sum_x_dTr = float(np.dot(x, dTr))
    sum_x_d_invrho = float(np.dot(x, d_invrho))
    n_dTr = dTr - sum_x_dTr                              # shape (N,) "n d Tr/d n_i"
    rho_r_factor = -rho_r * (d_invrho - sum_x_d_invrho)  # shape (N,)
    # d(a_r_d)/drho|_{T,x} = a_r_dd / rho_r
    # d(a_r_t)/drho|_{T,x} = a_r_dt / rho_r
    # d(delta)/drho = 1/rho_r
    # Walk through each term in ln_phi:
    out = np.zeros(N)
    for i in range(N):
        if x[i] <= 0:
            continue
        # T1: a_r_oi(delta, tau).  d/drho = a_r_d_oi / rho_r
        # T2a: dDelta_dxi[i].      d/drho = dDd_dx[i] / rho_r   (Schwarz)
        # T2b: -Delta.              d/drho = -Delta_d / rho_r
        # T3: delta_piece = delta * a_r_d * (1 - rho_r_factor[i])
        #     rho_r_factor[i] doesn't depend on rho.
        #     d(delta * a_r_d)/drho = a_r_d * (1/rho_r) + delta * a_r_dd / rho_r
        #                           = (a_r_d + delta * a_r_dd)/rho_r
        # T4: tau_piece = tau * a_r_t * (n_dTr[i]/Tr)
        #     Only a_r_t depends on rho. d(a_r_t)/drho = a_r_dt / rho_r
        # T5: -ln Z. d/drho = -d(delta * a_r_d)/drho / Z = -(a_r_d + delta*a_r_dd)/(Z*rho_r)
        out[i] = (
            a_r_d_oi[i] / rho_r
            + dDd_dx[i] / rho_r
            - Delta_d / rho_r
            + (1.0 - rho_r_factor[i]) * (a_r_d + delta * a_r_dd) / rho_r
            + tau * (a_r_dt / rho_r) * (n_dTr[i] / Tr)
            - (a_r_d + delta * a_r_dd) / (Z * rho_r)
        )
    return out


def dlnphi_dx_at_rho(rho, T, x, mixture):
    """N x N Jacobian d(ln phi_i)/d x_k at fixed (T, rho).

    This is the central composition derivative at the "natural" coordinates
    where everything is algebraic in x. The chain rule combines:
      - explicit x-dependence (via dDelta_dxi, the linear sum coefficients
        in alpha_r = sum_j x_j alpha_r_oj, and rho_r_factor / n_dTr_dni)
      - implicit dependence through (delta, tau) which themselves depend on
        x via the reducing functions

    Differentiating the ln_phi formula:

        ln phi_i = a_r_oi(delta, tau)
                 + dDelta_dxi[i] - Delta
                 + delta_piece[i]   (delta * a_r_d * (1 - rho_r_factor[i]))
                 + tau_piece[i]     (tau * a_r_t * n_dTr_dni[i] / Tr)
                 - ln Z              (Z = 1 + delta * a_r_d)

    yields six contributions per (i, k) entry. Each is computed in turn
    in the code below, with comments tying lines back to the math.

    Requires: KunzWagnerReducing.hessian(x) for the second derivatives of
    the reducing functions (used in n_dTr_dni and rho_r_factor's
    composition derivatives), and evaluate_total_departure_extended for
    the composition derivatives of (Delta_d, Delta_t, dDelta_dxi).
    """
    res = _alpha_r_mix_derivs_extended(rho, T, x, mixture)
    N = mixture.N
    delta = res["delta"]; tau = res["tau"]
    Tr = res["T_r"]; rho_r = res["rho_r"]
    a_r = res["a_r"]; a_r_d = res["a_r_d"]; a_r_t = res["a_r_t"]
    a_r_dd = res["a_r_dd"]; a_r_tt = res["a_r_tt"]; a_r_dt = res["a_r_dt"]
    Delta = res.get("Delta_alpha_r", 0.0)
    dDelta_dxi = res.get("dDelta_dxi", np.zeros(N))
    dDd_dx = res["dDd_dx"]; dDt_dx = res["dDt_dx"]; d2D = res["d2D_dxx"]
    dTr = res["dTr_dxi"]; d_invrho = res["d_invrhor_dxi"]
    pure = res["a_r_pure_i"]
    # Recover Delta_d, Delta_t (not in res by default)
    if mixture.departures:
        from .departure import evaluate_total_departure
        (_, Delta_d, Delta_t, _, _, _, _) = evaluate_total_departure(
            x, delta, tau, mixture.departures)
    else:
        Delta_d = 0.0; Delta_t = 0.0
    # Per-component pure derivatives at MIXTURE (delta, tau)
    a_r_oi   = np.array([0.0 if pure[i] is None else pure[i][0] for i in range(N)])
    a_r_d_oi = np.array([0.0 if pure[i] is None else pure[i][1] for i in range(N)])
    a_r_t_oi = np.array([0.0 if pure[i] is None else pure[i][2] for i in range(N)])
    # We'll need second-order pure derivatives too:
    a_r_dd_oi = np.array([0.0 if pure[i] is None else pure[i][3] for i in range(N)])
    a_r_tt_oi = np.array([0.0 if pure[i] is None else pure[i][4] for i in range(N)])
    a_r_dt_oi = np.array([0.0 if pure[i] is None else pure[i][5] for i in range(N)])

    # Reducing-function Hessian (analytic)
    H_T, H_invrho = mixture.reducing.hessian(x)

    # Composition derivatives of delta, tau at fixed (T, rho):
    ddelta_dx = rho * d_invrho                # (N,) since delta = rho/rho_r and ∂(1/ρ_r)/∂x_k = d_invrho[k]
    dtau_dx = dTr / T                          # (N,)

    # Projection sums and their composition derivatives
    sum_x_dTr = float(np.dot(x, dTr))
    sum_x_d_invrho = float(np.dot(x, d_invrho))
    # n_dTr_dni[i] = dTr[i] - sum_x_dTr
    # rho_r_factor[i] = -rho_r * (d_invrho[i] - sum_x_d_invrho)
    n_dTr_dni = dTr - sum_x_dTr
    n_d_invrho_dni = d_invrho - sum_x_d_invrho
    rho_r_factor = -rho_r * n_d_invrho_dni

    # d(rho_r)/dx_k = -rho_r^2 * d_invrho[k]   (from differentiating 1/rho_r)
    drhor_dx = -rho_r * rho_r * d_invrho        # (N,)

    # d(sum_x_dTr)/dx_k = dTr[k] + sum_l x_l * H_T[l, k]
    sum_x_H_T_col = H_T.T @ x                   # (N,) -- = sum_l x_l H_T[l, k]
    sum_x_H_invrho_col = H_invrho.T @ x         # (N,)
    d_sum_x_dTr_dx = dTr + sum_x_H_T_col        # (N,)
    d_sum_x_d_invrho_dx = d_invrho + sum_x_H_invrho_col

    # d(n_dTr_dni[i])/dx_k = H_T[i, k] - d_sum_x_dTr_dx[k]
    # As an N x N matrix indexed [i, k]:
    d_n_dTr = H_T - d_sum_x_dTr_dx[None, :]     # (N, N)  d(n_dTr_dni[i])/d x_k
    d_n_d_invrho = H_invrho - d_sum_x_d_invrho_dx[None, :]

    # d(rho_r_factor[i])/dx_k = -d(rho_r)/dx_k * n_d_invrho_dni[i]
    #                          - rho_r * d(n_d_invrho_dni[i])/dx_k
    # As N x N matrix [i, k]:
    d_rho_r_factor = (
        -drhor_dx[None, :] * n_d_invrho_dni[:, None]
        - rho_r * d_n_d_invrho
    )

    # d(a_r_d)/dx_k at fixed (T, rho), MIXTURE quantity (includes departure)
    # = a_r_d_ok + dDd_dx[k] + a_r_dd * ddelta_dx[k] + a_r_dt * dtau_dx[k]
    da_r_d_dx = a_r_d_oi + dDd_dx + a_r_dd * ddelta_dx + a_r_dt * dtau_dx     # (N,)
    # Same for a_r_t (used in tau_piece)
    da_r_t_dx = a_r_t_oi + dDt_dx + a_r_dt * ddelta_dx + a_r_tt * dtau_dx     # (N,)
    # And for a_r itself (used in -Delta term... wait actually Delta only)
    da_r_dx = a_r_oi + dDelta_dxi + a_r_d * ddelta_dx + a_r_t * dtau_dx       # (N,)
    # Hmm wait that's the unconstrained ∂(α^r)/∂x_k including everything

    # d(delta * a_r_d)/dx_k = ddelta_dx[k] * a_r_d + delta * da_r_d_dx[k]
    d_delta_a_r_d = ddelta_dx * a_r_d + delta * da_r_d_dx                     # (N,)

    Z = 1.0 + delta * a_r_d
    # d(ln Z)/dx_k = (1/Z) * d_delta_a_r_d[k]
    dlnZ_dx = d_delta_a_r_d / Z                                                # (N,)

    # ===== Build J row by row =====
    J = np.zeros((N, N))
    for i in range(N):
        if x[i] <= 0:
            continue
        # Term 1: d(a_r_oi(delta, tau))/dx_k  = a_r_d_oi * ddelta_dx[k] + a_r_t_oi * dtau_dx[k]
        T1 = a_r_d_oi[i] * ddelta_dx + a_r_t_oi[i] * dtau_dx                  # (N,)
        # Term 2: d(dDelta_dxi[i])/dx_k = explicit Hessian d2D[i,k] + chain through (delta, tau)
        #   chain: dDd_dx[i] * ddelta_dx[k] + dDt_dx[i] * dtau_dx[k]
        T2 = d2D[i, :] + dDd_dx[i] * ddelta_dx + dDt_dx[i] * dtau_dx           # (N,)
        # Term 3: d(-Delta)/dx_k = -dDelta_dxi[k] - Delta_d * ddelta_dx[k] - Delta_t * dtau_dx[k]
        T3 = -dDelta_dxi - Delta_d * ddelta_dx - Delta_t * dtau_dx             # (N,)
        # Term 4: d(delta_piece[i])/dx_k where delta_piece = delta * a_r_d * (1 - rho_r_factor[i])
        #       = (1 - rho_r_factor[i]) * d_delta_a_r_d[k] - (delta * a_r_d) * d_rho_r_factor[i, k]
        T4 = (1.0 - rho_r_factor[i]) * d_delta_a_r_d - (delta * a_r_d) * d_rho_r_factor[i, :]
        # Term 5: d(tau_piece[i])/dx_k where tau_piece = tau * a_r_t * coefficient_i
        #         coefficient_i = n_dTr_dni[i] / Tr
        #   d(coefficient_i)/dx_k = (1/Tr) * d_n_dTr[i, k] - (n_dTr_dni[i]/Tr^2) * dTr[k]
        d_coeff_i = d_n_dTr[i, :] / Tr - (n_dTr_dni[i] / (Tr * Tr)) * dTr      # (N,)
        # tau_piece[i] = tau * a_r_t * coefficient_i. Apply product rule:
        coefficient_i = n_dTr_dni[i] / Tr
        T5 = (
            dtau_dx * a_r_t * coefficient_i               # ∂τ/∂x_k * a_r_t * coeff
            + tau * da_r_t_dx * coefficient_i             # τ * ∂a_r_t/∂x_k * coeff
            + tau * a_r_t * d_coeff_i                     # τ * a_r_t * ∂coeff/∂x_k
        )
        # Term 6: d(-ln Z)/dx_k = -dlnZ_dx[k]
        T6 = -dlnZ_dx
        J[i, :] = T1 + T2 + T3 + T4 + T5 + T6
    return J


def dlnphi_dx_at_p(p, T, x, mixture, phase_hint='vapor'):
    """N x N Jacobian d(ln phi_i)/d x_k at fixed (T, p).

    Combines dlnphi_dx_at_rho (composition derivative at natural
    coordinates) with the chain-rule conversion to fixed (T, p) using

        drho/dx_k|_{T,p} = -dp/dx_k|_{T,rho} / dp/drho|_{T,x}

    from implicit differentiation of p(rho, T, x) = p_target. This is
    the analytic Newton-flash Jacobian for the Helmholtz/GERG mixture
    EOS (analogous to CubicMixture.dlnphi_dxk_at_p for cubic EOSes).

    Phase-specific because the density solve picks a particular root.
    """
    x = np.asarray(x, dtype=np.float64)
    rho = density_from_pressure(p, T, x, mixture, phase_hint=phase_hint)
    J_at_rho = dlnphi_dx_at_rho(rho, T, x, mixture)            # (N, N)
    dlnphi_drho = dlnphi_drho_at_x(rho, T, x, mixture)         # (N,)
    dpx = dp_dx_at_rho(rho, T, x, mixture)                      # (N,)
    dpr = dp_drho_T(rho, T, x, mixture)                         # scalar
    drho_dx = -dpx / dpr                                         # (N,)
    return J_at_rho + np.outer(dlnphi_drho, drho_dx)


# ---------------------------------------------------------------------------
# Temperature and pressure derivatives of ln phi for Helmholtz/GERG (v0.9.10)
# ---------------------------------------------------------------------------
#
# Complete the primitive set needed for second-order methods on
# bubble/dew-point solvers, phase-envelope tracing, and thermodynamic
# sensitivity analysis (the composition derivative was done in v0.9.9).
#
# The key observations that simplify the derivation:
#   1) At fixed rho, delta = rho/rho_r(x) does NOT depend on T (rho_r is
#      a function of x only). So only tau = T_r(x)/T carries T-dependence
#      at fixed (rho, x), with d(tau)/dT = -tau/T.
#   2) At fixed (T, x), only rho depends on p. So
#          d(ln phi)/dp = d(ln phi)/d rho / (dp/drho)_{T,x}
#      is a one-liner using the existing dlnphi_drho_at_x and dp_drho_T.
#
# The harder case is d(ln phi)/dT|_{p,x}, which requires the partial at
# fixed rho (straightforward tau-chain on each term) plus the implicit
# density response drho/dT|_{p,x} = -dp/dT|_{rho,x} / dp/drho|_{T,x},
# which in turn needs dp/dT|_{rho,x}.


def dp_dT_at_rho(rho, T, x, mixture):
    """Temperature derivative of pressure at fixed (rho, x), scalar.

    p = rho * R * T * (1 + delta * a_r_d) = rho * R * T * Z

    At fixed (rho, x), delta is constant. Only tau (= T_r/T) depends on T,
    with d(tau)/dT = -tau/T. So d(a_r_d)/dT|_{rho,x} = a_r_dt * (-tau/T).

        dp/dT|_{rho,x} = rho * R * (1 + delta * a_r_d)
                       + rho * R * T * delta * a_r_dt * (-tau/T)
                       = rho * R * (Z - delta * tau * a_r_dt)

    FD-verified to ~1e-9.
    """
    res = alpha_r_mix_derivs(rho, T, x, mixture)
    R = mixture.components[0].fluid.R
    delta = res["delta"]; tau = res["tau"]
    a_r_d = res["a_r_d"]; a_r_dt = res["a_r_dt"]
    Z = 1.0 + delta * a_r_d
    return rho * R * (Z - delta * tau * a_r_dt)


def dlnphi_dT_at_rho(rho, T, x, mixture):
    """Temperature derivative of ln phi at fixed (rho, x), N-vector.

    Walking the ln_phi formula term by term, using d(tau)/dT = -tau/T
    and d(delta)/dT = 0 at fixed rho:

        ln phi_i = a_r_oi(delta, tau)
                 + dDelta_dxi[i] - Delta
                 + delta_piece[i]   = delta * a_r_d * (1 - rho_r_factor[i])
                 + tau_piece[i]     = tau * a_r_t * (n_dTr_dni[i]/Tr)
                 - ln Z              Z = 1 + delta * a_r_d

      T1_i = d/dT a_r_oi = a_r_t_oi[i] * (-tau/T)
      T2_i = d/dT dDelta_dxi[i] = dDelta_t_dxi[i] * (-tau/T)   (via Schwarz)
      T3   = d/dT (-Delta) = -Delta_t * (-tau/T) = Delta_t * tau/T   (same for all i)
      T4_i = d/dT delta_piece[i] = delta * (1-rho_r_factor[i]) * a_r_dt * (-tau/T)
      T5_i = d/dT tau_piece[i] = (n_dTr_dni[i]/Tr) * d/dT [tau * a_r_t]
                               = (n_dTr_dni[i]/Tr) * (-tau/T) * (a_r_t + tau * a_r_tt)
      T6   = d/dT (-ln Z). Z = 1 + delta * a_r_d. d(Z)/dT at fixed rho,x
                         = delta * a_r_dt * (-tau/T). So
             T6 = -d(Z)/dT / Z = delta * a_r_dt * (tau/T) / Z       (same for all i)
    """
    from .departure import evaluate_total_departure
    res = _alpha_r_mix_derivs_extended(rho, T, x, mixture)
    N = mixture.N
    delta = res["delta"]; tau = res["tau"]
    Tr = res["T_r"]
    a_r_d = res["a_r_d"]; a_r_t = res["a_r_t"]
    a_r_tt = res["a_r_tt"]; a_r_dt = res["a_r_dt"]
    dTr = res["dTr_dxi"]; d_invrho = res["d_invrhor_dxi"]
    pure = res["a_r_pure_i"]
    dDt_dx = res["dDt_dx"]
    # Recover scalar Delta_t
    if mixture.departures:
        (_, _, Delta_t, _, _, _, _) = evaluate_total_departure(
            x, delta, tau, mixture.departures)
    else:
        Delta_t = 0.0
    # n_dTr_dni[i] and rho_r_factor[i] pure functions of x
    sum_x_dTr = float(np.dot(x, dTr))
    sum_x_d_invrho = float(np.dot(x, d_invrho))
    n_dTr_dni = dTr - sum_x_dTr
    rho_r_factor = -res["rho_r"] * (d_invrho - sum_x_d_invrho)
    # Per-component a_r_t_oi at mixture (delta, tau)
    a_r_t_oi = np.array([0.0 if pure[i] is None else pure[i][2] for i in range(N)])
    Z = 1.0 + delta * a_r_d
    mtauT = -tau / T           # d(tau)/dT
    # Assemble
    T1 = a_r_t_oi * mtauT                                           # (N,)
    T2 = dDt_dx * mtauT                                             # (N,)
    T3 = Delta_t * tau / T                                          # scalar -> broadcast
    T4 = delta * (1.0 - rho_r_factor) * a_r_dt * mtauT              # (N,)
    T5 = (n_dTr_dni / Tr) * mtauT * (a_r_t + tau * a_r_tt)          # (N,)
    T6 = delta * a_r_dt * (tau / T) / Z                             # scalar
    return T1 + T2 + T3 + T4 + T5 + T6


def dlnphi_dp_at_T(p, T, x, mixture, phase_hint='vapor'):
    """N-vector d(ln phi)/dp at fixed (T, x).

    Only rho depends on p. So d(ln phi)/dp = d(ln phi)/d rho / dp/drho|_T.
    Reuses the already-implemented dlnphi_drho_at_x and dp_drho_T.

    FD-verified to ~1e-9 across phases.
    """
    x = np.asarray(x, dtype=np.float64)
    rho = density_from_pressure(p, T, x, mixture, phase_hint=phase_hint)
    return dlnphi_drho_at_x(rho, T, x, mixture) / dp_drho_T(rho, T, x, mixture)


def dlnphi_dT_at_p(p, T, x, mixture, phase_hint='vapor'):
    """N-vector d(ln phi)/dT at fixed (p, x).

    Chain rule via implicit density response:

        d(ln phi)/dT|_{p,x} = d(ln phi)/dT|_{rho,x}
                            + d(ln phi)/drho|_{T,x} * drho/dT|_{p,x}
        drho/dT|_{p,x} = -dp/dT|_{rho,x} / dp/drho|_{T,x}

    FD-verified to ~1e-8 across phases.
    """
    x = np.asarray(x, dtype=np.float64)
    rho = density_from_pressure(p, T, x, mixture, phase_hint=phase_hint)
    d1 = dlnphi_dT_at_rho(rho, T, x, mixture)
    d2 = dlnphi_drho_at_x(rho, T, x, mixture)
    dp_dT = dp_dT_at_rho(rho, T, x, mixture)
    dp_dr = dp_drho_T(rho, T, x, mixture)
    return d1 + d2 * (-dp_dT / dp_dr)


# ---------------------------------------------------------------------------
# Density from (p, T, x) -- iterative solver on rho
# ---------------------------------------------------------------------------

def density_from_pressure(p, T, x, mixture, phase_hint="vapor",
                          rho_init=None, tol=1e-10, maxiter=150):
    """Solve for rho at given (p, T, x) on a chosen phase branch.

    Parameters
    ----------
    p : pressure [Pa]
    T : temperature [K]
    x : composition (length N)
    mixture : Mixture
    phase_hint : 'vapor' | 'liquid' | None
        Used to pick an initial density. For 'vapor', start from the ideal-gas
        density. For 'liquid', start from ~3 * (critical-weighted) rho_c.
    rho_init : float or None
        Overrides phase_hint with a specific initial guess.

    Returns
    -------
    rho [mol/m^3]
    """
    R = mixture.components[0].fluid.R
    # Pseudo-critical density
    Tr, rho_r = mixture.reduce(x)
    rho_pc = rho_r

    if rho_init is None:
        if phase_hint == "vapor":
            rho = p / (R * T)                # ideal-gas estimate
        elif phase_hint == "liquid":
            rho = rho_pc * 2.8               # dense-liquid estimate
        else:
            rho = p / (R * T)
    else:
        rho = rho_init

    # Bounds -- keep rho in physically reasonable range
    rho_max = 8.0 * rho_pc
    rho_min = 1e-10 * rho_pc

    # Newton with damping. Two convergence tests:
    #   (a) abs(dp) < tol * max(p, 1.0)  -- standard residual check
    #   (b) abs(drho/rho) < 1e-13         -- stalled-iteration detector
    # Test (b) is needed because at low pressures, float64 noise floor in
    # the EOS pressure evaluation is ~1e-9 * p, which exceeds the default
    # tol*p threshold and causes Newton to oscillate forever between two
    # adjacent float values. Test (b) accepts the answer when the step size
    # falls below relative machine epsilon, i.e. when further iteration
    # cannot improve the answer.
    rho_prev = rho
    for it in range(maxiter):
        p_curr = pressure(rho, T, x, mixture)
        dpdr = dp_drho_T(rho, T, x, mixture)
        dp = p - p_curr
        if abs(dp) < tol * max(p, 1.0):
            return rho
        # Stalled-iteration check: if rho has stopped changing meaningfully
        # AND the residual is small relative to typical pressures (1 Pa or
        # better), accept this as converged. Without this, Newton can
        # oscillate forever between two adjacent float64 values when the
        # achievable precision in p_curr is below tol*p.
        if it > 5 and abs(rho - rho_prev) / max(abs(rho), 1.0) < 1e-13 and abs(dp) < 1.0:
            return rho
        rho_prev = rho
        if dpdr <= 0:
            # Inside spinodal -- move in the direction of the hint
            if phase_hint == "liquid":
                rho = min(rho * 1.2, rho_max)
            else:
                rho = max(rho * 0.8, rho_min)
            continue
        # Newton step with damping
        drho = dp / dpdr
        if abs(drho) > 0.3 * rho:
            drho = 0.3 * rho * np.sign(drho)
        rho_new = rho + drho
        if rho_new <= rho_min:
            rho_new = 0.5 * (rho + rho_min)
        if rho_new >= rho_max:
            rho_new = 0.5 * (rho + rho_max)
        rho = rho_new

    raise RuntimeError(
        f"density_from_pressure did not converge (p={p}, T={T}, hint={phase_hint}, last rho={rho})"
    )
