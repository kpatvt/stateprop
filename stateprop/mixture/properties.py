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
    res_args = pack[3:25]
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
    ideal_codes = pack[25]
    ideal_a = pack[26]
    ideal_b = pack[27]
    # Use the pure-component's own critical delta, tau
    delta_pure = rho / component.fluid.rho_c
    tau_pure = component.fluid.T_c / T
    return alpha_0_derivs(delta_pure, tau_pure, ideal_codes, ideal_a, ideal_b)


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
        ideal_codes, ideal_a, ideal_b = pack[25], pack[26], pack[27]
        d = alpha_0_derivs(delta_i, tau_i, ideal_codes, ideal_a, ideal_b)
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

    for it in range(maxiter):
        p_curr = pressure(rho, T, x, mixture)
        dpdr = dp_drho_T(rho, T, x, mixture)
        dp = p - p_curr
        if abs(dp) < tol * max(p, 1.0):
            return rho
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
