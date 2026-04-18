"""
Core Numba-accelerated routines for evaluating the reduced Helmholtz energy
and its partial derivatives.

Conventions
-----------
    delta = rho / rho_c         reduced density
    tau   = T_c / T             inverse reduced temperature
    alpha = a / (R * T)         reduced Helmholtz energy
    alpha = alpha_0(delta, tau) + alpha_r(delta, tau)

The 6-tuple of derivatives returned everywhere is:
    (A, A_d, A_t, A_dd, A_tt, A_dt)
where A_d = d alpha / d delta, etc. These six values are sufficient to compute
all first- and second-order thermodynamic properties (cp, cv, sound speed,
Joule-Thomson coefficient, etc.).

Residual term types supported
-----------------------------
    type 1 -- polynomial:   n * delta^d * tau^t
    type 2 -- exponential:  n * delta^d * tau^t * exp(-delta^c)
    type 3 -- Gaussian:     n * delta^d * tau^t * exp(-eta*(delta-eps)^2
                                                      - beta*(tau-gamma)^2)

Ideal-gas term types supported
------------------------------
    'a1'         lead:         a1 + a2*tau          (added directly)
    'log_tau'    a * ln(tau)
    'log_delta'  ln(delta)  (coefficient implicitly 1; always present)
    'power_tau'  a * tau^b
    'PE'         Planck-Einstein: a * ln(1 - exp(-b*tau))    (b > 0)
    'PE_cosh'    a * ln( cosh(b*tau) )                       (GERG form)
    'PE_sinh'    a * ln(|sinh(b*tau)|)                       (GERG form)

These cover the vast majority of reference equations of state, including
Span-Wagner CO2, IAPWS-95 water, and the pure-fluid GERG-2008 entries.
"""
import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False
    def njit(*args, **kwargs):
        """No-op fallback when numba is unavailable."""
        # Support both @njit and @njit(cache=True, ...)
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def deco(fn):
            return fn
        return deco

# ---------------------------------------------------------------------------
# Residual part
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _alpha_r_polynomial(delta, tau, n, d, t):
    """Sum of n_i * delta^d_i * tau^t_i and its derivatives.

    Returns (A, A_d, A_t, A_dd, A_tt, A_dt).
    """
    A = 0.0
    A_d = 0.0
    A_t = 0.0
    A_dd = 0.0
    A_tt = 0.0
    A_dt = 0.0
    ln_delta = np.log(delta)
    ln_tau = np.log(tau)
    for i in range(n.shape[0]):
        # term = n * delta^d * tau^t
        term = n[i] * np.exp(d[i] * ln_delta + t[i] * ln_tau)
        A += term
        # delta derivatives bring down factors of d / delta
        A_d += d[i] * term
        A_dd += d[i] * (d[i] - 1.0) * term
        # tau derivatives bring down factors of t / tau
        A_t += t[i] * term
        A_tt += t[i] * (t[i] - 1.0) * term
        A_dt += d[i] * t[i] * term
    # Convert the "log-derivatives" (which carry factors of delta^k, tau^k) into
    # plain derivatives by dividing out powers of delta and tau.
    return (A,
            A_d / delta,
            A_t / tau,
            A_dd / (delta * delta),
            A_tt / (tau * tau),
            A_dt / (delta * tau))


@njit(cache=True, fastmath=False)
def _alpha_r_exponential(delta, tau, n, d, t, c):
    """Sum of n_i * delta^d_i * tau^t_i * exp(-delta^c_i).

    Uses the standard Span-Wagner derivative structure.
    """
    A = 0.0
    A_d = 0.0
    A_t = 0.0
    A_dd = 0.0
    A_tt = 0.0
    A_dt = 0.0
    ln_delta = np.log(delta)
    ln_tau = np.log(tau)
    for i in range(n.shape[0]):
        di = d[i]
        ti = t[i]
        ci = c[i]
        dc = delta ** ci
        # term = n * delta^d * tau^t * exp(-delta^c)
        term = n[i] * np.exp(di * ln_delta + ti * ln_tau - dc)
        A += term
        # d/d(delta): multiply by (d - c*delta^c) / delta
        f1 = di - ci * dc
        A_d += f1 * term / delta
        # d^2/d(delta)^2
        # second derivative factor (Span-Wagner eq. 7.16 style):
        #   (d - c*delta^c)*(d - 1 - c*delta^c) - c^2 * delta^c
        f2 = (di - ci * dc) * (di - 1.0 - ci * dc) - ci * ci * dc
        A_dd += f2 * term / (delta * delta)
        # tau derivatives are straightforward (the exp(-delta^c) factor is tau-independent)
        A_t += ti * term / tau
        A_tt += ti * (ti - 1.0) * term / (tau * tau)
        # cross derivative: t * (d - c*delta^c) / (delta*tau)
        A_dt += ti * f1 * term / (delta * tau)
    return (A, A_d, A_t, A_dd, A_tt, A_dt)


@njit(cache=True, fastmath=False)
def _alpha_r_gaussian(delta, tau, n, d, t, eta, eps, beta, gamma):
    """Gaussian-bell (non-analytic exponential) terms:

        n * delta^d * tau^t * exp( -eta*(delta-eps)^2 - beta*(tau-gamma)^2 )
    """
    A = 0.0
    A_d = 0.0
    A_t = 0.0
    A_dd = 0.0
    A_tt = 0.0
    A_dt = 0.0
    ln_delta = np.log(delta)
    ln_tau = np.log(tau)
    for i in range(n.shape[0]):
        di = d[i]
        ti = t[i]
        eta_i = eta[i]
        eps_i = eps[i]
        beta_i = beta[i]
        gamma_i = gamma[i]
        dd = delta - eps_i
        dt = tau - gamma_i
        exponent = di * ln_delta + ti * ln_tau - eta_i * dd * dd - beta_i * dt * dt
        term = n[i] * np.exp(exponent)
        A += term
        # First derivatives: ln(term) = d*ln(delta) + t*ln(tau) - eta*(delta-eps)^2 - beta*(tau-gamma)^2
        # d ln(term) / d(delta) = d/delta - 2*eta*(delta-eps)
        g_d = di / delta - 2.0 * eta_i * dd
        g_t = ti / tau - 2.0 * beta_i * dt
        A_d += term * g_d
        A_t += term * g_t
        # Second derivatives: (term)'' = term * (g' + g^2) where g' is derivative of g
        # d g_d / d(delta) = -d/delta^2 - 2*eta
        g_dd_prime = -di / (delta * delta) - 2.0 * eta_i
        g_tt_prime = -ti / (tau * tau) - 2.0 * beta_i
        A_dd += term * (g_dd_prime + g_d * g_d)
        A_tt += term * (g_tt_prime + g_t * g_t)
        # cross: d g_d / d(tau) = 0, so A_dt = term * g_d * g_t
        A_dt += term * g_d * g_t
    return (A, A_d, A_t, A_dd, A_tt, A_dt)


@njit(cache=True, fastmath=False)
def _alpha_r_nonanalytic(delta, tau, n, a, b, B, C, D, A_, beta):
    """Non-analytic near-critical terms of the IAPWS-95 / Span-Wagner form:

        phi_i = n_i * Delta^b * delta * psi

    where
        psi   = exp( -C (delta-1)^2 - D (tau-1)^2 )
        Delta = theta^2 + B ( (delta-1)^2 )^a
        theta = (1 - tau) + A ( (delta-1)^2 )^(1/(2 beta))

    All six derivatives follow from IAPWS-95 Table 5 (2018 Release, p. 13-14).

    The parameter names a, b, B, C, D, A_, beta match the release notation
    (``A_`` is used to avoid shadowing the accumulator ``A`` below).
    """
    A = 0.0
    A_d = 0.0
    A_t = 0.0
    A_dd = 0.0
    A_tt = 0.0
    A_dt = 0.0

    for i in range(n.shape[0]):
        ni   = n[i]
        ai   = a[i]
        bi   = b[i]
        Bi   = B[i]
        Ci   = C[i]
        Di   = D[i]
        Ai   = A_[i]
        beti = beta[i]

        # Core distances
        dm1  = delta - 1.0      # (delta - 1)
        tm1  = tau - 1.0        # (tau - 1)
        dm1_sq = dm1 * dm1      # (delta - 1)^2

        # psi = exp(-C (delta-1)^2 - D (tau-1)^2)
        psi = np.exp(-Ci * dm1_sq - Di * tm1 * tm1)

        # theta = (1 - tau) + A * ((delta-1)^2)^(1/(2 beta))
        # When delta=1 we have dm1_sq = 0 and the (dm1_sq)^x term vanishes.
        # When delta != 1, use pow; guard against FP anomalies at delta=1.
        if dm1_sq == 0.0:
            theta = -tm1   # = 1 - tau
        else:
            theta = -tm1 + Ai * dm1_sq ** (0.5 / beti)

        # Delta = theta^2 + B * ((delta-1)^2)^a
        # Same guard at delta=1
        if dm1_sq == 0.0:
            Delta = theta * theta
        else:
            Delta = theta * theta + Bi * dm1_sq ** ai

        # Delta^b  (only needed if Delta > 0; near delta=1 and tau=1, Delta -> 0
        # and Delta^b -> 0; contributions vanish cleanly.)
        if Delta == 0.0:
            # Term and all its derivatives are zero at exactly this point.
            # Any non-zero result here is a measure-zero edge case.
            continue
        Delta_b = Delta ** bi

        # ---- psi derivatives (IAPWS Table 5 footer) ----
        dpsi_dd   = -2.0 * Ci * dm1 * psi
        dpsi_dt   = -2.0 * Di * tm1 * psi
        d2psi_dd2 = (2.0 * Ci * dm1_sq - 1.0) * 2.0 * Ci * psi
        d2psi_dt2 = (2.0 * Di * tm1 * tm1 - 1.0) * 2.0 * Di * psi
        d2psi_ddt = 4.0 * Ci * Di * dm1 * tm1 * psi

        # ---- theta derivatives w.r.t. delta ----
        # dtheta/ddelta  (at delta != 1):
        #   theta = -tm1 + A * (dm1_sq)^(1/(2 beta))
        #   dtheta/d delta = A * (1/(2 beta)) * 2*(delta-1) * (dm1_sq)^(1/(2 beta) - 1)
        #                  = A * (1/beta) * (delta-1) * (dm1_sq)^(1/(2 beta) - 1)
        # Using the Release formulation in terms of (Delta^b) derivatives:
        #
        # From Table 5 (p. 14) footer:
        # d(Delta^b)/d(delta)   = b * Delta^(b-1) * dDelta/d(delta)
        # dDelta/d(delta)       = (delta-1) * [ A * theta * (2/beta) * (dm1_sq)^(1/(2 beta) - 1)
        #                                     + 2 B a (dm1_sq)^(a-1) ]
        # d^2 Delta/d(delta)^2  = (1/(delta-1)) * dDelta/d(delta)
        #                        + (delta-1)^2 * [4 B a (a-1) (dm1_sq)^(a-2)
        #                                         + 2 A^2 (1/beta)^2 ((dm1_sq)^(1/(2 beta) - 1))^2
        #                                         + A theta (4/beta) (1/(2 beta) - 1) (dm1_sq)^(1/(2 beta) - 2) ]
        # d^2 (Delta^b)/d(delta)^2 = b * [ Delta^(b-1) * d^2Delta/d(delta)^2
        #                                 + (b-1) Delta^(b-2) (dDelta/d(delta))^2 ]
        # d(Delta^b)/d(tau)         = -2 theta b Delta^(b-1)
        # d^2(Delta^b)/d(tau)^2     = 2 b Delta^(b-1) + 4 theta^2 b (b-1) Delta^(b-2)
        # d^2(Delta^b)/d(delta)d(tau) = -A b (2/beta) Delta^(b-1) (delta-1) (dm1_sq)^(1/(2 beta) - 1)
        #                               - 2 theta b (b-1) Delta^(b-2) dDelta/d(delta)

        inv_2beta = 0.5 / beti            # 1 / (2 beta)

        # pow exponents: carefully guard
        # (dm1_sq)^(1/(2 beta) - 1)  -- singular at delta=1 for beta < 0.5; not used if dm1_sq=0
        # (dm1_sq)^(a - 1)           -- singular at delta=1 for a < 1; not used if dm1_sq=0
        # (dm1_sq)^(a - 2), (dm1_sq)^(1/(2 beta) - 2): same guard

        if dm1_sq == 0.0:
            # At delta = 1, all terms with positive exponent of dm1_sq vanish.
            # Only the "2 B a (dm1_sq)^(a-1)" and similar pieces survive if a=1, etc.
            # For IAPWS-95 the exponents a >= 3.5, so these vanish cleanly.
            dDelta_dd = 0.0
            d2Delta_dd2 = 0.0
        else:
            pow_beta_m1 = dm1_sq ** (inv_2beta - 1.0)
            pow_a_m1    = dm1_sq ** (ai - 1.0)
            dDelta_dd   = dm1 * (
                Ai * theta * (2.0 / beti) * pow_beta_m1
                + 2.0 * Bi * ai * pow_a_m1
            )
            pow_beta_m2 = dm1_sq ** (inv_2beta - 2.0)
            pow_a_m2    = dm1_sq ** (ai - 2.0)
            d2Delta_dd2 = (
                (1.0 / dm1) * dDelta_dd
                + dm1_sq * (
                    4.0 * Bi * ai * (ai - 1.0) * pow_a_m2
                    + 2.0 * Ai * Ai * (1.0 / beti) * (1.0 / beti) * pow_beta_m1 * pow_beta_m1
                    + Ai * theta * (4.0 / beti) * (inv_2beta - 1.0) * pow_beta_m2
                )
            )

        # --- Delta^b derivatives ---
        # Delta^(b-1) and Delta^(b-2) if needed
        Delta_bm1 = Delta ** (bi - 1.0)
        Delta_bm2 = Delta ** (bi - 2.0)

        dDb_dd = bi * Delta_bm1 * dDelta_dd
        d2Db_dd2 = bi * (Delta_bm1 * d2Delta_dd2
                         + (bi - 1.0) * Delta_bm2 * dDelta_dd * dDelta_dd)
        dDb_dt = -2.0 * theta * bi * Delta_bm1
        d2Db_dt2 = 2.0 * bi * Delta_bm1 + 4.0 * theta * theta * bi * (bi - 1.0) * Delta_bm2
        if dm1_sq == 0.0:
            d2Db_ddt = 0.0
        else:
            d2Db_ddt = (
                -Ai * bi * (2.0 / beti) * Delta_bm1 * dm1 * pow_beta_m1
                - 2.0 * theta * bi * (bi - 1.0) * Delta_bm2 * dDelta_dd
            )

        # ---- phi = n * Delta^b * delta * psi ----
        # Using product rule (Delta^b) * (delta) * (psi):
        #
        # phi_delta = n * [ dDb_dd * delta * psi + Delta_b * psi + Delta_b * delta * dpsi_dd ]
        # phi_tau   = n * [ dDb_dt * delta * psi + Delta_b * delta * dpsi_dt ]
        # phi_dd    = n * [ d2Db_dd2 * delta * psi + 2 dDb_dd * psi + 2 dDb_dd * delta * dpsi_dd
        #                   + 2 Delta_b * dpsi_dd + Delta_b * delta * d2psi_dd2 ]
        # phi_tt    = n * [ d2Db_dt2 * delta * psi + 2 dDb_dt * delta * dpsi_dt + Delta_b * delta * d2psi_dt2 ]
        # phi_dt    = n * [ d2Db_ddt * delta * psi + dDb_dt * psi + dDb_dt * delta * dpsi_dd
        #                   + dDb_dd * delta * dpsi_dt + Delta_b * dpsi_dt + Delta_b * delta * d2psi_ddt ]

        phi     = ni * Delta_b * delta * psi
        phi_d   = ni * (dDb_dd * delta * psi + Delta_b * psi + Delta_b * delta * dpsi_dd)
        phi_t   = ni * (dDb_dt * delta * psi + Delta_b * delta * dpsi_dt)
        phi_dd  = ni * (d2Db_dd2 * delta * psi
                        + 2.0 * dDb_dd * psi
                        + 2.0 * dDb_dd * delta * dpsi_dd
                        + 2.0 * Delta_b * dpsi_dd
                        + Delta_b * delta * d2psi_dd2)
        phi_tt  = ni * (d2Db_dt2 * delta * psi
                        + 2.0 * dDb_dt * delta * dpsi_dt
                        + Delta_b * delta * d2psi_dt2)
        phi_dt  = ni * (d2Db_ddt * delta * psi
                        + dDb_dt * psi
                        + dDb_dt * delta * dpsi_dd
                        + dDb_dd * delta * dpsi_dt
                        + Delta_b * dpsi_dt
                        + Delta_b * delta * d2psi_ddt)

        A    += phi
        A_d  += phi_d
        A_t  += phi_t
        A_dd += phi_dd
        A_tt += phi_tt
        A_dt += phi_dt

    return (A, A_d, A_t, A_dd, A_tt, A_dt)


@njit(cache=True, fastmath=False)
def _alpha_r_kernel(delta, tau,
                    pn, pd, pt,                 # polynomial arrays
                    en, ed, et, ec,             # exponential arrays
                    gn, gd, gt, ge, geps, gb, ggam,   # gaussian arrays
                    na, naa, nb, nB, nC, nD, nA, nbeta):   # non-analytic arrays
    """Sum of all four residual term types.

    Non-analytic arg order matches pack(): (n, a, b, B, C, D, A, beta).
    """
    A1, A1d, A1t, A1dd, A1tt, A1dt = _alpha_r_polynomial(delta, tau, pn, pd, pt)
    A2, A2d, A2t, A2dd, A2tt, A2dt = _alpha_r_exponential(delta, tau, en, ed, et, ec)
    A3, A3d, A3t, A3dd, A3tt, A3dt = _alpha_r_gaussian(delta, tau, gn, gd, gt, ge, geps, gb, ggam)
    A4, A4d, A4t, A4dd, A4tt, A4dt = _alpha_r_nonanalytic(delta, tau, na, naa, nb, nB, nC, nD, nA, nbeta)
    return (A1 + A2 + A3 + A4,
            A1d + A2d + A3d + A4d,
            A1t + A2t + A3t + A4t,
            A1dd + A2dd + A3dd + A4dd,
            A1tt + A2tt + A3tt + A4tt,
            A1dt + A2dt + A3dt + A4dt)


# ---------------------------------------------------------------------------
# Ideal-gas part
# ---------------------------------------------------------------------------
#
# We encode each ideal-gas term as a (type_code, a, b) triplet so the whole
# ideal contribution can be represented with three flat float arrays. Codes:
#   0 = a1 lead:        a + 0              (b ignored)           contributes A += a
#   1 = a1 lead * tau:  a*tau              (b ignored)           (from a1 + a2*tau split)
#   2 = log_tau:        a * ln(tau)
#   3 = log_delta:      a * ln(delta)      (usually a = 1)
#   4 = power_tau:      a * tau^b
#   5 = PE:             a * ln(1 - exp(-b*tau))
#   6 = PE_cosh:        a * ln(cosh(b*tau))
#   7 = PE_sinh:        a * ln(|sinh(b*tau)|)

@njit(cache=True, fastmath=False)
def _alpha_0_kernel(delta, tau, codes, a_arr, b_arr):
    A = 0.0
    A_d = 0.0
    A_t = 0.0
    A_dd = 0.0
    A_tt = 0.0
    A_dt = 0.0  # cross-partials of alpha_0 vanish (it's separable in delta/tau)
    for i in range(codes.shape[0]):
        c = codes[i]
        a = a_arr[i]
        b = b_arr[i]
        if c == 0:
            # constant a (lead term, delta-independent and tau-independent)
            A += a
        elif c == 1:
            # a * tau
            A += a * tau
            A_t += a
        elif c == 2:
            # a * ln(tau)
            A += a * np.log(tau)
            A_t += a / tau
            A_tt += -a / (tau * tau)
        elif c == 3:
            # a * ln(delta)  (typically a = 1)
            A += a * np.log(delta)
            A_d += a / delta
            A_dd += -a / (delta * delta)
        elif c == 4:
            # a * tau^b
            tb = tau ** b
            A += a * tb
            A_t += a * b * tb / tau
            A_tt += a * b * (b - 1.0) * tb / (tau * tau)
        elif c == 5:
            # a * ln(1 - exp(-b*tau))
            e = np.exp(-b * tau)
            one_minus_e = 1.0 - e
            A += a * np.log(one_minus_e)
            # d/d(tau) of ln(1-e^{-b tau}) = b*e^{-b tau} / (1 - e^{-b tau})
            dA = a * b * e / one_minus_e
            A_t += dA
            # d^2/d(tau)^2 = -a * b^2 * e / (1 - e)^2
            A_tt += -a * b * b * e / (one_minus_e * one_minus_e)
        elif c == 6:
            # a * ln(cosh(b*tau)) -- first and second derivatives
            bt = b * tau
            # ln(cosh(bt))
            # use log(cosh) formula that avoids overflow
            if bt > 0.0:
                A += a * (bt + np.log(0.5 * (1.0 + np.exp(-2.0 * bt))))
            else:
                A += a * (-bt + np.log(0.5 * (1.0 + np.exp(2.0 * bt))))
            # d/d(tau) a*ln(cosh(bt)) = a*b*tanh(bt)
            A_t += a * b * np.tanh(bt)
            # d^2/d(tau)^2 = a*b^2*sech^2(bt) = a*b^2 * (1 - tanh^2)
            th = np.tanh(bt)
            A_tt += a * b * b * (1.0 - th * th)
        elif c == 7:
            # a * ln(|sinh(b*tau)|)
            bt = b * tau
            # ln(|sinh(bt)|)
            abs_bt = abs(bt)
            # sinh(x) = 0.5*(e^x - e^{-x}); for large |x|, ln|sinh| ~ |x| - ln2
            A += a * (abs_bt + np.log(0.5 * (1.0 - np.exp(-2.0 * abs_bt))))
            # d/d(tau) = a*b*coth(bt)
            th = np.tanh(bt)
            A_t += a * b / th
            # d^2/d(tau)^2 = -a*b^2*csch^2(bt) = -a*b^2*(1/sinh^2)
            sh = np.sinh(bt)
            A_tt += -a * b * b / (sh * sh)
        # (unknown codes silently skipped)
    return (A, A_d, A_t, A_dd, A_tt, A_dt)


# ---------------------------------------------------------------------------
# Public (still Numba-compiled) entry points
# ---------------------------------------------------------------------------

@njit(cache=True)
def alpha_r_derivs(delta, tau,
                   pn, pd, pt,
                   en, ed, et, ec,
                   gn, gd, gt, ge, geps, gb, ggam,
                   na, naa, nb, nB, nC, nD, nA, nbeta):
    """All six residual-part derivatives at (delta, tau)."""
    return _alpha_r_kernel(delta, tau,
                           pn, pd, pt,
                           en, ed, et, ec,
                           gn, gd, gt, ge, geps, gb, ggam,
                           na, naa, nb, nB, nC, nD, nA, nbeta)


@njit(cache=True)
def alpha_0_derivs(delta, tau, codes, a_arr, b_arr):
    """All six ideal-part derivatives at (delta, tau)."""
    return _alpha_0_kernel(delta, tau, codes, a_arr, b_arr)


@njit(cache=True)
def alpha_derivs(delta, tau,
                 pn, pd, pt,
                 en, ed, et, ec,
                 gn, gd, gt, ge, geps, gb, ggam,
                 na, naa, nb, nB, nC, nD, nA, nbeta,
                 codes, a_arr, b_arr):
    """Total alpha = alpha_0 + alpha_r and all derivatives."""
    R = _alpha_r_kernel(delta, tau,
                        pn, pd, pt,
                        en, ed, et, ec,
                        gn, gd, gt, ge, geps, gb, ggam,
                        na, naa, nb, nB, nC, nD, nA, nbeta)
    I = _alpha_0_kernel(delta, tau, codes, a_arr, b_arr)
    return (R[0] + I[0], R[1] + I[1], R[2] + I[2],
            R[3] + I[3], R[4] + I[4], R[5] + I[5])


@njit(cache=True)
def alpha(delta, tau,
          pn, pd, pt,
          en, ed, et, ec,
          gn, gd, gt, ge, geps, gb, ggam,
          na, naa, nb, nB, nC, nD, nA, nbeta,
          codes, a_arr, b_arr):
    """Scalar alpha(delta, tau) = alpha_0 + alpha_r."""
    return alpha_derivs(delta, tau,
                        pn, pd, pt,
                        en, ed, et, ec,
                        gn, gd, gt, ge, geps, gb, ggam,
                        na, naa, nb, nB, nC, nD, nA, nbeta,
                        codes, a_arr, b_arr)[0]


@njit(cache=True)
def alpha_r(delta, tau,
            pn, pd, pt,
            en, ed, et, ec,
            gn, gd, gt, ge, geps, gb, ggam,
            na, naa, nb, nB, nC, nD, nA, nbeta):
    """Scalar residual alpha_r(delta, tau)."""
    return _alpha_r_kernel(delta, tau,
                           pn, pd, pt,
                           en, ed, et, ec,
                           gn, gd, gt, ge, geps, gb, ggam,
                           na, naa, nb, nB, nC, nD, nA, nbeta)[0]


@njit(cache=True)
def alpha_0(delta, tau, codes, a_arr, b_arr):
    """Scalar ideal alpha_0(delta, tau)."""
    return _alpha_0_kernel(delta, tau, codes, a_arr, b_arr)[0]
